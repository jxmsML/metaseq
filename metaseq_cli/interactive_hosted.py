#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m metaseq_cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""

import os
import queue
import pkg_resources
import random
import threading
import traceback

import torch
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.queue import PriorityQueueRingShard
from metaseq.service.workers import WorkItem
from metaseq.service.constants import (
    MAX_SEQ_LEN,
    MAX_BATCH_TOKENS,
    MAX_BEAM,
    DEFAULT_PORT,
    TOTAL_WORLD_SIZE,
    LAUNCH_ARGS,
)
from metaseq.service.utils import get_my_ip, encode_fn, build_logger
from metaseq.service.responses import OAIResponse

import argparse
from metaseq_internal.projects.blenderbot3x.constants import MODEL_CONFIGS, MAX_BATCH_TOKENS
from metaseq_internal.projects.blenderbot3.workers import WorkItem

app = Flask(__name__)

# global state (mutable!)
cfg = None
port = DEFAULT_PORT
BATCH_QUEUE = PriorityQueueRingShard()

logger = build_logger()


def batching_loop(timeout=100, max_tokens=MAX_BATCH_TOKENS):
    """
    batching_loop is an infinite loop responsible for executing generations.

    GPUs benefit from batching requests, but we expect workloads to come
    in non-uniformly. This loop groups requests together (via BATCH_QUEUE)
    and executes them in one batch. In order to keep latency low, unfilled
    batches are executed within a window of :timeout: milliseconds.

    batching_loop also performs dynamic batching, in order to minimize the
    amount of padding by grouping like-sized workloads together. As a result
    batching loop will provide preferential treatment to smaller workloads.  At
    the current moment, there is no TTL logic to ensure a maximum wait time.

    For a rough overview of dynamic batching, see
    https://parl.ai/docs/tutorial_worlds.html#dynamic-batching.

    :param timeout: The max queue time before a non-full batch is launched.
    :param max_tokens: the maximum number of tokens that can be processed
        concurrently. model specific and empirical.
    """
    # TODO(roller):
    # - group by generation type, topp etc, as we cannot share these
    # - modify timeout logic to be cumulative
    global BATCH_QUEUE

    batch = []
    target_queue = None
    while True:
        try:
            # for now, we only have 1 worker, so can always index to shard 0
            if target_queue is None:
                target_queue = BATCH_QUEUE.queue_shards[0].get_largest_queue()
            if not target_queue:
                continue
            # dynamic batching: group like-sized items to reduce the cost
            # of padding. See PR#20 for additional context.
            item = target_queue.get(timeout=timeout / 1000)
            # accumulate the batch until it gets too big
            longest = max([item] + batch).cost
            batch_cost = longest * (len(batch) + 1)
            # overflow corresponds to whether max(prompt_len) + gen_len will
            # fit the max sequence length
            max_prompt_len = max(x.prompt_len for x in [item] + batch)
            max_gen_len = max(x.gen_len for x in [item] + batch)
            overflow = max_prompt_len + max_gen_len < MAX_SEQ_LEN
            if batch and (batch_cost > max_tokens or overflow):
                # we're over budget, put it back in the queue
                target_queue.put(item)
                raise queue.Empty
            else:
                # batch is empty or under budget
                batch.append(item)
        except queue.Empty:
            target_queue = None
            if batch:
                request_object = {
                    "inputs": [],
                    "min_tokens": [],
                    "max_tokens": [],
                }
                for work_item in batch:
                    ro = work_item.data
                    request_object["inputs"].append(ro["input"])
                    request_object["min_tokens"].append(ro.get("min_tokens", 0))
                    request_object["max_tokens"].append(
                        ro.get("max_tokens", MAX_SEQ_LEN)
                    )
                    # assumption: everyone has the same remaining args
                    for key in [
                        "temperature",
                        "top_p",
                        "n",
                        "best_of",
                        "echo",
                        "logprobs",
                        "stop",
                        "omega_bound",
                        "lambda_decay",
                        "alpha_presence",
                        "alpha_frequency",
                        "alpha_presence_src",
                        "alpha_frequency_src",
                        "alpha_src_penalty_end_idx",
                        "infer_mixing_weight",
                        "infer_gamma",
                    ]:
                        if key in ro:
                            request_object[key] = ro[key]
                # do the actual generations
                request_object["seed"] = random.randint(1, 20000)
                if torch.distributed.is_initialized():
                    distributed_utils.broadcast_object(
                        request_object,
                        src_rank=0,
                        group=distributed_utils.get_global_group(),
                    )
                try:
                    logger.info("Hi. I am about to call generations = generator.generate(**request_object)")
                    generations = generator.generate(**request_object)
                except RuntimeError:
                    # Probably cuda died. Unfortunately, we need to hard crash
                    # here to kick in our self-healing mechanisms.
                    raise
                except Exception as e:
                    # propagate any exceptions to the response so we can report it
                    logger.info(f"Hi. I received exception: {e} ")
                    generations = [e] * len(batch)
                # broadcast them back
                logger.info(f"Hi. Done with the generations in interactive hosted")
                for work_item, gen in zip(batch, generations):
                    work_item.return_queue.put((work_item.uid, gen))

                batch.clear()
            else:
                # back to the loop
                continue


def worker_main(cfg1: MetaseqConfig, namespace_args=None):
    # disable multithreading in tokenizers and torch, as different Flask threads
    # may then fight for resources.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    global generator
    global MODE

    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    MODE = "worker"
    cfg = cfg1

    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")
    if torch.distributed.is_initialized():
        request_object = distributed_utils.broadcast_object(
            None, src_rank=0, group=distributed_utils.get_global_group()
        )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(f"Worker engaged! {get_my_ip()}:{port}")
        # thread = threading.Thread(target=batching_loop, daemon=True)
        thread = threading.Thread(target=batching_loop, daemon=True, kwargs={'max_tokens': namespace_args.max_batch_toks})
        thread.start()
        app.run(host="0.0.0.0", port=port, threaded=True)
    else:
        # useful in FSDP setting
        logger.info(f"Looping engaged! {get_my_ip()}:{port}")
        while True:
            try:
                request_object = distributed_utils.broadcast_object(
                    None, src_rank=0, group=distributed_utils.get_global_group()
                )
                _ = generator.generate(**request_object)
            except Exception:
                # continue looping for the next generation so we don't lock up
                pass


@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    http_code = 400 if isinstance(e, ValueError) else 500
    return _create_error_response(
        str(e), http_code, stacktrace=traceback.format_tb(e.__traceback__)
    )


def _validate_key(key):
    # denylist a few placeholders various people have used
    if key == "":
        return False
    if "YOUR_NAME_HERE" in key:
        return False
    if "$USER" in key:
        return False
    if "your-key-here" in key:
        return False
    return True


def _create_error_response(msg, http_code, **others):
    error_dict = {
        "message": msg,
        "type": "invalid_request_error",
        "param": None,
        "code": None,
        **others,
    }
    response = jsonify({"error": error_dict})
    response.status = http_code
    return response


@app.route("/completions", methods=["POST"])
@app.route("/v1/engines/<engine>/completions", methods=["POST"])
@app.route("/v2/engines/<engine>/completions", methods=["POST"])
@app.route("/engines/<engine>/completions", methods=["POST"])
def completions(engine=None):
    # before anything else, check that we've got a valid API key
    if not _validate_key(request.headers.get("authorization", "")):
        return _create_error_response("Invalid API key or API key missing.", 401)

    # prompt can be 4 types:
    # - str. Basic case. Return one generation.
    # - list of ints. Pretokenized. Return one generation
    # - list of str. Multiple generations, one per prompt
    # - list of list of ints. Pretokenized multiple generations.

    # our approach is to turn everything into the last case

    prompts = request.json["prompt"]
    del request.json["prompt"]
    generation_args = request.json

    if isinstance(prompts, str):
        # single string. tokenize and turn it to the single pre-tokenized case
        prompts = [encode_fn(generator, prompts)]
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    if isinstance(prompts[0], str):
        # multi string
        prompts = [encode_fn(generator, p) for p in prompts]
    elif isinstance(prompts[0], int):
        # single pre-tokenized
        prompts = [prompts]
    assert isinstance(prompts[0], list)
    # final case: multi pre-tokenized
    assert len(prompts[0]) > 0

    if "min_tokens" in generation_args:
        generation_args["min_tokens"] = int(generation_args["min_tokens"])
    if "max_tokens" in generation_args:
        generation_args["max_tokens"] = int(generation_args["max_tokens"])
    if "stop" in generation_args:
        stop = generation_args["stop"]
        if stop is None:
            pass
        elif isinstance(stop, str):
            stop = [encode_fn(generator, stop)[0]]
        else:
            stop = [encode_fn(generator, s)[0] for s in stop]
        generation_args["stop"] = stop
    if "temperature" in generation_args:
        generation_args["temperature"] = round(float(generation_args["temperature"]), 1)
    else:
        generation_args["temperature"] = 1.0
    if "top_p" in generation_args:
        generation_args["top_p"] = round(float(generation_args["top_p"]), 1)
    else:
        generation_args["top_p"] = 1.0
    # beam search top n
    if "n" in generation_args:
        generation_args["n"] = min(MAX_BEAM, max(1, int(generation_args["n"])))
    else:
        generation_args["n"] = 1
    
    if "logprobs" in generation_args:
        generation_args["logprobs"] = int(generation_args["logprobs"])
    else:
        generation_args["logprobs"] = 0

    if "omega_bound" in generation_args:
        generation_args["omega_bound"] = round(float(generation_args["omega_bound"]), 1)
    else:
        generation_args["omega_bound"] = 0.3

    if "lambda_decay" in generation_args:
        generation_args["lambda_decay"] = round(float(generation_args["lambda_decay"]), 1)
    else:
        generation_args["lambda_decay"] = -1

    for key in ["alpha_frequency", "alpha_presence"]:
        for suffix in ["", "_src"]:
            _gen_arg = f"{key}{suffix}"
            if _gen_arg in generation_args:
                generation_args[_gen_arg] = round(float(generation_args[_gen_arg]), 1)
            else:
                generation_args[_gen_arg] = 0

    if "alpha_src_penalty_end_idx" in generation_args:
        generation_args["alpha_src_penalty_end_idx"] = int(generation_args["alpha_src_penalty_end_idx"])
    else:
        generation_args["alpha_src_penalty_end_idx"] = -1
    
    if "infer_mixing_weight" in generation_args:
        generation_args["infer_mixing_weight"] = round(float(generation_args["infer_mixing_weight"]), 3)
        assert generation_args["infer_mixing_weight"] <= 1.0
    else:
        logger.info("No infer_mixing_weight preset, set infer_mixing_weight to -1")
        generation_args["infer_mixing_weight"] = -1
    
    if "infer_gamma" in generation_args:
        generation_args["infer_gamma"] = round(float(generation_args["infer_gamma"]), 3)
    else:
        logger.info("No infer_gamma preset, set infer_gamma to -1")
        generation_args["infer_gamma"] = -1


    ret_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        gen_len = generation_args.get("max_tokens", 0)
        if gen_len + len(prompt) + 1 > MAX_SEQ_LEN:
            # cut off the prompt to always fit with number of generations we need
            # +1 to always have the EOS token
            prompt = prompt[-(MAX_SEQ_LEN - gen_len - 1) :]
        request_object = {"input": prompt, **generation_args}
        BATCH_QUEUE.put(
            WorkItem(
                cost=len(prompt) + gen_len,
                uid=i,
                return_queue=ret_queue,
                data=request_object,
                prompt_len=len(prompt),
                gen_len=gen_len,
            )
        )
    unordered_results = []
    for _ in prompts:
        unordered_results.append(ret_queue.get())
    # resort results by the original ordering
    # weirdly, openai returns to you a flat list if you gave multiple prompts
    reordered = sorted(unordered_results, key=lambda x: x[0])
    results = []
    for prompt, (_, generations) in zip(prompts, reordered):
        if isinstance(generations, Exception):
            raise generations
        results += generations
    # transform the result into the openai format
    return OAIResponse(results).__dict__()


@app.route("/")
def index():
    # TODO(roller): decouple demopage.html
    fn = pkg_resources.resource_filename("metaseq", "service/index.html")
    with open(fn) as f:
        return f.read()

def get_launch_args(base_launch_args, path, model_parallel_size, total_world_size, bf16, task, ddp_backend, distributed_port):
    launch_args = base_launch_args.copy()
    dp_size = total_world_size // model_parallel_size
    assert total_world_size % model_parallel_size == 0
    # proceed
    launch_args.insert(-2, f'--path {path}')
    launch_args.insert(-2, f'--model-parallel-size {model_parallel_size}')
    launch_args.insert(-2, f'--distributed-world-size {total_world_size}')
    launch_args.insert(-2, f'--task {task}')
    if dp_size > 1:
        launch_args.insert(-2, f'--use-sharded-state')
        launch_args.insert(-2, f'--load-checkpoint-on-all-dp-ranks')
    if bf16:
        launch_args.insert(-2, f'--fp16')
        launch_args.insert(-2, f'--bf16')
    else:
        launch_args.insert(-2, f'--memory-efficient-fp16')
    launch_args.insert(-2, f'--ddp-backend {ddp_backend}')
    launch_args.insert(-2, f'--distributed-port {distributed_port}')
    logger.warning(launch_args)
    return launch_args

def cli_main(model_size, config_key, max_toks, model_overrides, task='language_modeling', distributed_port=13000):
    """
    Hosted version of the web UI for generation.
    """
    launch_args = LAUNCH_ARGS.copy()
    total_world_size = TOTAL_WORLD_SIZE
    if model_size:
        checkpoint = MODEL_CONFIGS[model_size][config_key]['checkpoint']
        mp_size = MODEL_CONFIGS[model_size][config_key]['mp']
        dp_size = MODEL_CONFIGS[model_size][config_key].get('dp', 1)
        bf16 = MODEL_CONFIGS[model_size][config_key].get('bf16', False)
        total_world_size = mp_size * dp_size
        # proceed
        launch_args = get_launch_args(
            base_launch_args=launch_args,
            path=checkpoint,
            model_parallel_size=mp_size,
            total_world_size=total_world_size,
            bf16=bf16,
            task=task,
            ddp_backend=getattr(MODEL_CONFIGS[model_size][config_key], 'ddp_backend', 'pytorch_ddp'),
            distributed_port=distributed_port,
        )
    else:
        dp_size = 1
        raise RuntimeError('not allowed here')
        _copy_checkpoint_cache()

    global port, MODE, cfg
    parser = options.get_generation_parser()

    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    for s in launch_args:
        flat_launch_args += s.split()
    print(f'flat_launch_args = {flat_launch_args}')
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    args.max_batch_toks = max_toks
    args.arch = "transformer_lm_megatron"
    port = DEFAULT_PORT
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = total_world_size
    # inference_override
    if str(model_overrides) != '{}':
        cfg.common_eval.model_overrides = model_overrides
        logger.info(f'Model Overrides: {model_overrides}')
    distributed_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    # cli_main()
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        '--interactive-model-size',
        type=str
    )
    parser.add_argument(
        '--interactive-model-key',
        type=str
    )
    parser.add_argument(
        '--interactive-model-overrides',
        type=str
    )
    parser.add_argument(
        '--max-batch-tokens',
        type=int,
        default=MAX_BATCH_TOKENS
    )
    parser.add_argument(
        '--task',
        type=str,
        default="language_modeling"
    )
    parser.add_argument(
        '--dp-size',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--interactive-distributed-port',
        type=int,
        default=13000,
    )
    args = parser.parse_args()
    logger.warning(args)
    cli_main(args.interactive_model_size, args.interactive_model_key, args.max_batch_tokens, args.interactive_model_overrides, task=args.task, distributed_port=args.interactive_distributed_port)
