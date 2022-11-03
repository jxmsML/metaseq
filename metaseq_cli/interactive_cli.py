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
import random
import sys
import logging

import torch

from metaseq import options
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.constants import (
    TOTAL_WORLD_SIZE,
    LAUNCH_ARGS,
    MAX_BATCH_TOKENS,
)
from metaseq.service.utils import encode_fn, build_logger
from metaseq_cli.interactive_hosted import get_launch_args

logger = build_logger()


def input_loop():
    inp = []
    while False:
        try:
            # green display, bold user prompt
            display = (
                "\033[32mPrompt (ctrl-D to end input, ctrl-C to quit):\n\033[0;1m"
                if not inp
                else ""
            )
            data = input(display)
            inp.append(data)
        except KeyboardInterrupt:
            # reset the formatting
            sys.stdout.write("\033[0m")
            raise
        except EOFError:
            break
        # reset the formatting
        sys.stdout.write("\033[0m")
    inp = ['How are you today?']
    logger.debug(f"Input: {inp}")
    return "\n".join(inp)


def worker_main(cfg: MetaseqConfig, namespace_args=None):
    global generator
    # quiet some of the stuff for visual aspects
    logging.getLogger("metaseq.hub_utils").setLevel(logging.WARNING)

    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))

    logger.info(f"begin loading model {cfg.distributed_training.distributed_rank}")

    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

    # quiet some of the stuff for visual aspects
    logging.getLogger("metaseq.hub_utils").setLevel(logging.WARNING)

    logger.info(f"loaded model {cfg.distributed_training.distributed_rank}")
    request_object = distributed_utils.broadcast_object(
        None, src_rank=0, group=distributed_utils.get_global_group()
    )
    logger.info(f"request_object DONE")
    if torch.distributed.get_rank() == 0:
        while True:
            prompt = input_loop()
            logger.info(f'prompt = {prompt}, start generation')
            tokens = encode_fn(generator, prompt)
            logger.info(f'-- done encoding start generation')
            request_object = {
                "inputs": [tokens],
                "max_tokens": [128],
            }
            distributed_utils.broadcast_object(
                request_object, src_rank=0, group=distributed_utils.get_global_group()
            )
            logger.info(f'-- about to call generator.generate with {request_object}')
            generations = generator.generate(**request_object)
            print(generations[0][0]["text"])
    else:
        # useful in FSDP setting
        while True:
            request_object = distributed_utils.broadcast_object(
                None, src_rank=0, group=distributed_utils.get_global_group()
            )
            _ = generator.generate(**request_object)


def cli_main():
    """
    Command line interactive.
    """
    parser = options.get_generation_parser()

    parser.add_argument(
        '--max-batch-tokens',
        type=int,
    )
    parser.add_argument(
        '--dp-size',
        type=int,
        default="1"
    )
    parser.add_argument(
        '--use-sharded-states',
        type=bool,
        default=None
    )

    cmd_args = parser.parse_args()
    checkpoint = cmd_args.path
    mp_size = cmd_args.model_parallel_size
    dp_size = getattr(cmd_args, 'dp_size', 1)  # assuming dp_size = 1 by default    
    total_world_size = mp_size * dp_size
    logger.warning(f'mp_size = {mp_size}, dp_size = {dp_size}, total_world_size = {total_world_size}')

    bf16 = getattr(cmd_args, 'bf16', False)
    # proceed
    launch_args = get_launch_args(
        base_launch_args=LAUNCH_ARGS,
        path=checkpoint,
        model_parallel_size=mp_size,
        total_world_size=total_world_size,
        bf16=bf16,
        task=getattr(cmd_args, 'task', 'language_modeling'),
        ddp_backend=getattr(cmd_args, 'ddp_backend', 'pytorch_ddp'),
        distributed_port=getattr(cmd_args, 'ddp_backend', 'distributed_port'),
    )
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    for s in launch_args:
        flat_launch_args += s.split()
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    MAX_BATCH_TOKENS = 2048
    args.max_batch_toks = getattr(cmd_args, 'max_batch_tokens', MAX_BATCH_TOKENS)
    args.arch = "transformer_lm_megatron"
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = total_world_size
    if getattr(cmd_args, 'use_sharded_states', None) is not None:
        cfg.distributed_training.use_sharded_state = cmd_args.use_sharded_states
    cfg.model.tensor_parallel_init_model_on_gpu = True
    if getattr(cmd_args, 'model_overrides', None):
        cfg.common_eval.model_overrides = cmd_args.model_overrides
    distributed_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    cli_main()

"""
python -m metaseq_cli.interactive_cli \
 --path /checkpoint/jingxu23/projects/bb3/models/BB3_models/08_30_2022_aws_from_pt_6/reshard_checkpoint3_mp2/reshard.pt  \
 --model-parallel-size 2 \
 --task language_modeling # language_modeling_inference_for_models_trained_with_streaming


python -m metaseq_cli.interactive_cli \
 --path /checkpoint/jingxu23/projects/bb3/models/BB3_models/08_30_2022_aws_from_pt_6/consolidated_checkpoint3_mp1/consolidated.pt  \
 --model-parallel-size 1 \
 --task language_modeling


python -m metaseq_cli.interactive_cli \
 --path /private/home/jingxu23/src_metaseqpy38_20220328/metaseq/3b_opt/reshard.pt  \
 --model-parallel-size 4 \
 --task language_modeling \
 --use-sharded-states True \


python -m metaseq_cli.interactive_cli \
 --path /checkpoint/jingxu23/projects/bb3/3B_OPT/reshard_mp2/reshard.pt  \
 --model-parallel-size 2 \
 --task language_modeling

 python -m metaseq_cli.interactive_cli \
 --path /checkpoint/jingxu23/projects/bb3/models/3B_OPT/reshard_mp4_ddp4/reshard/reshard.pt  \
 --model-parallel-size 2 \
 --task language_modeling

 # director
 python -m metaseq_cli.interactive_cli \
 --path /checkpoint/jingxu23/projects/bb3/FT_BASE_MODELS_1/pt_director.lm_and_classification_cross_entropy.streaming_finetune_language_modeling_and_classification.adam.endlr0.ms8.edclinear.2.7b.ngpu8/reshard_checkpoint_last_mp2/reshard.pt  \
 --model-parallel-size 2 \
 --task language_modeling \
 --model-overrides '{"infer_mixing_weight": 0.45}'

 python -m metaseq_cli.interactive_cli \
 --path /checkpoint/jingxu23/projects/bb3/FT_BASE_MODELS/pt_director.fp16adam.edclinear.3b.ngpu8/reshard_checkpoint_last_mp2/reshard
/reshard.pt  \
 --model-parallel-size 2 \
 --task language_modeling


python3 -m metaseq_internal.scripts.launch_api_helper \
 --interactive-model-size 175b \
 --interactive-model-key 07_12_2022_aws_from_pt_23_checkpoint1 \
 --max-batch-tokens 2048
"""
