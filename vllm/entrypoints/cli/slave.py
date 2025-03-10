import argparse
from typing import List
import sys

import xfastertransformer

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


class SlaveSubcommand(CLISubcommand):
    """The `slave` subcommand for the vLLM-xft CLI."""

    def __init__(self):
        self.name = "serve"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        print("Starting slave rank")        
        print(f"model: {args.model}")
        print(f"dtype: {args.dtype}")
        print(f"kv_cache_dtype: {args.kv_cache_dtype}") 

        # model = xfastertransformer.AutoModel.from_pretrained(
        #     args.model, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype
        # )

        # if model.rank == 0:
        #     print("Error: slave shouldn't be rank 0")
        #     sys.exit(0)

        # while True:
        #     model.set_input_cb()
        #     model.forward_cb()
        #     model.free_seqs()

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "slave", help="Start the salve rank for xfastertransformer"
        )
        return make_arg_parser(serve_parser)


def cmd_init() -> List[CLISubcommand]:
    return [SlaveSubcommand()]
