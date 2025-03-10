import argparse
from typing import List
import sys

import xfastertransformer

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


class MasterSubcommand(CLISubcommand):
    """The `master` subcommand for the vLLM-xft CLI."""

    def __init__(self):
        self.name = "serve"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        print("Starting master rank")
        print(f"input_path: {args.input_path}")
        print(f"output_path: {args.output_path}")

    def validate(self, args: argparse.Namespace) -> None:
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            "master", help="Start the master rank for xfastertransformer"
        )
        serve_parser.add_argument("--input_path", type=str, help="ZMQ input path")
        serve_parser.add_argument("--output_path", type=str, help="ZMQ output path")
        return make_arg_parser(serve_parser)


def cmd_init() -> List[CLISubcommand]:
    return [MasterSubcommand()]
