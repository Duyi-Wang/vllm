import sys
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser
import xfastertransformer



def parse_args():
    parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.model)
    print(args.dtype)
    print(args.kv_cache_dtype)

    model = xfastertransformer.AutoModel.from_pretrained(
        args.model, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype
    )

    if model.rank == 0:
        print("[ERROR] llm.entrypoints.slave shouldn't be Master!")
        sys.exit(0)

    while True:
        model.set_input_cb()
        model.forward_cb()
        model.free_seqs()
