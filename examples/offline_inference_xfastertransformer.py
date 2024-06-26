from vllm import LLM
from vllm import SamplingParams

import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, top_p=0.8, top_k=20, max_tokens=16)

# Create an LLM.
llm = LLM(
    model="/data/llama-2-7b-chat-cpu",
    tokenizer="/data/llama-2-7b-chat-hf",
    trust_remote_code=True,
    max_num_batched_tokens=None,
    max_num_seqs=256,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")