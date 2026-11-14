import os
from multiprocessing import freeze_support
from vllm import LLM, SamplingParams


MODEL_PATH = os.environ.get("QWEN05B_PATH")

if not MODEL_PATH:
    raise RuntimeError(
        "QWEN05B_PATH is empty. Please run: source env.sh, "
        "and make sure the model has been downloaded."
    )


def main():
    print("Using local model path:")
    print(MODEL_PATH)

    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.55,
        enable_prefix_caching=True,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
    )

    prompts = [
        "请用一句话解释什么是 KV Cache。",
        "请用一句话解释什么是 Prefix Cache。",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print("=" * 60)
        print("Prompt:", output.prompt)
        print("Output:", output.outputs[0].text.strip())


if __name__ == "__main__":
    freeze_support()
    main()
