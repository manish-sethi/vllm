from vllm import LLM, SamplingParams

import os
import time

import contextlib
import torch
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
)

model_path = '/net/storage149/autofs/css22/manish/models/Llama-2-13b-hf'
#model_path = '/nvme/manish/models/Llama-2-13b-hf'
#model_path = '/nvme/manish/models/Llama-7b-hf'
#model_path = '/nvme/manish/models/Llama-2-70b-hf'
tensor_parallel_size = 1
os.environ['USE_FASTSAFETENSOR'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

def print_gpus():
    import torch
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()

    # Print each GPU's name
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def drop_cache():
    total = 0
    filenames = [f for f in os.listdir(model_path) if not f.startswith('.')]
    print(f'filenames={filenames}')
    for filename in filenames:
        fd = os.open(f'{model_path}/{filename}', os.O_RDONLY)
        s = os.fstat(fd)
        os.posix_fadvise(fd, 0, s.st_size, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        print(f"DROP_CACHE: {filename}, {s.st_size/1024/1024/1024} GiB")
        total += s.st_size
    print(f"DROP_CACHE: {s.st_size/1024/1024/1024} GiB")


if __name__ == '__main__': # wrap under main for this bug during TP https://github.com/vllm-project/vllm/issues/5637
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    drop_cache()
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


    # Create an LLM.
    start_time = time.time()
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time model construction: {elapsed_time:.4f} seconds")

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    destroy_model_parallel()
    del llm.llm_engine.model_executor
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
