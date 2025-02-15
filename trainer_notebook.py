# %%
import argparse
import gc
import platform
import psutil
import time
import os
from typing import List, Dict, Tuple, Optional

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM, LlamaSkipMLP, FastLoRAProjection
from src.models.configuration_llama_skip import LlamaSkipConnectionConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLaMA model inference benchmark')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                      help='Device to run inference on')
    parser.add_argument('--num_runs', type=int, default=50,
                      help='Number of inference runs')
    parser.add_argument('--verbose', type=bool, default=False,
                      help='Verbose output')
    parser.add_argument('--config', type=str, default='configs/llama_skip_causal_3b.json',
                      help='Config file')
    return parser.parse_args()


def get_gpu_info() -> Optional[List[Dict]]:
    """Get GPU information if CUDA is available."""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_info = torch.cuda.mem_get_info(i)
        free_memory = mem_info[0] / 1024**3  # Convert to GB
        total_memory = mem_info[1] / 1024**3  # Convert to GB
        gpu_info.append({
            'name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory': f"{total_memory:.2f}GB",
            'free_memory': f"{free_memory:.2f}GB",
            'multi_processor_count': props.multi_processor_count
        })
    return gpu_info


def get_system_info() -> Dict:
    """Get system information including CPU and RAM details."""
    cpu_info = {
        'processor': platform.processor(),
        'physical_cores': psutil.cpu_count(logical=False),
        'total_cores': psutil.cpu_count(logical=True),
        'max_frequency': f"{psutil.cpu_freq().max:.0f}MHz" if psutil.cpu_freq() else "Unknown",
        'current_frequency': f"{psutil.cpu_freq().current:.0f}MHz" if psutil.cpu_freq() else "Unknown"
    }
    
    memory = psutil.virtual_memory()
    ram_info = {
        'total': f"{memory.total / (1024**3):.2f}GB",
        'available': f"{memory.available / (1024**3):.2f}GB",
        'used_percent': f"{memory.percent}%"
    }
    
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'cpu': cpu_info,
        'ram': ram_info
    }


def print_system_info(args: argparse.Namespace) -> None:
    """Print system configuration information."""
    print("\nSystem Configuration:")
    print("-" * 50)
    system_info = get_system_info()
    print(f"OS: {system_info['system']} {system_info['release']}")
    print(f"CPU: {system_info['cpu']['processor']}")
    print(f"Physical cores: {system_info['cpu']['physical_cores']}")
    print(f"Total cores: {system_info['cpu']['total_cores']}")
    print(f"Max CPU frequency: {system_info['cpu']['max_frequency']}")
    print(f"Current CPU frequency: {system_info['cpu']['current_frequency']}")
    print(f"RAM: Total={system_info['ram']['total']}, Available={system_info['ram']['available']} ({system_info['ram']['used_percent']} used)")

    if args.device == 'cuda':
        print("\nGPU Configuration:")
        print("-" * 50)
        gpu_info = get_gpu_info()
        for i, gpu in enumerate(gpu_info or []):
            print(f"\nGPU {i}: {gpu['name']}")
            print(f"Compute capability: {gpu['compute_capability']}")
            print(f"Total memory: {gpu['total_memory']}")
            print(f"Free memory: {gpu['free_memory']}")
            print(f"Multi processors: {gpu['multi_processor_count']}")

    print("\nPyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
    print("-" * 50)


def setup_torch_optimizations() -> None:
    """Enable TorchScript optimizations."""
    torch.jit.enable_onednn_fusion(True)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(True)
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)


def setup_devices(args: argparse.Namespace) -> Tuple[torch.device, torch.device, torch.device]:
    """Setup devices for model inference."""
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    if args.device == 'cuda':
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        if num_gpus > 1:
            standard_device = torch.device('cuda:1')
            skip_device = torch.device('cuda:0')
        else:
            standard_device = skip_device = device
    else:
        standard_device = skip_device = device
    
    return device, standard_device, skip_device


def create_pipeline(model, tokenizer: AutoTokenizer, device: torch.device, **kwargs):
    """Create a text generation pipeline."""
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=1000,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs
    )


def setup_cuda_debugging(verbose: bool = False):
    """Setup CUDA debugging flags."""
    # Enable CUDA launch blocking for synchronous error reporting
    torch.cuda.set_device(0)  # Set primary device
    
    if verbose:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # Enable CUDA memory stats
        torch.cuda.memory.set_per_process_memory_fraction(0.9)  # Leave some GPU memory free
        torch.cuda.memory._record_memory_history(max_entries=10000)


def run_inference(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    model_device: Optional[torch.device],
    num_runs: int = 50,
    verbose: bool = False
) -> Tuple[List[float], List[float]]:
    """Run inference benchmark on the model."""    
    # Setup model on device
    model = model.to(model_device).to(torch.float16 if model_device.type == 'cuda' else torch.float32)
    base_input_ids = input_ids.to(model_device)
    base_attention_mask = attention_mask.to(model_device)
    
    print(f"\nModel type: {type(model)}")
    print(f"Model device: {model_device}")
    print(f"Model path: {model.config._name_or_path}")
    
    times = []
    mlp_times = []

    # Add MLP timing hooks if needed
    if model_device.type == 'cpu' and verbose:
        def forward_hook(module, input, output):
            start = time.perf_counter()
            result = module.forward(*input)
            end = time.perf_counter()
            mlp_times.append(end - start)
            return result
            
        for module in model.modules():
            if isinstance(module, (LlamaSkipMLP, LlamaMLP)):
                module.register_forward_hook(forward_hook)

    # Warmup for CUDA
    if model_device.type == 'cuda':
        for _ in range(3):
            with torch.amp.autocast(device_type='cuda'):
                with torch.no_grad():
                    _ = model(base_input_ids, attention_mask=base_attention_mask, return_dict=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    gc.collect()

    # Main inference loop
    for i in range(num_runs):
        random_shift = torch.randint(-100, 100, base_input_ids.shape, device=model_device)
        input_ids = torch.clamp(base_input_ids + random_shift, min=0, max=tokenizer.vocab_size-1)
        
        # Reset model state
        if hasattr(model, 'past_key_values'):
            model.past_key_values = None
        model._past_length = 0
        model.config.use_cache = False

        if model_device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            if model_device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    _ = model(input_ids, attention_mask=base_attention_mask, return_dict=False)
            else:
                _ = model(input_ids, attention_mask=base_attention_mask, return_dict=False)
        
        if model_device.type == 'cuda':
            torch.cuda.synchronize()
            
        times.append(time.perf_counter() - start)
        
        # Periodic cache clearing
        if i % 10 == 0:
            if model_device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    return times, mlp_times


def print_results(name: str, times: List[float], device_type: str) -> None:
    """Print benchmark results."""
    print(f"\n{name} {device_type} Results:")
    print(f"Average time: {sum(times)/len(times):.3f}s")
    print(f"Min time: {min(times):.3f}s")
    print(f"Max time: {max(times):.3f}s")
    print(f"Individual times: {[f'{t:.3f}s' for t in times]}")


def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    # Setup CUDA debugging if using CUDA
    if args.device == 'cuda':
        setup_cuda_debugging(verbose=args.verbose)
    
    setup_torch_optimizations()
    print_system_info(args)
    
    device, standard_device, skip_device = setup_devices(args)
    print(f"Using devices: {device}, {standard_device}, {skip_device}")

    # Register custom models
    AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
    AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)

    # Load models and tokenizer
    config = LlamaSkipConnectionConfig.from_json_file(args.config)
    checkpoint = config._name_or_path
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    scripted_model = LlamaSkipConnectionForCausalLM.from_pretrained(
        checkpoint, 
        config=config
    ).to(skip_device)

    # Script CPU modules if needed
    if args.device == 'cpu':
        for module in scripted_model.modules():
            if isinstance(module, (LlamaSkipMLP, FastLoRAProjection)):
                module.eval()
                try:
                    scripted_module = torch.jit.script(module)
                    module.forward = scripted_module.forward
                except Exception as e:
                    print(f"Failed to script module {type(module).__name__}: {str(e)}")

    # Prepare input
    sequence = "Give recipe of burrito including all the ingredients and their quantity."
    inputs = tokenizer(
        sequence, 
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )

    # Create pipelines
    llamaSkipScriptedPipe = create_pipeline(scripted_model, tokenizer, skip_device)
    llamaPipe = create_pipeline(checkpoint, tokenizer, standard_device)

    # Run benchmarks
    print(f"\nRunning {args.device.upper()} inference benchmarks...")
    print("-" * 50)

    print("Warming up models...")
    _, _ = run_inference(llamaPipe.model, inputs["input_ids"], inputs["attention_mask"], 
                        tokenizer, standard_device, num_runs=2, verbose=args.verbose)
    _, _ = run_inference(llamaSkipScriptedPipe.model, inputs["input_ids"], inputs["attention_mask"], 
                        tokenizer, skip_device, num_runs=2, verbose=args.verbose)

    skip_scripted_times, skip_scripted_mlp_times = run_inference(
        llamaSkipScriptedPipe.model, inputs["input_ids"], inputs["attention_mask"], 
        tokenizer, skip_device, args.num_runs, args.verbose
    )

    std_times, std_mlp_times = run_inference(
        llamaPipe.model, inputs["input_ids"], inputs["attention_mask"], 
        tokenizer, standard_device, args.num_runs, args.verbose
    )

    # Print results
    if device.type == 'cpu' and args.verbose:
        print_results("SkipLLaMA Scripted MLP", skip_scripted_mlp_times, args.device.upper())
        print_results("Standard LLaMA MLP", std_mlp_times, args.device.upper())

    print_results("SkipLLaMA Scripted", skip_scripted_times, args.device.upper())
    print_results("Standard LLaMA", std_times, args.device.upper())
    
    speedup = (sum(std_times)/len(std_times))/(sum(skip_scripted_times)/len(skip_scripted_times))
    print(f"\n{args.device.upper()} Speedups:")
    print(f"Scripted vs Standard: {speedup:.2f}x")


if __name__ == "__main__":
    main()
