# %%
import argparse
import gc
import platform
import psutil
import time
import os
import threading
import statistics
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import subprocess
import json
import tempfile

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
from src.models.llama.modelling_llama_skip import LlamaSkipConnectionForCausalLM, LlamaSkipMLP, FastLoRAProjection
from src.models.llama.configuration_llama_skip import LlamaSkipConnectionConfig


class GPUMonitor:
    """Monitor GPU usage during inference."""
    
    def __init__(self, monitoring_interval: float = 0.1):
        self.monitoring_interval = monitoring_interval
        self._gpu_memory_usage = []
        self._gpu_utilization = []
        self._is_monitoring = False
        self._monitor_thread = None
        
    def _monitor_gpu(self):
        """Background monitoring of GPU metrics."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            while self._is_monitoring:
                # Get memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = memory_info.used / 1024**2
                
                # Get utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                
                self._gpu_memory_usage.append(memory_used_mb)
                self._gpu_utilization.append(gpu_util)
                
                time.sleep(self.monitoring_interval)
                
        except ImportError:
            # Fallback to torch methods if pynvml not available
            while self._is_monitoring:
                if torch.cuda.is_available():
                    memory_used_mb = torch.cuda.memory_allocated() / 1024**2
                    self._gpu_memory_usage.append(memory_used_mb)
                time.sleep(self.monitoring_interval)
    
    def start(self):
        """Start GPU monitoring."""
        self._is_monitoring = True
        self._gpu_memory_usage.clear()
        self._gpu_utilization.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_gpu)
        self._monitor_thread.start()
    
    def stop(self):
        """Stop GPU monitoring."""
        self._is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def get_peak_usage(self) -> Dict:
        """Get peak GPU usage metrics."""
        if not self._gpu_memory_usage:
            return {"peak_gpu_memory_mb": 0, "p90_gpu_utilization": 0}
        
        return {
            "peak_gpu_memory_mb": max(self._gpu_memory_usage),
            "p90_gpu_memory_mb": np.percentile(self._gpu_memory_usage, 90),
            "max_gpu_utilization": max(self._gpu_utilization) if self._gpu_utilization else 0,
            "p90_gpu_utilization": np.percentile(self._gpu_utilization, 90) if self._gpu_utilization else 0
        }


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
    parser.add_argument('--compare_with_llamacpp', action='store_true',
                      help='Compare with llama.cpp implementation instead of HuggingFace')
    parser.add_argument('--llamacpp_model_path', type=str, default=None,
                      help='Path to GGUF model file for llama.cpp')
    parser.add_argument('--llamacpp_binary', type=str, default='llama.cpp/build/bin/llama-cli',
                      help='Path to llama.cpp binary (default: llama.cpp/build/bin/llama-cli)')
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


def get_diverse_test_prompts() -> List[Dict[str, Union[str, int]]]:
    """Get diverse test prompts for comprehensive benchmarking."""
    return [
        {
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "description": "Short simple prompt"
        },
        {
            "prompt": "Give me a detailed recipe for making a burrito including all ingredients and their quantities.",
            "max_tokens": 200,
            "description": "Medium recipe prompt"
        },
        {
            "prompt": "Explain the concept of machine learning, its applications in modern technology, and how it differs from traditional programming approaches. Include examples.",
            "max_tokens": 300,
            "description": "Long technical explanation"
        },
        {
            "prompt": "Write a short story about a robot discovering emotions.",
            "max_tokens": 400,
            "description": "Creative writing prompt"
        },
        {
            "prompt": "Analyze the economic implications of artificial intelligence on job markets, considering both positive and negative effects, and suggest potential policy responses.",
            "max_tokens": 500,
            "description": "Complex analytical prompt"
        }
    ]


def reset_model_state(model: AutoModelForCausalLM):
    """Reset model state between independent sequences."""
    # Clear past key values cache
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None
    
    # Reset internal state
    if hasattr(model, '_past_length'):
        model._past_length = 0
    
    # Disable caching for fresh inference
    model.config.use_cache = False

    # Clear GPU cache if using CUDA
    if next(model.parameters()).device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Trigger garbage collection
    gc.collect()


def benchmark_single_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.7
) -> Dict:
    """Benchmark a single prompt for TTFT and TPS metrics."""
    
    # Reset model state for independent sequence
    reset_model_state(model)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    input_tokens = input_ids.shape[1]
    
    # Setup GPU monitoring
    gpu_monitor = None
    if device.type == 'cuda':
        gpu_monitor = GPUMonitor()
        gpu_monitor.start()
    
    # Setup CUDA events for accurate timing
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        first_token_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        start_time = time.perf_counter()
    
    # Generate tokens with streaming to measure TTFT
    generated_tokens = []
    first_token_time = None
    
    model.eval()
    with torch.no_grad():
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        
        for step in range(max_new_tokens):
            # Forward pass
            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(current_input_ids, attention_mask=current_attention_mask, return_dict=True)
            else:
                outputs = model(current_input_ids, attention_mask=current_attention_mask, return_dict=True)
            
            # Get next token
            logits = outputs.logits[:, -1, :]
            
            # Apply temperature sampling
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Record first token time
            if step == 0:
                if device.type == 'cuda':
                    first_token_event.record()
                    torch.cuda.synchronize()
                else:
                    first_token_time = time.perf_counter() - start_time
            
            generated_tokens.append(next_token.item())
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Update input for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            current_attention_mask = torch.cat([
                current_attention_mask, 
                torch.ones(1, 1, device=device, dtype=current_attention_mask.dtype)
            ], dim=-1)
    
    # Record end time
    if device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate times using CUDA events (in milliseconds)
        total_time_ms = start_event.elapsed_time(end_event)
        first_token_time_ms = start_event.elapsed_time(first_token_event)
        generation_time_ms = first_token_event.elapsed_time(end_event)
        
        # Convert to seconds
        total_time = total_time_ms / 1000
        first_token_time = first_token_time_ms / 1000
        generation_time = generation_time_ms / 1000
    else:
        total_time = time.perf_counter() - start_time
        generation_time = total_time - first_token_time
    
    # Stop GPU monitoring
    gpu_usage = {}
    if gpu_monitor:
        gpu_monitor.stop()
        gpu_usage = gpu_monitor.get_peak_usage()
    
    output_tokens = len(generated_tokens)
    total_tokens = input_tokens + output_tokens
    
    # Calculate metrics
    results = {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'time_to_first_token_seconds': first_token_time,
        'total_generation_time_seconds': generation_time,
        'total_time_seconds': total_time,
        'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
        'output_tokens_per_second': output_tokens / generation_time if generation_time > 0 else 0,
        'input_tokens_per_second': input_tokens / first_token_time if first_token_time > 0 else 0,
        **gpu_usage
    }
    
    return results


def benchmark_language_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_prompts: List[Dict],
    device: torch.device,
    temperature: float = 0.7
) -> Dict:
    """Benchmark language model across multiple prompts."""
    
    all_results = []
    
    print(f"\nRunning comprehensive benchmark on {len(test_prompts)} prompts...")
    
    for i, prompt_config in enumerate(test_prompts):
        print(f"\nPrompt {i+1}/{len(test_prompts)}: {prompt_config['description']}")
        print(f"Max tokens: {prompt_config['max_tokens']}")
        
        try:
            result = benchmark_single_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_config['prompt'],
                max_new_tokens=prompt_config['max_tokens'],
                device=device,
                temperature=temperature
            )
            
            result['prompt_description'] = prompt_config['description']
            all_results.append(result)
            
            # Print individual results
            print(f"TTFT: {result['time_to_first_token_seconds']:.3f}s")
            print(f"Output TPS: {result['output_tokens_per_second']:.1f}")
            print(f"Total TPS: {result['tokens_per_second']:.1f}")
            if 'peak_gpu_memory_mb' in result:
                print(f"Peak GPU Memory: {result['peak_gpu_memory_mb']:.1f}MB")
            
        except Exception as e:
            print(f"Error benchmarking prompt {i+1}: {str(e)}")
            continue
    
    if not all_results:
        return {}
    
    # Aggregate results
    metrics = ['time_to_first_token_seconds', 'tokens_per_second', 'output_tokens_per_second']
    gpu_metrics = ['peak_gpu_memory_mb', 'p90_gpu_utilization']
    
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results if metric in r and r[metric] > 0]
        if values:
            aggregated[f'p50_{metric}'] = statistics.median(values)
            aggregated[f'p90_{metric}'] = np.percentile(values, 90)
            aggregated[f'mean_{metric}'] = statistics.mean(values)
    
    for metric in gpu_metrics:
        values = [r[metric] for r in all_results if metric in r]
        if values:
            aggregated[f'max_{metric}'] = max(values)
            aggregated[f'mean_{metric}'] = statistics.mean(values)
    
    aggregated['total_prompts'] = len(all_results)
    aggregated['individual_results'] = all_results
    
    return aggregated


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_prompts: List[Dict],
    model_device: torch.device,
    model_name: str,
    verbose: bool = False
) -> Dict:
    """Run comprehensive inference benchmark."""
    
    print(f"\n=== Benchmarking {model_name} ===")
    
    # Move model to device with appropriate precision
    model = model.to(model_device)
    if model_device.type == 'cuda':
        model = model.to(torch.float16)
    else:
        model = model.to(torch.float32)
    
    print(f"Model device: {model_device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Warmup runs
    print("Warming up model...")
    warmup_prompt = test_prompts[0]
    for _ in range(2):
        try:
            benchmark_single_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=warmup_prompt['prompt'],
                max_new_tokens=min(50, warmup_prompt['max_tokens']),
                device=model_device,
                temperature=0.7
            )
        except Exception as e:
            print(f"Warmup failed: {e}")
    
    # Main benchmark
    results = benchmark_language_model(
        model=model,
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        device=model_device,
        temperature=0.7
    )
    
    return results


def print_comprehensive_results(results: Dict, model_name: str):
    """Print comprehensive benchmark results."""
    print(f"\n{'='*60}")
    print(f"📊 {model_name} Benchmark Results")
    print(f"{'='*60}")
    
    if not results:
        print("❌ No results available")
        return
    
    print(f"📈 Performance Metrics (n={results.get('total_prompts', 0)} prompts):")
    print("-" * 40)
    
    # TTFT metrics
    if 'p50_time_to_first_token_seconds' in results:
        print(f"⚡ Time to First Token:")
        print(f"   P50: {results['p50_time_to_first_token_seconds']:.3f}s")
        print(f"   P90: {results['p90_time_to_first_token_seconds']:.3f}s")
        print(f"   Mean: {results['mean_time_to_first_token_seconds']:.3f}s")
    
    # TPS metrics
    if 'p50_output_tokens_per_second' in results:
        print(f"🚀 Output Generation Speed:")
        print(f"   P50: {results['p50_output_tokens_per_second']:.1f} tokens/sec")
        print(f"   P90: {results['p90_output_tokens_per_second']:.1f} tokens/sec")
        print(f"   Mean: {results['mean_output_tokens_per_second']:.1f} tokens/sec")
    
    if 'p50_tokens_per_second' in results:
        print(f"📊 Total Throughput:")
        print(f"   P50: {results['p50_tokens_per_second']:.1f} tokens/sec")
        print(f"   P90: {results['p90_tokens_per_second']:.1f} tokens/sec")
        print(f"   Mean: {results['mean_tokens_per_second']:.1f} tokens/sec")
    
    # GPU metrics
    if 'max_peak_gpu_memory_mb' in results:
        print(f"🖥️  GPU Usage:")
        print(f"   Max Memory: {results['max_peak_gpu_memory_mb']:.1f}MB")
        print(f"   Mean Memory: {results['mean_peak_gpu_memory_mb']:.1f}MB")
        if 'max_p90_gpu_utilization' in results:
            print(f"   Max Utilization: {results['max_p90_gpu_utilization']:.1f}%")


def check_llamacpp_available(binary_path: str) -> bool:
    """Check if llama.cpp binary is available."""
    try:
        result = subprocess.run([binary_path, '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def convert_model_to_gguf(model_path: str, output_path: str) -> bool:
    """Convert HuggingFace model to GGUF format using llama.cpp convert script."""
    try:
        # Look for convert script in the local llama.cpp installation
        convert_script = "llama.cpp/convert_hf_to_gguf.py"
        if not os.path.exists(convert_script):
            # Try without underscore (older versions)
            convert_script = "llama.cpp/convert-hf-to-gguf.py"
            if not os.path.exists(convert_script):
                # Try the home directory as fallback
                convert_script = os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py")
                if not os.path.exists(convert_script):
                    print(f"Convert script not found. Tried:")
                    print(f"  - llama.cpp/convert_hf_to_gguf.py")
                    print(f"  - llama.cpp/convert-hf-to-gguf.py")
                    print(f"  - ~/llama.cpp/convert_hf_to_gguf.py")
                    return False
        
        cmd = [
            "python", convert_script,
            model_path,
            "--outfile", output_path,
            "--outtype", "f16"  # Use FP16 for fair comparison
        ]
        
        print(f"Converting model to GGUF format...")
        print(f"Using script: {convert_script}")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Conversion failed: {result.stderr}")
            return False
            
        print(f"✓ Model converted successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"Error converting model: {e}")
        return False


def benchmark_llamacpp_single_prompt(
    model_path: str,
    prompt: str,
    max_tokens: int,
    binary_path: str = "llama-cli",
    temperature: float = 0.7,
    device: str = "cpu"
) -> Dict:
    """Benchmark a single prompt using llama.cpp."""
    
    # Prepare command
    cmd = [
        binary_path,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--seed", "42",  # Fixed seed for reproducibility
        "--log-disable",  # Disable logging for cleaner output
        "--simple-io",  # Simple input/output mode
        "-ngl", "99" if device == "cuda" else "0",  # GPU layers
        "--no-display-prompt"  # Don't display the prompt in output
    ]
    
    # Add timing flag if available in the llama.cpp version
    cmd.extend(["--timings"])
    
    # Run llama.cpp
    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        total_time = time.perf_counter() - start_time
        
        if result.returncode != 0:
            raise RuntimeError(f"llama.cpp failed: {result.stderr}")
        
        # Parse output and timing information
        output_text = result.stdout
        stderr_lines = result.stderr.strip().split('\n')
        
        # Extract timing metrics from stderr
        metrics = {
            'total_time_seconds': total_time,
            'output_text': output_text
        }
        
        # Parse timing information from llama.cpp output
        for line in stderr_lines:
            if "prompt eval time" in line:
                # Extract prompt evaluation time
                parts = line.split()
                if "ms" in line:
                    time_idx = parts.index("ms") - 1
                    prompt_time_ms = float(parts[time_idx])
                    metrics['prompt_eval_time_seconds'] = prompt_time_ms / 1000
                    
            elif "eval time" in line and "prompt eval" not in line:
                # Extract generation time
                parts = line.split()
                if "ms" in line:
                    time_idx = parts.index("ms") - 1
                    gen_time_ms = float(parts[time_idx])
                    metrics['generation_time_seconds'] = gen_time_ms / 1000
                    
            elif "total time" in line:
                parts = line.split()
                if "ms" in line:
                    time_idx = parts.index("ms") - 1
                    total_ms = float(parts[time_idx])
                    metrics['reported_total_time_seconds'] = total_ms / 1000
                    
            elif "speed" in line or "tok/s" in line:
                # Extract tokens per second
                parts = line.split()
                for i, part in enumerate(parts):
                    if "tok/s" in part or (i > 0 and parts[i-1] == "tok/s"):
                        try:
                            tps = float(parts[i-2] if "tok/s" in part else part)
                            if "prompt" in line:
                                metrics['prompt_tokens_per_second'] = tps
                            else:
                                metrics['output_tokens_per_second'] = tps
                        except (ValueError, IndexError):
                            pass
        
        # Count tokens (approximate)
        output_tokens = len(output_text.split())  # Rough approximation
        input_tokens = len(prompt.split())  # Rough approximation
        
        metrics.update({
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        })
        
        # Calculate TTFT if we have the metrics
        if 'prompt_eval_time_seconds' in metrics:
            metrics['time_to_first_token_seconds'] = metrics['prompt_eval_time_seconds']
        
        # Calculate tokens per second if not provided
        if 'output_tokens_per_second' not in metrics and 'generation_time_seconds' in metrics:
            if metrics['generation_time_seconds'] > 0:
                metrics['output_tokens_per_second'] = output_tokens / metrics['generation_time_seconds']
        
        if 'tokens_per_second' not in metrics:
            metrics['tokens_per_second'] = metrics['total_tokens'] / total_time
        
        return metrics
        
    except subprocess.TimeoutExpired:
        return {
            'error': 'Timeout',
            'total_time_seconds': 300,
            'input_tokens': len(prompt.split()),
            'output_tokens': 0,
            'total_tokens': len(prompt.split())
        }
    except Exception as e:
        return {
            'error': str(e),
            'total_time_seconds': time.perf_counter() - start_time,
            'input_tokens': len(prompt.split()),
            'output_tokens': 0,
            'total_tokens': len(prompt.split())
        }


def benchmark_llamacpp_model(
    model_path: str,
    test_prompts: List[Dict],
    binary_path: str = "llama-cli",
    device: str = "cpu",
    temperature: float = 0.7
) -> Dict:
    """Benchmark llama.cpp model across multiple prompts."""
    
    all_results = []
    
    print(f"\nRunning llama.cpp benchmark on {len(test_prompts)} prompts...")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    
    for i, prompt_config in enumerate(test_prompts):
        print(f"\nPrompt {i+1}/{len(test_prompts)}: {prompt_config['description']}")
        print(f"Max tokens: {prompt_config['max_tokens']}")
        
        try:
            result = benchmark_llamacpp_single_prompt(
                model_path=model_path,
                prompt=prompt_config['prompt'],
                max_tokens=prompt_config['max_tokens'],
                binary_path=binary_path,
                temperature=temperature,
                device=device
            )
            
            if 'error' not in result:
                result['prompt_description'] = prompt_config['description']
                all_results.append(result)
                
                # Print individual results
                if 'time_to_first_token_seconds' in result:
                    print(f"TTFT: {result['time_to_first_token_seconds']:.3f}s")
                if 'output_tokens_per_second' in result:
                    print(f"Output TPS: {result['output_tokens_per_second']:.1f}")
                print(f"Total time: {result['total_time_seconds']:.3f}s")
            else:
                print(f"Error: {result['error']}")
                
        except Exception as e:
            print(f"Error benchmarking prompt {i+1}: {str(e)}")
            continue
    
    if not all_results:
        return {}
    
    # Aggregate results
    metrics = ['time_to_first_token_seconds', 'tokens_per_second', 'output_tokens_per_second', 'total_time_seconds']
    
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results if metric in r and r[metric] > 0]
        if values:
            aggregated[f'p50_{metric}'] = statistics.median(values)
            aggregated[f'p90_{metric}'] = np.percentile(values, 90)
            aggregated[f'mean_{metric}'] = statistics.mean(values)
    
    aggregated['total_prompts'] = len(all_results)
    aggregated['individual_results'] = all_results
    
    return aggregated


def main():
    """Main function to run the comprehensive benchmark."""
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

    # Get test prompts
    test_prompts = get_diverse_test_prompts()
    
    print(f"\n🎯 Running comprehensive benchmark with {len(test_prompts)} diverse prompts...")
    print(f"📝 Test prompts: {[p['description'] for p in test_prompts]}")

    # Always run SkipLLaMA benchmark with HuggingFace
    skip_model = LlamaSkipConnectionForCausalLM.from_pretrained(checkpoint, config=config)
    skip_results = run_inference(
        model=skip_model,
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        model_device=skip_device,
        model_name="SkipLLaMA",
        verbose=args.verbose
    )

    # Run standard model benchmark
    if args.compare_with_llamacpp:
        # Check if llama.cpp is available
        if not check_llamacpp_available(args.llamacpp_binary):
            print(f"\n❌ llama.cpp binary '{args.llamacpp_binary}' not found!")
            print("Please install llama.cpp and ensure the binary is in your PATH")
            print("Or specify the full path using --llamacpp_binary")
            return
        
        # Determine GGUF model path
        if args.llamacpp_model_path:
            gguf_model_path = args.llamacpp_model_path
        else:
            # Try to convert the model
            gguf_model_path = os.path.join(tempfile.gettempdir(), "model.gguf")
            print(f"\n📦 Converting model to GGUF format...")
            if not convert_model_to_gguf(checkpoint, gguf_model_path):
                print("❌ Failed to convert model to GGUF format")
                print("Please provide a pre-converted GGUF model using --llamacpp_model_path")
                return
        
        # Run llama.cpp benchmark
        standard_results = benchmark_llamacpp_model(
            model_path=gguf_model_path,
            test_prompts=test_prompts,
            binary_path=args.llamacpp_binary,
            device=args.device,
            temperature=0.7
        )
        standard_name = "Standard LLaMA (llama.cpp)"
    else:
        # Use HuggingFace implementation
        standard_model = AutoModelForCausalLM.from_pretrained(checkpoint)
        standard_results = run_inference(
            model=standard_model,
            tokenizer=tokenizer,
            test_prompts=test_prompts,
            model_device=standard_device,
            model_name="Standard LLaMA (HuggingFace)",
            verbose=args.verbose
        )
        standard_name = "Standard LLaMA (HuggingFace)"

    # Print results
    print_comprehensive_results(skip_results, "SkipLLaMA")
    print_comprehensive_results(standard_results, standard_name)
    
    # Calculate speedups
    if skip_results and standard_results:
        print(f"\n{'='*60}")
        print(f"🏁 Performance Comparison")
        print(f"{'='*60}")
        
        if ('mean_time_to_first_token_seconds' in skip_results and 
            'mean_time_to_first_token_seconds' in standard_results):
            ttft_speedup = standard_results['mean_time_to_first_token_seconds'] / skip_results['mean_time_to_first_token_seconds']
            print(f"⚡ TTFT Speedup: {ttft_speedup:.2f}x")
        
        if ('mean_output_tokens_per_second' in skip_results and 
            'mean_output_tokens_per_second' in standard_results):
            tps_speedup = skip_results['mean_output_tokens_per_second'] / standard_results['mean_output_tokens_per_second']
            print(f"🚀 Output TPS Speedup: {tps_speedup:.2f}x")
        
        if ('mean_tokens_per_second' in skip_results and 
            'mean_tokens_per_second' in standard_results):
            total_speedup = skip_results['mean_tokens_per_second'] / standard_results['mean_tokens_per_second']
            print(f"📊 Total Throughput Speedup: {total_speedup:.2f}x")


if __name__ == "__main__":
    main()
