import torch
import time
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM, LlamaSkipDecoderLayer, global_timer, LlamaSkipMLP, FastLoRAProjection
from src.models.configuration_llama_skip import LlamaSkipConnectionConfig
import numpy as np

# Enable TorchScript optimization
torch.jit.enable_onednn_fusion(True)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(True)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
# Set device
device = torch.device("cpu")

# Register custom model and config
AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)

# Load model IDs and config
model_id = "vkkhare/llama-skip"
checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
config = LlamaSkipConnectionConfig.from_pretrained(model_id)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

class LayerTimer:
    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.layer_times = {i: [] for i in range(num_layers)}
        self.current_layer = 0
        
    def start_layer(self):
        self.start_time = time.perf_counter()
        
    def end_layer(self):
        end_time = time.perf_counter()
        self.layer_times[self.current_layer].append(end_time - self.start_time)
        self.current_layer = (self.current_layer + 1) % self.num_layers
    
    def print_stats(self, model_name):
        print(f"\n{model_name} Layer-wise Statistics:")
        for layer_idx in range(self.num_layers):
            times = self.layer_times[layer_idx]
            if times:
                avg_time = np.mean(times) * 1000  # Convert to ms
                min_time = np.min(times) * 1000
                max_time = np.max(times) * 1000
                std_time = np.std(times) * 1000
                print(f"\nLayer {layer_idx}:")
                print(f"  Avg: {avg_time:.3f}ms")
                print(f"  Min: {min_time:.3f}ms")
                print(f"  Max: {max_time:.3f}ms")
                print(f"  Std: {std_time:.3f}ms")
                print(f"  Count: {len(times)}")

def add_timing_hooks(model, layer_timer):
    def forward_pre_hook(module, input):
        layer_timer.start_layer()
        return None
    
    def forward_hook(module, input, output):
        layer_timer.end_layer()
        return None
    
    # Add hooks to each decoder layer
    for name, module in model.named_modules():
        if "layers" in name and isinstance(module, (LlamaSkipDecoderLayer, LlamaDecoderLayer)):
            module.register_forward_pre_hook(forward_pre_hook)
            module.register_forward_hook(forward_hook)

def run_inference(model, input_ids, attention_mask, tokenizer, layer_timer, num_runs=10):
    times = []
    
    for i in range(num_runs):
        start = time.perf_counter()
        
        # Clear component timings
        global_timer.timings = {k: [] for k in global_timer.timings.keys()}
        _ = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
                return_dict_in_generate=False
            )
        end = time.perf_counter()
        times.append(end - start)
        
        print(f"\nRun {i+1} timing breakdown:")
        global_timer.print_stats()
        layer_timer.print_stats(type(model).__name__)
    
    return times

def main():
    # Load models
    print("Loading models...")
    llama_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    skip_model = LlamaSkipConnectionForCausalLM.from_pretrained(checkpoint, config=config).to(device)
    
    skip_model.eval()
    # Move all masks to the correct device
    for module in skip_model.modules():
        if hasattr(module, 'mask'):
            module.mask = module.mask.to(device)

    for module in skip_model.modules():
        if isinstance(module, LlamaSkipMLP) or isinstance(module, FastLoRAProjection):
            module.eval()  # Ensure in eval mode before scripting
            try:
                scripted_module = torch.jit.script(module)
                module.forward = scripted_module.forward
            except Exception as e:
                print(f"Failed to script module {type(module).__name__}: {str(e)}")
                continue
    # Create layer timers
    llama_timer = LayerTimer(len(llama_model.model.layers))
    skip_timer = LayerTimer(len(skip_model.model.layers))
    
    # Add timing hooks
    add_timing_hooks(llama_model, llama_timer)
    add_timing_hooks(skip_model, skip_timer)
    
    # Prepare input
    sequence = "Give recipe of burrito including all the ingredients and their quantity."
    inputs = tokenizer(
        sequence, 
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("\nRunning CPU inference benchmarks...")
    print("-" * 50)
    
    # Warm up runs
    print("Warming up models...")
    _ = run_inference(llama_model, input_ids, attention_mask, tokenizer, llama_timer, num_runs=2)
    _ = run_inference(skip_model, input_ids, attention_mask, tokenizer, skip_timer, num_runs=2)
    
    # Actual benchmarks
    print("\nStandard LLaMA Benchmark:")
    std_times = run_inference(llama_model, input_ids, attention_mask, tokenizer, llama_timer)
    
    print("\nSkipLLaMA Benchmark:")
    skip_times = run_inference(skip_model, input_ids, attention_mask, tokenizer, skip_timer)
    
    # Print comparative results
    print("\nComparative Results:")
    print("Standard LLaMA vs SkipLLaMA Layer-wise Comparison:")
    for layer_idx in range(len(llama_model.model.layers)):
        std_avg = np.mean(llama_timer.layer_times[layer_idx])
        skip_avg = np.mean(skip_timer.layer_times[layer_idx])
        speedup = std_avg / skip_avg
        print(f"\nLayer {layer_idx}:")
        print(f"  Standard: {std_avg:.3f}ms")
        print(f"  Skip: {skip_avg:.3f}ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    print("\nStandard LLaMA:")
    print(f"Average time: {sum(std_times)/len(std_times):.3f}s")
    print(f"Min time: {min(std_times):.3f}s")
    print(f"Max time: {max(std_times):.3f}s")
    
    print("\nSkipLLaMA:")
    print(f"Average time: {sum(skip_times)/len(skip_times):.3f}s")
    print(f"Min time: {min(skip_times):.3f}s")
    print(f"Max time: {max(skip_times):.3f}s")
    
    print(f"\nSpeedup: {(sum(std_times)/len(std_times))/(sum(skip_times)/len(skip_times)):.2f}x")

if __name__ == "__main__":
    main() 