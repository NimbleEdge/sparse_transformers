# %%
import argparse
import torch

from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaMLP
from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM, LlamaSkipMLP, FastLoRAProjection
from src.models.configuration_llama_skip import LlamaSkipConnectionConfig
import gc
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu',
                      help='Device to run inference on')
    parser.add_argument('--num_runs', type=int, default=50,
                      help='Number of inference runs')
    return parser.parse_args()

args = parse_args()

# Enable TorchScript optimization
torch.jit.enable_onednn_fusion(True)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(True)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)

# Set device based on args and availability
if args.device == 'cuda' and not torch.cuda.is_available():
    print("CUDA requested but not available. Falling back to CPU.")
    device = torch.device('cpu')
else:
    device = torch.device(args.device)

print(f"Using device: {device}")

# Register the custom model and config
AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)

# Load base model and tokenizer
model_id = "vkkhare/llama-skip"
checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

# Set padding token to be the same as EOS token
tokenizer.pad_token = tokenizer.eos_token

# Create custom config and model
config = LlamaSkipConnectionConfig.from_pretrained(model_id)

scripted_model = LlamaSkipConnectionForCausalLM.from_pretrained(
    checkpoint, 
    config=config
).to(device)
scripted_model.eval()

# Move all masks to the correct device
for module in scripted_model.modules():
    if hasattr(module, 'mask'):
        module.mask = module.mask.to(device)

for module in scripted_model.modules():
    if isinstance(module, LlamaSkipMLP) or isinstance(module, FastLoRAProjection):
        module.eval()  # Ensure in eval mode before scripting
        try:
            scripted_module = torch.jit.script(module)
            module.forward = scripted_module.forward
        except Exception as e:
            print(f"Failed to script module {type(module).__name__}: {str(e)}")
            continue

# Generate text
sequence = "Give recipe of burrito including all the ingredients and their quantity."
inputs = tokenizer(
    sequence, 
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)

# Explicitly move all input tensors to the same device as model
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Debug prints
print(f"Model device: {next(scripted_model.parameters()).device}")
print(f"Input IDs device: {input_ids.device}")
print(f"Attention Mask device: {attention_mask.device}")

def create_pipeline(model, **kwargs):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=1000,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs
    )

llamaSkipScriptedPipe = create_pipeline(scripted_model)
llamaPipe = create_pipeline(checkpoint)

# Convert to float32
llamaPipe.model.to(torch.float32)
llamaSkipScriptedPipe.model.to(torch.float32)

def run_inference(model, input_ids, attention_mask, tokenizer, num_runs=args.num_runs):
    model = model.to(device)
    base_input_ids = input_ids.to(device)
    base_attention_mask = attention_mask.to(device)
    
    print(f"\nModel type: {type(model)}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    times = []
    mlp_times = []  # Track MLP times separately
    for module in model.modules():
        if isinstance(module, LlamaSkipMLP) or isinstance(module, LlamaMLP):
            # Add forward hook to track MLP time
            def forward_hook(module, input, output):
                start = time.perf_counter()
                result = module.forward(*input)
                end = time.perf_counter()
                mlp_times.append(end - start)
                return result
            
            module.register_forward_hook(forward_hook)

    for i in range(num_runs):
        torch.cuda.empty_cache()
        gc.collect()
        
        # Randomize input for each run
        random_shift = torch.randint(-100, 100, base_input_ids.shape, device=device)
        input_ids = torch.clamp(base_input_ids + random_shift, min=0, max=tokenizer.vocab_size-1)
        attention_mask = base_attention_mask
        
        # Reset model state
        if hasattr(model, 'past_key_values'):
            model.past_key_values = None
        model._past_length = 0
        model.config.use_cache = False
        
        # Warmup iteration
        if i == 0:
            with torch.no_grad():
                _ = model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_dict=False
                )
            continue
        
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=False
            )
            
        end = time.perf_counter()
        times.append(end - start)    
    
    return times, mlp_times

print(f"\nRunning {args.device.upper()} inference benchmarks...")
print("-" * 50)

# Warm up runs
print("Warming up models...")
_, _ = run_inference(llamaPipe.model, input_ids, attention_mask, tokenizer, num_runs=2)
_, _ = run_inference(llamaSkipScriptedPipe.model, input_ids, attention_mask, tokenizer, num_runs=2)

# Actual benchmarks
skip_scripted_times, skip_scripted_mlp_times = run_inference(llamaSkipScriptedPipe.model, input_ids, attention_mask, tokenizer)
std_times, std_mlp_times = run_inference(llamaPipe.model, input_ids, attention_mask, tokenizer)

print_results = lambda name, times: (
    print(f"\n{name} {args.device.upper()} Results:"),
    print(f"Average time: {sum(times)/len(times):.3f}s"),
    print(f"Min time: {min(times):.3f}s"), 
    print(f"Max time: {max(times):.3f}s"),
    print(f"Individual times: {[f'{t:.3f}s' for t in times]}")
)

print_results("SkipLLaMA Scripted MLP", skip_scripted_mlp_times)
print_results("Standard LLaMA MLP", std_mlp_times)

calc_speedup = lambda t1, t2: (sum(t1)/len(t1))/(sum(t2)/len(t2))

print_results("SkipLLaMA Scripted", skip_scripted_times)
print_results("Standard LLaMA", std_times)
print(f"\n{args.device.upper()} Speedups:")
print(f"Scripted vs Standard: {calc_speedup(std_times, skip_scripted_times):.2f}x")
