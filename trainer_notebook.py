# %%
import torch

from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoTokenizer
from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM, LlamaSkipMLP, FastLoRAProjection, global_timer
from src.models.configuration_llama_skip import LlamaSkipConnectionConfig
import gc
import time

# Enable TorchScript optimization
torch.jit.enable_onednn_fusion(True)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
torch._C._jit_set_texpr_fuser_enabled(True)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)

# Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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

# # Load model without device_map
# model = LlamaSkipConnectionForCausalLM.from_pretrained(
#     checkpoint, 
#     config=config
# ).to(device)



# model.eval()
# Create scripted version of the model
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

# Create pipelines for each model variant
# llamaSkipPipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens = 1000,
#     device=device,
#     eos_token_id=tokenizer.eos_token_id
# )
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

llamaSkipScriptedPipe = pipeline(
    "text-generation",
    model=scripted_model,
    tokenizer=tokenizer,
    max_new_tokens = 1000,
    device=device,
    eos_token_id=tokenizer.eos_token_id
)

llamaPipe = pipeline(
    "text-generation",
    device=device,
    model=checkpoint,
    tokenizer=tokenizer,
    max_new_tokens = 1000,
    eos_token_id=tokenizer.eos_token_id
)

llamaPipe.model.to(torch.float32)
# llamaSkipPipe.model.to(torch.float32)
llamaSkipScriptedPipe.model.to(torch.float32)

def run_inference(model, input_ids, attention_mask, tokenizer, num_runs=10):
    model = model.cpu()
    base_input_ids = input_ids.cpu()
    base_attention_mask = attention_mask.cpu()
    
    print(f"\nModel type: {type(model)}")
    # print(f"Model config: {model.config}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    times = []
    
    for i in range(num_runs):
        torch.cuda.empty_cache()
        gc.collect()
        
        # Randomize input for each run
        random_shift = torch.randint(-100, 100, base_input_ids.shape)
        input_ids = torch.clamp(base_input_ids + random_shift, min=0, max=tokenizer.vocab_size-1)
        attention_mask = base_attention_mask  # Keep attention mask same
        
        if hasattr(model, 'past_key_values'):
            model.past_key_values = None
        model._past_length = 0
        model.config.use_cache = False
        
        # Warmup iteration for more accurate timing
        if i == 0:
            with torch.no_grad():
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
            continue
        
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
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

    # After inference, print global timing stats
    global_timer.print_stats()
    
    # Clear timings for next run
    global_timer.timings = {k: [] for k in global_timer.timings.keys()}
    
    return times

print("Running CPU inference benchmarks...")
print("-" * 50)

# Warm up runs
print("Warming up models...")
_ = run_inference(llamaPipe.model, input_ids, attention_mask, tokenizer, num_runs=2)
# _ = run_inference(llamaSkipPipe.model, input_ids, attention_mask, tokenizer, num_runs=2)
_ = run_inference(llamaSkipScriptedPipe.model, input_ids, attention_mask, tokenizer, num_runs=2)

# Actual benchmarks
skip_scripted_times = run_inference(llamaSkipScriptedPipe.model, input_ids, attention_mask, tokenizer)
std_times = run_inference(llamaPipe.model, input_ids, attention_mask, tokenizer)
# skip_times = run_inference(llamaSkipPipe.model, input_ids, attention_mask, tokenizer)

print_results = lambda name, times: (
    print(f"\n{name} CPU Results:"),
    print(f"Average time: {sum(times)/len(times):.3f}s"),
    print(f"Min time: {min(times):.3f}s"), 
    print(f"Max time: {max(times):.3f}s"),
    print(f"Individual times: {[f'{t:.3f}s' for t in times]}")
)

calc_speedup = lambda t1, t2: (sum(t1)/len(t1))/(sum(t2)/len(t2))

print_results("SkipLLaMA Scripted", skip_scripted_times)
print_results("Standard LLaMA", std_times)
# print_results("SkipLLaMA", skip_times)
print("\nCPU Speedups:")
print(f"Scripted vs Standard: {calc_speedup(std_times, skip_scripted_times):.2f}x")
# print(f"SkipLLaMA vs Standard: {calc_speedup(std_times, skip_times):.2f}x")
# print(f"SkipLLaMA vs Scripted: {calc_speedup(skip_times, skip_scripted_times):.2f}x")
