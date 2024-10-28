from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM, LlamaSkipConnectionConfig
from transformers.models.llama import LlamaForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoModel, BitsAndBytesConfig
import time

checkpoint = "meta-llama/Llama-3.2-1B-Instruct"

AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
AutoModel.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)
AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)

llamaSkipConfig = LlamaSkipConnectionConfig.from_json_file("./configs/llama_skip_causal.json")
llamaSkipModel = LlamaSkipConnectionForCausalLM.from_pretrained(checkpoint, config=llamaSkipConfig)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequence = "Give recipe of burrito including all the ingredients and their quantity."
input= tokenizer(sequence, return_tensors='pt').input_ids

pipe = pipeline(
    "text-generation",
    model=llamaSkipModel,
    tokenizer=tokenizer,
    max_new_tokens = 1000,
    eos_token_id=tokenizer.eos_token_id,
)

standardPipe = pipeline(
    "text-generation",
    model=checkpoint,
    tokenizer=tokenizer,
    max_new_tokens = 1000,
    eos_token_id=tokenizer.eos_token_id,
)
iterations = [15, 100, 1000]
for iteration in iterations:
    start1 = time.time()
    for i in range(iteration):
        out = standardPipe.model.forward(input, use_cache=True)
    start2 = time.time()
    print(f"Time taken for {iteration} model runs.")

    print("Standard pipeline time: ", start2 - start1)
    for i in range(iteration):
        out = pipe.model.forward(input,use_cache=True)
    start3 = time.time()
    print("Skip Model Pipeline time: ", start3 - start2)
