from src.models.modelling_llama_skip import LlamaSkipConnectionForCausalLM, LlamaSkipConnectionConfig
import torch
from torch.export import Dim
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime
from transformers import pipeline, AutoModelForCausalLM, AutoConfig, AutoModel, AutoTokenizer

checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
AutoConfig.register("llama-skip", LlamaSkipConnectionConfig)
AutoModel.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)
AutoModelForCausalLM.register(LlamaSkipConnectionConfig, LlamaSkipConnectionForCausalLM)
llamaSkipConfig = LlamaSkipConnectionConfig.from_json_file("./configs/llama_skip_causal.json")
llamaSkipModel = LlamaSkipConnectionForCausalLM.from_pretrained(checkpoint, config=llamaSkipConfig)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequence = "Give recipe of burrito including all the ingredients and their quantity."
input= tokenizer(sequence, return_tensors='pt')

runtime = Runtime.get()
model = llamaSkipModel.eval() # turn into evaluation mode

# passing dynamic_shapes={"input_ids": {1: 14}, "attention_mask": {1: 14}} in torch.export.export does not help
exported_graph = torch.export.export(model.eval(), args=(input["input_ids"], input["attention_mask"]), strict=False) # Core Aten graph
print("torch.export.export done")

edge_delegated = to_edge_transform_and_lower(exported_graph, partitioner=[XnnpackPartitioner()])
print("to_edge_transform_and_lower done")

executorch_program = edge_delegated.to_executorch() # ExecuTorch program
print("to_executorch done")

pte_path = "/home/azureuser/weight_caching/checkpoints/llamaskipmodel/llama_skip_model.pte"

with open(pte_path, "wb") as file:
    executorch_program.write_to_file(file) # Serializing into .pte file
print("File created")

program = runtime.load_program(pte_path)
print("load_program successfull")

method = program.load_method("forward")
print("load_method done")

# Breaking here with 
# F 00:02:14.997224 executorch:pybindings.cpp:749] In function run_method(), assert failed (false): Execution should not reach this point. <class 'transformers.tokenization_utils_base.BatchEncoding'>
# Aborted (core dumped)
output = method.execute([input])
print("method.execute done")