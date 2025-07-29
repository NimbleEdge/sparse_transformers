import argparse
import os
import logging

import torch

from transformers import AutoConfig, AutoModelForCausalLM
from lm_eval import simple_evaluate
from lm_eval.utils import make_table
from lm_eval.models.huggingface import HFLM

import src.models

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on common LM datasets using LM Eval Harness.")
    parser.add_argument("--model_type", type=str, choices=["hf", "sparse"], default="hf")
    parser.add_argument("--model_name_or_config", type=str, required=True,
                       help="Name or path of the base model (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--sp_dir", type=str, default="",
                        help="Path to trained predictor dir for sparse model.")
    parser.add_argument("--tasks", nargs='+', default=["hellaswag"], 
                        help="Tasks on which to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--lora_size", type=float, default=4.0,
                       help="Size of lora predictors to use as percentage of total hidden size")
    parser.add_argument("--sp_layers", default="all", nargs='+',
                       help="Which layers to use sparse predictors for")
    parser.add_argument("--disable_weight_cache", action="store_true", 
                        help="Disable weight cache and compute sparse mlp manually")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")

    # Load pretrained model
    logging.info("Loading pretrained model for evaluation...")
    
    if args.model_type == "hf":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_config)
    if args.model_type == "sparse":
        config = AutoConfig.from_pretrained(args.model_name_or_config)
        if args.sp_layers != "all":
            args.sp_layers = [int(x) for x in args.sp_layers]
        config.lora_size = args.lora_size / 100.0
        config.sp_layers = args.sp_layers
        if args.disable_weight_cache:
            config.use_weight_cache = False
        model = AutoModelForCausalLM.from_pretrained(config._name_or_path, config=config)
        for layer_idx in model.get_decoder().sp_layers:
            layer = model.get_decoder().layers[layer_idx]
            layer_path = os.path.join(args.sp_dir, f"final_predictor_layer_{layer_idx}_lora_{args.lora_size}pct.pt")
            if not os.path.exists(layer_path):
                logger.error(f"Pretrained weights for sparse predictor at layer {layer_idx} do not exist.")
                return
            pretrained_dict = torch.load(layer_path)
            layer.mlp_lora_proj.load_state_dict(pretrained_dict)
        model.tie_weights()
        model.reset_cache()

    wrapped_model = HFLM(
        pretrained=model,
        backend="causal",
        batch_size=args.batch_size,
        device=device
    )

    logging.info("Beginning evaluation...")
    results = simple_evaluate(
        wrapped_model,
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=device
    )

    if results is not None:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

if __name__ == '__main__':
    main()
