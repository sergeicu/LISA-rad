import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.LISA import LISAForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

def filter_state_dict(state_dict, model_dict):
    """
    Filter the state dictionary to only include keys that exist in the model
    and have matching shapes.
    """
    filtered_dict = {}
    for key, value in state_dict.items():
        if key in model_dict:
            if value.shape == model_dict[key].shape:
                filtered_dict[key] = value
            else:
                print(f"Skipping {key} due to shape mismatch: "
                      f"{value.shape} vs {model_dict[key].shape}")
    return filtered_dict

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--grounded", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./lisa_model", type=str, required=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    # num_added_tokens = tokenizer.add_tokens("[SEG]")  
    phrase_tokens = ['<p>', '</p>']
    segmentation_tokens = ['[SEG]']
    if args.grounded: 
        segmentation_tokens = segmentation_tokens + phrase_tokens
    num_added_tokens = tokenizer.add_tokens(segmentation_tokens, special_tokens=True) # adding tokens to align vocab to phrase grounding
    #print('[SEG]' in tokenizer.get_vocab())
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.grounded: 
        args.p_start_idx = tokenizer.convert_tokens_to_ids('<p>')
        args.p_end_idx = tokenizer.convert_tokens_to_ids('</p>')        
        
        
    # from IPython import embed; embed()

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "vision_tower": args.vision_tower,
    }
    if args.grounded:         
        model_args['p_start_idx'] = args.p_start_idx
        model_args['p_end_idx'] = args.p_end_idx        

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.weight, map_location="cpu")
    
    model_dict = model.state_dict()




    ##############################
    # check if token sizes match
    ##############################
    # Check state dict token count
    if 'base_model.model.model.embed_tokens.weight' in state_dict:
        state_dict_tokens = state_dict['base_model.model.model.embed_tokens.weight'].shape[0] # these are the lora weights 
    elif 'model.embed_tokens.weight' in state_dict:
        state_dict_tokens = state_dict['model.embed_tokens.weight'].shape[0]
    # else:
    #     print("Couldn't find embedding layer in state dict. Check the layer name.")
    #     return

    # Check model's current token count
    model_tokens = model.get_input_embeddings().weight.shape[0] # these are the base weights 

    # Check tokenizer's vocabulary size
    tokenizer_tokens = len(tokenizer)

    print(f"Tokens in state dict: {state_dict_tokens}")
    print(f"Tokens in current model: {model_tokens}")
    print(f"Tokens in tokenizer: {tokenizer_tokens}")

    if state_dict_tokens != model_tokens:
        print(f"Mismatch: State dict has {state_dict_tokens} tokens, "
              f"but model has {model_tokens} tokens.")
    
    if model_tokens != tokenizer_tokens:
        print(f"Warning: Model has {model_tokens} tokens, "
              f"but tokenizer has {tokenizer_tokens} tokens.")

  


    
    # filtered_state_dict = {k: v for k, v in state_dict.items() 
    #                     if k in model_dict and v.shape == model_dict[k].shape}
    
    
    if model_tokens != tokenizer_tokens or state_dict_tokens != model_tokens: 
        print("WARNING: trying to resize the token sizes - please exit to proceed ")
        from IPython import embed; embed()
        # try to fix 
        filtered_state_dict = filter_state_dict(state_dict, model_dict)
        
        
        model.load_state_dict(filtered_state_dict, strict=False)
    else: 
        model.load_state_dict(state_dict, strict=True)
        

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
