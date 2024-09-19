# latest model 
cd ./runs/lisa-7b62ddd33333333343g/ckpt_model/
cd ./runs/lisa-7b/ckpt_model # messed up model - may have been grounding / not grounding 
cd ./runs/lisa-7b-full_run1_1b/ckpt_model # grounded model 
cd ./runs/lisa-7b-full_run1_1a/ckpt_model # grounded model 

# STEP 1: export weights 
python zero_to_fp32.py . ../pytorch_model.bin

# STEP 2: merge lora weights -> using the original model on which we trained 
export HF_HOME=/lab-share/Rad-Afacan-e2/Public/serge/code/huggingface_cache/
python ../../../merge_lora_weights_and_save_hf_model.py \
--version "xinlai/LISA-7B-v1" \
--vision_pretrained ../../../sam_vit_h_4b8939.pth \
--weight ../pytorch_model.bin \
--save_path "../" --grounded --train_mask_decoder --use_mm_start_end


# STEP 3: run inference 
    #HF - basic test 
    CUDA_VISIBLE_DEVICES=2 python chat.py --version="xinlai/LISA-7B-v1" --precision='bf16' --use_mm_start_end 
    cd /lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/runs/lisa-7b-test_run1_inf_og/code/
    CUDA_VISIBLE_DEVICES=1 python chat.py --version="xinlai/LISA-7B-v1" --precision='bf16'  --use_mm_start_end 

    # local 
    CUDA_VISIBLE_DEVICES=0 python chat.py --model_path ./runs/lisa-7b62ddd33333333343g/ --conv_type conv_bch_v1 --use_mm_start_end 

    CUDA_VISIBLE_DEVICES=1 python chat.py --model_path ./runs/lisa-7b-full_run1_1b/ --conv_type conv_bch_v1 --use_mm_start_end 
    
    
    # File: 




###############################################################################

# merge lora weights -> using the actual model that we were supposed to train on (but we did not merge the weights)
export HF_HOME=/lab-share/Rad-Afacan-e2/Public/serge/code/huggingface_cache/
python merge_lora_weights_and_save_hf_model.py \
--version "liuhaotian/LLaVA-Lightning-7B-delta-v1-1" \
--vision_pretrained sam_vit_h_4b8939.pth \
--weight runs/lisa-7b/pytorch_model.bin \
--save_path "./lisa_model2"



llava="xinlai/LISA-7B-v1"
w=~/ww/code/llm/experiments/LISA/runs/lisa-7b/pytorch_model.bin
CUDA_VISIBLE_DEVICES=0 python chat.py --version=$w --precision='bf16' 

# model 1 - original - tokenizer fails - out of scope 
python chat.py --local_model_path "./lisa_model" --precision bf16

# model 2 - xinlai/LISA-7B-v1 - on which we trained - 
python chat.py --local_model_path "./lisa_model2" --precision bf16

# image to check 
dataset/bchwrist_images/train/26317536-1_PA-2.png

