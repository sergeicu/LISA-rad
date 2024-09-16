# latest model 
./runs/lisa-7b62ddd33333333343g/ckpt_model/global_step1/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt.


# export weights 
cd ~/ww/code/llm/experiments/LISA/runs/lisa-7b/ckpt_model
python zero_to_fp32.py . ../pytorch_model.bin

# merge lora weights -> using the original model on which we trained 
export HF_HOME=/lab-share/Rad-Afacan-e2/Public/serge/code/huggingface_cache/
python ../../../merge_lora_weights_and_save_hf_model.py \
--version "xinlai/LISA-7B-v1" \
--vision_pretrained ../../../sam_vit_h_4b8939.pth \
--weight ../pytorch_model.bin \
--save_path "../"

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

