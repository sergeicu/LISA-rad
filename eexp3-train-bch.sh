
# train normal 
export HF_HOME=/lab-share/Rad-Afacan-e2/Public/serge/code/huggingface_cache/
llava="xinlai/LISA-7B-v1"
sam=sam_vit_h_4b8939.pth
data="cocostuff"
data="bchwrist"

# python train_ds.py \
deepspeed --master_port=24999 train_ds.py \
  --version=$llava \
  --vision_pretrained=$sam \
  --dataset="sem_seg" \
  --val_dataset="val" \
  --log_base_dir="./runs" \
  --sem_seg_data=$data \
  --sample_rates="1" \
  --no_eval \
  --batch_size=1 \
  --exp_name="lisa-7b" 



# checkpoint issue of training 
https://github.com/microsoft/DeepSpeed/issues/3810
