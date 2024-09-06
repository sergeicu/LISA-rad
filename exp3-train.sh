# see available machines 
sinfo -o "%30N %30c %30m %30G"
sinfo 

# all gpus currently in use 
squeue -o %b | grep gpu

# show jobs 
squeue -U $USER
scontrol show job <jobid>


# activate 
ssh ch215616@e3-login.tch.harvard.edu

srun -A crl -p bch-compute -t 1:00:00 --qos=crl --pty /bin/bash 

srun -A crl -p crl-gpu -t 1:00:00 --qos=crl --gres=gpu:4  --pty /bin/bash 
srun -A crl -p bch-gpu -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:NVIDIA_A40:4  --pty /bin/bash 

srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:NVIDIA_A100:1  --pty /bin/bash 

# alternative export 



# original export 
conda activate glamm_e2 
cd /home/ch215616/w/code/llm/experiments/LISA
source venv_lisa/bin/activate

# export 
export HF_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/
export CUDA_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/cuda-11.7/
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH

# check weights 
huggingface-cli scan-cache     
ls /lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/hub/

# we have the following models 
models--liuhaotian--llava-v1.5-7b # original llava 1.5 weights (already trained) but not lightning? 
models--xinlai--LISA-7B-v1 # original lisa weights obtained for lisa paper - for inference 
models--xinlai--LISA-7B-v1-explanatory  # this is not downloaded but we can use it if necessary... 

gpu-b07-0


# train 
llava="liuhaotian/llava-llama-2-13b-chat-lightning-preview"  # default 
llava="models--xinlai--LISA-7B-v1"
llava="xinlai/LISA-7B-v1"
sam=sam_vit_h_4b8939.pth
# sam=medsam_vit_b.pth
# deepspeed --master_port=24999 train_ds.py \
python train_ds.py \
  --version=$llava \
  --dataset_dir='./dataset' \
  --vision_pretrained=$sam \
  --dataset="sem_seg" \
  --val_dataset="val" \
  --log_base_dir="./runs" \
  --sem_seg_data="cocostuff" \
  --sample_rates="9,3,3,1" \
  --no_eval \
  --exp_name="lisa-7b" 

  --vis_save_path="./vis_output" \
  --precision="bf16" \
  --image_size="1024" \
  --model_max_length="512" \
  --lora_r="8" \
  --vision-tower="openai/clip-vit-large-patch14" \
  --epochs="10" \
  --steps_per_epoch="500" \
  --batch_size="2" \
  --grad_accumulation_steps="10" \
  --workers="4" \
  --lr="0.0003" \
  --ce_loss_weight="1.0" \
  --dice_loss_weight="0.5" \
  --bce_loss_weight="2.0" \
  --lora_alpha="16" \
  --lora_dropout="0.05" \
  --lora_target_modules="q_proj,v_proj" \
  --explanatory="0.1" \
  --beta1="0.9" \
  --beta2="0.95" \
  --num_classes_per_sample="3" \
  --out_dim="256" \
  --print_freq="1" \
  --start_epoch="0" \
  --conv_type="llava_v1" 
  

# note on coco stuff 
  sem_seg -> cocostuff # https://github.com/dvlab-research/LISA



/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/venv_lisa/lib/python3.10/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We recommend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565
Loading checkpoint shards:   0%|                                                             | 0/2 [00:00<?, ?it/s]Killed




Traceback (most recent call last):
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/train_ds.py", line 584, in <module>
    main(sys.argv[1:])
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/train_ds.py", line 161, in main
    model.get_model().initialize_vision_modules(model.get_model().config)
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/model/llava/model/llava_arch.py", line 51, in initialize_vision_modules
    vision_tower = build_vision_tower(model_args)
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/model/llava/model/multimodal_encoder/builder.py", line 15, in build_vision_tower
    return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/model/llava/model/multimodal_encoder/clip_encoder.py", line 17, in __init__
    self.load_model()
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/model/llava/model/multimodal_encoder/clip_encoder.py", line 25, in load_model
    self.vision_tower = CLIPVisionModel.from_pretrained(
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/venv_lisa/lib/python3.10/site-packages/transformers/modeling_utils.py", line 2629, in from_pretrained
    state_dict = load_state_dict(resolved_archive_file)
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/venv_lisa/lib/python3.10/site-packages/transformers/modeling_utils.py", line 447, in load_state_dict
    with safe_open(checkpoint_file, framework="pt") as f:
  File "/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/venv_lisa/lib/python3.10/site-packages/torch/storage.py", line 776, in from_file
    untyped_storage: UntypedStorage = UntypedStorage.from_file(
RuntimeError: unable to open file </home/ch215616/w/huggingface/transformers/cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41/model.safetensors> in read-only mode: No such file or directory (2)
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2/2 [03:41<00:00, 110.84s/it]
[2024-09-02 13:12:51,420] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 730704
[2024-09-02 13:12:51,484] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 730705
[2024-09-02 13:12:52,938] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 730706
[2024-09-02 13:12:54,265] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 730707
[2024-09-02 13:12:55,685] [ERROR] [launch.py:325:sigkill_handler] ['/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/venv_lisa/bin/python', '-u', 'train_ds.py', '--local_rank=3', '--version=xinlai/LISA-7B-v1', '--dataset_dir=./dataset', '--vision_pretrained=sam_vit_h_4b8939.pth', '--dataset=sem_seg', '--val_dataset=val', '--log_base_dir=./runs', '--sem_seg_data=cocostuff', '--sample_rates=9,3,3,1', '--exp_name=lisa-7b'] exits with return code = 1