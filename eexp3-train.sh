

########################################################
# TESTING - 5hrs 

# 1 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 5:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:NVIDIA_A100:1 
salloc -A bch -p bch-gpu-pe -t 5:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:NVIDIA_A40:1 
salloc -A crl -p crl-gpu -t 5:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --qos=crl --gres=gpu:NVIDIA_A40:1 

# bch-gpu (not A100 or A40)
salloc -A bch -p bch-gpu -t 5:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --gres=gpu:1 

########################################################
# TESTING - 200hrs 

# 4 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 200:00:00 --nodes=1 --ntasks=4 --cpus-per-task=32 --mem=256G --gres=gpu:NVIDIA_A100:4 
salloc -A bch -p bch-gpu-pe -t 200:00:00 --nodes=1 --ntasks=4 --cpus-per-task=32 --mem=256G --gres=gpu:NVIDIA_A40:4 
salloc -A crl -p crl-gpu -t 200:00:00 --nodes=1 --ntasks=4 --cpus-per-task=32 --mem=256G --qos=crl --gres=gpu:NVIDIA_A40:4 
# salloc -A crl -p crl-gpu -t 200:00:00 --nodes=1 --ntasks=4 --cpus-per-node=96 --mem=256G --qos=crl --gres=gpu:NVIDIA_A40:4


# 2 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 200:00:00 --nodes=1 --ntasks=2 --cpus-per-task=32 --mem=128G --gres=gpu:NVIDIA_A100:2
salloc -A bch -p bch-gpu-pe -t 200:00:00 --nodes=1 --ntasks=2 --cpus-per-task=32 --mem=128G --gres=gpu:NVIDIA_A40:2 
salloc -A crl -p crl-gpu -t 200:00:00 --nodes=1 --ntasks=2 --cpus-per-task=32 --mem=128G --qos=crl --gres=gpu:NVIDIA_A40:2 


# 1 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 200:00:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --gres=gpu:NVIDIA_A100:1 
salloc -A bch -p bch-gpu-pe -t 200:00:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --gres=gpu:NVIDIA_A40:1 
salloc -A crl -p crl-gpu -t 200:00:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --qos=crl --gres=gpu:NVIDIA_A40:1 

# 1 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 200:00:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --gres=gpu:NVIDIA_A100:1 
salloc -A bch -p bch-gpu-pe -t 200:00:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --gres=gpu:NVIDIA_A40:1 
salloc -A crl -p crl-gpu -t 200:00:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --qos=crl --gres=gpu:NVIDIA_A40:1 

# bch-gpu (not A100 or A40)
salloc -A bch -p bch-gpu -t 200:00:00 --nodes=1 --ntasks=4 --cpus-per-task=32 --mem=256G --gres=gpu:4 
salloc -A bch -p bch-gpu -t 200:00:00 --nodes=1 --ntasks=2 --cpus-per-task=32 --mem=128G --gres=gpu:2 
salloc -A bch -p bch-gpu -t 200:00:00 --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --gres=gpu:1 


########################################################
# TESTING - 200hrs (BAD MEMORY)

# 4 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A100:4 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A40:4 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A crl -p crl-gpu -t 200:00:00 --qos=crl --gres=gpu:NVIDIA_A40:4 #--pty /bin/bash -c "while true; do sleep 60; done"

# 2 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A100:2 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A40:2 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A crl -p crl-gpu -t 200:00:00 --qos=crl --gres=gpu:NVIDIA_A40:2 #--pty /bin/bash -c "while true; do sleep 60; done"

# 1 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A100:1 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A40:1 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A crl -p crl-gpu -t 200:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 #--pty /bin/bash -c "while true; do sleep 60; done"

# 1 GPU - A100,A40 x crl & bch-gpu-pe
salloc -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A100:1 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A40:1 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A crl -p crl-gpu -t 200:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 #--pty /bin/bash -c "while true; do sleep 60; done"

# bch-gpu (not A100 or A40)
salloc -A bch -p bch-gpu -t 200:00:00 --gres=gpu:4 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A bch -p bch-gpu -t 200:00:00 --gres=gpu:2 #--pty /bin/bash -c "while true; do sleep 60; done"
salloc -A bch -p bch-gpu -t 200:00:00 --gres=gpu:1 #--pty /bin/bash -c "while true; do sleep 60; done"


########################################################
# SRUN - do not use it unless you want to lose it. 

srun -A bch -p bch-compute -t 2:00:00 --pty /bin/bash -c "top" 

# 4 GPU - A100,A40 x crl & bch-gpu-pe
srun -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A100:4 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A40:4 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A crl -p crl-gpu -t 200:00:00 --qos=crl --gres=gpu:NVIDIA_A40:4 --pty /bin/bash -c "while true; do sleep 60; done"

# 2 GPU - A100,A40 x crl & bch-gpu-pe
srun -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A100:2 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A40:2 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A crl -p crl-gpu -t 200:00:00 --qos=crl --gres=gpu:NVIDIA_A40:2 --pty /bin/bash -c "while true; do sleep 60; done"

# 1 GPU - A100,A40 x crl & bch-gpu-pe
srun -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A100:1 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A40:1 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A crl -p crl-gpu -t 200:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 --pty /bin/bash -c "while true; do sleep 60; done"

# 1 GPU - A100,A40 x crl & bch-gpu-pe
srun -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A100:1 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A bch -p bch-gpu-pe -t 200:00:00 --gres=gpu:NVIDIA_A40:1 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A crl -p crl-gpu -t 200:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 --pty /bin/bash -c "while true; do sleep 60; done"

# bch-gpu (not A100 or A40)
srun -A bch -p bch-gpu -t 200:00:00 --gres=gpu:4 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A bch -p bch-gpu -t 200:00:00 --gres=gpu:2 --pty /bin/bash -c "while true; do sleep 60; done"
srun -A bch -p bch-gpu -t 200:00:00 --gres=gpu:1 --pty /bin/bash -c "while true; do sleep 60; done"


########################################################
# misc slurm commands 


# show remaining jobs 
squeue --user $USER --sort=-t,e,P -o "%.18i %.2t %.12M %.12L %.6C %.8m %.9P %.12R" | (sed -u 1q; sort -k5 -r)
          # squeue --user $USER --sort=-t,e,P -o "%.18i %.20j %.9u %.2t %.12M %.12L %.6C %.8m %.8G %.9P" | (sed -u 1q; sort -k5 -r)

# how many jobs 
(
  squeue --user $USER --sort=-t,e,P -o "%.18i %.20j %.9u %.2t %.12M %.12L %.6C %.8m %.8G %.9P" | sed -u 1q
  squeue --user $USER --sort=-t,e,P -o "%.18i %.20j %.9u %.2t %.12M %.12L %.6C %.8m %.8G %.9P" | tail -n +2 | sort -k5 -r | 
    while IFS= read -r line; do
      partition=$(echo "$line" | awk '{print $10}')
      count=$(squeue --user $USER --partition=$partition --state=RUNNING | wc -l)
      printf "%s %5d\n" "$line" "$count"
    done
) | column -t


srun --pty tmux new-session -d 'watch -n 60 date'
srun -A bch -p bch-compute --pty tmux new-session -d 'watch -n 60 date'

srun --jobid=2716185 --pty /bin/bash

salloc  --time=01:00:00




# monitor slurm 
watch -n 10 squeue -u $USER
watch -n 10 'squeue | grep gpu'
squeue --sort=P -O jobid,name,partition,state,priority | grep gpu
squeue --sort=-P -o "%.18i %.20j %.9u %.9P %.12T %.6p" | grep gpu

squeue --sort=-P -o "%.18i %.20j %.9u %.9P %.12T %.6p %V" | grep gpu




# see available machines 
sinfo -o "%30N %30c %30m %30G"
sinfo -o "%30N %30c %30m %30G %30P"

# how many jobs were submitted to partition in last 7 days 
sacct --starttime=$(date --date='7 days ago' +%Y-%m-%d) --partition=fnndsc-gpu --format=JobID --noheader | wc -l

# fnndsc-gpu usage last 7 days 
$ sacct --starttime=$(date --date='7 days ago' +%Y-%m-%d) --partition=fnndsc-gpu --format=JobID --noheader | wc -l
0
# crl-gpu usage last 7 days 
$ sacct --starttime=$(date --date='7 days ago' +%Y-%m-%d) --partition=crl-gpu --format=JobID --noheader | wc -l
76

# fnndsc-gpu usage last 30 days 
$ sacct --starttime=$(date --date='30 days ago' +%Y-%m-%d) --partition=fnndsc-gpu --format=JobID --noheader | wc -l
0
# crl-gpu usage last 30 days 
$ sacct --starttime=$(date --date='7 days ago' +%Y-%m-%d) --partition=crl-gpu --format=JobID --noheader | wc -l
206

p=bchen-gpu; sacct --starttime=$(date --date='7 days ago' +%Y-%m-%d) --partition=$p --format=JobID --noheader | wc -l
p=chip-gpu; sacct --starttime=$(date --date='7 days ago' +%Y-%m-%d) --partition=$p --format=JobID --noheader | wc -l

sacct --partition=bch-gpu --format=JobID,JobName,Partition,AllocCPUs,State,Elapsed,Start,End --starttime=<start_date> --endtime=<end_date>





# sleep 
while true; do sleep 60; done



# all gpus currently in use 
squeue -o %b | grep gpu

# show jobs 
squeue -U $USER
scontrol show job <jobid>




########################################################
# train 


# activate 
ssh ch215616@e3-login.tch.harvard.edu



srun -A crl -p bch-compute -t 1:00:00 --qos=crl --pty /bin/bash 
srun -A crl -p crl-gpu -t 1:00:00 --qos=crl --gres=gpu:4  --pty /bin/bash 
srun -A crl -p bch-gpu -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:NVIDIA_A40:4  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:NVIDIA_A100:1  --pty /bin/bash 

# alternative export 
cd /home/ch215616/ww/code/llm/experiments/LISA
# source vvenv_lisa_torch2_cuda/bin/activate
conda activate glamm_e2

# export 
export HF_HOME=/lab-share/Rad-Afacan-e2/Public/serge/code/huggingface_cache/

# check weights 
huggingface-cli scan-cache     
ls /lab-share/Rad-Afacan-e2/Public/serge/code/huggingface_cache/hub/
 
# we have the following models 
models--liuhaotian--llava-v1.5-7b # original llava 1.5 weights (already trained) but not lightning? 
models--xinlai--LISA-7B-v1 # original lisa weights obtained for lisa paper - for inference 
models--xinlai--LISA-7B-v1-explanatory  # this is not downloaded but we can use it if necessary... 

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