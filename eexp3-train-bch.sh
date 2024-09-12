# user shares 
https://chatgpt.com/share/89e40b5f-e94c-4bd4-af0b-b6fbed721ca5


# mount 
mount -t nfs -o noauto,_netdev,hard,comment=systemd.automount,nofail rc-fs-nfs.tch.harvard.edu:/Rad-Afacan-e2 /lab-share/Rad-Afacan-e2


# init 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:NVIDIA_A40:1  --pty /bin/bash 
cd /home/ch215616/ww/code/llm/experiments/LISA
# conda activate glamm_e2
conda activate lisa2


# train normal 
export HF_HOME=/lab-share/Rad-Afacan-e2/Public/serge/code/huggingface_cache/
llava="xinlai/LISA-7B-v1"
sam=sam_vit_h_4b8939.pth
data="cocostuff"
data="bchwrist"

# deepspeed --master_port=24999 train_ds.py \
python train_ds.py \
  --version=$llava \
  --vision_pretrained=$sam \
  --dataset="sem_seg" \
  --val_dataset="val" \
  --log_base_dir="./runs" \
  --sem_seg_data=$data \
  --sample_rates="1" \
  --no_eval \
  --batch_size=1 \
  --exp_name="lisa-7b6233g" 



# tried installing 
cudatoolkit-11.8.0


# checkpoint issue of training 
https://github.com/microsoft/DeepSpeed/issues/3810






# issues 
[2024-09-10 15:06:49,478] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Warning: The cache directory for DeepSpeed Triton autotune, /home/ch215616/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when
 DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
/home/ch215616/w/miniconda2/envs/glamm_e2/lib/python3.10/site-packages/huggingface_hub/file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads
 always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
You are using the legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This means that tokens that come after special tokens will not be properly handled. We reco
mmend you to read the related pull request available at https://github.com/huggingface/transformers/pull/24565
/lab-share/Rad-Afacan-e2/Public/serge/code/llm/experiments/LISA/model/segment_anything/build_sam.py:107: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), wh
ich uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECUR
ITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrar
y objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=T
rue` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state_dict = torch.load(f)