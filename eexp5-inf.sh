# activate 
ssh ch215616@e3-login.tch.harvard.edu
srun -A crl -p crl-gpu -t 1:00:00 --qos=crl --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:NVIDIA_A40:4  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:NVIDIA_A40:4 -N 2  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:NVIDIA_A100:1  --pty /bin/bash 


cd /home/ch215616/ww/code/llm/experiments/LISA
source venv_lisa_torch2_cuda/bin/activate

# export 
# export HF_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/
# export CUDA_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/cuda-11.7/
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64
# export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/lab-share/Rad-Afacan-e2/Public/serge/code/huggingface_cache/

# inferrence - working 
llava="xinlai/LISA-7B-v1"
CUDA_VISIBLE_DEVICES=0 python chat.py --version=$llava --precision='bf16' 

# inferrence - investigation 
    # inferrence 
    llava="xinlai/LISA-7B-v1"
    CUDA_VISIBLE_DEVICES=0 python chat.py --version=$llava

    CUDA_VISIBLE_DEVICES=0 python chat.py --version=$llava --precision='fp16' --load_in_4bit # ends up with black images 

    CUDA_VISIBLE_DEVICES=0 python chat.py --version=$llava --precision='fp16' # breaks 
    CUDA_VISIBLE_DEVICES=0 python chat.py --version=$llava --precision='bf16' # produces similar results as full model 
    CUDA_VISIBLE_DEVICES=0 python chat.py --version=$llava #full model


# app inferrence 
llava="xinlai/LISA-7B-v1"
CUDA_VISIBLE_DEVICES=0 python app.py --version=$llava --load_in_8bit --precision='bf16' # starts a port on 7860 
ssh -N -f -L 7862:localhost:7862 ch215616@gamakichi # from mac to CRL machine 
ssh -N -f -L 7862:localhost:7862 ch215616@e3-login.tch.harvard.edu # from CRL machine to E3 NODE  
ssh -N -f -L 7862:localhost:7860 ch215616@gpu-b07-0 # from e3 node to ALLOCATED GPU cluster 
ssh -N -f -L LOCALPORT:HOST:REMOTEPORT 
# kind of works but kind of broken 
