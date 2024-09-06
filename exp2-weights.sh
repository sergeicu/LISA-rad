
# download weights of SAM 
pip install -U "huggingface_hub[cli]"
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth # in LISA folder 


#######################################################
# llava - llama - lisa weights descriptions 
#######################################################


# list available HF models 
ls ~/w/code/huggingface_cache/hub/

# we have the following models 
models--liuhaotian--LLaVA-Lightning-7B-delta-v1-1 # delta that needs to be applied on top of llama weights to get llava to work 
models--huggyllama--llama-7b # original llama weights downloaded from non official repo - https://huggingface.co/huggyllama/llama-7b
models--liuhaotian--llava-v1.5-7b # original llava 1.5 weights (already trained) but not lightning? 
models--xinlai--LISA-7B-v1 # original lisa weights obtained for lisa paper - for inference 
models--xinlai--LISA-7B-v1-explanatory  # this is not downloaded but we can use it if necessary... 





#######################################################
# download weights 
#######################################################
    # download llava delta weights from HF -> the ones that must be added to llama to make a complete llava model 
    delta=liuhaotian/LLaVA-Lightning-7B-delta-v1-1
    export HF_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/
    huggingface-cli download $delta 
        # check if correctly stored 
        ls ~/w/code/huggingface_cache/hub/models--liuhaotian--LLaVA-Lightning-7B-delta-v1-1
        huggingface-cli scan-cache      # scan cache 
        # move the weights if $HF_HOME was not defined properly
        # mv ~/.cache/huggingface/hub/models--liuhaotian--LLaVA-Lightning-7B-delta-v1-1 ~/w/code/huggingface_cache/hub/

    # download llama weights from HG -> the ones that will server as a base for adding llava delta weights (to make a complete model)
    w=huggyllama/llama-7b
    huggingface-cli download $w --cache-dir $HF_HOME

    # download pretrained llava weights (already merged on top of llama)
    w=liuhaotian/llava-v1.5-7b
    huggingface-cli download $w --cache-dir $HF_HOME


    # [unused] download llavav.16 vicuna weights 
    huggingface-cli download liuhaotian/llava-v1.6-vicuna-7b # WARNING: must move weights to local location 

    # download LISA-7B-v1 weights (already trained)
    huggingface-cli download --cache-dir $HF_HOME xinlai/LISA-7B-v1 # https://huggingface.co/xinlai/LISA-7B-v1


    # we will also need to download llama tokenizer (which the training script does automatically tbh)
    transformers.models.llama.tokenization_llama.LlamaTokenizer

    # download all the models that went to wrong place... ~/w/huggingface/
    export HF_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/
    huggingface-cli scan-cache 
    models--openai/clip-vit-large-patch14

    # download bigger models for lisa 
    w=xinlai/LISA-13B-llama2-v1-explanatory
    huggingface-cli download $w --cache-dir $HF_HOME


#######################################################

# merge and move 
#######################################################


    # apply delta 
    python3 -m llava.model.apply_delta \
        --base /path/to/llama-7b \
        --target /output/path/to/LLaVA-7B-v0 \
        --delta liuhaotian/LLaVA-7b-delta-v0


    # how move hugginface directory 
    https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory
    export HF_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/






#######################################################
# activate 
#######################################################

# activate 
ssh ch215616@e3-login.tch.harvard.edu
srun -A crl -p crl-gpu -t 1:00:00 --qos=crl --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
conda activate glamm_e2 
cd /home/ch215616/w/code/llm/experiments/LISA
source venv_lisa/bin/activate

# export 
export HF_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/huggingface_cache/
export CUDA_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/cuda-11.7/
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH



