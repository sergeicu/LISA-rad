

# LISA
ssh ch215616@e3-login.tch.harvard.edu # ssh ch215616@e3-ondemand.tch.harvard.edu

srun -A crl -p crl-gpu -t 3:00:00 --qos=crl --gres=gpu:1  --pty /bin/bash 
conda activate glamm_e2 
cd /home/ch215616/w/code/llm/experiments/LISA
python -m venv venv_lisa
source venv_lisa/bin/activate
pip install -r requirements.txt
pip install wheel 
pip install ninja
pip install deepspeed 
pip install flash-attn --no-build-isolation


  

# try to reisntall flash-attn 
# to install 
export CUDA_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/cuda-11.7/
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
# export CUDA_HOME=/lab-share/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/llm/experiments/LISA/cuda-11.7/targets/x86_64-linux/lib

export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export PATH=$CUDA_HOME/bin:$PATH
cd flash-attention
python setup.py install
# pip install flash-attn --no-build-isolation

# try to reinstall flash once again 
https://github.com/Dao-AILab/flash-attention/issues/317
https://pypi.org/project/flash-attn/2.6.3/

# activate 
ssh ch215616@e3-login.tch.harvard.edu # ssh ch215616@e3-ondemand.tch.harvard.edu
srun -A crl -p crl-gpu -t 1:00:00 --qos=crl --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu -t 1:00:00 --gres=gpu:1  --pty /bin/bash 
srun -A crl -p bch-gpu-pe -t 1:00:00 --gres=gpu:1  --pty /bin/bash 

srun -A crl -p bch-gpu -t 6:00:00 --gres=gpu:1  --pty /bin/bash 
conda activate glamm_e2 
cd /home/ch215616/w/code/llm/experiments/LISA
source venv_lisa/bin/activate

# additional libraries to be installed 
pip install tensorboard 
pip install scikit-learn 
pip install scikit-image
pip install bleach # for app.py




# nvcc vs nvidia-smi 
https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi

