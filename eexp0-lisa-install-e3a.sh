

# LISA
ssh ch215616@e3-login.tch.harvard.edu # ssh ch215616@e3-ondemand.tch.harvard.edu
srun -A crl -p bch-gpu-pe -t 2:00:00 --gres=gpu:NVIDIA_A40:1  --pty /bin/bash 


# install pythons 
cd ~/ww/code/llm/experiments/LISA
conda activate /temp_work/ch215616/miniconda33/envs/glamm_e2
python -m venv vvenv_lisa
conda deactivate 
source vvenv_lisa/bin/activate
python -m venv vvenv_lisa_torch2
source vvenv_lisa_torch2/bin/activate
python -m venv vvenv_lisa_torch2_cuda
source vvenv_lisa_torch2_cuda/bin/activate

# install pips 
cd ~/ww/code/llm/experiments/LISA
source vvenv_lisa/bin/activate; pip install -r requirements.txt
source vvenv_lisa_torch2/bin/activate; pip install -r source vvenv_lisa_torch2/bin/activate; 
source vvenv_lisa_torch2_cuda/bin/activate; pip install -r source vvenv_lisa_torch2_cuda/bin/activate; 


# install from pip 
cd /temp_work/ch215616
cd llm/experiments/LISA
git clone https://github.com/dvlab-research/LISA.git
python -m venv venv_lisa | venv_lisa_torch2
source venv_lisa/bin/activate
pip install -r requirements.txt
pip install wheel ninja deepspeed
pip install tensorboard scikit-image bleach 
pip install flash-attn --no-build-isolation
# 
pip uninstall transformers # transformers-4.44.2 -> transformers==4.31.0
pip install transformers==4.31.0
pip install mpi4py # this one does not work - see instructions below on how to turn glamm_e2 conda env to compatible env with mpi4py and pi 



# installing inside glamm_e2 conda environment (because we need mpi4py which is not available without conda)
pip freeze > requirements_final_vvenv_lisa_torch2_cuda.txt # then go an uncomment +cu.... in torch 
# then manually add first line of code from requirements_torch_cuda.txt 
conda activate glamm_e2
conda install -c conda-forge mpich mpi4py
pip install -r requirements_final_vvenv_lisa_torch2_cuda.txt




conda install -c conda-forge gcc=11 gxx=11


/temp_work/ch215616/miniconda33/bin/conda init bash
/temp_work/ch215616/miniconda33/bin/conda activate /temp_work/ch215616/miniconda33/envs/glamm_e2


/lab-share/Rad-Afacan-e2/Public/serge/conda/bin/conda init bash
exec $SHELL
conda activate /lab-share/Rad-Afacan-e2/Public/serge/conda/envs/glamm_e2
$ conda create --prefix /lab-share/Rad-Afacan-e2/Public/serge/conda/envs/lisa1 python=3.10



WARNING:
    You currently have a PYTHONPATH environment variable set. This may cause
    unexpected behavior when running the Python interpreter in Miniconda3.
    For best results, please verify that your PYTHONPATH only points to
    directories of packages that are compatible with the Python interpreter
    in Miniconda3: /lab-share/Rad-Afacan-e2/Public/serge/conda
Do you wish the installer to initialize Miniconda3







ERROR: This cross-compiler package contains no program /home/ch215616/w/miniconda2/envs/glamm_e2/bin/x86_64-conda_cos6-linux-gnu-cc
ERROR: deactivate-gcc_linux-64.sh failed, see above for details
ERROR: This cross-compiler package contains no program /home/ch215616/w/miniconda2/envs/glamm_e2/bin/x86_64-conda_cos6-linux-gnu-cc
ERROR: activate-gcc_linux-64.sh failed, see above for details