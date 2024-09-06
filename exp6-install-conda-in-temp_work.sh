[UNUSED BECAUSE WE MOVED AWAY FROM TEMP_WORK]

# which environments were installed 
conda activate /temp_work/ch215616/miniconda33/envs/glamm_e2
/temp_work/ch215616/llm/experiments/LISA/venv_lisa # installs torch1.13.1 -> lisa provided 
/temp_work/ch215616/llm/experiments/LISA/venv_lisa_torch2  # install torch 2 (latest), without cuda in requirements.txt 
/temp_work/ch215616/llm/experiments/LISA/venv_lisa_torch2_cuda # with cuda121



# install copy of conda 
cd /temp_work/ch215616
miniconda33/bin/conda env list

conda activate glamm_e2 
conda env export > /temp_work/ch215616/glamm_e2_environment.yml
conda deactivate 
/temp_work/ch215616/miniconda33/bin/conda env create -f /temp_work/ch215616/glamm_e2_environment.yml -p /temp_work/ch215616/miniconda33/envs/glamm_e2

# install from pip 
cd /temp_work/ch215616
cd llm/experiments/LISA
git clone https://github.com/dvlab-research/LISA.git
python -m venv venv_lisa | venv_lisa_torch2
source venv_lisa/bin/activate
pip install -r requirements.txt
pip install wheel ninja deepspeed
pip install flash-attn --no-build-isolation





# 
alias python=/temp_work/ch215616/miniconda33/bin/python3
alias pip=/temp_work/ch215616/miniconda33/bin/pip3