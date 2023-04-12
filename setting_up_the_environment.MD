Create conda environments without GPUs

´´´ python
conda info --envs
conda create --name tf python=3.9
conda activate tf
´´

´´´ python
pip install tensorflow==2.12.*
conda deactivate
´´
