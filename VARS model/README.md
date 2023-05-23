
# VARS model

# Installation

conda create -n vars python=3.9
conda activate vars

conda install pytorch torchvision cudatoolkit=11.7 -c pytorch -c conda-forge
pip install -r requirements.txt

python main.py
