# VARS Interface

# Installation

The following instructions will help you install the required libraries to run the tools. The code runs in python 3 and was tested in a conda environment.
```
conda create -n vars python=3.9

conda activate vars

pip install git+https://github.com/ajhamdi/mvtorch

pip install -r requirements.txt
```

# Download the dataset weights

Download the weights of the model: https://drive.google.com/drive/folders/1N0Lv-lcpW8w34_iySc7pnlQ6eFMSDvXn?usp=share_link

And save the 8_model.pth.tar file in the folder "interface"

# Run the tool
Once the environment is ready, you can simply run the annotation tool for camera shots and replays with the following commands:

```
python main.py
```

Then select one or several clips in the folder "Dataset". The clips should be 5 seconds long and the foul should be around the 3th second.

