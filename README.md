

# Video Assistant Referee System - VARS

The Video Assistant Referee (VAR) has revolutionized association football, enabling referees to review incidents on the pitch, making informed decisions, and ensuring fairness. However, due to the lack of referees in many countries and the high cost of the VAR infrastructure, only professional leagues can benefit from it. 
We propose a first step towards a fully automated “Video Assistant Referee System” (VARS) which could support or replace the current VAR.

This repository contains:
 * the SoccerNet-MVFoul, a new multi-view video dataset containing video clips of fouls captured by multiple cameras, annotated with 10 properties.
 * the code for the VARS, a new multi-camera video recognition system for classifying the type of fouls and their severity. 
* the VARS interface, which shows the ground truth of the action and the top 2 predictions for the foul classification task, and the offence and severity classification task with the corresponding confidence scores.

For more information
* Paper: [VARS: Video Assistant Referee System for Automated Soccer Decision Making from Multiple Views](https://arxiv.org/abs/2304.04617).


![My Image](images/abstract_image.jpg)
## SoccerNet-MVFoul

Follow the [link](https://pypi.org/project/SoccerNet/) to easily download the SoccerNet pip package.

If you want to download the data and annotations, you will need to fill a [NDA](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) to get the password.

Then use the API to downlaod the data:

```
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
mySNdl = SNdl(LocalDirectory="path/to/SoccerNet")
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="enter password")
```
To obtain the data in 720p, add version = "720p" to the input arguments. If you face issues extracting data from the train_720p.zip folder, the error may come from using the default unzip extractor. Using the app "The Unarchiver" should enable you to unzip it successfully.

The dataset consists of 3901 available actions. Each action is composed of at least two videos depicting the live action and at least one replay. 
The dataset is divided into a training set (2916 actions), validation set (411 actions), test set (301 actions) and challenge set (273 actions without the annotations).

![My Image](images/dataset_example.png)

The actions are annotated with 10 different properties describing the characteristics of the foul from a referee
perspective (e.g. the severity of the foul, the type of foul,
etc.). \
To ensure high-quality annotations, all these properties were manually annotated by a professional soccer referee with 6 years of experience and more than 300 official
games.
## VARS

Our VARS is a multi-view multi-task video architecture, that automatically identifies the type of foul and their severity. 

![My Image](images/pipeline_mvfoul.jpg)

Our system encodes per-view video features (E), aggregates the view features (A), and classifies different properties of the foul action (C).

Run the following lines to install all the dependencies:
```
conda create -n vars python=3.9

conda activate vars

Install Pytorch with CUDA : https://pytorch.org/get-started/locally/

pip install SoccerNet

pip install -r requirements.txt

pip install pyav

```
Unzip the dataset and to start the training, run the following command:

```
python main.py --path "path/to/dataset" 
```
You can download our pre-trained weights to recreate the baseline: https://drive.google.com/drive/folders/1N0Lv-lcpW8w34_iySc7pnlQ6eFMSDvXn?usp=share_link

Run the following command:

```
python main.py --pooling_type "attention" --start_frame 63 --end_frame 87 --fps 17 --path "path/to/dataset" --pre_model "mvit_v2_s" --path_to_model_weights "14_model.pth.tar"
```

## VARS interface

The VARS interface enables easy access to all available
perspectives for a particular action. The multi-task VARS,
which achieved the best results on the test set, is built directly into the interface, allowing for immediate analysis of
selected videos. The VARS interface offers top two predictions for the type of foul classification, as well as the offence and severity classification for the selected videos. Furthermore, for each prediction, the VARS interface shows the confidence score of his prediction.

![My Image](images/vars_interface.png)

Run the following lines to instal all the dependencies
```
conda create -n vars python=3.9

conda activate vars

pip install -r requirements.txt
pip install av
```
Download the weights of the model: https://drive.google.com/drive/folders/1N0Lv-lcpW8w34_iySc7pnlQ6eFMSDvXn?usp=share_link

And save the 14_model.pth.tar file in the folder "interface".

Once the environment is ready, you can simply run the interface with the following command:
```
python main.py
```
Then select one or several clips in the folder "Dataset".


## Demo
The VARS gives his top two predictions with the corresponding confidence score.

Example 1:
![My Demo](images/HighLeg_RedCard_GIF.gif)

Example 2:
![My Demo](images/Tackling_YellowCard_GIF.gif)

## License
See the [License](LICENSE) file for details.



## Citation

For further information check out our [paper](https://arxiv.org/abs/2304.04617) and supplementary material.

Please cite our work if you use our dataset or code:
```
@InProceedings{Held2023VARS,
    author    = {Held, Jan and Cioppa, Anthony and Giancola, Silvio and Hamdi, Abdullah and Ghanem, Bernard and Van Droogenbroeck, Marc},
    title     = {{VARS}: Video Assistant Referee System for Automated Soccer Decision Making From Multiple Views},
    booktitle = cvsports,
    month     = Jun,
    year      = {2023},
	publisher = ieee,
	address = seattle,
    pages     = {5085-5096}
}
```
