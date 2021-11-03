# 2021-VRDL_HW1

Visual Recognition using Deep Learning HW1

##  Hardware

The following specs were used to create the original solution.

* Windows 10 Home
* AMD Ryzen™ 7 4800H Processor 2.9 GHz
* NVIDIA® GeForce RTX™ 3050 Laptop GPU 4GB GDDR6

## Reproducing Submission
📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Requirements

```train
# python version: Python 3.7.11
pip3 install -r requirements.txt
```

## Dataset Preparation
You can download the data on the Codalab website：https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit

Label 
```label
  training_labels.txt
```

Data
```data
  +- training_images
    +- 0003.jpg
    +- 0008.jpg
    ...
  +- testing_images
    +- 0001.jpg
    +- 0002.jpg
    ...
```

Classes predict

```predict
  classes.txt
  testing_img_order.txt
```

## Training

You need to download Preprocessing.py, and then run the model by following:

```train
$ python3 Train.py
```

## Models

You can download models here:

- https://drive.google.com/drive/u/1/folders/14ik-vMLMJNqJ781JZSzMhLXi6fjwUPwn

## Make Submission

You can get the predict result by following:

```eval
$ python3 Predict.py
```
