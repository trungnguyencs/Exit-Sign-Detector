 # Table of Contents
1. [Introduction](README.md#introduction)
1. [Directory structure](README.md#directory-structure)
1. [Program structure](README.md#program-structure)
1. [Run the code](README.md#run-the-code)

# Introduction
This program calculates the distance from a camera to an exit sign in real-life, given the image containing the exit sign taken by the camera and coordinates of its four corners, together with the sign dimensions and the camera intrinsic parameters.

This program serves two purposes. First, this distance estimation model placed on top of a deep learning model, which assuming can accurately segment the four corners of the sign automatically, would be able to calculate the distance from the exit sign to the camera. Second, this distance estimation model works as a “labeler”: given a large dataset of exit sign images, with this model, we can obtain the exit sign distance from each image and feed it into a deep learning model that detects exit signs and predicts its distance from regression learning.

# Requirements
Requires Python 2.7 and the following python libraries:
* numpy
* cv2
* json
* glob
* yaml

# Directory structure
```
├── annotations
│   └── label_map.pbtxt
├── ckpt
│   ├── checkpoint
│   ├── graph.pbtxt
│   └── (some model.ckpt files)
├── data
│   ├── csv
│   │   ├── quadrilateral-1787.csv
│   │   ├── quadrilateral-test.csv
│   │   └── quadrilateral-train.csv
│   └── images
│       └── test
│           └── (some jpg images)
├── export-inference-graph
│   ├── ckpt
│   │   ├── model.ckpt-5715.data-00000-of-00001
│   │   ├── model.ckpt-5715.index
│   │   └── model.ckpt-5715.meta
│   ├── export_inference_graph.py
│   ├── inference_graph
│   │   ├── checkpoint
│   │   ├── frozen_inference_graph.pb
│   │   ├── model.ckpt.data-00000-of-00001
│   │   ├── model.ckpt.index
│   │   ├── model.ckpt.meta
│   │   ├── pipeline.config
│   │   └── saved_model
│   │       ├── saved_model.pb
│   │       └── variables
│   └── run_export.sh
├── model_main.py
├── preprocessing
│   ├── generate_tfrecord.py
│   └── split_train_test.ipynb
├── pre-trained-model
│   ├── ssd_inception_v2_coco_2018_01_28
│   │   ├── checkpoint
│   │   ├── frozen_inference_graph.pb
│   │   ├── model.ckpt.data-00000-of-00001
│   │   ├── model.ckpt.index
│   │   ├── model.ckpt.meta
│   │   ├── pipeline.config
│   │   └── saved_model
│   │       ├── saved_model.pb
│   │       └── variables
│   └── ssd_inception_v2_coco.config
├── README.md
├── results
│   └── boxed_images
│       └── confidence_0.01
│           └── (some jpg images)
├── run.sh
└── test_model
    ├── exit_sign_detector_demo.ipynb
    ├── run_notebook.sh
    └── utils
```

# Program structure
## Main files
## Data
## Other folders:

# Installation
## Install pip + Anaconda
* Install pip:
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```
* Install Anaconda: 
Download Anaconda Python 3.7 version for Linux: https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh 
```
bash Anaconda3-2018.12-Linux-x86_64.sh
```

## Install Tensorflow CPU
Create a new environment:
```
conda create -n tfcpu pip python=3.7
conda activate tfcpu 
pip install --ignore-installed --upgrade tensorflow==1.14
```

## Install Tensorflow GPU:
### Install CUDA v10.0
Note that Tensorflow 1.14 is only compatible with CUDA 10.0, so in order to run this program, you will need to install this version of CUDA.
In addition, our lab already has CUDA 10.2 installed so you may have to change the ```$PATH``` in ```~/.bashrc``` so that it runs CUDA 10.0 and not CUDA 10.2.
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-toolkit-10-0
```
### Add the path to your environment
```
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Check that you have installed the right version
```
nvcc -V
```

### Check out details about your video card
```
nvidia-smi
```
If it shows "Failed to initialize NVML: Driver/library version mismatch" then reboot:
```
sudo reboot
```

### Install cuDNN v7.6.5 Library for Linux
- Create a user NVIDIA developer profile and log in
- Download the tar from:
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.0_20191031/cudnn-10.0-linux-x64-v7.6.5.32.tgz
```
tar xvf cudnn*.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

### Install Tensorflow GPU
```
conda create -n tfgpu pip python=3.7
conda activate tfgpu
pip install --upgrade tensorflow-gpu==1.14
```

### Test to see if the installation was successful:
Run python
```
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

## Install the following necessary packages
```
conda install pillow, lxml, jupyter, matplotlib, opencv, cython
```
or 
```
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
pip install cython
pip install opencv-python
```
(opencv-python instead of opencv)

## Install object detection package
Install the Tensorflow\models\research\object_detection package by running the following from Tensorflow\models\research:
```
# From within TensorFlow/models/research/
pip install .
```

## Add research/slim to your PYTHONPATH
```
# From within tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/slim
```

## Install Protobuf
* Head to the protoc releases page: https://github.com/google/protobuf/releases
* Download the latest protoc-*-*.zip release
* Extract the contents of the downloaded protoc-*-*.zip in a directory TensorFlow/Protobuf
* Extract the contents of the downloaded protoc-*-*.zip, inside TensorFlow/Protobuf
* Add to your Path environment variable:
  ```
  export PATH=/home/trung/TensorFlow/Protobuf/bin:$PATH
  ```
* In a new Terminal, cd into TensorFlow/models/research/ directory and run the following command: 
  ```
  # From within TensorFlow/models/research/ 
  protoc object_detection/protos/*.proto --python_out=.
  ```

## Install COCO API (Optional)
* The pycocotools package should be installed if you are interested in using COCO evaluation metrics, as discussed in Evaluating the Model (Optional).
* Download cocoapi to a directory of your choice, then make and copy the pycocotools subfolder to the Tensorflow/models/research directory, as such:
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/
```