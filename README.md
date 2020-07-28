 # Table of Contents
1. [Introduction](README.md#Introduction)
2. [Directory](README.md#Directory)
3. [Program structure](README.md#Program-structure)
4. [Setup](README.md#Setup)
5. [Run the program](README.md#Run-the-program)

# Introduction
This program takes an image taken indoor as input and uses an SSD-inception deep neural network to predict whether there is an exit sign and where it locates in the image (with a confidence level between 0-1). This is a part of a vision-based indoor navigation system for the blinds developed by CVLab lead by professor Roberto Manduchi at UC Santa Cruz.

Demo image:

![alt text](https://github.com/trungnguyencs/Exit-Sign-Detector/blob/master/git_img/demo.png)

![alt text](https://github.com/trungnguyencs/Exit-Sign-Detector/blob/master/git_img/loss.png)

# Directory
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
│   ├── images
│   │   └── test
│   │       └── (some jpg images)
│   └── records
│       ├── train.record
│       └── test.record
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
* ```run.sh```: run this script and it will run ```model_main.py``` with some specified parameters to start training the DNN
* ```annotations```: containing ```label_map.pbtxt```, which is where you specify the number of classes (in our case there's only 1 class) and the classes' names
* ```ckpt```: the check point data files generated during the training process
* ```data```: containing:
  - ```records```: containing the ```train.record``` and ```test.record```. These are the data used to train our model
  - ```csv```: containing the csv data files, which were used to generate the above ```.record``` files using the python script in ```preprocessing/generate_tfrecord.py```
  - ```images```: here I only store test images so that they can be used to test and demo the working DNN using the ```exit_sign_detector_demo.ipynb``` file in ```test_model```
* ```export-inference-graph```: containing the ```run_export.sh``` which runs the ```export_inference_graph.py``` (with some specified parameters) in order to generate an inference graph (which is used to predict the labels in ```exit_sign_detector_demo.ipynb```) from a checkpoint generated during the training process
* ```preprocessing```: containing: 
  - ```split_train_test.ipynb``` to remdomly split the data to training set + testing set (ratio 80/20) and save them to the csv files in ```data/csv```
  - ```generate_tfrecord.py``` to convert the csv files to ```.record``` files
* ```pre-trained-model```: containing the downloaded pre-trained model so that we can use transfer learning and not having to train our exit sign detector from scratch
* ```results```: containing the result predicted images. Confident 0.01 means that the images show all the labels which have confidence >= 1 percent
* ```test_model```: the ```exit_sign_detector_demo.ipynb``` runs the exported inference graph and makes detector prediction. It uses the images in ```data/images/test``` and write the results to ```results```

# Setup
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
Note that Tensorflow 1.14 is only compatible with CUDA 10.0, so in order to run this program, you need to install this version of CUDA.
In addition, our lab already has CUDA 10.2 installed so you may have to change the ```$PATH``` in ```~/.bashrc``` so that it runs CUDA 10.0 instead of CUDA 10.2.
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

## Your ```~/.bashrc``` file

After all the installation above, my ```~/.bashrc``` looks like this. Copied it here for your reference:

```
# export PATH=/home/trung/anaconda3/bin:$PATH
export PATH=/home/trung/TensorFlow/Protobuf/bin:$PATH
export PYTHONPATH=$PYTHONPATH:/home/trung/TensorFlow/models/research/slim
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

# Run the program
Please refer to this page regarding how to run the program:

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

It shows you how to:
* Organise your workspace/training files
* Prepare/annotate image datasets
* Generate tf records from such datasets
* Configure a simple training pipeline
* Train a model and monitor it’s progress
* Export the resulting model and use it to detect objects

You should follow all the steps listed above, EXCEPT the ```Annotate image datasets``` part, since we did not use their tool ```labelImg``` to annotate the images as we used ```LabelBox``` instead.

I have also written a script that converts the json data file to the correct csv file formatted correctly so that it can be comsumed directly by their python ```generate_tfrecord.py``` script. You can find my code here:

https://github.com/trungnguyencs/Exit-Sign-Distance-Measurement/blob/master/preprocessing/json_to_csv.py

Hope that this setup tutorial saves your some time. Good luck!
