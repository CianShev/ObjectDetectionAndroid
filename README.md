# TensorFlow Lite Object Detection for Android on Windows 10
### Overview
The purpose of this document is to detail the pipeline for configuring and local training a machine learning model for object detection on Windows 10 and deploying this model to an Android mobile device.
This document and repo has been created as a modified and updated version of the official TF repo in conjuction with a 3rd party repo [here](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

## Potential Issues
There are several potential pitfalls to encounter when using TensorFlow and especially when attempting to deploy the model to mobile. Common issues are addressed at the end of this README. 
Note: At the time of writing, TF2.0 currently *does not* support Object Detection as noted in the official GitHub release notes [here (esp. "Breaking Changes"](https://github.com/tensorflow/tensorflow/releases/tag/v2.0.0-alpha0), [here](https://github.com/tensorflow/models/issues/7036) and [here](https://github.com/tensorflow/models/issues/6423)

## Getting Started
Firstly, download Anaconda virtual environment. There are several version conflict issues (notably with Pip, Python and TensorFlow itself) that will prevent initialisation or training of the model, so utilising a virtual environment is neccessary.  

Navigate to [the Anaconda downloads page](https://www.anaconda.com/distribution/) and select version 3.x.x (currently 3.7), the version suitable for Python 3.x. Install Anaconda After installation, run the Anaconda prompt with administrator privileges and enter the following commands to create our new virtual environment named tensorflow1: 

```
(base) C:\Windows\system32> cd /
(base) C:\Windows\system32> conda create -n tensorflow1 pip python=3.5
```

Once installed, activate the virtual environment and run some updates with the below commands:
```
(base) C:\> conda activate tensorflow1
(tensorflow1) C:\> python -m pip install --upgrade pip
```
Failing to update pip or duplicate installs in the virtual environment will break the process. 
Then, we can install TensorFlow using pip:

```
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow
```
Check the version of TensorFlow with the below command. This tutorial was achieved using 1.14 but future versions should work fine. If you are running into errors that are not addressed in this document, rolling back to 1.14 may fix some issues

```
(tensorflow1) C:\> pip show tensorflow
(tensorflow1) C:\> pip show tensorflow
Name: tensorflow
Version: 1.14.0
Summary: TensorFlow is an open source machine learning framework for everyone.
Home-page: https://www.tensorflow.org/
Author: Google Inc.
Author-email: packages@tensorflow.org
License: Apache 2.0
Location: c:\programdata\anaconda3\envs\tensorflow1\lib\site-packages
Requires: grpcio, keras-preprocessing, protobuf, termcolor, gast, wheel, astor, google-pasta, keras-applications, wrapt, absl-py, tensorflow-estimator, numpy, six, tensorboard
``` 
Then, install the following packages. Some are used for running your model on a laptop or desktop using a webcam and some are neccessary for TensorFlow training and testing or generating TFRecord files: 

```
(tensorflow1) C:\> conda install -c anaconda protobuf
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install jupyter
(tensorflow1) C:\> pip install matplotlib
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install opencv-python
```
Next, we need to clone the official TF repo. This can be done from https://github.com/tensorflow/models or with the following command: 
```
(tensorflow1) C:\> git clone https://github.com/tensorflow/models
```
Extract the files if compressed and rename "models-master" to "models" if it has not changed already. 

