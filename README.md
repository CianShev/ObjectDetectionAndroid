# TensorFlow Lite Object Detection for Android on Windows 10
### Overview
The purpose of this document is to detail the pipeline for configuring and local training a machine learning model for object detection on Windows 10 and deploying this model to an Android mobile device.
This document and repo has been created as a modified and updated version of the official TF repo in conjuction with a 3rd party repo [here](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

## Getting Started
Firstly, download Anaconda virtual environment. There are several version conflict issues (notably with Pip, Python and TensorFlow itself) that will prevent initialisation or training of the model, so utilising a virtual environment is neccessary.  

Navigate to [the Anaconda downloads page](https://www.anaconda.com/distribution/) and select version 3.x.x (currently 3.7), the version suitable for Python 3.x. After installation, run the Anaconda prompt with administrator privileges and enter the following commands to create our new virtual environment named tensorflow1: 

```
(base) C:\Windows\system32> cd /
(base) C:\Windows\system32> conda create -n tensorflow1 pip python=3.5
```

Once installed, activate the virtual environment and run some updates with the below commands:
```
(base) C:\> conda activate tensorflow1
(tensorflow1) C:\> python -m pip install --upgrade pip
```

Then, we can install TensorFlow using pip

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
Location: c:\programdata\anaconda3\envs\tensorflow69\lib\site-packages
Requires: grpcio, keras-preprocessing, protobuf, termcolor, gast, wheel, astor, google-pasta, keras-applications, wrapt, absl-py, tensorflow-estimator, numpy, six, tensorboard
``` 




