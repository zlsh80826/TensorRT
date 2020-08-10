[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)



# TensorRT Open Source Software

* This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT.
* This branch provides instructions to **build TensorRT-OSS 7.1 based on TensorRT 6 release**

## Build

### Step 1
Download the OSS 7.1 release

```
$ git clone 'https://github.com/NVIDIA/TensorRT.git'
$ cd TensorRT
$ git checkout origin/release/7.1
$ git submodule update --init --recursive
```

### Step 2
Download the [TensorRT release](https://developer.nvidia.com/zh-cn/tensorrt) with both 6.0 and 7.1

* For TensorRT 6.0, select the tarfile installation with the target architecture and cuda version, e.g Ubuntu 18.04/CUDA 10.1
* For TensorRT 7.1, any version is ok, we just need it to bypass the myelin library check (will not be used)

```
$ # after downloading TensorRT release ...
$ tar -xf TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz
$ tar -xf TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
$ cp -a TensorRT-7.1.3.4/lib/libmyelin.so* TensorRT-6.0.1.5 # TensorRT-7.1 CMakefile will check whether myelin libraries exist
```

### Step 3
Set `$TRT_RELEASE` to the path of `TensorRT-6.0.1.5` and `$TRT_SOURCE` to the path of this repo

```
$ export TRT_RELEASE=`pwd`/TensorRT-6.0.1.5
$ export TRT_SOURCE=`pwd`
```

### Step 4
Configure cmake and build


## Test

python build.py --fp16 && python inference.py --fp16
