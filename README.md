[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)



# TensorRT Open Source Software

* This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT.
* This branch provides instructions to **build TensorRT-OSS 7.1 based on TensorRT 6 release**

## Setup test environment
Clone and checkout this branch and setup the docker environment
```
$ git clone https://github.com/zlsh80826/TensorRT.git
$ cd TensorRT
$ git checkout origin/release7.1-nvinfer6
$ bash scripts/build.sh
$ bash scripts/launch
```

## Build
After above steps, you should enter the docker environment and the working dir = `/workspace/qkv_test`

### Pre-check
run `nvidia-smi` and check the Driver Version and CUDA Version

### Step 1
Download the OSS 7.1 release

```
$ git clone 'https://github.com/NVIDIA/TensorRT.git'
$ cd TensorRT
$ git checkout origin/release/7.1
$ git submodule update --init --recursive -j 4
```

### Step 2
Download the [TensorRT-6.0 release](https://developer.nvidia.com/zh-cn/tensorrt) and locate the tar.gz file at `/workspace/qkv_test/TensorRT`

* Please select the tarfile installation with the target architecture and cuda version, e.g Ubuntu 18.04/CUDA 10.1

```
$ # after downloading TensorRT release ...
$ tar -xf TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz
```

### Step 3
Set `$TRT_RELEASE` to the path of `TensorRT-6.0.1.5` and `$TRT_SOURCE` to the path of this repo

```
$ export TRT_RELEASE=`pwd`/TensorRT-6.0.1.5
$ export TRT_SOURCE=`pwd`
```

### Step 4
Configure cmake and build, please make sure `nvcc` is in the `PATH`
```
$ cd $TRT_SOURCE
$ mkdir -p build && cd build
$ cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=$TRT_SOURCE/lib -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF
$ make -j`nproc` nvinfer_plugin
```

## Step 5 relink libnvinfer
```
$ cd $TRT_SOURCE/lib
$ unlink libnvinfer_plugin.so && unlink libnvinfer_plugin.so.7
$ ln -sf libnvinfer_plugin.so.7.1.3 libnvinfer_plugin.so.6 && ln -sf libnvinfer_plugin.so.6 libnvinfer_plugin.so
```

## Test
Please test following scripts on the GPU with architecture `Turing`

```
$ export LD_LIBRARY_PATH=$TRT_SOURCE/lib:$LD_LIBRARY_PATH
$ cd /workspace/qkv_test/test # enter test directory
$ python build.py --fp16 && python inference.py --fp16
```

if success, it should output:
```
[ 4.8813505e-05  2.1518937e-04  1.0276338e-04  4.4883182e-05
 -7.6345197e-05  1.4589411e-04 -6.2412786e-05  3.9177301e-04
  4.6366276e-04 -1.1655848e-04]
[-1.8144418e-10            nan -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10            nan
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10            nan            nan
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10            nan            nan -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10            nan -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10 -1.8144418e-10
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
  0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00]
```

## Problem Shooting

### docker: Unknown runtime specified nvidia
If you meet follow message, please change the option `--runtime=nvidia` to `--gpus all` in the `scripts/launch.sh`
```
docker: Error response from daemon: Unknown runtime specified nvidia.
See 'docker run --help'.
```
