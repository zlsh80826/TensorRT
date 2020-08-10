[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)



# TensorRT Open Source Software

* This repository contains the Open Source Software (OSS) components of NVIDIA TensorRT.
* This branch provides instructions to **build TensorRT-OSS 7.1 based on TensorRT 6 release**

### Step 1
Download the OSS 7.1 release

```
git clone 'https://github.com/NVIDIA/TensorRT.git'
cd TensorRT
git checkout origin/release/7.1
git submodule update --init --recursive
```

### Step 2
Download the [TensorRT release](https://developer.nvidia.com/zh-cn/tensorrt) with both 6 and 7

E.g. Ubuntu 18.04 / CUDA 10.1 / TarFile Installation
