#!/bin/sh

if [ -d '/usr/local/cuda' ]; then
    with_cuda='=/usr/local/cuda'
elif [ -d '/opt/cuda' ]; then
    with_cuda='=/opt/cuda'
else
    echo "Cuda dir not found, you may need to add your own --with-cuda flag to ./configure."
fi

extracflags="-march=native -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"

CUDA_CFLAGS="-O3 -lineno -Xcompiler -Wall -D_FORCE_INLINES" ./configure CXXFLAGS="-O3 $extracflags" --with-cuda${with_cuda} --with-nvml=libnvidia-ml.so

