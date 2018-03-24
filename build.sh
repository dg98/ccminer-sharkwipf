#!/bin/bash

# Simple script to create the Makefile and build

# export PATH="$PATH:/usr/local/cuda/bin/"

CORES="$(grep -c ^processor /proc/cpuinfo)"

make distclean || echo clean

rm -f Makefile.in
rm -f config.status
./autogen.sh || echo done

# CFLAGS="-O2" ./configure
./configure.sh

make -j $CORES
