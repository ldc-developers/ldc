#!/bin/bash

set -euxo pipefail

HOST_LDC_VERSION=1.9.0
ls -al $SEMAPHORE_CACHE_DIR
rm -rf $SEMAPHORE_CACHE_DIR/{ldc2-*,llvm*}
ls -al $SEMAPHORE_CACHE_DIR
git submodule update --init
sudo apt-get update
install-package ninja-build g++-4.9-multilib libcurl3:i386 libconfig++8-dev
pip install --user lit
wget https://cmake.org/files/v3.11/cmake-3.11.1-Linux-x86_64.sh && sudo sh cmake-*.sh --prefix=/usr --skip-license && rm cmake-*.sh
export CC=gcc-4.9
export CXX=g++-4.9
sudo update-alternatives --install /usr/bin/ld ld /usr/bin/ld.gold 99
wget -O llvm.tar.xz https://github.com/ldc-developers/llvm/releases/download/ldc-v6.0.0/llvm-6.0.0-linux-x86_64-withAsserts.tar.xz && mkdir llvm-x64 && tar -xf llvm.tar.xz --strip 1 -C llvm-x64 && rm llvm.tar.xz
wget https://github.com/ldc-developers/ldc/releases/download/v$HOST_LDC_VERSION/ldc2-$HOST_LDC_VERSION-linux-x86_64.tar.xz && tar -xf ldc2-*.tar.xz && rm ldc2-*.tar.xz
mkdir build
cd build
cmake -G Ninja -DMULTILIB=ON -DLDC_INSTALL_LTOPLUGIN=ON -DLDC_INSTALL_LLVM_RUNTIME_LIBS=ON -DLLVM_ROOT_DIR=$PWD/../llvm-x64 -DD_COMPILER=$PWD/../ldc2-$HOST_LDC_VERSION-linux-x86_64/bin/ldmd2 ..
