#!/bin/bash

set -euxo pipefail

HOST_LDC_VERSION=1.9.0

export DEBIAN_FRONTEND=noninteractive
dpkg --add-architecture i386
apt-get -q update
apt-get -yq install git-core cmake ninja-build g++-multilib libcurl4 libcurl4:i386 curl gdb python-pip unzip zip
pip install --user lit
update-alternatives --install /usr/bin/ld ld /usr/bin/ld.gold 99
curl -L -o llvm.tar.xz https://github.com/ldc-developers/llvm/releases/download/ldc-v6.0.0/llvm-6.0.0-linux-x86_64-withAsserts.tar.xz && \
	mkdir llvm-x64 && \
	tar -xf llvm.tar.xz --strip 1 -C llvm-x64 && \
	rm llvm.tar.xz
curl -L -o ldc.tar.xz https://github.com/ldc-developers/ldc/releases/download/v$HOST_LDC_VERSION/ldc2-$HOST_LDC_VERSION-linux-x86_64.tar.xz && \
	mkdir ldc-x64 && \
	tar -xf ldc.tar.xz --strip 1 -C ldc-x64 && \
	rm ldc.tar.xz

git submodule update --init
mkdir build
cd build
cmake -G Ninja -DMULTILIB=ON -DLDC_INSTALL_LTOPLUGIN=ON -DLDC_INSTALL_LLVM_RUNTIME_LIBS=ON -DLLVM_ROOT_DIR=$PWD/../llvm-x64 -DD_COMPILER=$PWD/../ldc-x64/bin/ldmd2 ..
