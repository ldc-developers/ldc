#!/bin/bash

set -euxo pipefail

HOST_LDC_VERSION=1.19.0

export DEBIAN_FRONTEND=noninteractive
dpkg --add-architecture i386
apt-get -q update
apt-get -yq install \
  git-core cmake ninja-build g++-multilib \
  llvm-dev zlib1g-dev libclang-common-6.0-dev \
  libcurl4 libcurl4:i386 \
  curl gdb python-pip tzdata unzip zip
pip install --user lit
update-alternatives --install /usr/bin/ld ld /usr/bin/ld.gold 99
curl -L -o ldc.tar.xz https://github.com/ldc-developers/ldc/releases/download/v$HOST_LDC_VERSION/ldc2-$HOST_LDC_VERSION-linux-x86_64.tar.xz && \
  mkdir ldc-x64 && \
  tar -xf ldc.tar.xz --strip 1 -C ldc-x64 && \
  rm ldc.tar.xz

git submodule update --init
mkdir build
cd build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMULTILIB=ON \
  -DLDC_INSTALL_LTOPLUGIN=ON \
  -DLDC_INSTALL_LLVM_RUNTIME_LIBS=ON \
  -DD_COMPILER=$PWD/../ldc-x64/bin/ldmd2 \
  ..
