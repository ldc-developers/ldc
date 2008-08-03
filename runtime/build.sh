#!/bin/bash

# I'm no good bash scripter ...

# copy imports
cp -u internal/llvmdc/bitmanip.d ../import/llvmdc/bitmanip.di
cp -u internal/llvmdc/vararg.d ../import/llvmdc/vararg.di

# make the runtime
cp -Ru lib ../tango
cd ../tango/lib
make -f llvmdc-posix.mak clean
make -f llvmdc-posix.mak

# install the runtime
rm -f ../../lib/libtango-base-llvmdc-native.a
cp `pwd`/libtango-base-llvmdc-native.a ../../lib
