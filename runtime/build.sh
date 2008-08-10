#!/bin/bash

# I'm no good bash scripter ...

# copy imports
mkdir -p ../tango/llvmdc
cp -u internal/llvmdc/bitmanip.d ../tango/llvmdc/bitmanip.di
cp -u internal/llvmdc/vararg.d ../tango/llvmdc/vararg.di
cp -u import/llvmdc/* ../tango/llvmdc

# make the runtime
cp -Ru lib ../tango
cd ../tango/lib
make -f llvmdc-posix.mak clean
make -f llvmdc-posix.mak

# install the runtime
rm -f ../../lib/libtango-base-llvmdc-native.a
cp `pwd`/libtango-base-llvmdc-native.a ../../lib
