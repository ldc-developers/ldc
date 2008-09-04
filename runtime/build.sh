#!/bin/bash

# I'm no good bash scripter ...

# copy imports
mkdir -p ../tango/llvmdc
cp internal/llvmdc/bitmanip.d ../tango/llvmdc/bitmanip.di
cp internal/llvmdc/vararg.d ../tango/llvmdc/vararg.di
cp import/llvmdc/* ../tango/llvmdc

# make the runtime
cp -R lib ../tango
cd ../tango/lib
make -f llvmdc-posix.mak clean
make -f llvmdc-posix.mak
cd ../..

# install the runtime
rm -f lib/libllvmdc-runtime*.a
cp runtime/internal/libllvmdc-runtime*.a lib
rm -f lib/libtango-gc-basic*.a
cp tango/lib/gc/basic/libtango-gc-basic*.a lib
rm -f lib/libtango-cc-tango*.a
cp tango/lib/common/tango/libtango-cc-tango*.a lib
