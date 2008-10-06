#!/bin/bash

# I'm no good bash scripter ...

# copy imports
mkdir -p ../tango/ldc
cp internal/ldc/bitmanip.d ../tango/ldc/bitmanip.di
cp internal/ldc/vararg.d ../tango/ldc/vararg.di
cp import/ldc/* ../tango/ldc

# make the runtime
cp -R lib ../tango
cd ../tango/lib
make -f ldc-posix.mak clean
make -f ldc-posix.mak sharedlib
cd ../..

# install the runtime
rm -f lib/libldc-runtime-shared.so
cp runtime/internal/libldc-runtime-shared.so lib
rm -f lib/libtango-gc-basic-shared.so
cp tango/lib/gc/basic/libtango-gc-basic-shared.so lib
rm -f lib/libtango-cc-tango-shared.so
cp tango/lib/common/tango/libtango-cc-tango-shared.so lib
