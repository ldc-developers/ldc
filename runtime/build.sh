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
make -f ldc-posix.mak lib
cd ../..

# install the runtime
rm -f lib/libldc-runtime*.a
cp runtime/internal/libldc-runtime*.a lib
rm -f lib/libtango-gc-basic*.a
cp tango/lib/gc/basic/libtango-gc-basic*.a lib
rm -f lib/libtango-cc-tango*.a
cp tango/lib/common/tango/libtango-cc-tango*.a lib
