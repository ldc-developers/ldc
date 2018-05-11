#!/bin/bash

set -euxo pipefail

free -m
cd build
ninja -j2
bin/ldc2 -version
ctest --output-on-failure -R ldc2-unittest
ctest -V -R lit-tests
DMD_TESTSUITE_MAKE_ARGS=-j4 ctest -V -R dmd-testsuite
