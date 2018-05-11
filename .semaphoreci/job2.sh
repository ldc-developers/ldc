#!/bin/bash

set -euxo pipefail

free -m
cd build
ninja -j2
ninja -j1 all-test-runners
ctest -j4 --output-on-failure -E "dmd-testsuite|ldc2-unittest|lit-tests"
