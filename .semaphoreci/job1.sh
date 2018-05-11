#!/bin/bash

set -euxo pipefail

free -m
cd build
ninja -j2
bin/ldc2 -version

EC=0

# LDC D unittests
ctest --output-on-failure -R ldc2-unittest || EC=1

# LIT tests
ctest -V -R lit-tests || EC=1

# DMD testsuite
DMD_TESTSUITE_MAKE_ARGS=-j4 ctest -V -R dmd-testsuite || EC=1

# Build & run defaultlib debug unittests.
ninja -j1 \
	druntime-test-runner-debug \
	druntime-test-runner-debug_32 \
	druntime-test-runner-debug-shared \
	druntime-test-runner-debug-shared_32 \
	phobos2-test-runner-debug \
	phobos2-test-runner-debug_32 \
	phobos2-test-runner-debug-shared \
	phobos2-test-runner-debug-shared_32
ctest -j4 --output-on-failure -R "-debug" -E "ldc2-unittest|lit-tests|dmd-testsuite"

exit $EC
