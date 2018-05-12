#!/bin/bash

set -euxo pipefail

free -m
cd build
ninja -j2

# Build & run defaultlib release unittests.
ninja -j1 \
	druntime-test-runner \
	druntime-test-runner_32 \
	druntime-test-runner-shared \
	druntime-test-runner-shared_32 \
	phobos2-test-runner \
	phobos2-test-runner_32 \
	phobos2-test-runner-shared \
	phobos2-test-runner-shared_32
ctest -j4 --output-on-failure -E "ldc2-unittest|lit-tests|dmd-testsuite|-debug"
