#!/bin/bash

set -euxo pipefail

cd build
ninja -j2 all runtime/objects-unittest/std/algorithm/searching.o runtime/objects-unittest_32/std/algorithm/searching.o runtime/objects-unittest-debug/std/algorithm/searching.o runtime/objects-unittest-debug_32/std/algorithm/searching.o runtime/objects-unittest/std/range/package.o runtime/objects-unittest_32/std/range/package.o runtime/objects-unittest-debug/std/range/package.o runtime/objects-unittest-debug_32/std/range/package.o runtime/objects-unittest/std/regex/internal/tests.o runtime/objects-unittest_32/std/regex/internal/tests.o runtime/objects-unittest-debug/std/regex/internal/tests.o runtime/objects-unittest-debug_32/std/regex/internal/tests.o
ninja -j3 all-test-runners
ctest -j4 --output-on-failure -E "dmd-testsuite|ldc2-unittest|lit-tests"
