#!/bin/bash

if [ "$1" = "gdb" ]; then
dc_cmd="gdb --args llvmdc"
else
dc_cmd="llvmdc"
fi

$dc_cmd internal/contract.d \
        internal/arrays.d \
        internal/moduleinit.d \
        -c -noruntime -odobj || exit 1

llvm-as -f -o=obj/moduleinit_backend.bc internal/moduleinit_backend.ll || exit 1
llvm-link -f -o=../lib/llvmdcore.bc obj/contract.bc obj/arrays.bc obj/moduleinit.bc obj/moduleinit_backend.bc || exit 1

$dc_cmd internal/objectimpl.d -c -odobj || exit 1
llvm-link -f -o=obj/all.bc obj/contract.bc obj/arrays.bc obj/moduleinit.bc obj/objectimpl.bc obj/moduleinit_backend.bc || exit 1

opt -f -std-compile-opts -o=../lib/llvmdcore.bc obj/all.bc || exit 1

if [ "$1" = "ll" ]; then
    llvm-dis -f -o=all.ll ../lib/llvmdcore.bc || exit 1
fi

echo SUCCESS
