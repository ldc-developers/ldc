#!/bin/bash

echo "removing old objects"
rm -f obj/*.bc
rm -f ../lib/*.bc

echo "compiling contract runtime"
llvmdc internal/contract.d -c -of../lib/llvmdcore.bc -noruntime || exit 1

echo "compiling common runtime"
rebuild internal/arrays.d \
        internal/mem.d \
        internal/moduleinit.d \
        -c -oqobj -dc=llvmdc-posix || exit 1

echo "compiling module init backend"
llvm-as -f -o=obj/moduleinit_backend.bc internal/moduleinit_backend.ll || exit 1
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/internal.*.bc` ../lib/llvmdcore.bc obj/moduleinit_backend.bc || exit 1

echo "compiling object implementation"
llvmdc internal/objectimpl.d -c -odobj || exit 1
llvm-link -f -o=../lib/llvmdcore.bc obj/objectimpl.bc ../lib/llvmdcore.bc || exit 1

echo "compiling typeinfos"
rebuild typeinfos.d -c -oqobj -dc=llvmdc-posix || exit 1
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/typeinfo.*.bc` ../lib/llvmdcore.bc || exit 1

echo "compiling llvm runtime support"
rebuild llvmsupport.d -c -oqobj -dc=llvmdc-posix || exit
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/llvm.*.bc` ../lib/llvmdcore.bc || exit 1

echo "optimizing"
opt -f -std-compile-opts -o=../lib/llvmdcore.bc ../lib/llvmdcore.bc || exit 1

# build phobos
echo "compiling phobos"
rebuild phobos.d -c -oqobj -dc=llvmdc-posix || exit 1
llvm-link -f -o=../lib/lphobos.bc `ls phobos_obj/*.bc` || exit 1
opt -f -std-compile-opts -o=../lib/lphobos.bc ../lib/lphobos.bc || exit 1

echo "SUCCESS"
