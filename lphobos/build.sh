#!/bin/bash

echo "removing old objects"
mkdir -p obj
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


echo "compiling typeinfo 1"
rebuild typeinfos1.d -c -oqobj -dc=llvmdc-posix -v || exit 1
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/typeinfo1.*.bc` ../lib/llvmdcore.bc || exit 1

echo "compiling typeinfo 2"
rebuild typeinfos2.d -c -oqobj -dc=llvmdc-posix || exit 1
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/typeinfo2.*.bc` ../lib/llvmdcore.bc || exit 1

echo "compiling string foreach runtime support"
llvmdc internal/aApply.d -c -odobj || exit 1
llvmdc internal/aApplyR.d -c -odobj || exit 1
llvm-link -f -o=../lib/llvmdcore.bc obj/aApply.bc obj/aApplyR.bc ../lib/llvmdcore.bc || exit 1

echo "compiling array runtime support"
llvmdc internal/qsort2.d -c -odobj || exit
llvm-link -f -o=../lib/llvmdcore.bc obj/qsort2.bc ../lib/llvmdcore.bc || exit 1
llvmdc internal/adi.d -c -odobj || exit
llvm-link -f -o=../lib/llvmdcore.bc obj/adi.bc ../lib/llvmdcore.bc || exit 1

echo "compiling object implementation"
llvmdc internal/objectimpl.d -c -odobj || exit 1
llvm-link -f -o=../lib/llvmdcore.bc obj/objectimpl.bc ../lib/llvmdcore.bc || exit 1

echo "compiling llvm runtime support"
rebuild llvmsupport.d -c -oqobj -dc=llvmdc-posix || exit 1
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/llvm.*.bc` ../lib/llvmdcore.bc || exit 1

echo "compiling garbage collector"
llvmdc gc/gclinux.d -c -odobj || exit 1
llvmdc gc/gcstub.d -c -odobj -Igc || exit 1
llvm-link -f -o=../lib/llvmdcore.bc obj/gclinux.bc obj/gcstub.bc ../lib/llvmdcore.bc || exit 1

echo "compiling phobos"
rebuild phobos.d -c -oqobj -dc=llvmdc-posix -v || exit 1
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/std.*.bc` ../lib/llvmdcore.bc || exit 1

echo "optimizing"
opt -f -std-compile-opts -o=../lib/llvmdcore.bc ../lib/llvmdcore.bc || exit 1


echo "SUCCESS"
