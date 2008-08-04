#!/bin/bash

echo "removing old objects"
mkdir -p obj
rm -f obj/*.bc
rm -f ../lib/*.bc

LLVMDCFLAGS="-c -odobj -oq -gc -noasm"
LLVMDCFLAGS_ASM="-c -odobj -oq -gc"

echo "compiling contract runtime"
llvmdc internal/contract.d -c -of../lib/llvmdcore.bc || exit 1 #-noruntime || exit 1

echo "compiling common runtime"
./llvmdc-build internal/arrays.d \
        internal/mem.d \
        internal/critical.d \
        internal/dmain2.d \
        $LLVMDCFLAGS_ASM || exit 1
mv *.bc obj

echo "compiling typeinfo 1"
./llvmdc-build typeinfos1.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/typeinfo1.*.bc` ../lib/llvmdcore.bc || exit 1

echo "compiling typeinfo 2"
./llvmdc-build typeinfos2.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/typeinfo2.*.bc` ../lib/llvmdcore.bc || exit 1

echo "compiling exceptions"
./llvmdc-build internal/eh.d $LLVMDCFLAGS -debug || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc obj/*eh.bc ../lib/llvmdcore.bc || exit 1

echo "compiling object/interface casting runtime support"
llvmdc internal/cast.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc obj/cast.bc ../lib/llvmdcore.bc || exit 1

echo "compiling string foreach/switch runtime support"
llvmdc internal/aApply.d $LLVMDCFLAGS || exit 1
llvmdc internal/aApplyR.d $LLVMDCFLAGS || exit 1
llvmdc internal/switch.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc obj/aApply.bc obj/aApplyR.bc obj/switch.bc ../lib/llvmdcore.bc || exit 1

echo "compiling array runtime support"
llvmdc internal/qsort2.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc obj/qsort2.bc ../lib/llvmdcore.bc || exit 1
llvmdc internal/adi.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc obj/adi.bc ../lib/llvmdcore.bc || exit 1
llvmdc internal/aaA.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc obj/aaA.bc ../lib/llvmdcore.bc || exit 1

echo "compiling object implementation"
llvmdc internal/objectimpl.d -c -odobj -g || exit 1
llvm-link -f -o=../lib/llvmdcore.bc obj/objectimpl.bc ../lib/llvmdcore.bc || exit 1

echo "compiling llvm runtime support"
./llvmdc-build llvmsupport.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/llvm.*.bc` ../lib/llvmdcore.bc || exit 1

echo "compiling garbage collector"
cd gc
llvmdc $(ls *.d |grep -v win32) $LLVMDCFLAGS_ASM -I.. ||exit 1
# llvmdc gclinux.d $LLVMDCFLAGS -I.. || exit 1
# llvmdc gcx.d $LLVMDCFLAGS -I.. || exit 1
# llvmdc gcbits.d $LLVMDCFLAGS -I.. || exit 1
# llvmdc gc.d $LLVMDCFLAGS -I.. || exit 1
mv std.gc.bc gc.bc
mv *.bc ../obj
cd ..
llvm-link -f -o=../lib/llvmdcore.bc obj/gclinux.bc obj/gcx.bc obj/gcbits.bc obj/gc.bc ../lib/llvmdcore.bc || exit 1

echo "compiling phobos"
./llvmdc-build phobos.d $LLVMDCFLAGS || exit 1
mv *.bc obj
echo "linking phobos"
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/std.*.bc` ../lib/llvmdcore.bc || exit 1

echo "Compiling auxiliary"
./llvmdc-build etc/c/zlib.d $LLVMDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/llvmdcore.bc `ls obj/etc.*.bc` ../lib/llvmdcore.bc || exit 1

echo "optimizing"
opt -stats -p -f -std-compile-opts -disable-inlining -o=../lib/llvmdcore.bc ../lib/llvmdcore.bc || exit 1


echo "SUCCESS"
