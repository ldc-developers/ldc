#!/bin/bash

echo "removing old objects"
mkdir -p obj
rm -f obj/*.bc
rm -f ../lib/*.bc

LDCFLAGS_ASM="-c -oq -release"
LDCFLAGS="$LDCFLAGS_ASM -noasm"

echo "compiling contract runtime"
ldc internal/contract.d -c -of../lib/ldcore.bc || exit 1 #-noruntime || exit 1

echo "compiling common runtime"
./ldc-build internal/arrays.d \
        internal/mem.d \
        internal/critical.d \
        internal/dmain2.d \
        internal/inv.d \
        $LDCFLAGS_ASM || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc obj/internal.*.bc ../lib/ldcore.bc

echo "compiling typeinfo 1"
./ldc-build typeinfos1.d $LDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc `ls obj/typeinfo1.*.bc` ../lib/ldcore.bc || exit 1

echo "compiling typeinfo 2"
./ldc-build typeinfos2.d $LDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc `ls obj/typeinfo2.*.bc` ../lib/ldcore.bc || exit 1

echo "compiling exceptions"
./ldc-build internal/eh.d $LDCFLAGS -debug || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc obj/*eh.bc ../lib/ldcore.bc || exit 1

echo "compiling object/interface casting runtime support"
ldc internal/cast.d $LDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc obj/cast.bc ../lib/ldcore.bc || exit 1

echo "compiling string foreach/switch runtime support"
ldc internal/aApply.d $LDCFLAGS || exit 1
ldc internal/aApplyR.d $LDCFLAGS || exit 1
ldc internal/switch.d $LDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc obj/aApply.bc obj/aApplyR.bc obj/switch.bc ../lib/ldcore.bc || exit 1

echo "compiling array runtime support"
ldc internal/qsort2.d internal/adi.d internal/aaA.d $LDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc obj/qsort2.bc obj/adi.bc obj/aaA.bc ../lib/ldcore.bc || exit 1

echo "compiling object implementation"
ldc internal/objectimpl.d $LDCFLAGS || exit 1
mv object.bc objectimpl.bc
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc obj/objectimpl.bc ../lib/ldcore.bc || exit 1

echo "compiling crc32"
ldc crc32.d $LDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc ../lib/ldcore.bc obj/crc32.bc || exit 1

echo "compiling llvm runtime support"
# ./ldc-build llvmsupport.d $LDCFLAGS || exit 1
ldc llvmsupport.d -oq -c || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc `ls obj/llvm*.bc` ../lib/ldcore.bc || exit 1

echo "compiling garbage collector"
cd gc
ldc $(ls *.d |grep -v win32) $LDCFLAGS_ASM -I.. ||exit 1
# ldc gclinux.d $LDCFLAGS -I.. || exit 1
# ldc gcx.d $LDCFLAGS -I.. || exit 1
# ldc gcbits.d $LDCFLAGS -I.. || exit 1
# ldc gc.d -oq -c -I.. || exit 1
mv std.gc.bc std_gc.bc
mv *.bc ../obj
# mv -v obj/*.bc ../obj 
cd ..
llvm-link -f -o=../lib/ldcore.bc obj/gclinux.bc obj/gcx.bc obj/gcbits.bc obj/std_gc.bc ../lib/ldcore.bc || exit 1

echo "compiling phobos"
./ldc-build phobos.d $LDCFLAGS || exit 1
mv *.bc obj
echo "linking phobos"
# llvm-link -f -o=../lib/ldcore.bc `ls obj/std.*.bc` ../lib/ldcore.bc || exit 1
for i in $(ls obj/std.*.bc); do
	echo $i
	llvm-link -f -o=../lib/ldcore.bc ../lib/ldcore.bc $i || exit 1
done

echo "Compiling auxiliary"
./ldc-build etc/c/zlib.d $LDCFLAGS || exit 1
mv *.bc obj
llvm-link -f -o=../lib/ldcore.bc `ls obj/etc.*.bc` ../lib/ldcore.bc || exit 1

echo "optimizing"
opt -stats -p -f -std-compile-opts -o=../lib/ldcore.bc ../lib/ldcore.bc || exit 1

echo "SUCCESS"
