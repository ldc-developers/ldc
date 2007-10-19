#!/bin/bash

if [ -z $1 ]; then
    echo "you need to specify the test name"
    exit 1
fi

if [ "$2" = "ll" ]; then
    llvmdc $1 -Itest -odtest -c -vv &&
    llvm-dis -f $1.bc &&
    cat $1.ll
    exit $?
elif [ "$2" = "llopt" ]; then
    llvmdc $1 -Itest -odtest -c -vv &&
    opt -f -o=$1.bc -std-compile-opts $1.bc &&
    llvm-dis -f $1.bc &&
    cat $1.ll
    exit $?
elif [ "$2" = "run" ]; then
    llvmdc $1 lib/lphobos.bc -Itest -odtest -of$1 -vv &&
    $1
    exit $?
elif [ "$2" = "c" ]; then
    llvmdc $1 -Itest -odtest -c -vv
    exit $?
elif [ "$2" = "gdb" ]; then
    gdb --args llvmdc $1 -Itest -odtest -c -vv
    exit $?
elif [ "$2" = "gdbrun" ]; then
    llvmdc $1 -Itest -odtest -c -vv &&
    gdb $1
    exit $?
else
    echo "bad command or filename"
fi
