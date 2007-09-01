#!/bin/bash

if [ -z $1 ]; then
    echo "you need to specify the test name"
    exit 1
fi

if [ "$2" = "ll" ]; then
    make &&
    llvmdc $1 -Itest -odtest -c &&
    llvm-dis -f $1.bc &&
    cat $1.ll
    exit $?
elif [ "$2" = "run" ]; then
    make &&
    llvmdc $1 -Itest -odtest -of$1 &&
    $1
    exit $?
elif [ "$2" = "c" ]; then
    make &&
    llvmdc $1 -Itest -odtest -c
    exit $?
elif [ "$2" = "gdb" ]; then
    make &&
    gdb --args llvmdc $1 -Itest -odtest '-c'
    exit $?
else
    echo "bad command or filename"
fi
