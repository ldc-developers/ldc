// ThinLTO: Test that module ctors/dtors are called

// REQUIRES: LTO
// UNSUPPORTED: FreeBSD

// RUN: %ldc -flto=thin -O3 -run %s | FileCheck %s

// CHECK: ctor
// CHECK: main
// CHECK: dtor

import core.stdc.stdio;

static this()
{
    puts("ctor\n");
}

static ~this()
{
    puts("dtor\n");
}

void main() {
    puts("main\n");
}
