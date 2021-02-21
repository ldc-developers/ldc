// REQUIRES: LTO
// UNSUPPORTED: FreeBSD

// RUN: %ldc -flto=thin -O3 %S/inputs/thinlto_ctor.d -run %s | FileCheck --check-prefix=EXECUTE %s

// EXECUTE: ctor
// EXECUTE: main
// EXECUTE: dtor

import core.stdc.stdio;

static ~this()
{
    puts("dtor\n");
}

void main() {
    puts("main\n");
}
