// REQUIRES: atleast_llvm309
// REQUIRES: LTO

// RUN: %ldc -flto=thin -O3 -Xcc -fuse-ld=gold %S/inputs/thinlto_ctor.d -run %s | FileCheck --check-prefix=EXECUTE %s

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
