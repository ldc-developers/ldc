/*
EXTRA_SOURCES: imports/testminitAA.d imports/testminitBB.d
PERMUTE_ARGS:
LDC: adapt output to reversed AA/BB order (benign, due to reversed modules codegen order)
RUN_OUTPUT:
---
BB
AA
hello
Success
---
*/

import core.stdc.stdio;

import imports.testminitAA;
private import imports.testminitBB;

static this()
{
    printf("hello\n");
    assert(aa == 1);
    assert(bb == 1);
}

int main()
{
    printf("Success\n");
    return 0;
}
