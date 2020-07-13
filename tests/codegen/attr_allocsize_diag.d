// Test ldc.attributes.allocSize diagnostics

// RUN: not %ldc -d-version=NORMAL %s 2>&1 | FileCheck %s --check-prefix=NORMAL
// RUN: not %ldc -d-version=THIS   %s 2>&1 | FileCheck %s --check-prefix=THIS

import ldc.attributes;

version(NORMAL)
{
// NORMAL: attr_allocsize_diag.d([[@LINE+2]]): Error: `@ldc.attributes.allocSize.sizeArgIdx=2` too large for function `my_calloc` with 2 arguments.
// NORMAL: attr_allocsize_diag.d([[@LINE+1]]): Error: `@ldc.attributes.allocSize.numArgIdx=2` too large for function `my_calloc` with 2 arguments.
extern (C) void* my_calloc(size_t num, size_t size) @allocSize(2, 2)
{
    return null;
}
}

version(THIS)
{
// Test function type with hidden `this` argument
class A
{
    // THIS: attr_allocsize_diag.d([[@LINE+2]]): Error: `@ldc.attributes.allocSize.sizeArgIdx=4` too large for function `this_calloc` with 4 arguments.
    // THIS: attr_allocsize_diag.d([[@LINE+1]]): Error: `@ldc.attributes.allocSize.numArgIdx=4` too large for function `this_calloc` with 4 arguments.
    void* this_calloc(int size, int b, size_t num, int c) @allocSize(4, 4)
    {
        return null;
    }
}
}
