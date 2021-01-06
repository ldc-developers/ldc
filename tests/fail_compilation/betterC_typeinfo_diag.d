// https://github.com/ldc-developers/ldc/issues/3631

// RUN: not %ldc -betterC %s 2> %t.stderr
// RUN: FileCheck %s < %t.stderr

extern(C) void main()
{
    // CHECK: betterC_typeinfo_diag.d([[@LINE+1]]): Error: `TypeInfo` cannot be used with -betterC
    int[] foo = [1];
}
