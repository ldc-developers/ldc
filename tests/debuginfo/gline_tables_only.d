// RUN: %ldc -gline-tables-only --output-ll -of%t.ll %s && FileCheck %s < %t.ll
// checks that ldc with -gline-tables-only do not emit debug info

// REQUIRES: atleast_llvm309
// reason: different llvm version emits far different metadata in IR code

int main()
{
    immutable int fact=6;
    int res=1;
    for(int i=1; i<=fact; i++) res *= i;
    return res;
}

// CHECK-NOT: DW_TAG
// CHECK-NOT: DW_AT
