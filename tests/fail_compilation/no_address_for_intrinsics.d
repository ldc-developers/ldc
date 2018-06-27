// RUN: not %ldc -c %s 2>&1 | FileCheck %s

import ldc.intrinsics;
alias exp = llvm_exp!real;
alias exp = llvm_exp!double;
alias exp = llvm_exp!float;

void foo()
{
    import core.math;
    // CHECK: no_address_for_intrinsics.d(12): Error: cannot take the address of intrinsic function `llvm_sin`
    real function(real) psin = &sin;

    // CHECK: no_address_for_intrinsics.d(15): Error: cannot take the address of intrinsic function `llvm_exp`
    real function(real) pexp = &exp;
}
