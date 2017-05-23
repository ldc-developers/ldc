// REQUIRES: atleast_llvm307

// RUN: %ldc -output-ll -od=%T -I%S -O0 -release -enable-cross-module-inlining %s %S/inputs/inlining_gh1712_string.d \
// RUN:   && FileCheck %s < %T/inlining_gh1712_originalbug.ll

import inputs.inlinables;

void main()
{
    call_template_foo(1);
}

// Make sure that the pragma(inline, true) function `call_template_foo` is defined, and not just declared.
// When it is correctly defined, optimization (available_externally+alwaysinline) even at -O0 means that it will dissappear from IR.
// CHECK-NOT: declare {{.*}} @call_template_foo