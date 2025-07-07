/*
REQUIRED_ARGS: runnable/imports/imp21241a.c runnable/imports/imp21241b.c
*/
// https://github.com/dlang/dmd/issues/21241

/* LDC: no 'fix' for LDC (moved from runnable/ to fail_compilation/); make sure a warning is emitted instead
        see https://github.com/ldc-developers/ldc/pull/4949#issuecomment-2972894481
REQUIRED_ARGS: -w
TEST_OUTPUT:
---
runnable/imports/imp21241a.c(3): Warning: skipping definition of function `imp21241a.foo` due to previous definition for the same mangled name: foo
---
*/

import imp21241a;
import imp21241b;

void main(){
    int x = getA();
    assert(x==aValue);
    x = getB();
    assert(x==bValue);
}
