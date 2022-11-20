/*
REQUIRED_ARGS: -m64 -o-
PERMUTE_ARGS:
TEST_OUTPUT:
---
fail_compilation/ldc_diag8425.d(10): Error: T in __vector(T) must be a static array, not `void`
---
*/

alias a = __vector(void); // not static array
