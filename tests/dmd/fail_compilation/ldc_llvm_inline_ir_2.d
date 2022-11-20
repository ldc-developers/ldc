/* TEST_OUTPUT:
---
fail_compilation/ldc_llvm_inline_ir_2.d(10): Error: All parameters of a template defined with pragma `LDC_inline_ir`, except for the first one or the first three, should be types
---
*/

pragma(LDC_inline_ir)
    R inlineIR(string s, R, P...)();

alias foo = inlineIR!(``, void, 1, 2, 3);

void bar() { foo(); }
