/* TEST_OUTPUT:
---
fail_compilation/ldc_llvm_inline_ir_1.d(8): Error: the `LDC_inline_ir` pragma template must have three (string, type and type tuple) or five (string, string, string, type and type tuple) parameters
---
*/

pragma(LDC_inline_ir)
    R inlineIR(int i, R, P...)(P);
