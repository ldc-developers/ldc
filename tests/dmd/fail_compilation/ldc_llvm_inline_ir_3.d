/* TEST_OUTPUT:
---
fail_compilation/ldc_llvm_inline_ir_3.d(18): Error: can't parse inline LLVM IR:
`ret i32 %0`
    ^
value doesn't match function result type 'float'
The input string was:
`define float @inline.ir.0(i32)
{
ret i32 %0
}`
---
*/

pragma(LDC_inline_ir)
    R inlineIR(string s, R, P...)(P);

alias foo = inlineIR!(`ret i32 %0`, float, int);

float bar(int p) { return foo(p); }
