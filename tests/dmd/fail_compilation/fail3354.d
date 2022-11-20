/* DISABLED: LDC_not_x86
TEST_OUTPUT:
---
fail_compilation/fail3354.d(18): Error: too many operands for instruction
fail_compilation/fail3354.d(18): Error: wrong number of operands
fail_compilation/fail3354.d(19): Error: too many operands for instruction
fail_compilation/fail3354.d(19): Error: wrong number of operands
---
*/

void main()
{
    version(D_InlineAsm_X86) {}
    else version(D_InlineAsm_X86_64) {}
    else static assert(0);

    asm {
        fldz ST(0), ST(1), ST(2), ST(3);
        fld ST(0), ST(1), ST(2), ST(3);
    }
}
