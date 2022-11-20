// REQUIRED_ARGS: -de

/* DISABLED: LDC_not_x86
TEST_OUTPUT:
---
fail_compilation/deprecate12979a.d(12): Error: `asm` statement is assumed to throw - mark it with `nothrow` if it does not
---
*/

void foo() nothrow
{
    asm
    {
        ret;
    }
}
