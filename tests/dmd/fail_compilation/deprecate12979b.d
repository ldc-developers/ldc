// REQUIRED_ARGS: -de

/* DISABLED: LDC_not_x86
TEST_OUTPUT:
---
fail_compilation/deprecate12979b.d(12): Error: `asm` statement is assumed to be impure - mark it with `pure` if it is not
---
*/

void foo() pure
{
    asm
    {
        ret;
    }
}
