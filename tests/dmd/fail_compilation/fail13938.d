// REQUIRED_ARGS: -o-
/* DISABLED: LDC // works as expected
TEST_OUTPUT:
---
fail_compilation/fail13938.d(14): Error: cannot directly load TLS variable `val`
---
*/

void test1()
{
    static int val;
    asm
    {
        mov EAX, val;
    }
}
