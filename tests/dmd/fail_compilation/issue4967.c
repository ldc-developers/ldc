/*
TEST_OUTPUT:
---
fail_compilation/issue4967.c(12): Error: cannot take address of register variable `ax`
fail_compilation/issue4967.c(13): Error: cannot take address of register variable `ax`
---
*/

void f()
{
    register int ax;
    asm("" : "=m" (ax));
    asm("" :: "m" (ax));
}
