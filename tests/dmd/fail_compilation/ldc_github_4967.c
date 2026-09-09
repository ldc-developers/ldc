/*
TEST_OUTPUT:
---
fail_compilation/ldc_github_4967.c(12): Error: cannot take address of register variable `ax`
fail_compilation/ldc_github_4967.c(13): Error: cannot take address of register variable `ax`
---
*/

void f()
{
    register int ax;
    asm("" : "=m" (ax));
    asm("" :: "m" (ax));
}
