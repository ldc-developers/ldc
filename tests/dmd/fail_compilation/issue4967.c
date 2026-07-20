/* TEST_OUTPUT:
---
fail_compilation/issue4967.c(10): Error: cannot take address of register variable `ax`
fail_compilation/issue4967.c(11): Error: cannot take address of register variable `ax`
---
 */

void f()
{
    register int ax;
    asm("" : "=m" (ax));
    asm("" :: "m" (ax));
}
