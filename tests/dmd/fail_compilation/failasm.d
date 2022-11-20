/* DISABLED: LDC_not_x86
REQUIRED_ARGS: -m32
TEST_OUTPUT:
---
fail_compilation/failasm.d(111): Error: unknown operand size `long`
---
*/

#line 100

// https://issues.dlang.org/show_bug.cgi?id=21181

uint func()
{
    asm
    {
        naked;
        inc byte ptr [EAX];
        inc short ptr [EAX];
        inc int ptr [EAX];
        inc long ptr [EAX];
    }
}
