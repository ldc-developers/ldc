// https://issues.dlang.org/show_bug.cgi?id=23816
// DISABLED: LDC_not_x86
/*
TEST_OUTPUT:
---
fail_compilation/fail23816.d(14): Error: unknown opcode `NOP`
---
*/

void main()
{
    asm
    {
        NOP;
    }
}
