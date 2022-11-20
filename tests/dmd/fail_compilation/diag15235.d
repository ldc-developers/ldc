/* DISABLED: LDC_not_x86
TEST_OUTPUT:
---
fail_compilation/diag15235.d(11): Error: too many registers memory operand
---
*/

void main()
{
    asm {
        mov [EBX+EBX+EBX], EAX; // prints the same error message 20 times
    }
}
