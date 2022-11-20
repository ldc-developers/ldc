/* DISABLED: LDC_not_x86
TEST_OUTPUT:
---
fail_compilation/fail8168.d(9): Error: unknown opcode `unknown`
---
*/
void main() {
    asm {
        unknown; // wrong opcode
    }
}
