// REQUIRED_ARGS: -o- -fPIC
// DISABLED: win32 win64 LDC // LDC: works as expected
/*
TEST_OUTPUT:
---
fail_compilation/fail13939.d(15): Error: cannot directly load global variable `val` with PIC or PIE code
---
*/
version(Windows) static assert(0);
void test1()
{
    __gshared int val;
    asm
    {
        mov EAX, val;
    }
}
