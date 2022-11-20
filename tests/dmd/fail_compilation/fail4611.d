/* DISABLED: LDC // no arraysize limit to work around an optlink bug (https://issues.dlang.org/show_bug.cgi?id=14859)
TEST_OUTPUT:
---
fail_compilation/fail4611.d(15): Error: `Vec[$n$]` size 4 * $n$ exceeds $?:windows+32=0x1000000|0x7fffffff$ size limit for static array
---
*/

struct Vec
{
    int x;
}

void main()
{
    Vec[ptrdiff_t.max] a;
}
