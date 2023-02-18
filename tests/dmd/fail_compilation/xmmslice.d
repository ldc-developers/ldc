
/* REQUIRED_ARGS: -mcpu=avx
DISABLED: win32 freebsd32 linux32 osx32 LDC_not_x86
TEST_OUTPUT:
---
fail_compilation/xmmslice.d(110): Error: `__vector(int[4])` cannot be sliced with `[]`
---
 */

#line 100

import core.simd;

int4 testz4()
{
    return [0,0,0,0];
}

void test()
{
    assert(testz4()[] == [0,0,0,0]);
}
