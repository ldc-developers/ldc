/* DISABLED: LDC_not_x86 // because of -m32
REQUIRED_ARGS: -m32
TEST_OUTPUT:
---
fail_compilation/test15703.d(16): Error: cast from `Object[]` to `uint[]` not allowed in safe code
fail_compilation/test15703.d(18): Error: cast from `object.Object` to `const(uint)*` not allowed in safe code
fail_compilation/test15703.d(21): Error: cast from `uint[]` to `Object[]` not allowed in safe code
---
*/

// https://issues.dlang.org/show_bug.cgi?id=15703

void test() @safe
{
     auto objs = [ new Object() ];
     auto longs = cast(size_t[]) objs;          // error
     auto longc = cast(const(size_t)[]) objs;   // ok
     auto longp = cast(const(size_t)*) objs[0]; // error

     size_t[] al;
     objs = cast(Object[]) al;                  // error

     auto am = cast(int[])[];
}

void test2() @safe
{
    const(ubyte)[] a;
    auto b = cast(const(uint[])) a;
}
