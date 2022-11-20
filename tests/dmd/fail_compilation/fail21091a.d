// https://issues.dlang.org/show_bug.cgi?id=21091

/*
TRANSFORM_OUTPUT: remove_lines("^import path")
TEST_OUTPUT:
----
fail_compilation/fail21091a.d(14): Error: unable to read module `Ternary`
fail_compilation/fail21091a.d(14):        Expected 'Ternary.d' or 'Ternary/package.d' in one of the following import paths:
----
*/

struct NullAllocator
{
    import Ternary;
    Ternary owns() { }
}
