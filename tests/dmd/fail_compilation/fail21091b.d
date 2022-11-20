// https://issues.dlang.org/show_bug.cgi?id=21091

/*
TRANSFORM_OUTPUT: remove_lines("^import path")
TEST_OUTPUT:
----
fail_compilation/fail21091b.d(14): Error: unable to read module `Tid`
fail_compilation/fail21091b.d(14):        Expected 'Tid.d' or 'Tid/package.d' in one of the following import paths:
----
*/

class Logger
{
    import Tid;
    Tid threadId;
}
