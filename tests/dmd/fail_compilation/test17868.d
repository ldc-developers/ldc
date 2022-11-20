/* DISABLED: LDC // differing output as LDC allows for a single optional integer argument (priority)
TEST_OUTPUT:
----
fail_compilation/test17868.d(10): Error: pragma `crt_constructor` takes no argument
fail_compilation/test17868.d(11): Error: pragma `crt_constructor` takes no argument
fail_compilation/test17868.d(12): Error: pragma `crt_constructor` takes no argument
fail_compilation/test17868.d(13): Error: pragma `crt_constructor` takes no argument
----
 */
pragma(crt_constructor, ctfe())
pragma(crt_constructor, 1.5f)
pragma(crt_constructor, "foobar")
pragma(crt_constructor, S())
void foo()
{
}

int ctfe()
{
    __gshared int val;
    return val;
}

struct S {}
