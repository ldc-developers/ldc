module inputs.inlinables;

import ldc.attributes;

extern (C): // simplify mangling for easier function name matching

int easily_inlinable(int i)
{
    if (i > 0)
        return easily_inlinable(i - 1);
    return 2;
}

pragma(inline, false) int never_inline()
{
    return 1;
}

@weak int external()
{
    return 1;
}

pragma(inline, true) int always_inline()
{
    int a;
    foreach (i; 1 .. 10)
    {
        foreach (ii; 1 .. 10)
        {
            foreach (iii; 1 .. 10)
            {
                a += i * external();
            }
        }
    }
    return a;
}

pragma(inline, true) int always_inline_chain0()
{
    return always_inline_chain1();
}

pragma(inline, true) int always_inline_chain1()
{
    return always_inline_chain2();
}

pragma(inline, true) int always_inline_chain2()
{
    return 345;
}

class A
{
    int virtual_func()
    {
        return 12345;
    }

    pragma(inline, true) final int final_func()
    {
        return virtual_func();
    }

    final int final_class_function()
    {
        return 12345;
    }
}

// Weak-linkage functions can not be inlined.
@weak int weak_function()
{
    return 654;
}

int module_variable = 666;
pragma(inline, true) void write_module_variable(int i)
{
    module_variable = i;
}

pragma(inline, true) void write_function_static_variable(int i)
{
    static int static_func_var = 5;
    static_func_var = i;
}

pragma(inline, true) auto get_typeid_A()
{
    return typeid(A);
}

pragma(inline, true) int call_template_foo(int i)
{
    return template_foo(i);
}

pragma(inline, true) T template_foo(T)(T i)
{
    return i;
}

void call_enforce_with_default_template_params()
{
    import std.exception;
    enforce(true, { assert(0); });
}
