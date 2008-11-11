/*
 * vararg support for extern(C) functions
 */

module ldc.cstdarg;

// Check for the right compiler
version(LDC)
{
    // OK
}
else
{
    static assert(false, "This module is only valid for LDC");
}

alias void* va_list;

pragma(va_start)
    void va_start(T)(va_list ap, ref T);

pragma(va_arg)
    T va_arg(T)(va_list ap);

pragma(va_end)
    void va_end(va_list args);

pragma(va_copy)
    void va_copy(va_list dst, va_list src);
