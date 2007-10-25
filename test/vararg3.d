module vararg3;

import std.stdarg;

void func(...)
{
    assert(_arguments.length == 3);
    assert(_arguments[0] is typeid(int));
    assert(_arguments[1] is typeid(float));
    assert(_arguments[2] is typeid(long));
    assert(va_arg!(int)(_argptr) == 4);
    assert(va_arg!(float)(_argptr) == 2.5f);
    assert(va_arg!(long)(_argptr) == 42L);
}

void main()
{
    func(4, 2.5f, 42L);
}
