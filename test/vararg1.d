module vararg1;

import std.c.stdarg;

extern(C) int add(int n, ...)
{
    va_list ap=void;
    va_start(ap, n);
    int r;
    //for (int i=0; i<n; i++)
    //    r += va_arg!(int)(ap);
    r = va_arg!(int)(ap);
    va_end(ap);
    return r;
}

void main()
{
    int i = add(4,1,2,3,4);
    assert(i == 10);
}
