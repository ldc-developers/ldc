module tangotests.vararg2;

extern(C) int printf(char*, ...);

import tango.core.Vararg;

void main()
{
    func(0xf00, 1, " ", 2, " ", 3, "\n", 0.3, "\n");
}

void func(int foo, ...)
{
    foreach(t; _arguments)
    {
        if (t == typeid(char[]))
        {
            char[] str = va_arg!(char[])(_argptr);
            printf("%.*s", str.length, str.ptr);
        }
        else if (t == typeid(int))
        {
            printf("%d", va_arg!(int)(_argptr));
        }
        else if (t == typeid(float))
        {
            printf("%f", va_arg!(float)(_argptr));
        }
        else if (t == typeid(double))
        {
            printf("%f", va_arg!(double)(_argptr));
        }
        else if (t == typeid(real))
        {
            printf("%f", va_arg!(real)(_argptr));
        }
        else
        {
            assert(0, "not int");
        }
    }
}
