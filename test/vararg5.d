module vararg5;
import std.stdarg;
void func(...)
{
    char[] str = va_arg!(char[])(_argptr);
    {printf("%.*s\n", str.length, str.ptr);}
}
void main()
{
    char[] str = "hello";
    func(str);
}
