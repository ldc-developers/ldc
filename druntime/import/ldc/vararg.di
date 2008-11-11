// D import file generated from 'vararg.d'
module ldc.Vararg;
version (LDC)
{
}
else
{
    static assert(false,"This module is only valid for LDC");
}
alias void* va_list;
template va_start(T)
{
void va_start(out va_list ap, ref T parmn)
{
}
}
template va_arg(T)
{
T va_arg(ref va_list vp)
{
T* arg = cast(T*)vp;
vp = cast(va_list)(cast(void*)vp + (T.sizeof + size_t.sizeof - 1 & ~(size_t.sizeof - 1)));
return *arg;
}
}
void va_end(va_list ap)
{
}
void va_copy(out va_list dst, va_list src)
{
dst = src;
}
