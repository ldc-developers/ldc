module internal.arrays;

extern(C):

void _d_array_init_i1(bool* a, size_t n, bool v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_i8(ubyte* a, size_t n, ubyte v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_i16(ushort* a, size_t n, ushort v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_i32(uint* a, size_t n, uint v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_i64(ulong* a, size_t n, ulong v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_float(float* a, size_t n, float v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_double(double* a, size_t n, double v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_pointer(void** a, size_t n, void* v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}
