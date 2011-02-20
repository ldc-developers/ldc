private import ldc.intrinsics;

extern(C):

int memcmp(void*,void*,size_t);
size_t strlen(char*);

// per-element array init routines

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

void _d_array_init_real(real* a, size_t n, real v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_cfloat(cfloat* a, size_t n, cfloat v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_cdouble(cdouble* a, size_t n, cdouble v)
{
    auto p = a;
    auto end = a+n;
    while (p !is end)
        *p++ = v;
}

void _d_array_init_creal(creal* a, size_t n, creal v)
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

void _d_array_init_mem(void* a, size_t na, void* v, size_t nv)
{
    auto p = a;
    auto end = a + na*nv;
    while (p !is end) {
        llvm_memcpy(p,v,nv,0);
        p += nv;
    }
}

/*
void _d_array_init(TypeInfo ti, void* a)
{
    auto initializer = ti.next.init();
    auto isize = initializer.length;
    auto q = initializer.ptr;

    if (isize == 1)
        memset(p, *cast(ubyte*)q, size);
    else if (isize == int.sizeof)
    {
        int init = *cast(int*)q;
        size /= int.sizeof;
        for (size_t u = 0; u < size; u++)
        {
            (cast(int*)p)[u] = init;
        }
    }
    else
    {
        for (size_t u = 0; u < size; u += isize)
        {
            memcpy(p + u, q, isize);
        }
    }
}*/

// for array cast
size_t _d_array_cast_len(size_t len, size_t elemsz, size_t newelemsz)
{
    if (newelemsz == 1) {
        return len*elemsz;
    }
    else if ((len*elemsz) % newelemsz) {
        throw new Exception("Bad array cast");
    }
    return (len*elemsz)/newelemsz;
}

// slice copy when assertions are enabled
void _d_array_slice_copy(void* dst, size_t dstlen, void* src, size_t srclen)
{
    if (dstlen != 0) assert(dst);
    if (dstlen != 0) assert(src);
    if (dstlen != srclen)
        throw new Exception("lengths don't match for array copy");
    else if (dst+dstlen <= src || src+srclen <= dst)
        llvm_memcpy!size_t(dst, src, dstlen, 0);
    else
        throw new Exception("overlapping array copy");
}
