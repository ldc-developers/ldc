module mini.packed1;

extern(C) int printf(char*, ...);

align(1)
struct PackedStruct
{
    ubyte ub;
    float f;
    long l;
    short s;
    ubyte ub2;
    short s2;
}

void main()
{
    PackedStruct[2] a = void;
    void* begin = a.ptr;
    void* end = &a[1];
    ptrdiff_t sz = end - begin;
    printf("size = 18 = %u = %u\n", PackedStruct.sizeof, sz);
    assert(sz == 18);
}