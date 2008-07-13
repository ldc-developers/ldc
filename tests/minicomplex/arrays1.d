module tangotests.arrays1;

import tango.stdc.stdio;

void main()
{
    real[] arr;
    print(arr);
    main2();
}

void main2()
{
    real[] arr = void;
    fill(arr);
    print(arr);
    main3();
}

void main3()
{
}

void print(real[] arr)
{
    printf("len=%u ; ptr=%p\n", arr.length, arr.ptr);
}

void fill(ref real[] arr)
{
    auto ptr = cast(void**)&arr;
    *ptr++ = cast(void*)0xbeefc0de;
    *ptr = cast(void*)0xbeefc0de;
}

void dg1(void delegate(int[]) dg)
{
    dg2(dg);
}

void dg2(void delegate(int[]) dg)
{
    dg(null);
}

void sarr1(int[16] sa)
{
    sarr1(sa);
}

struct Str
{
    size_t length;
    char* ptr;
}

void str1(Str str)
{
    str1(str);
}

void str2(ref Str str)
{
    str2(str);
}

void str3(out Str str)
{
    str3(str);
}

void str4(Str* str)
{
    str4(str);
}

void str5(Str);

