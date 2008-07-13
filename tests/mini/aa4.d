module aa4;

void main()
{
    int[int] aa;
    aa = addkey(aa,42,12);
    int* p = haskey(aa,42);
    assert(p && *p == 12);
}

int[int] addkey(int[int] aa, int key, int val)
{
    aa[key] = val;
    return aa;
}

int* haskey(int[int] aa, int key)
{
    return key in aa;
}
