module aa3;

void main()
{
    int[string] aa;
    {aa["hello"] = 1;}
    {int* p = "" in aa;}
    aa[" worl"] = 2;
    aa["d"] = 3;
    aa["thisisgreat"] = 10;
    int sum;
    string cat;
    {
    foreach(k,v;aa)
    {
        printf("int[%.*s] = %d\n", k.length, k.ptr, v);
        sum += v;
        cat ~= k;
    }
    }
    assert(sum == 16);
    printf("cat = %.*s\n", cat.length, cat.ptr);
    assert(cat.length == 22);
}
