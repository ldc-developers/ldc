void numbers()
{
    bool[8] bools;
    char[8] chars;
    byte[8] bytes;
    short[8] shorts;
    int[8] ints;
    long[8] longs;
    float[8] floats;
    double[8] doubles;
    real[8] reals;
    {
        bools[7] = true;
        floats[7] = 3.14159265;
        {
            printf("bools[0] = %d, bools[7] = %d\n", bools[0], bools[7]);
            printf("floats[0] = %f, floats[7] = %f\n", floats[0], floats[7]);
        }
    }
}

struct S
{
    int i = 42;
    void print()
    {
        printf("S.i = %d\n", i);
    }
}

class C
{
    int i;
    this()
    {
        i = 3;
    }
    void print()
    {
        printf("C.i = %d\n", i);
    }
}

void refs()
{
    void*[5] voids;
    S*[5] structs;
    C[5] classes;
    
    {
        voids[0] = cast(void*)0xA;
        printf("void* = %p\n", voids[0]);
    }
    {
        structs[0] = new S;
        structs[0].print();
        delete structs[0];
    }
    {
        classes[0] = new C;
        classes[0].print();
        delete classes[0];
    }
}

void main()
{
    numbers();
    refs();
}
