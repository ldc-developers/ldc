module classes6;
extern(C) int printf(char*, ...);

class C
{
    void f()
    {
        printf("world\n");
    }
}

class D : C
{
    void f()
    {
        printf("moon\n");
    }
}


extern(C)
{
    void srand(uint seed);
    int rand();
}

import llvm.intrinsic;

void main()
{
    C c;
    srand(readcyclecounter());
    if (rand() % 2)
        c = new C;
    else
        c = new D;
    c.f();
}
