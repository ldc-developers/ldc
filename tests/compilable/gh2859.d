// RUN: %ldc -c %s

void foo()
{
    static struct S { int a; }

    S s;
    const aa = [1 : &s.a];
}
