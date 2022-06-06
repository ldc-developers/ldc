// RUN: %ldc -c %s

interface I
{
    static void staticFoo();
    final void finalFoo();
}

void foo()
{
    I.staticFoo();
    (new class I{}).finalFoo();
}
