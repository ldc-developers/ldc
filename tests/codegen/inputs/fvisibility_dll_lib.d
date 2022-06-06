import ldc.attributes;

__gshared int dllGlobal = 123;

double dllSum(double a, double b)
{
    return a + b;
}

void dllWeakFoo() @weak {}

// extern(C++) for -betterC
extern(C++) class MyClass
{
    int myInt = 456;
}
