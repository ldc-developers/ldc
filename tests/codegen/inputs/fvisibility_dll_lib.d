import ldc.attributes;

__gshared int dllGlobal = 123;

double dllSum(double a, double b)
{
    return a + b;
}

void dllWeakFoo() @weak {}
