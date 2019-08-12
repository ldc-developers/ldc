// RUN: %ldc -c %s

void foo()
{
    alias V = __vector(void[16]);
    V v;
}
