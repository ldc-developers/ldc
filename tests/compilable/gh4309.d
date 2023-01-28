// RUN: %ldc -c %s

void foo(void* p)
{
    auto q = p++;
}
