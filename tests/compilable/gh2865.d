// RUN: %ldc -c %s

void foo()
{
    auto addr = (cast(size_t) &foo) - 10;
}
