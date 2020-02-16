// RUN: %ldc -c %s

extern (C++) interface I
{
    extern (System) void func();
}

void foo(I i) { i.func(); }
