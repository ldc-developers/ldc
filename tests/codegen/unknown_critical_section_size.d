// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64 -c -w %s

// no target OS
version (FreeStanding) {} else static assert(0);

void foo()
{
    __gshared int global;
    synchronized
    {
        global += 10;
    }
}
