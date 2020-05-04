// REQUIRES: target_X86
// RUN:     %ldc -mtriple=x86_64 -c -w %s
// RUN: not %ldc -mtriple=x86_64 -c -w -d-version=WithSynchronized %s 2>&1 | FileCheck %s

void foo() {}

version (WithSynchronized)
void bar()
{
    __gshared int global;
    // CHECK: unknown_critical_section_size.d(12): Error: unknown critical section size for the selected target
    synchronized
    {
        global += 10;
    }
}
