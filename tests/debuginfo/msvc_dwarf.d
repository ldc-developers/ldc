// REQUIRES: Windows

// RUN: %ldc -gdwarf -of=%t.exe %s 2> %t_stderr.log
// RUN: FileCheck %s < %t_stderr.log
// CHECK: lld-link: warning: section name .debug_info is longer than 8 characters and will use a non-standard string table

// The test sometimes fails on Win32 due to an lld assertion; no problems on Win64.
// ALLOW_RETRIES: 2

int foo(int p)
{
    auto r = 2 * p;
    return r;
}

void main()
{
    foo(123);
}
