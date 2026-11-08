// REQUIRES: Linux
// RUN: mkdir -p %t && cp %S/../dmd/runnable/extra-files/lib13742a.d %S/../dmd/runnable/extra-files/lib13742b.d %t
// RUN: cd %t && %ldc -cov -cov-increment=boolean -lib lib13742a.d lib13742b.d -of=libtest.a
// RUN: cd %t && %ldc -cov -cov-increment=boolean %s libtest.a -I. -of=program
// RUN: cd %t && ./program && ./program --DRT-covopt=merge:1
// RUN: FileCheck %s < %t/lib13742a.lst

import lib13742b : clear;

void main()
{
    clear();
    clear();
}

// CHECK: {{^ *1\| *PROC\(\);}}
