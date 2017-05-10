// Test if global order of flags passed with -Xcc, -L and pragma(lib) is preserved

// UNSUPPORTED: Windows

// RUN: %ldc %s -of=%t -Xcc -DOPT1,-DOPT2 -L-L/usr/lib -L--defsym -Lfoo=5 -Xcc -DOPT3 -v | FileCheck %s

// CHECK: -DOPT1 -DOPT2 -L/usr/lib -Xlinker --defsym -Xlinker foo=5 -DOPT3 {{.*}}-lpthread

pragma(lib, "pthread");

void main()
{
}
