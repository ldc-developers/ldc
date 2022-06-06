// REQUIRES: Plugins

// RUN: %gnu_make -f %S/Makefile
// RUN: %ldc -c -output-ll -plugin=./addFuncEntryCallPass.so -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// CHECK: define {{.*}}testfunction
int testfunction(int i)
{
    // CHECK-NEXT: call {{.*}}__test_funcentrycall
    return i * 2;
}

// CHECK-DAG: declare {{.*}}__test_funcentrycall
