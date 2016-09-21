// REQUIRES: atleast_llvm308
// REQUIRES: lldb, sed

// RUN: %ldc -g -of=%t%exe %s \
// RUN: && sed -e "/^\\/\\/ LLDB:/!d" -e "s,// LLDB:,," %s > %t.lldb \
// RUN: && %lldb %t%exe -s %t.lldb > %t.out.txt 2>&1 \
// RUN: && FileCheck %s -check-prefix=CHECK < %t.out.txt

int globalvar = 123;

void main()
{
    int a = 42;
    return;
}

// CHECK: Current executable set to {{.*}}lldb
// L LDB: break set --file lldb.d --line 14
// LLDB: version
// LLDB: run
// CHECK:      void main()
// CHECK-NEXT: {
// CHECK-NEXT: int a = 42;
// CHECK-NEXT: -> {{[0-9]+}} return;
// CHECK-NEXT: }
// L LDB: frame variable
// check for frame variable name 'a', broken atm
// L LDB: target variable
// check for global variable name 'globalvar', broken atm
// L LDB: continue
// LLDB: exit
