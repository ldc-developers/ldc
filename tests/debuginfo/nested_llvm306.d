// Tests debug info generation for nested functions
// REQUIRES: atmost_llvm306
// RUN: %ldc -g -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define {{.*}} @_D{{.*}}8encloserFiiZv
void encloser(int arg0, int arg1)
{
    // CHECK: @llvm.dbg.declare{{.*}}%.frame{{.*}}enc_n
    int enc_n;

    // CHECK-LABEL: define {{.*}} @_D{{.*}}encloser{{.*}}nested
    void nested(int nes_i)
    {
        // CHECK: @llvm.dbg.declare{{.*}}%nestedFrame{{.*}}arg1
        arg0 = arg1 = enc_n = nes_i; // accessing arg0, arg1 and enc_n from a nested function turns them into closure variables
    }
}

// CHECK: @_D{{.*}}8encloserFiiZv{{.*}}DW_TAG_subprogram
// CHECK: @_D{{.*}}8encloserFiiZ6nestedMFiZv{{.*}}DW_TAG_subprogram
// CHECK: nes_i{{.*}}DW_TAG_arg_variable
// CHECK: arg1{{.*}}DW_TAG_auto_variable
