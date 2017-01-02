// Tests debug info generation for nested functions
// REQUIRES: llvm307
// RUN: %ldc -g -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define {{.*}} @_D{{.*}}8encloser
void encloser(int arg0, int arg1)
{
    // CHECK: @llvm.dbg.declare{{.*}}%.frame{{.*}}enc_n
    int enc_n;

    // CHECK-LABEL: define {{.*}} @_D{{.*}}8encloser{{.*}}nested
    void nested(int nes_i)
    {
        // CHECK: @llvm.dbg.declare{{.*}}%nestedFrame{{.*}}arg1
        arg0 = arg1 = enc_n = nes_i; // accessing arg0, arg1 and enc_n from a nested function turns them into closure variables

        // nes_i and arg1 have the same parameter index in the generated IR, if both get declared as
        // function parameters this triggers off an assert in LLVM >=3.8 (see Github PR #1597)
    }
}

// CHECK: !DISubprogram(name:{{.*}}"{{.*}}.encloser"
// CHECK-SAME: function: void {{.*}} @_D{{.*}}8encloserFiiZv
// CHECK-LABEL: !DISubprogram(name:{{.*}}"{{.*}}.encloser.nested"
// CHECK: !DILocalVariable{{.*}}DW_TAG_arg_variable{{.*}}nes_i
// CHECK: !DILocalVariable{{.*}}DW_TAG_auto_variable{{.*}}arg1
