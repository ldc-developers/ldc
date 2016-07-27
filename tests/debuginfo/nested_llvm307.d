// Tests debug info generation for nested functions
// REQUIRES: llvm307
// RUN: %ldc -g -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -g -c -O3 -of=%t %s

// See test "nested.d".

// CHECK-LABEL: define {{.*}} @_D{{.*}}8encloser
void encloser(int arg0, int arg1)
{
    // CHECK: @llvm.dbg.declare{{.*}}%.frame{{.*}}
    int enc_n;

    // CHECK-LABEL: define {{.*}} @_D{{.*}}8encloser{{.*}}nested
    void nested(int nes_i)
    {
        arg0 = arg1 = enc_n = nes_i; // accessing arg0, arg1 and enc_n from a nested function turns them into closure variables
    }
}

// CHECK: !DISubprogram(name:{{.*}}"{{.*}}.encloser"
// CHECK-SAME: function: void {{.*}} @_D{{.*}}8encloserFiiZv
// CHECK-LABEL: !DISubprogram(name:{{.*}}"{{.*}}.encloser.nested"
// CHECK: !DILocalVariable{{.*}}DW_TAG_arg_variable{{.*}}nes_i
