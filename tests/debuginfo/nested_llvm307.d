// Tests debug info generation for nested functions
// REQUIRES: llvm307
// RUN: %ldc -g -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK-LABEL: define {{.*}} @{{.*}}_D14nested_llvm3078encloserFiiZv
void encloser(int arg0, int arg1)
{
    // CHECK: @llvm.dbg.declare{{.*}}%enc_n{{.*}}enc_n
    int enc_n;

    // CHECK-LABEL: define {{.*}} @{{.*}}_D14nested_llvm3078encloserFiiZ6nestedMFNaNbNiNfiZv
    void nested(int nes_i)
    {
        // CHECK: %arg0 = getelementptr inbounds %nest.encloser
        // CHECK: @llvm.dbg.declare{{.*}}%arg0
        // CHECK: %arg1 = getelementptr inbounds %nest.encloser
        // CHECK: @llvm.dbg.declare{{.*}}%arg1
        // CHECK: %enc_n = getelementptr inbounds %nest.encloser
        // CHECK: @llvm.dbg.declare{{.*}}%enc_n
        arg0 = arg1 = enc_n = nes_i; // accessing arg0, arg1 and enc_n from a nested function turns them into closure variables
    }
}

// CHECK: !DISubprogram(name:{{.*}}"{{.*}}.encloser"
// CHECK-SAME: function: void {{.*}} @_D{{.*}}8encloserFiiZv
// CHECK-LABEL: !DISubprogram(name:{{.*}}"{{.*}}.encloser.nested"
// CHECK: !DILocalVariable{{.*}}DW_TAG_arg_variable{{.*}}nes_i
// CHECK: !DILocalVariable{{.*}}DW_TAG_auto_variable{{.*}}arg1
