// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// REQUIRES: MLIR

import ldc.llvmasm;

// CHECK-LABEL: define {{.*}}add
int add(int a, int b) {

    // CHECK: add i32
    // CHECK: ret i32
    return __mlir!(`
        %res = arith.addi %arg0, %arg1: i32
        return %res: i32
    `, int, int, int)(a, b);
}

// CHECK-LABEL: define {{.*}}testStore
void testStore(int* ptr, int val)
{
    enum mlirCode = q{
        llvm.store %arg1, %arg0 : i32, !llvm.ptr
    };
    // CHECK: store i32 {{.*}}, ptr
    // CHECK: ret void
    __mlir!(`llvm.store %arg1, %arg0 : i32, !llvm.ptr`, void, int*, int)(ptr, val);
}

// CHECK-NOT: define {{.*}}inline.mlir
