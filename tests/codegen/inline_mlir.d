// REQUIRES: MLIR
// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

import ldc.llvmasm;

// CHECK-LABEL: define{{.*}} i32 @mlir_add(
extern (C) int mlir_add(int a, int b)
{
    // CHECK: %[[R:[0-9]+]] = add i32 %{{[0-9]+}}, %{{[0-9]+}}
    // CHECK-NEXT: ret i32 %[[R]]
    return __mlir!(`
        %res = arith.addi %arg0, %arg1 : i32
        return %res : i32
    `, int, int, int)(a, b);
}

// CHECK-LABEL: define{{.*}} float @mlir_mulf(
extern (C) float mlir_mulf(float a, float b)
{
    // CHECK: %[[R:[0-9]+]] = fmul float %{{[0-9]+}}, %{{[0-9]+}}
    // CHECK-NEXT: ret float %[[R]]
    return __mlir!(`
        %res = arith.mulf %arg0, %arg1 : f32
        return %res : f32
    `, float, float, float)(a, b);
}

// CHECK-LABEL: define{{.*}} void @mlir_store(
extern (C) void mlir_store(int* ptr, int val)
{
    // CHECK: %[[P:[0-9]+]] = load ptr, ptr %ptr
    // CHECK: %[[V:[0-9]+]] = load i32, ptr %val
    // CHECK-NEXT: store i32 %[[V]], ptr %[[P]]
    // CHECK-NEXT: ret void
    __mlir!(`llvm.store %arg1, %arg0 : i32, !llvm.ptr`, void, int*, int)(ptr, val);
}

// CHECK-LABEL: define{{.*}} i32 @mlir_max(
extern (C) int mlir_max(int a, int b)
{
    enum code = q{
        %cmp = arith.cmpi sgt, %arg0, %arg1 : i32
        %res = scf.if %cmp -> (i32) {
            scf.yield %arg0 : i32
        } else {
            scf.yield %arg1 : i32
        }
        return %res : i32
    };

    // CHECK: %[[A:[0-9]+]] = load i32, ptr %a
    // CHECK: %[[B:[0-9]+]] = load i32, ptr %b
    // CHECK: %[[CMP:[0-9]+]] = icmp sgt i32 %[[A]], %[[B]]
    // CHECK: br i1 %[[CMP]], label %[[THEN:[0-9]+]], label %[[ELSE:[0-9]+]]
    // CHECK: phi i32 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ %{{[0-9]+}}, %{{[0-9]+}} ]
    // CHECK-NEXT: ret i32
    return __mlir!(code, int, int, int)(a, b);
}

// CHECK-LABEL: define{{.*}} i32 @mlir_sum_below(
extern (C) int mlir_sum_below(int n)
{
    // CHECK: icmp slt i32
    // CHECK: add i32
    // CHECK: add i32 %{{[0-9]+}}, 1
    // CHECK: ret i32
    return __mlir!(`
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %r = scf.for %i = %c0 to %arg0 step %c1 iter_args(%acc = %c0) -> (i32) : i32 {
            %next = arith.addi %acc, %i : i32
            scf.yield %next : i32
        }
        return %r : i32
    `, int, int)(n);
}

// CHECK-LABEL: define{{.*}} i32 @mlir_abs(
extern (C) int mlir_abs(int a)
{
    // CHECK: icmp slt i32 %{{[0-9]+}}, 0
    // CHECK: sub i32 0,
    // CHECK: phi i32
    // CHECK-NEXT: ret i32
    return __mlir!(`
        %zero = arith.constant 0 : i32
        %isneg = arith.cmpi slt, %arg0, %zero : i32
        cf.cond_br %isneg, ^neg, ^pos
      ^neg:
        %n = arith.subi %zero, %arg0 : i32
        return %n : i32
      ^pos:
        return %arg0 : i32
    `, int, int)(a);
}

// CHECK-LABEL: define{{.*}} i32 @mlir_chain(
extern (C) int mlir_chain(int a)
{
    // CHECK: %[[X:[0-9]+]] = add i32
    // CHECK: store i32 %[[X]], ptr %x
    // CHECK: %[[Y:[0-9]+]] = load i32, ptr %x
    // CHECK: mul i32 %[[Y]], %[[Y]]
    const x = __mlir!(`
        %r = arith.addi %arg0, %arg0 : i32
        return %r : i32
    `, int, int)(a);
    return __mlir!(`
        %r = arith.muli %arg0, %arg0 : i32
        return %r : i32
    `, int, int)(x);
}

// Alias declared at module scope, used from several functions.
alias mlirNeg = __mlir!(`
    %zero = arith.constant 0 : i64
    %r = arith.subi %zero, %arg0 : i64
    return %r : i64
`, long, long);

// CHECK-LABEL: define{{.*}} i64 @mlir_neg1(
// CHECK: sub i64 0,
extern (C) long mlir_neg1(long a) { return mlirNeg(a); }

// CHECK-LABEL: define{{.*}} i64 @mlir_neg2(
// CHECK: sub i64 0,
extern (C) long mlir_neg2(long a) { return mlirNeg(a) + 1; }

void main()
{
    assert(mlir_add(2, 3) == 5);
    assert(mlir_mulf(1.5f, 4.0f) == 6.0f);

    int i;
    mlir_store(&i, 42);
    assert(i == 42);

    assert(mlir_max(3, 7) == 7);
    assert(mlir_max(7, 3) == 7);
    assert(mlir_max(-1, -5) == -1);

    assert(mlir_sum_below(0) == 0);
    assert(mlir_sum_below(5) == 0 + 1 + 2 + 3 + 4);

    assert(mlir_abs(-9) == 9);
    assert(mlir_abs(9) == 9);

    assert(mlir_chain(3) == 36);

    assert(mlir_neg1(5) == -5);
    assert(mlir_neg2(5) == -4);
}
