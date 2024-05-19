// https://github.com/ldc-developers/ldc/issues/3692

// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll


// D `int[3]` rewritten to LL `{ i64, i32 }` for SysV ABI - mismatching size and alignment
// CHECK-LABEL: define void @_D6gh36924takeFG3iZv({ i64, i32 } %a_arg)
void take(int[3] a)
{
    // the `{ i64, i32 }` size is 16 bytes, so we need a padded alloca (with 8-bytes alignment)
    // CHECK-NEXT: = alloca { i64, i32 }, align 8
}

// CHECK-LABEL: define void @_D6gh36924passFZv()
void pass()
{
    // CHECK-NEXT: %arrayliteral = alloca [3 x i32], align 4
    // we need an extra padded alloca with proper alignment
    // CHECK-NEXT: %.BaseBitcastABIRewrite_padded_arg_storage = alloca { i64, i32 }, align 8
    // CHECK:      %.BaseBitcastABIRewrite_arg = load { i64, i32 }, ptr %.BaseBitcastABIRewrite_padded_arg_storage
    take([1, 2, 3]);
}


// D `int[4]` rewritten to LL `{ i64, i64 }` for SysV ABI - mismatching alignment only
// CHECK-LABEL: define void @_D6gh36925take4FG4iZv({ i64, i64 } %a_arg)
void take4(int[4] a)
{
    // the alloca should have 8-bytes alignment, even though a.alignof == 4
    // CHECK-NEXT: %a = alloca [4 x i32], align 8
    // CHECK: store { i64, i64 } %a_arg, ptr %
}

// CHECK-LABEL: define void @_D6gh36925pass4FZv()
void pass4()
{
    // CHECK-NEXT: %arrayliteral = alloca [4 x i32], align 4
    // we need an extra alloca with 8-bytes alignment
    // CHECK-NEXT: %.BaseBitcastABIRewrite_padded_arg_storage = alloca { i64, i64 }, align 8
    // CHECK:      %.BaseBitcastABIRewrite_arg = load { i64, i64 }, ptr %.BaseBitcastABIRewrite_padded_arg_storage
    take4([1, 2, 3, 4]);
}
