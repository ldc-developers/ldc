// Integral / POD-element array `==` in @compute device code takes the memcmp
// "fast path" (it does NOT instantiate the __equals hook). On the host that path
// lowers to a `memcmp` runtime call, but device targets have no libc `memcmp`
// and there is no `llvm.memcmp` intrinsic, so DtoArrayEqCmp_memcmp/callMemcmp
// must emit the comparison as an INLINE byte-wise loop instead. This test pins
// that lowering down:
//   * the inline loop blocks (memcmp.cond/body/diff/inc/end) are emitted,
//   * the byte count is numElements * elementSize (and == numElements when the
//     element is 1 byte, i.e. no multiply),
//   * static arrays skip the runtime length guard and use a constant byte count,
//   * the merge phi reads its memcmp result from the loop's real exit block
//     (regression guard for the GetInsertBlock() fix: callMemcmp now emits extra
//     blocks, so the result no longer comes from the pre-loop block),
//   * NO `memcmp` symbol/call and NO __equals hook are emitted for the device.
//
// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-700 -m64 -output-ll -output-o -c \
// RUN:   -mdcompute-file-prefix=dcompute_eqinline %s
// RUN: FileCheck %s < dcompute_eqinline_cuda700_64.ll
// RUN: FileCheck %s --check-prefix=NOCALL < dcompute_eqinline_cuda700_64.ll

@compute(CompileFor.deviceOnly) module dcompute_array_eq_inline;
import ldc.dcompute;
 
// Dynamic int[]: runtime length guard, byte count = len*4, full inline loop,
// and the result phi must take the memcmp answer from %memcmp.end.
// CHECK-LABEL: define{{.*}}ptx_device{{.*}}dyn_int
// CHECK: icmp eq i64
// CHECK: br i1 {{.*}}, label %domemcmp, label %memcmpend
// CHECK: domemcmp:
// CHECK: mul i64 {{.*}}, 4
// CHECK: br label %memcmp.cond
// CHECK: memcmp.cond:
// CHECK: icmp ult i64
// CHECK: br i1 {{.*}}, label %memcmp.body, label %memcmp.end
// CHECK: memcmp.body:
// CHECK: load i8
// CHECK: load i8
// CHECK: icmp eq i8
// CHECK: br i1 {{.*}}, label %memcmp.inc, label %memcmp.diff
// CHECK: memcmp.diff:
// CHECK: zext i8
// CHECK: sub i32
// CHECK: memcmp.inc:
// CHECK: add i64 {{.*}}, 1
// CHECK: memcmp.end:
// CHECK: phi i32 [ 0, %memcmp.cond ]
// CHECK: memcmpend:
// CHECK: phi i32 [ 1, {{.*}} ], [ {{.*}}, %memcmp.end ]
@kernel void dyn_int(int[] a, int[] b, bool* o) { *o = (a == b); }
 
// Dynamic byte[]: element size 1, so the byte count IS numElements -- there must
// be NO multiply between the length guard and the loop.
// CHECK-LABEL: define{{.*}}ptx_device{{.*}}dyn_byte
// CHECK: domemcmp:
// CHECK-NOT: mul i64
// CHECK: memcmp.cond:
// CHECK: memcmp.body:
// CHECK: load i8
// CHECK: icmp eq i8
@kernel void dyn_byte(byte[] a, byte[] b, bool* o) { *o = (a == b); }
 
// Dynamic short[]: element size 2 -> byte count = len*2.
// CHECK-LABEL: define{{.*}}ptx_device{{.*}}dyn_short
// CHECK: domemcmp:
// CHECK: mul i64 {{.*}}, 2
// CHECK: memcmp.cond:
@kernel void dyn_short(short[] a, short[] b, bool* o) { *o = (a == b); }
 
// Static int[4]: lengths are statically equal, so NO runtime length guard and a
// CONSTANT byte count of 16 (4 elements * 4 bytes).
// CHECK-LABEL: define{{.*}}ptx_device{{.*}}stat_int4
// CHECK-NOT: domemcmp
// CHECK: icmp ult i64 {{.*}}, 16
// CHECK: memcmp.body:
// CHECK: load i8
@kernel void stat_int4(int[4] a, int[4] b, bool* o) { *o = (a == b); }
 
// Static long[3]: constant byte count 24 (3 elements * 8 bytes).
// CHECK-LABEL: define{{.*}}ptx_device{{.*}}stat_long3
// CHECK-NOT: domemcmp
// CHECK: icmp ult i64 {{.*}}, 24
@kernel void stat_long3(long[3] a, long[3] b, bool* o) { *o = (a == b); }
 
// Device code must contain NO libc memcmp call/declare and must NOT fall back to
// the __equals hook for these integral/POD element types.
// NOCALL-NOT: @memcmp
// NOCALL-NOT: __equals
