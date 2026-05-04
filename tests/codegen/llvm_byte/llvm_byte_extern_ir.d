// Verify -fc-interop-llvm-byte lowers extern(C) ubyte/char parameters and returns
// to LLVM b8 on AArch64 (see gen/abi/aarch64.cpp).
//
// CI / older LLVM: unmet REQUIRES => UNSUPPORTED (skipped), not FAIL.
// - atleast_llvm23, llvm_ir_b8: from lit.site.cfg when LLVM >= 23 (b8 in IR).
// - target_AArch64: from LLVM_TARGETS_TO_BUILD (cross-compile uses AArch64 backend).

// REQUIRES: atleast_llvm23 && llvm_ir_b8 && target_AArch64

// RUN: %ldc -fc-interop-llvm-byte -mtriple=aarch64-linux-gnu -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

extern (C) void import_ubyte(ubyte x);
extern (C) void import_char(char x);

// Call sites: D passes constants; IR should pass b8 to the call.
void call_sites() {
  import_ubyte(cast(ubyte) 3);
  import_char(cast(char) 4);
}

// Callee definitions: parameters and return should be b8; body uses i8 storage + bitcasts.
extern (C) ubyte export_ubyte_param(ubyte x) {
  return cast(ubyte)(x + 1);
}

extern (C) char export_char_param(char x) {
  return cast(char)(x - 1);
}

// IR order: definitions before forward declares for callees.

// CHECK-LABEL: define{{.*}} @{{.*}}call_sites
// CHECK: call void @import_ubyte(b8 zeroext
// CHECK: call void @import_char(b8 zeroext

// CHECK: declare void @import_ubyte(b8 zeroext
// CHECK: declare void @import_char(b8 zeroext

// CHECK-LABEL: define{{.*}} @export_ubyte_param
// CHECK-SAME: (b8 zeroext
// CHECK: bitcast b8 %x_arg to i8
// CHECK: bitcast i8{{.*}} to b8
// CHECK: ret b8

// CHECK-LABEL: define{{.*}} @export_char_param
// CHECK-SAME: (b8 zeroext
// CHECK: ret b8
