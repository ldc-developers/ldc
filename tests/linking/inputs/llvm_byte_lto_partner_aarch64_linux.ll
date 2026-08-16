; Partner for llvm_byte_full_lto_linux.d: b8 signatures for Full LTO with LDC
; (-fc-interop-llvm-byte). Full LTO (not ThinLTO): llvm-as has no module summary.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define zeroext b8 @llvm_byte_lto_add_one(b8 zeroext %x) #0 {
entry:
  %xi = bitcast b8 %x to i8
  %y = add i8 %xi, 1
  %r = bitcast i8 %y to b8
  ret b8 %r
}

define void @llvm_byte_lto_sink_uchar(b8 zeroext %x) #0 {
entry:
  ret void
}

attributes #0 = { nounwind uwtable }
