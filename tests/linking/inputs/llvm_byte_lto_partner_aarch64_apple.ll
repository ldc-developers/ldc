; Partner for llvm_byte_full_lto_apple.d: b8 signatures for Full LTO with LDC.
; Triple/layout match arm64-apple-macos (Apple AArch64 hosts).

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macos11.0"

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
