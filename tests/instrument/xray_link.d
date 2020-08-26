// REQUIRES: XRay_RT

// fails on macOS with LLVM 11 due to a linker error, see
// https://github.com/llvm/llvm-test-suite/commit/2c3c4a6286d453f763c0245c6536ddd368f0db99
// XFAIL: Darwin && atleast_llvm1100

// RUN: %ldc -fxray-instrument -fxray-instruction-threshold=1 -of=%t%exe %s -vv | FileCheck %s

void foo()
{
}

void main()
{
    foo();
}

// CHECK: Linking with:
// CHECK-NEXT: rt.xray
