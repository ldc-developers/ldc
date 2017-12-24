// REQUIRES: XRay_RT

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
