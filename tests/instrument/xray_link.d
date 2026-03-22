// REQUIRES: XRay_RT

// RUN: %ldc -fxray-instrument -fxray-instruction-threshold=1 -of=%t%exe %s -vv 2>&1 | FileCheck %s

void foo()
{
}

void main()
{
    foo();
}

// CHECK-NOT: error
// CHECK: Linking with:
// CHECK-NEXT: rt.xray
