// Test druntime real-time violations

// REQUIRES: RTSan

// RUN: %ldc -g -O2 -fsanitize=realtime %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s
import ldc.attributes;

// CHECK: ERROR: RealtimeSanitizer: blocking-call
void main()
{
    // CHECK-NEXT: Call to blocking function `rt_init`
}
