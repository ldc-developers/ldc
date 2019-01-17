// Check basic execution of XRay to verify the basic system is working.

// REQUIRES: Linux
// REQUIRES: XRay_RT
// REQUIRES: atleast_llvm700

// RUN: %ldc -fxray-instrument -fxray-instruction-threshold=1 %s -of=%t%exe
// RUN: env XRAY_OPTIONS="patch_premain=true xray_mode=xray-basic verbosity=1" %t%exe  2>&1 | FileCheck %s
// This last command should give some output on stderr, one line of which containing:
// CHECK: XRay: Log file
// If stderr is empty, things are not working.

void foo()
{
}

void main()
{
    foo();
}
