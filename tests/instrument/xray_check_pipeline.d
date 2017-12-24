// Check that the LLVM pipeline is set up correctly to generate XRay sleds.

// If we have the XRay runtime lib for this platform, then we can also do machinecodegen:
// REQUIRES: XRay_RT

// RUN: %ldc -c -output-s -betterC -fxray-instrument -fxray-instruction-threshold=1 -of=%t.s %s && FileCheck %s < %t.s

// CHECK: xray_sled
void instrument()
{
}
