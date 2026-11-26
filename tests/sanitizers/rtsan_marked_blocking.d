// Test simple realtime violations (marked blocking)

// REQUIRES: RTSan

// RUN: %ldc -g -O0 -fsanitize=realtime %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s
import ldc.attributes;

// CHECK: ERROR: RealtimeSanitizer: blocking-call
@realTimeUnsafe
int blocking_function()
{
    return 42;
}

extern (C) int main(int argc, char** argv)
{
    // CHECK: Call to blocking function `_D21rtsan_marked_blocking17blocking_functionFZi`
    return blocking_function();
}
