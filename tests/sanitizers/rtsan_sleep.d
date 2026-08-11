// Test simple realtime violations

// REQUIRES: RTSan

// RUN: %ldc -g -O0 -fsanitize=realtime %s -of=%t%exe
// RUN: not %t%exe 2>&1 | FileCheck %s

extern (C) uint sleep(uint seconds);

// CHECK: ERROR: RealtimeSanitizer: unsafe-library-call
void blocking_function()
{
    cast(void) sleep(3);
}

extern (C) int main(int argc, char** argv)
{
    blocking_function();
    return 0;
}
