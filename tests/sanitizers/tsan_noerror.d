// Test that a simple program passes ThreadSanitizer without error

// REQUIRES: TSan
// REQUIRES: atleast_llvm800

// RUN: %ldc -fsanitize=thread %s -of=%t%exe
// RUN: %t%exe

void main()
{
}
