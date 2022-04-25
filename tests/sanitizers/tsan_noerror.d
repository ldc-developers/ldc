// Test that a simple program passes ThreadSanitizer without error

// REQUIRES: TSan

// RUN: %ldc -fsanitize=thread %s -of=%t%exe
// RUN: %t%exe

void main()
{
}
