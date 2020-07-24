// Test that a simple program passes ThreadSanitizer without error

// REQUIRES: TSan

// XFAIL: *
// Druntime does not yet work with ThreadSanitizer.
// See Github issue 3519 (https://github.com/ldc-developers/ldc/issues/3519)

// RUN: %ldc -fsanitize=thread %s -of=%t%exe
// RUN: %t%exe

void main()
{
}
