// Test that a simple progam passes AddressSanitizer without error

// REQUIRES: ASan

// RUN: %ldc -g -fsanitize=address %s -of=%t%exe
// RUN: %t%exe

void main()
{
}
