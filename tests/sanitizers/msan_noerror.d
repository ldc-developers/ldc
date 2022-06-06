// Test that a simple progam passes MemorySanitizer without error

// REQUIRES: MSan

// RUN: %ldc -g -fsanitize=memory %s -of=%t%exe
// RUN: %t%exe

void main()
{
}
