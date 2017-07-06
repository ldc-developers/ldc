// REQUIRES: target_MSP430

// RUN: %ldc -mtriple=msp430 -o- %s

void* ptr;
static assert(ptr.sizeof == 2);

version(D_P16) {}
else static assert(0);