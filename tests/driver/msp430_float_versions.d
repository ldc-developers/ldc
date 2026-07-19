// REQUIRES: target_MSP430

// RUN: %ldc -c -o- %s -mtriple=msp430-unknown-elf

version (D_SoftFloat) {} else static assert(0);
version (D_HardFloat) static assert(0);
