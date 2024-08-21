// REQUIRES: target_Xtensa

// RUN: %ldc -mtriple=xtensa -betterC

version (Xtensa) {} else static assert(0);

