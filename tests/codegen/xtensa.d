// REQUIRES: target_Xtensa

// RUN: %ldc -mtriple=xtensa -betterC %s -c -of=%t.o

version (Xtensa) {} else static assert(0);

