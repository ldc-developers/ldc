// Test instrumentation codegen with -O
// (tests correct setup of LLVM optimizer passes)

// RUN: %ldc -c -O -fprofile-instr-generate -of=%t.o %s

void main() {}
