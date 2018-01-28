// Test that maximum function counts are set correctly (LLVM >= 3.8)

// REQUIRES: PGO_RT
// REQUIRES: atleast_llvm308
// For LLVM > 3.8, a summary is emitted, see summary.d

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck %s < %t2.ll

// CHECK: !{{[0-9]+}} = {{.*}}!"MaxFunctionCount", i{{(32|64)}} 2}

void foo() {}
void bar() {}

void main() {
  foo();
  bar();
  bar();
}
