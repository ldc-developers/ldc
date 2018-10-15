// Test that maximum function counts are set correctly
// REQUIRES: PGO_RT

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck %s < %t2.ll

// CHECK: !{{[0-9]+}} = !{i32 1, !"ProfileSummary", !{{[0-9]+}}}
// CHECK: !{{[0-9]+}} = !{!"MaxFunctionCount", i64 2}

void foo() {}
void bar() {}

void main() {
  foo();
  bar();
  bar();
}
