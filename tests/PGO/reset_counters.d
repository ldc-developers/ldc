// Tests (the availability of) the runtime lib function to reset all profile counters.

// REQUIRES: PGO_RT

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck %s < %t2.ll

extern(C) void foo(int N) {
  // CHECK-LABEL: define void @foo(
  // CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[FOO:[0-9]+]]
  if (N) {}
}

// CHECK-LABEL: define i32 @_Dmain(
void main() {
  import ldc.profile;
  foo(0);
  resetAll();
  foo(1);
}

// CHECK: ![[FOO]] = !{!"branch_weights", i32 2, i32 1}
