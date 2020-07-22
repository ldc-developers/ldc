// Test that a small code change changes the function hash.

// See: https://github.com/ldc-developers/ldc/pull/3511 and https://reviews.llvm.org/D79961

// REQUIRES: PGO_RT

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -d-version=WithChange -c -wi -fprofile-instr-use=%t.profdata %s 2>&1 | FileCheck %s

// CHECK: Warning: Ignoring profile data for function {{.*}}.foo{{.*}} control-flow hash mismatch
// CHECK: Warning: Ignoring profile data for function {{.*}}.longerfunction{{.*}} control-flow hash mismatch

bool bar;

void foo() {
  if (bar) {}
  version(WithChange)
    if (bar) {}
}

// Function with more controlflow to trigger MD5 hashing
void longerfunction() {
  version(WithChange)
    if (bar) {}

  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
  if (bar) {}
}

void main() {
  foo();
  longerfunction();
}
