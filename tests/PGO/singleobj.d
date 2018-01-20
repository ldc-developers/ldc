// Tests simulaneous compilation into singleobj

// REQUIRES: PGO_RT

// RUN: %ldc -fprofile-instr-generate=%t.profraw %S/inputs/singleobj_input.d -c -of=%t%obj \
// RUN:   &&  %ldc -fprofile-instr-generate=%t.profraw %t%obj -run %s \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -singleobj -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %S/inputs/singleobj_input.d %s

void foo() {}
void bar() {}

void main() {
  foo();
  bar();
  bar();
}
