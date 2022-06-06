// Test value name discarding when creating non-textual IR.

// RUN: %ldc %S/inputs/input_discard_valuename.d -c -output-ll -of=%t.bar.ll && FileCheck %S/inputs/input_discard_valuename.d < %t.bar.ll

// Output a bitcode file (i.e. with discarded names) and input it into a second LDC command that outputs textual IR.
// RUN: %ldc %S/inputs/input_discard_valuename.d -g -c -output-bc -of=%t.bar.bc \
// RUN: && %ldc %s %t.bar.bc -g -c -output-ll -of=%t.ll && FileCheck %s < %t.ll

// IR imported from the bitcode file should not have local value names:
// CHECK-LABEL: define{{.*}} @foo
// CHECK: %localfoovar
// CHECK-LABEL: define{{.*}} @bar
// CHECK-NOT: %localbarvar

// But the imported IR should still have debug names:
// CHECK: DILocalVariable{{.*}}"localfoovar"
// CHECK: DILocalVariable{{.*}}"localbarvar"

extern(C) void foo()
{
    int localfoovar;
}
