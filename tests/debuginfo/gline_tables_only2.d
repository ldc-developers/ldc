// RUN: %ldc -gline-tables-only --output-ll -of%t.ll %s && FileCheck %s < %t.ll
// Checks that ldc with "-gline-tables-only" emits metadata for
// compile unit, subprogram and file.

int main() {
  // CHECK: ret i32 0, !dbg
  return 0;
}

// CHECK: !llvm.dbg.cu = !{!0}
// CHECK: !DICompileUnit(
// CHECK: !DISubprogram(
// CHECK: !DIFile(
