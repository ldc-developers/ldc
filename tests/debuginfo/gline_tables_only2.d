// RUN: %ldc -gline-tables-only --output-ll -of%t.ll %s && FileCheck %s < %t.ll
// Checks that ldc with "-gline-tables-only" emits metadata for
// compile unit, subprogram and file.
// Also checks that no type attributes are emitted

// REQUIRES: atleast_llvm309
// reason: different llvm version emits far different metadata in IR code

int main() {
  // CHECK: ret i32 0, !dbg
  return 0;
}

// CHECK: !llvm.dbg.cu = !{
// CHECK: !DICompileUnit(
// CHECK: !DIFile(
// CHECK: !DISubprogram(
// CHECK-NOT: !DIBasicType(
// CHECK: !DILocation
// CHECK-NOT: !DILocalVariable(
// CHECK-NOT: !DIDerivedType(
// CHECK-NOT: !DIBasicType(
