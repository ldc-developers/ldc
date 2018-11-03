// Test if compiling without -export-marked-symbols exports all symbols on non-Windows targets

// RUN: ldc2 %s -betterC -shared -of=lib%t.so
// RUN: nm lib%t.so | FileCheck %s

// UNSUPPORTED: Windows
// CHECK: test__exportedFunDef
// CHECK: test__exportedVarDef
// CHECK: test__nonExportedFunDef
// CHECK: test__nonExportedVarDef
// CHECK-NOT: test__nonExportedFunDecl
// CHECK-NOT: test__nonExportedVarDecl


extern(C) export int test__exportedFunDef() { return 42; }
extern(C) int test__nonExportedFunDef() { return 101; }

extern(C) export int test__exportedFunDecl();
extern(C) int test__nonExportedFunDecl();

extern(C) export int test__exportedVarDef;
extern(C) int test__nonExportedVarDef;

extern(C) extern export int test__exportedVarDecl;
extern(C) extern int test__nonExportedVarDecl;
