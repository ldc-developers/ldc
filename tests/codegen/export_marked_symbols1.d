// Test if passing -export-marked-symbols hides all unexported symbols

// UNSUPPORTED: Windows

// RUN: ldc2 %s -betterC -shared -export-marked-symbols -of=lib%t.so
// RUN: nm lib%t.so | FileCheck %s

// CHECK: test__exportedFunDef
// CHECK: test__exportedVarDef
// CHECK-NOT: test__nonExportedFunDef
// CHECK-NOT: test__nonExportedFunDecl
// CHECK-NOT: test__nonExportedVarDef
// CHECK-NOT: test__nonExportedVarDecl


extern(C) export int test__exportedFunDef() { return 42; }
extern(C) int test__nonExportedFunDef() { return 101; }

extern(C) export int test__exportedFunDecl();
extern(C) int test__nonExportedFunDecl();

extern(C) export int test__exportedVarDef;
extern(C) int test__nonExportedVarDef;

extern(C) extern export int test__exportedVarDecl;
extern(C) extern int test__nonExportedVarDecl;
