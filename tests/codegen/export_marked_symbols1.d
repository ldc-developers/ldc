// Tests -fvisibility={default,hidden} for function definitions and
// (non-extern) globals on non-Windows targets.

// UNSUPPORTED: Windows

// RUN: %ldc %s -betterC -shared -fvisibility=default -of=lib%t_default%so
// RUN: nm -g lib%t_default%so | FileCheck -check-prefix=DEFAULT %s

// RUN: %ldc %s -betterC -shared -fvisibility=hidden -of=lib%t_hidden%so
// RUN: nm -g lib%t_hidden%so | FileCheck -check-prefix=HIDDEN %s

extern(C) export int test__exportedFun() { return 42; }
// DEFAULT: test__exportedFun
// HIDDEN: test__exportedFun
extern(C) export int test__exportedVar;
// DEFAULT: test__exportedVar
// HIDDEN: test__exportedVar

extern(C) int test__nonExportedFun() { return 101; }
// DEFAULT: test__nonExportedFun
// HIDDEN-NOT: test__nonExportedFun
extern(C) int test__nonExportedVar;
// DEFAULT: test__nonExportedVar
// HIDDEN-NOT: test__nonExportedVar
