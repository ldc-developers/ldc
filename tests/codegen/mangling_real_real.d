// Tests that repeated `real` return types are treated as built-in types in C++ mangling (no substitution).

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux   -c -output-ll -of=%t.ll         %s && FileCheck %s --check-prefix=LINUX   < %t.ll
// RUN: %ldc -mtriple=x86_64-android -c -output-ll -of=%t.android.ll %s && FileCheck %s --check-prefix=ANDROID < %t.android.ll
// RUN: %ldc -mtriple=x86_64-windows -c -output-ll -of=%t.windows.ll %s && FileCheck %s --check-prefix=WINDOWS < %t.windows.ll

import core.stdc.config;

// LINUX: define {{.*}}Z8withrealee
// ANDROID: define {{.*}}Z8withrealgg
// WINDOWS: define {{.*}}?withreal@@YAXOO@Z
extern (C++) void withreal(real a, real b)
{
}

// LINUX: define {{.*}}Z15withclongdoubleee
// ANDROID: define {{.*}}Z15withclongdoublegg
// WINDOWS: define {{.*}}?withclongdouble@@YAXOO@Z
extern (C++) void withclongdouble(c_long_double a, c_long_double b)
{
}
