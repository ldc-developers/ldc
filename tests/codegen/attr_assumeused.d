// Tests @assumeUsed attribute

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: @llvm.used = appending global {{.*}} @some_function {{.*}} @some_variable

static import ldc.attributes;

extern (C): // For easier name mangling

@(ldc.attributes.assumeUsed) void some_function()
{
}

@(ldc.attributes.assumeUsed) int some_variable;
