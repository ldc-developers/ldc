// Test pragma(LDC_extern_weak) on function declarations.

// RUN: %ldc -d-version=DECLARATION -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll --check-prefix=DECLARATION
// RUN: not %ldc -d-version=DEFINITION %s 2>&1 | FileCheck %s --check-prefix=DEFINITION

version(DECLARATION)
{
// DECLARATION: declare{{.*}} extern_weak {{.*}}weakreffunction
pragma(LDC_extern_weak) extern(C) void weakreffunction();
}

version(DEFINITION)
{
// DEFINITION: Error: `LDC_extern_weak` cannot be applied to function definitions
pragma(LDC_extern_weak) extern(C) void weakreffunction() {};
}

void foo()
{
    auto a = &weakreffunction;
}
