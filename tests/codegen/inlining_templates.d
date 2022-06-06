// Test inlining of templates
// Templates that would otherwise not be codegenned, _should_ be codegenned for inlining when pragma(inline, true) is specified.

// RUN: %ldc %s -I%S -c -output-ll -release -enable-inlining -enable-cross-module-inlining -O0 -of=%t.O0.ll && FileCheck %s < %t.O0.ll

// RUN: %ldc -singleobj %S/inputs/inlinables.d %s -I%S -c -output-ll -release -enable-inlining -enable-cross-module-inlining -O0 -of=%t.singleobj.O0.ll && FileCheck %s < %t.singleobj.O0.ll

// Test linking too.
// Separate compilation
//   RUN: %ldc -c -enable-inlining -enable-cross-module-inlining %S/inputs/inlinables.d -of=%t.inlinables%obj \
//   RUN: && %ldc -I%S -enable-inlining -enable-cross-module-inlining %t.inlinables%obj %s -of=%t%exe
// Singleobj compilation
//   RUN: %ldc -I%S -enable-inlining -enable-cross-module-inlining -singleobj %S/inputs/inlinables.d %s -of=%t2%exe

import inputs.inlinables;

int foo(int i)
{
    return call_template_foo(i);
}

// Call a function calling std.exception.enforce with default __FILE__ template parameter.
// Make sure the symbol is inlined/defined and not declared (which will lead to linker errors if the location
// of the stdlib is different from where LDC was built from)
void ggg()
{
    call_enforce_with_default_template_params();
}

void main()
{
}

// CHECK-NOT: declare {{.*}}template_foo

// CHECK-NOT: declare {{.*}}call_enforce_with_default_template_params
// CHECK-NOT: declare {{.*}}__lambda
// CHECK-NOT: declare {{.*}}_D3std9exception__T7enforce
