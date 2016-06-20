// Test inlining of templates
// Templates that would otherwise not be codegenned, _should_ be codegenned for inlining when pragma(inline, true) is specified.

// RUN: %ldc %s -I%S -c -output-ll -release -enable-inlining -O0 -of=%t.O0.ll && FileCheck %s < %t.O0.ll

// Test linking too (separate compilation)
// RUN: %ldc -c -enable-inlining %S/inputs/inlinables.d -of=%t.inlinables%obj \
// RUN: && %ldc -I%S -enable-inlining %t.inlinables%obj %s -of=%t%exe

import inputs.inlinables;
import std.stdio;
import std.exception;

int foo(int i)
{
    return call_template_foo(i);
}
// CHECK-NOT: declare{{.*}}D6inputs10inlinables20__T12template_fooTiZ12template_fooUNaNbNiNfiZi
// CHECK: define{{.*}}D6inputs10inlinables20__T12template_fooTiZ12template_fooUNaNbNiNfiZi
// CHECK-SAME: #[[ATTR1:[0-9]+]]

// stdio.File.flush contains a call to errnoException, which contains __FILE__ as default template parameter.
// Make sure the symbol is inlined/defined and not declared (which will lead to linker errors if the location
// of the stdlib is different from where LDC was built from)
void ggg(ref File f)
{
    f.flush();
}

// CHECK-NOT: declare{{.*}}D3std9exception{{[0-9]+}}__T12errnoEnforce
// CHECK: define{{.*}}D3std9exception{{[0-9]+}}__T12errnoEnforce
// CHECK-SAME: #[[ATTR2:[0-9]+]]

void hhh()
{
    auto f = File("filename","r");
}

void main()
{
}

// CHECK-DAG: attributes #[[ATTR1]] ={{.*}} alwaysinline
// CHECK-DAG: attributes #[[ATTR2]] ={{.*}} alwaysinline
