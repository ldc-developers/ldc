// Test ldc.attributes.callingConvention

// To make testing easier (no exhaustive test for all recognized calling convention names), we choose to test only X86.

// REQUIRES: target_X86
// RUN: %ldc -mtriple=x86_64-linux-gnu -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;


//////////////////////////////////////////////////////////
/// Normal (D) function calls and invokes:

// CHECK-LABEL: define x86_vectorcallcc void @{{.*}}foofoofoo
@callingConvention("vectorcall") void foofoofoo()
{
}

// CHECK-LABEL: define{{.*}} @{{.*}}foocallsfoofoofoo
void foocallsfoofoofoo()
{
  // CHECK: call x86_vectorcallcc void @{{.*}}attr_callingconvention9foofoofoo
  foofoofoo();
}

// CHECK-LABEL: define{{.*}} @{{.*}}fooinvokesfoofoofoo
void fooinvokesfoofoofoo()
{
  RAII force_invoke;
  // CHECK: invoke x86_vectorcallcc void @{{.*}}attr_callingconvention9foofoofoo
  foofoofoo();
}

//////////////////////////////////////////////////////////
/// Extern (D) function calls and invokes:

// CHECK-LABEL: define x86_vectorcallcc void @fooexternCfoofoo
extern(C) @callingConvention("vectorcall") void fooexternCfoofoo()
{
}

// CHECK-LABEL: define{{.*}} @{{.*}}foocallsexternCfoofoofoo
void foocallsexternCfoofoofoo()
{
  // CHECK: call x86_vectorcallcc void @{{.*}}fooexternCfoofoo
  fooexternCfoofoo();
}

// CHECK-LABEL: define{{.*}} @{{.*}}fooinvokesexternCfoofoofoo
void fooinvokesexternCfoofoofoo()
{
  RAII force_invoke;
  // CHECK: invoke x86_vectorcallcc void @{{.*}}fooexternCfoofoo
  fooexternCfoofoo();
}



//////////////////////////////////////////////////////////
/// Forward-declared function calls and invokes:

// CHECK-LABEL: define{{.*}} @{{.*}}attr_callingconvention34foocalls_forward_declared_function
void foocalls_forward_declared_function()
{
  // CHECK: call x86_vectorcallcc void @{{.*}}attr_callingconvention25forward_declared_function
  forward_declared_function();
}

// CHECK-LABEL: declare x86_vectorcallcc void @{{.*}}forward_declared_function
@callingConvention("vectorcall") void forward_declared_function();

// CHECK-LABEL: define{{.*}} @{{.*}}attr_callingconvention36fooinvokes_forward_declared_function
void fooinvokes_forward_declared_function()
{
  RAII force_invoke;
  // CHECK: invoke x86_vectorcallcc void @{{.*}}attr_callingconvention25forward_declared_function
  forward_declared_function();
}


//////////////////////////////////////////////////////////
/// Struct function calls and invokes:
struct A {
  @callingConvention("vectorcall") void struct_method() {}
}

// CHECK-LABEL: define{{.*}} @{{.*}}foocalls_struct_method
void foocalls_struct_method()
{
  A a;
  // CHECK: call x86_vectorcallcc void @{{.*}}struct_method
  a.struct_method();
}

// CHECK-LABEL: define{{.*}} @{{.*}}fooinvokes_struct_method
void fooinvokes_struct_method()
{
  RAII force_invoke;
  A a;
  // CHECK: invoke x86_vectorcallcc void @{{.*}}struct_method
  a.struct_method();
}



//////////////////////////////////////////////////////////
/// Class virtual function calls and invokes:
class C {
  @callingConvention("default") @callingConvention("vectorcall") void virtual_method() {}
}

// CHECK-LABEL: define{{.*}} @{{.*}}foocalls_virtual_method
void foocalls_virtual_method()
{
  C c = new C;
  // CHECK: call x86_vectorcallcc void %virtual_method
  c.virtual_method();
}

// CHECK-LABEL: define{{.*}} @{{.*}}fooinvokes_virtual_method
void fooinvokes_virtual_method()
{
  RAII force_invoke;
  C c;
  // CHECK: invoke x86_vectorcallcc void %virtual_method
  c.virtual_method();
}

//////////////////////////////////////////////////////////
// Check scope rule, multiple application
@callingConvention("vectorcall"):
// CHECK-LABEL: define void @{{.*}}gggscopeggg
@callingConvention("default") void gggscopeggg()
{
}
// CHECK-LABEL: define x86_vectorcallcc void @{{.*}}fooscopefoofoo
void fooscopefoofoo()
{
}
@callingConvention("default"):

//////////////////////////////////////////////////////////
// RAII struct to force `invoke` on function calls.
struct RAII {
  ~this() {}
}
