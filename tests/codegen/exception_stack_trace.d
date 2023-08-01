// RUN: %ldc -g -frame-pointer=all -link-defaultlib-debug %s -of=%t%exe
// RUN: %t%exe | FileCheck %s

void bar()
{
    throw new Exception("lala");
}

void foo()
{
    bar();
}

void main()
{
    try
    {
        foo();
    }
    catch (Exception e)
    {
        import core.stdc.stdio;
        auto s = e.toString();
        printf("%.*s\n", s.length, s.ptr);
    }
}

// CHECK:      object.Exception@{{.*}}exception_stack_trace.d(6): lala
// CHECK-NEXT: ----------------
/* Hiding all frames up to and including the first _d_throw_exception()
 * one doesn't work reliably on all platforms, so don't enforce
 * CHECK-*NEXT* for the bar() frame.
 * On Win32, the bar() frame is missing altogether.
 * So be very generous and only check for 2 consecutive lines containing
 * 'exception_stack_trace' each (in function name and/or source file).
 */
// CHECK:      exception_stack_trace{{.*$}}
// CHECK-NEXT: exception_stack_trace{{.*$}}
