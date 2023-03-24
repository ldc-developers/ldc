// RUN: %ldc %s -c --output-ll -of=%t.ll --mtriple=arm-none-newlibeabi --betterC --checkaction=C > %t && FileCheck %s < %t.ll

extern (C) void main()
{
    assert(false);
}

// CHECK: declare void @__assert_func(i8*, i32, i8*, i8*)
