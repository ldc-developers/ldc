// Make sure the file can be compiled and linked successfully with -betterC.
// Also test that druntime and Phobos aren't in the linker command line.
// RUN: %ldc -betterC %s -v > %t.log
// RUN: FileCheck %s < %t.log
// CHECK-NOT: druntime-ldc
// CHECK-NOT: phobos2-ldc

struct MyStruct { int a; }

extern (C) void main()
{
    auto s = MyStruct();
}
