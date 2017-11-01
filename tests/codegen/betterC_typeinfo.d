// Make sure the file can be compiled and linked successfully with -betterC.
// Also test that druntime and Phobos aren't in the linker command line.
// RUN: %ldc -betterC %s -v > %t.log
// RUN: FileCheck %s --check-prefix=WITHOUT_TI < %t.log
// WITHOUT_TI-NOT: druntime-ldc
// WITHOUT_TI-NOT: phobos2-ldc

// With version=WITH_TI, make sure the file can be compiled with -betterC...
// RUN: %ldc -betterC -d-version=WITH_TI -c -of=%t%obj %s
// ... but not linked due to the undefined TypeInfo.
// RUN: not %ldc -betterC %t%obj > %t.fail 2>&1
// RUN: FileCheck %s --check-prefix=WITH_TI < %t.fail
// WITH_TI: _D37TypeInfo_S16betterC_typeinfo8MyStruct6__initZ

struct MyStruct { int a; }

extern (C) void main()
{
    auto s = MyStruct();
    version (WITH_TI)
        auto ti = typeid(MyStruct);
}
