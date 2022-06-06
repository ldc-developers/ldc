// Tests that -fvisibility=public for Windows targets exports defined symbols
// (dllexport) without explicit `export` visibility, and that linking an app
// against such a DLL via import library works as expected.

// REQUIRES: Windows

// generate DLL and import lib (public visibility by default)
// RUN: %ldc %S/inputs/fvisibility_dll_lib.d -betterC -shared -of=%t_lib.dll

// compile, link and run the app; `-dllimport=all` for dllimporting data symbols
// RUN: %ldc %s -I%S/inputs -betterC -dllimport=all %t_lib.lib -of=%t.exe
// RUN: %t.exe

import fvisibility_dll_lib;

// test manual 'relocation' of references to dllimported globals in static data initializers:
__gshared int* dllimportRef = &dllGlobal;
__gshared void* castDllimportRef = &dllGlobal;
immutable void*[2] arrayOfRefs = [ null, cast(immutable) &dllGlobal ];

struct MyStruct
{
    int* dllimportRef = &dllGlobal; // init symbol references dllimported global
}

extern(C) void main()
{
    assert(dllGlobal == 123);

    const x = dllSum(1, 2);
    assert(x == 3);

    dllWeakFoo();

    scope c = new MyClass;
    assert(c.myInt == 456);

    assert(dllimportRef == &dllGlobal);
    assert(castDllimportRef == &dllGlobal);
    assert(arrayOfRefs[1] == &dllGlobal);
    assert(MyStruct.init.dllimportRef == &dllGlobal);
}
