// Tests that -fvisibility=public for Windows targets exports defined symbols
// (dllexport) without explicit `export` visibility, and that linking an app
// against such a DLL via import library works as expected.

// REQUIRES: Windows

// generate DLL and import lib
// RUN: %ldc %S/inputs/fvisibility_dll_lib.d -betterC -shared -fvisibility=public -of=%t_lib.dll

// compile, link and run the app
// RUN: %ldc %s -I%S/inputs -betterC %t_lib.lib -of=%t.exe
// RUN: %t.exe

import fvisibility_dll_lib;

extern(C) void main()
{
    //assert(dllGlobal == 123);
    const x = dllSum(1, 2);
    assert(x == 3);
}
