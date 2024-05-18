// Makes sure T.mangleof and pragma(mangle, …) are in sync.
// inputs/mangling_definitions.d features C/C++/D symbols and validates their .mangleof.
// This module forward-declares these symbols, using an identical pragma(mangle, …) string,
// and references the symbols in main().
// Compile both modules separately and make sure the 2 objects can be linked successfully.

// RUN: %ldc %S/inputs/mangling_definitions.d -c -of=%t-dir/mangling_definitions%obj
// RUN: %ldc %t-dir/mangling_definitions%obj -run %s

module mangling;

// Variables:

extern(C) pragma(mangle, "cGlobal")
extern __gshared int decl_cGlobal;

version(CRuntime_Microsoft)
    enum cppGlobalMangle = "?cppGlobal@cpp_vars@@3HA";
else
    enum cppGlobalMangle = "_ZN8cpp_vars9cppGlobalE";

extern(C++, decl_cpp_vars) pragma(mangle, cppGlobalMangle)
extern __gshared int decl_cppGlobal;

pragma(mangle, "_D11definitions7dGlobali")
extern __gshared int decl_dGlobal;

// Functions:

extern(C) pragma(mangle, "cFunc")
int decl_cFunc(double a);

version(CRuntime_Microsoft)
    enum cppFuncMangle = "?cppFunc@cpp_funcs@@YAHN@Z";
else
    enum cppFuncMangle = "_ZN9cpp_funcs7cppFuncEd";

extern(C++, decl_cpp_funcs) pragma(mangle, cppFuncMangle)
int decl_cppFunc(double a);

pragma(mangle, "_D11definitions5dFuncFdZi")
int decl_dFunc(double a);

// Naked functions:

version(D_InlineAsm_X86)
    version = AsmX86;
else version(D_InlineAsm_X86_64)
    version = AsmX86;

version(AsmX86)
{
    extern(C) pragma(mangle, "naked_cFunc")
    int decl_naked_cFunc(double a);

    version(CRuntime_Microsoft)
        enum nakedCppFuncMangle = "?naked_cppFunc@cpp_naked_funcs@@YAHN@Z";
    else
        enum nakedCppFuncMangle = "_ZN15cpp_naked_funcs13naked_cppFuncEd";

    extern(C++, decl_cpp_naked_funcs) pragma(mangle, nakedCppFuncMangle)
    int decl_naked_cppFunc(double a);

    pragma(mangle, "_D11definitions11naked_dFuncFiZi")
    int decl_naked_dFunc(int a);
}

// Interfacing with C via pragma(mangle, …), without having to take care
// of a potential target-specific C prefix (underscore on Win32 and OSX):
extern(C) pragma(mangle, "cos")
double decl_cos(double x);

void main()
{
    assert(decl_cGlobal == 1);
    assert(decl_cppGlobal == 2);
    assert(decl_dGlobal == 3);

    assert(decl_cFunc(1.0) == 1);
    assert(decl_cppFunc(2.0) == 2);
    assert(decl_dFunc(3.0) == 3);

    version(AsmX86)
    {
        decl_naked_cFunc(1.0);
        decl_naked_cppFunc(2.0);
        decl_naked_dFunc(3);
    }

    assert(decl_cos(0.0) == 1.0);
}
