module definitions;

// variables:

extern(C) __gshared int cGlobal = 1;
static assert(cGlobal.mangleof == "cGlobal");

extern(C++, cpp_vars) __gshared int cppGlobal = 2;
version(CRuntime_Microsoft)
    static assert(cppGlobal.mangleof == "?cppGlobal@cpp_vars@@3HA");
else
    static assert(cppGlobal.mangleof == "_ZN8cpp_vars9cppGlobalE");

__gshared int dGlobal = 3;
static assert(dGlobal.mangleof == "_D11definitions7dGlobali");

// functions:

extern(C) int cFunc(double a) { return cGlobal; }
static assert(cFunc.mangleof == "cFunc");

extern(C++, cpp_funcs) int cppFunc(double a) { return cppGlobal; }
version(CRuntime_Microsoft)
    static assert(cppFunc.mangleof == "?cppFunc@cpp_funcs@@YAHN@Z");
else
    static assert(cppFunc.mangleof == "_ZN9cpp_funcs7cppFuncEd");

int dFunc(double a) { return dGlobal; }
static assert(dFunc.mangleof == "_D11definitions5dFuncFdZi");

// naked functions:

version(D_InlineAsm_X86)
    version = AsmX86;
else version(D_InlineAsm_X86_64)
    version = AsmX86;

version(AsmX86)
{
    extern(C) int naked_cFunc(double a) { asm { naked; ret; } }
    static assert(naked_cFunc.mangleof == "naked_cFunc");

    extern(C++, cpp_naked_funcs) int naked_cppFunc(double a) { asm { naked; ret; } }
    version(CRuntime_Microsoft)
        static assert(naked_cppFunc.mangleof == "?naked_cppFunc@cpp_naked_funcs@@YAHN@Z");
    else
        static assert(naked_cppFunc.mangleof == "_ZN15cpp_naked_funcs13naked_cppFuncEd");

    // Pass an int instead of a double due to x86 calling convetion
    // See: https://github.com/ldc-developers/ldc/pull/4661
    int naked_dFunc(int a) { asm { naked; ret; } }
    static assert(naked_dFunc.mangleof == "_D11definitions11naked_dFuncFiZi");
}
