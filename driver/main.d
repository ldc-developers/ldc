//===-- driver/main.d - D entry point -----------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// D entry point for LDC/LDMD, just forwarding to cppmain().
//
//===----------------------------------------------------------------------===//

module driver.main;

// In driver/main.cpp or driver/ldmd.cpp
extern(C++) int cppmain();

/+ We use this manual D main for druntime initialization via a manual
 + _d_run_main() call in the C main() in driver/{main,ldmd}.cpp.
 +/
extern(C) int _Dmain(string[])
{
    version (Windows)
        switchConsoleCodePageToUTF8();

    return cppmain();
}

// We use UTF-8 for narrow strings, on Windows too.
version (Windows)
{
    import core.sys.windows.wincon;
    import core.sys.windows.windef : UINT;

    private:

    __gshared UINT originalCP, originalOutputCP;

    void switchConsoleCodePageToUTF8()
    {
        import core.stdc.stdlib : atexit;
        import core.sys.windows.winnls : CP_UTF8;

        originalCP = GetConsoleCP();
        originalOutputCP = GetConsoleOutputCP();

        SetConsoleCP(CP_UTF8);
        SetConsoleOutputCP(CP_UTF8);

        // atexit handlers are also called when exiting via exit() etc.;
        // that's the reason this isn't a RAII struct.
        atexit(&resetConsoleCodePage);
    }

    extern(C) void resetConsoleCodePage()
    {
        SetConsoleCP(originalCP);
        SetConsoleOutputCP(originalOutputCP);
    }
}
