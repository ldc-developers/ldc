// Tests that -fvisibility=hidden doesn't affect imported and fwd declared symbols,
// so that they can still be linked in from a shared library.

// UNSUPPORTED: Windows

// RUN: %ldc %S/inputs/export_marked_symbols_lib.d -shared -of=%t_lib%so
// RUN: %ldc %s -I%S/inputs -fvisibility=hidden -of=%t%exe %t_lib%so
// RUN: %ldc %s -I%S/inputs -fvisibility=hidden -of=%t%exe %t_lib%so -d-version=DECLARE_MANUALLY

version (DECLARE_MANUALLY)
{
    extern(C++):

    export extern __gshared int exportedGlobal;
    extern __gshared int normalGlobal;

    export void exportedFoo();
    void normalFoo();
}
else
{
    import export_marked_symbols_lib;
}

void main()
{
    exportedGlobal = 1;
    normalGlobal = 2;

    exportedFoo();
    normalFoo();
}
