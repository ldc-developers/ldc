// Tests that naked template functions get correct symbol mangling on Windows.
//
// The bug: When a naked template function is called, DtoDeclareFunction creates
// a declaration with the proper IR mangle name (including \01 prefix for
// vectorcall). Without the fix, DtoDefineNakedFunction would create a definition
// with a DIFFERENT mangle name (missing the \01 prefix), leaving the declared
// function undefined.
//
// This test verifies that calling and defining a naked template in the same
// module produces a linkable object file.
//
// See: https://github.com/ldc-developers/ldc/issues/4294

// REQUIRES: target_X86
// REQUIRES: atleast_llvm1600
// REQUIRES: ld.lld

// Compile for Windows with LTO and link into a DLL.
// LTO is critical for this test - without LTO, the linker resolves both
// symbol variants to the same name, masking the bug.
// RUN: %ldc -mtriple=x86_64-windows-msvc -betterC -flto=full -c %s -of=%t.obj
// RUN: ld.lld -flavor link /dll /noentry /export:caller %t.obj /out:%t.dll

module naked_lambda_linkage;

// Non-exported naked template function.
// The call in caller() triggers DtoDeclareFunction first, then the function
// is defined by DtoDefineNakedFunction. Without the fix, these use different
// mangle names and the declared symbol remains undefined.
uint nakedTemplateFunc(int N)() pure @safe @nogc {
    asm pure nothrow @nogc @trusted {
        naked;
        mov EAX, N;
        ret;
    }
}

// This function calls the naked template, triggering the declaration before definition
extern(C) export uint caller() {
    return nakedTemplateFunc!42();
}
