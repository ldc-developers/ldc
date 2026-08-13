//===-- tools/runtime-adapt/source/ldcmods/native.d ---------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Non-D ldc/* listed from runtime/CMakeLists.txt (eh_asm.S) and compiler refs
// (arm_unwind.c, msvc.c). Bodies stay stubs — never copied from goal.
//
//===----------------------------------------------------------------------===//

module ldcmods.native;

import compilerparse;
import ldcmods.common;

bool wantNative(const CompilerModel)
{
    return true;
}

string renderArmUnwind(const CompilerModel, string tag)
{
    return renderCStub(tag, "arm unwind helpers");
}

string renderEhAsm(const CompilerModel, string tag)
{
    return renderCStub(tag, "EH assembly");
}

string renderMsvcC(const CompilerModel, string tag)
{
    return renderCStub(tag, "MSVC helpers (linker-msvc / ms-cxx-helper)");
}
