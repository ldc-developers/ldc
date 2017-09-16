//===-- gen/llvmhelpers.d - General LLVM codegen helpers ----------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// General codegen helper constructs for the D frontend.
//
//===----------------------------------------------------------------------===//

module gen.llvmhelpers;

import ddmd.func;
import ddmd.dtemplate;

/// Fixup an overloaded intrinsic name string.
extern (C++) void DtoSetFuncDeclIntrinsicName(TemplateInstance ti, TemplateDeclaration td, FuncDeclaration fd);

extern (C++) bool isArchx86_64();
extern (C++) bool isTargetWindowsMSVC();