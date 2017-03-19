//===-- gen/modules.h - Entry points for D module codegen -------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

struct IRState;
class Module;

/// Generates code for the contents of module m into the LLVM module associated
/// with irs.
void codegenModule(IRState *irs, Module *m);
