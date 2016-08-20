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

/// Sets the m->*file members to the output file names, as derived from platform
/// and command line switches, making sure that they are valid paths and don't
/// conflict with the source file.
void buildTargetFiles(Module *m, bool singleObj, bool library);

/// Generates code for the contents of module m into the LLVM module associated
/// with irs.
void codegenModule(IRState *irs, Module *m);
