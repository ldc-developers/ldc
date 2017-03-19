//===-- gen/moduleinfo.h - ModuleInfo instance data emission ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

namespace llvm {
class GlobalVariable;
}
class Module;

/// Creates a global variable containing the ModuleInfo data for the given
/// module.
///
/// Note that this just creates data itself, and is not concerned with emitting
/// a reference pointing to it to register the module with the runtime.
llvm::GlobalVariable *genModuleInfo(Module *m);
