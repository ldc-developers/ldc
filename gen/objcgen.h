//===-- gen/objcgen.cpp -----------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for generating Objective-C method calls.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_GEN_OBJCGEN_H
#define LDC_GEN_OBJCGEN_H

struct ObjcSelector;
namespace llvm {
class GlobalVariable;
class Triple;
}

bool objc_isSupported(const llvm::Triple &triple);
void objc_init();
void objc_Module_genmoduleinfo_classes();
llvm::GlobalVariable *objc_getMethVarRef(const ObjcSelector &sel);

#endif // LDC_GEN_OBJCGEN_H
