//===-- ir/irmodule.h - Codegen state for top-level D modules ---*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Represents the state of a D module on its way through code generation. Also
// see the TODO in gen/module.cpp – parts of IRState really belong here.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_IR_IRMODULE_H
#define LDC_IR_IRMODULE_H

class Module;
namespace llvm
{
    class GlobalVariable;
}

struct IrModule
{
    IrModule(Module* module, const char* srcfilename);
    virtual ~IrModule();

    Module* M;

    llvm::GlobalVariable* fileName;
};

IrModule *getIrModule(Module *m);

#endif
