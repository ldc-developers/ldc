//===-- ms-cxx-helper.h ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "gen/irstate.h"

llvm::StructType *getTypeDescriptorType(IRState &irs,
                                        llvm::Constant *classInfoPtr,
                                        llvm::StringRef TypeInfoString);
llvm::GlobalVariable *getTypeDescriptor(IRState &irs, ClassDeclaration *cd);

void findSuccessors(std::vector<llvm::BasicBlock *> &blocks,
                    llvm::BasicBlock *bb, llvm::BasicBlock *ebb);

void remapBlocksValue(std::vector<llvm::BasicBlock *> &blocks,
                      llvm::Value *from, llvm::Value *to);

void cloneBlocks(const std::vector<llvm::BasicBlock *> &srcblocks,
                 std::vector<llvm::BasicBlock *> &blocks,
                 llvm::BasicBlock *continueWith, llvm::BasicBlock *unwindTo,
                 llvm::Value *funclet);

bool isCatchSwitchBlock(llvm::BasicBlock* bb);
