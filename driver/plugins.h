//===-- driver/plugins.h ---------------------------------------*-  C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Passes/PassBuilder.h"

void loadAllPlugins();
void registerAllPluginsWithPassBuilder(llvm::PassBuilder &PB);
