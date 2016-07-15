//===-- dcompute/codegenmanager.cpp ---------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dcompute/codegenmanager.h"
#include "ir/irdsymbol.h"
std::vector<DComputeCodeGenManager::target> DComputeCodeGenManager::clTargets = {{1, 210},{ 2, 350} };
DComputeTarget *DComputeCodeGenManager::createComputeTarget(target t) {
  switch (t.platform) {
    case 1:
      return createOCLTarget(ctx,t._version);
    case 2:
      return createCUDATarget(ctx,t._version);
    default:
      llvm_unreachable("no such compute target");

  }
  
}

DComputeCodeGenManager::DComputeCodeGenManager(llvm::LLVMContext &c): ctx(c) {
    for (int i = 0; i<clTargets.size() ; i++) {
        targets.push_back(createComputeTarget(clTargets[i]));
    }
}
                          
void DComputeCodeGenManager::emit(Module *m)
{
    for (int  i = 0 ; i < targets.size(); i++) {
        targets[i]->emit(m);
        IrDsymbol::resetAll();
    }
}

DComputeCodeGenManager::~DComputeCodeGenManager() {
    
}

void DComputeCodeGenManager::writeModules()
{
    for (int i = 0; i<targets.size() ; i++) {
        targets[i]->writeModule();
    }
}

