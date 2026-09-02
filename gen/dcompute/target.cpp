//===-- gen/dcompute/target.cpp -------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX

#include "dmd/dsymbol.h"
#include "dmd/errors.h"
#include "dmd/module.h"
#include "dmd/scope.h"
#include "driver/linker.h"
#include "driver/toobj.h"
#include "driver/cl_options.h"
#include "gen/dcompute/target.h"
#include "gen/llvmhelpers.h"
#include "gen/runtime.h"
#include "ir/irtypestruct.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/Module.h"
#include <algorithm>


void DComputeTarget::doCodeGen(Module *m) {
  // Reset any generated type info for dcompute types.
  // The ll types get generated when the host code gets
  // gen'd which means the address space info is not
  // properly set.
  IrTypeStruct::resetDComputeTypes();

  // process module members
  for (unsigned k = 0; k < m->members->length; k++) {
    Dsymbol *dsym = (*m->members)[k];
    assert(dsym);
    Declaration_codegen(dsym, _ir);
  }

  if (global.errors)
    fatal();
}

void DComputeTarget::emit(Module *m) {
  // Reset the global ABI to the target's ABI. Necessary because we have
  // multiple ABI we are trying to target. Also reset gIR. These are both
  // reused. MAJOR HACK.
  gABI = abi;
  gIR = _ir;
  gTargetMachine = targetMachine;
  modules.push_back(m);
  doCodeGen(m);
}

#include "llvm/Support/Program.h"
#include "gen/optimizer.h"

void DComputeTarget::writeModule(llvm::Module *hostModule) {
  addMetadata();
  gABI = abi;
  gIR = _ir;
  gTargetMachine = targetMachine;
  std::string filename;
  llvm::raw_string_ostream os(filename);
  const bool is64 = global.params.targetTriple->isArch64Bit();
  os << opts::dcomputeFilePrefix << '_' << short_name << tversion << '_'
     << (is64 ? 64 : 32) << '.' << binSuffix;

  const char *path =
      FileName::combine(global.params.objdir.ptr, os.str().c_str());

  ::writeModule(&_ir->module, path);

  if (optLevel() > 0 &&
      (target == ID::Vulkan || target == ID::OpenCL)) {
    if (auto spirvOpt = llvm::sys::findProgramByName("spirv-opt")) {
      std::vector<llvm::StringRef> optArgs;
      optArgs.push_back(*spirvOpt);
      optArgs.push_back("-O");
      optArgs.push_back(path);
      optArgs.push_back("-o");
      optArgs.push_back(path);
      std::string errorMsg;
      llvm::sys::ExecuteAndWait(*spirvOpt, optArgs, std::nullopt, {}, 0, 0, &errorMsg);
    }
  }
  
  delete _ir;
  _ir = nullptr;

  if (!hostModule)
    return;

  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return;

  llvm::StringRef deviceString = bufferOrErr.get()->getBuffer();
  llvm::Constant *deviceConst = llvm::ConstantDataArray::getString(
      ctx, deviceString, true);

  std::string prefix = "__dcompute_" + std::string(binSuffix) + "_" +
                       std::string(short_name) + std::to_string(tversion);
  std::string internalName = prefix + "_internal_" + opts::dcomputeFilePrefix;
  
  auto *gv = new llvm::GlobalVariable(
      *hostModule,
      deviceConst->getType(),
      true, // isConstant
      llvm::GlobalValue::PrivateLinkage,
      deviceConst,
      internalName);
  gv->setAlignment(llvm::Align(4));

  for (auto *m : modules) {
    std::string modName = m->toPrettyChars();
    std::replace(modName.begin(), modName.end(), '.', '_');
    std::string symName = prefix + "_" + modName;

    auto *alias = llvm::GlobalAlias::create(
        deviceConst->getType(),
        0, // address space
        llvm::GlobalValue::ExternalLinkage,
        symName,
        gv,
        hostModule);

    if (alias->getName() != symName) {
      if (auto *existing = hostModule->getNamedValue(symName)) {
        existing->replaceAllUsesWith(alias);
        existing->eraseFromParent();
      }
      alias->setName(symName);
    }
  }
}

#endif
