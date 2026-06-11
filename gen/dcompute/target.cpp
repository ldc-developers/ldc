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

void DComputeTarget::writeModule(llvm::Module *hostModule) {
  addMetadata();

  std::string filename;
  llvm::raw_string_ostream os(filename);
  const bool is64 = global.params.targetTriple->isArch64Bit();
  os << opts::dcomputeFilePrefix << '_' << short_name << tversion << '_'
     << (is64 ? 64 : 32) << '.' << binSuffix;

  const char *path =
      FileName::combine(global.params.objdir.ptr, os.str().c_str());

  ::writeModule(&_ir->module, path);

  if (hostModule) {
    auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
    if (bufferOrErr) {
      llvm::StringRef ptxString = bufferOrErr.get()->getBuffer();
      llvm::Constant *ptxConst = llvm::ConstantDataArray::getString(
          ctx, ptxString, true);

      std::string internalName = "__dcompute_ptx_internal_" + std::string(short_name) + std::to_string(tversion) + "_" + opts::dcomputeFilePrefix;
      
      auto *gv = new llvm::GlobalVariable(
          *hostModule,
          ptxConst->getType(),
          true, // isConstant
          llvm::GlobalValue::PrivateLinkage,
          ptxConst,
          internalName);

      for (auto *m : modules) {
        std::string modName = m->toPrettyChars();
        std::replace(modName.begin(), modName.end(), '.', '_');
        std::string symName = "__dcompute_ptx_" + std::string(short_name) + std::to_string(tversion) + "_" + modName;

        auto *alias = llvm::GlobalAlias::create(
            ptxConst->getType(),
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
  }

  delete _ir;
  _ir = nullptr;
}

#endif
