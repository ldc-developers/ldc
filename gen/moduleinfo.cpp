//===-- moduleinfo.cpp ----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "gen/moduleinfo.h"

#include "gen/abi.h"
#include "gen/classes.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/objcgen.h"
#include "gen/rttibuilder.h"
#include "ir/irfunction.h"
#include "ir/irmodule.h"
#include "ir/irtype.h"
#include "module.h"

// These must match the values in druntime/src/object_.d
#define MIstandalone 0x4
#define MItlsctor 0x8
#define MItlsdtor 0x10
#define MIctor 0x20
#define MIdtor 0x40
#define MIxgetMembers 0x80
#define MIictor 0x100
#define MIunitTest 0x200
#define MIimportedModules 0x400
#define MIlocalClasses 0x800
#define MInew 0x80000000 // it's the "new" layout

namespace {
/// Creates a function in the current llvm::Module that dispatches to the given
/// functions one after each other and then increments the gate variables, if
/// any.
llvm::Function *buildForwarderFunction(
    const std::string &name, const std::list<FuncDeclaration *> &funcs,
    const std::list<VarDeclaration *> &gates = std::list<VarDeclaration *>()) {
  // If there is no gates, we might get away without creating a function at all.
  if (gates.empty()) {
    if (funcs.empty()) {
      return nullptr;
    }

    if (funcs.size() == 1) {
      return DtoCallee(funcs.front());
    }
  }

  // Create an internal-linkage void() function.
  const auto fnTy =
      LLFunctionType::get(LLType::getVoidTy(gIR->context()), {}, false);

  std::string const symbolName = gABI->mangleFunctionForLLVM(name, LINKd);
  assert(gIR->module.getFunction(symbolName) == NULL);
  llvm::Function *fn = llvm::Function::Create(
      fnTy, llvm::GlobalValue::InternalLinkage, symbolName, &gIR->module);
  fn->setCallingConv(gABI->callingConv(fn->getFunctionType(), LINKd));

  // Emit the body, consisting of...
  const auto bb = llvm::BasicBlock::Create(gIR->context(), "", fn);
  IRBuilder<> builder(bb);

  ldc::DISubprogram dis = gIR->DBuilder.EmitModuleCTor(fn, name.c_str());
  if (global.params.symdebug) {
    // Need _some_ debug info to avoid inliner bug, see GitHub issue #998.
    builder.SetCurrentDebugLocation(llvm::DebugLoc::get(0, 0, dis));
  }

  // ... calling the given functions, and...
  for (auto func : funcs) {
    const auto f = DtoCallee(func);
    const auto call = builder.CreateCall(f, {});
    const auto ft = call->getFunctionType();
    call->setCallingConv(gABI->callingConv(ft, LINKd));
  }

  // ... incrementing the gate variables.
  for (auto gate : gates) {
    assert(getIrGlobal(gate));
    const auto val = getIrGlobal(gate)->value;
    const auto rval = builder.CreateLoad(val, "vgate");
    const auto res = builder.CreateAdd(rval, DtoConstUint(1), "vgate");
    builder.CreateStore(res, val);
  }

  builder.CreateRetVoid();
  return fn;
}

namespace {
std::string getMangledName(Module *m, const char *suffix) {
  OutBuffer buf;
  buf.writestring("_D");
  mangleToBuffer(m, &buf);
  if (suffix)
    buf.writestring(suffix);
  return buf.peekString();
}
}

llvm::Function *buildModuleCtor(Module *m) {
  std::string name = getMangledName(m, "6__ctorZ");
  IrModule *irm = getIrModule(m);
  return buildForwarderFunction(name, irm->ctors, irm->gates);
}

llvm::Function *buildModuleDtor(Module *m) {
  std::string name = getMangledName(m, "6__dtorZ");
  return buildForwarderFunction(name, getIrModule(m)->dtors);
}

llvm::Function *buildModuleUnittest(Module *m) {
  std::string name = getMangledName(m, "10__unittestZ");
  return buildForwarderFunction(name, getIrModule(m)->unitTests);
}

llvm::Function *buildModuleSharedCtor(Module *m) {
  std::string name = getMangledName(m, "13__shared_ctorZ");
  IrModule *irm = getIrModule(m);
  return buildForwarderFunction(name, irm->sharedCtors, irm->sharedGates);
}

llvm::Function *buildModuleSharedDtor(Module *m) {
  std::string name = getMangledName(m, "13__shared_dtorZ");
  return buildForwarderFunction(name, getIrModule(m)->sharedDtors);
}

/// Builds the (constant) data content for the importedModules[] array.
llvm::Constant *buildImportedModules(Module *m, size_t &count) {
  const auto moduleInfoPtrTy = DtoPtrToType(Module::moduleinfo->type);

  std::vector<LLConstant *> importInits;
  for (auto mod : m->aimports) {
    if (!mod->needModuleInfo() || mod == m) {
      continue;
    }

    importInits.push_back(
        DtoBitCast(getIrModule(mod)->moduleInfoSymbol(), moduleInfoPtrTy));
  }
  count = importInits.size();

  if (importInits.empty())
    return nullptr;

  const auto type = llvm::ArrayType::get(moduleInfoPtrTy, importInits.size());
  return LLConstantArray::get(type, importInits);
}

/// Builds the (constant) data content for the localClasses[] array.
llvm::Constant *buildLocalClasses(Module *m, size_t &count) {
  const auto classinfoTy = Type::typeinfoclass->type->ctype->getLLType();

  ClassDeclarations aclasses;
  for (auto s : *m->members) {
    s->addLocalClass(&aclasses);
  }

  std::vector<LLConstant *> classInfoRefs;
  for (auto cd : aclasses) {
    DtoResolveClass(cd);

    if (cd->isInterfaceDeclaration()) {
      IF_LOG Logger::println("skipping interface '%s' in moduleinfo",
                             cd->toPrettyChars());
      continue;
    }

    if (cd->sizeok != SIZEOKdone) {
      IF_LOG Logger::println(
          "skipping opaque class declaration '%s' in moduleinfo",
          cd->toPrettyChars());
      continue;
    }

    IF_LOG Logger::println("class: %s", cd->toPrettyChars());
    classInfoRefs.push_back(
        DtoBitCast(getIrAggr(cd)->getClassInfoSymbol(), classinfoTy));
  }
  count = classInfoRefs.size();

  if (classInfoRefs.empty())
    return nullptr;

  const auto type = llvm::ArrayType::get(classinfoTy, classInfoRefs.size());
  return LLConstantArray::get(type, classInfoRefs);
}
}

llvm::GlobalVariable *genModuleInfo(Module *m) {
  if (!Module::moduleinfo) {
    m->error("object.d is missing the ModuleInfo struct");
    fatal();
  }

  // The "new-style" ModuleInfo records are variable-length, with the presence
  // of the various fields indicated by a certain flag bit. The base struct
  // should consist only of the _flags/_index fields (the latter of which is
  // unused).
  if (Module::moduleinfo->structsize != 4 + 4) {
    m->error("Unexpected size of struct object.ModuleInfo; "
             "druntime version does not match compiler (see -v)");
    fatal();
  }

  // First, figure out which fields are present and set the flags accordingly.
  unsigned flags = MInew;

  const auto fctor = buildModuleCtor(m);
  if (fctor) {
    flags |= MItlsctor;
  }

  const auto fdtor = buildModuleDtor(m);
  if (fdtor) {
    flags |= MItlsdtor;
  }

  const auto fsharedctor = buildModuleSharedCtor(m);
  if (fsharedctor) {
    flags |= MIctor;
  }

  const auto fshareddtor = buildModuleSharedDtor(m);
  if (fshareddtor) {
    flags |= MIdtor;
  }

#if 0
  if (fgetmembers)
    flags |= MIxgetMembers;
  if (fictor)
    flags |= MIictor;
#endif

  const auto funittest = buildModuleUnittest(m);
  if (funittest) {
    flags |= MIunitTest;
  }

  size_t importedModulesCount;
  const auto importedModules = buildImportedModules(m, importedModulesCount);
  if (importedModules) {
    flags |= MIimportedModules;
  }

  size_t localClassesCount;
  const auto localClasses = buildLocalClasses(m, localClassesCount);
  if (localClasses) {
    flags |= MIlocalClasses;
  }

  if (!m->needmoduleinfo) {
    flags |= MIstandalone;
  }

  // Now, start building the initialiser for the ModuleInfo instance.
  RTTIBuilder b(Module::moduleinfo);

  b.push_uint(flags);
  b.push_uint(0); // index

  if (fctor) {
    b.push(fctor);
  }
  if (fdtor) {
    b.push(fdtor);
  }
  if (fsharedctor) {
    b.push(fsharedctor);
  }
  if (fshareddtor) {
    b.push(fshareddtor);
  }
#if 0
    if (fgetmembers)
        b.push(fgetmembers);
    if (fictor)
        b.push(fictor);
#endif
  if (funittest) {
    b.push(funittest);
  }
  if (importedModules) {
    b.push_size(importedModulesCount);
    b.push(importedModules);
  }
  if (localClasses) {
    b.push_size(localClassesCount);
    b.push(localClasses);
  }

  // Put out module name as a 0-terminated string.
  const char *name = m->toPrettyChars();
  const size_t len = strlen(name) + 1;
  const auto it = llvm::IntegerType::getInt8Ty(gIR->context());
  const auto at = llvm::ArrayType::get(it, len);
  b.push(toConstantArray(it, at, name, len, false));

  objc_Module_genmoduleinfo_classes();

  // Create a global symbol with the above initialiser.
  LLGlobalVariable *moduleInfoSym = getIrModule(m)->moduleInfoSymbol();
  b.finalize(moduleInfoSym->getType()->getPointerElementType(), moduleInfoSym);
  setLinkage({LLGlobalValue::ExternalLinkage, false}, moduleInfoSym);
  return moduleInfoSym;
}
