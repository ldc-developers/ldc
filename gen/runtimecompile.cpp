#include "runtimecompile.h"

#if defined(LDC_RUNTIME_COMPILE)

#include <unordered_map>
#include <unordered_set>

#include "driver/cl_options.h"

#include "gen/irstate.h"
#include "gen/llvm.h"
#include "ir/irfunction.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/TypeBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace {

const char *RuntimeCompileModulesHeadName = "runtimecompile_modules_head";

llvm::GlobalValue *getPredefinedSymbol(llvm::Module &module,
                                       llvm::StringRef name, llvm::Type *type) {
  assert(nullptr != type);
  auto ret = module.getNamedValue(name);
  if (nullptr != ret) {
    return ret;
  }
  if (type->isFunctionTy()) {
    ret = llvm::Function::Create(llvm::cast<llvm::FunctionType>(type),
                                 llvm::GlobalValue::ExternalLinkage, name,
                                 &module);
  } else {
    ret = new llvm::GlobalVariable(
        module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr, name);
  }
  return ret;
}

template <typename C, typename T> bool contains(const C &cont, const T &val) {
  return cont.end() != cont.find(val);
}

template <typename F> void enumOperands(const llvm::User &usr, F &&handler) {
  for (auto &&op : usr.operands()) {
    llvm::Value *val = op.get();
    if (auto opusr = llvm::dyn_cast<llvm::User>(val)) {
      if (auto gv = llvm::dyn_cast<llvm::GlobalValue>(opusr)) {
        handler(gv);
      }
      enumOperands(*opusr, std::forward<F>(handler));
    }
  }
}

template <typename F> void enumFuncSymbols(llvm::Function *fun, F &&handler) {
  assert(nullptr != fun);
  if (fun->hasPersonalityFn()) {
    if (auto personality =
            llvm::dyn_cast<llvm::GlobalValue>(fun->getPersonalityFn())) {
      handler(personality);
    }
  }

  for (auto &&bb : *fun) {
    for (auto &&instr : bb) {
      enumOperands(instr, std::forward<F>(handler));
    }
  }
}

enum class GlobalValVisibility {
  Internal,
  External,
  Declaration,
};

using GlobalValsMap =
    std::unordered_map<llvm::GlobalValue *, GlobalValVisibility>;

void getPredefinedSymbols(IRState *irs, GlobalValsMap &symList) {
  assert(nullptr != irs);
  const llvm::Triple *triple = global.params.targetTriple;
  if (!opts::runtimeCompileTlsWorkaround) {
    if (triple->isWindowsMSVCEnvironment() ||
        triple->isWindowsGNUEnvironment()) {
      symList.insert(std::make_pair(
          getPredefinedSymbol(irs->module, "_tls_index",
                              llvm::Type::getInt32Ty(irs->context())),
          GlobalValVisibility::Declaration));
      if (triple->isArch32Bit()) {
        symList.insert(std::make_pair(
            getPredefinedSymbol(irs->module, "_tls_array",
                                llvm::Type::getInt32Ty(irs->context())),
            GlobalValVisibility::Declaration));
      }
    }
  }
}

GlobalValsMap createGlobalValsFilter(IRState *irs) {
  assert(nullptr != irs);
  GlobalValsMap ret;
  getPredefinedSymbols(irs, ret);
  std::vector<llvm::Function *> newFunctions;
  newFunctions.reserve(irs->runtimeCompiledFunctions.size());

  for (auto &&it : irs->runtimeCompiledFunctions) {
    ret.insert({it.first, GlobalValVisibility::External});
    newFunctions.push_back(it.first);
  }

  std::unordered_set<llvm::GlobalValue *> runtimeCompiledVars;
  for (auto &&var : irs->runtimeCompiledVars) {
    assert(nullptr != var);
    assert(nullptr != var->value);
    runtimeCompiledVars.insert(llvm::cast<llvm::GlobalValue>(var->value));
  }

  std::vector<llvm::Function *> functionsToAdd;
  while (!newFunctions.empty()) {
    for (auto &&fun : newFunctions) {
      enumFuncSymbols(fun, [&](llvm::GlobalValue *gv) {
        if (!contains(runtimeCompiledVars, gv)) {
          auto it = ret.insert({gv, GlobalValVisibility::Declaration});
          if (it.second && !gv->isDeclaration()) {
            if (auto newFun = llvm::dyn_cast<llvm::Function>(gv)) {
              if (!newFun->isIntrinsic()) {
                it.first->second = GlobalValVisibility::Internal;
                functionsToAdd.push_back(newFun);
              }
            }
          }
        }
      });
    }
    newFunctions.swap(functionsToAdd);
    functionsToAdd.clear();
  }
  return ret;
}

template <typename F>
void iterateFuncInstructions(llvm::Function &func, F &&handler) {
  for (auto &&bb : func) {
    // We can change bb contents in this loop
    // so we reiterate it from start after each change
    bool bbChanged = true;
    while (bbChanged) {
      bbChanged = false;
      for (auto &&instr : bb) {
        if (handler(instr)) {
          bbChanged = true;
          break;
        }
      } // for (auto &&instr : bb)
    }   // while (bbChanged)
  }     // for (auto &&bb : fun)
}

void fixRtModule(llvm::Module &newModule,
                 const decltype(IRState::runtimeCompiledFunctions) &funcs) {
  std::unordered_map<std::string, std::string> thunkVar2func;
  std::unordered_map<std::string, std::string> thunkFun2func;
  std::unordered_set<std::string> externalFuncs;
  for (auto &&it : funcs) {
    assert(nullptr != it.first);
    assert(nullptr != it.second.thunkVar);
    assert(nullptr != it.second.thunkFunc);
    assert(!contains(thunkVar2func, it.second.thunkVar->getName()));
    thunkVar2func.insert({it.second.thunkVar->getName(), it.first->getName()});
    thunkFun2func.insert({it.second.thunkFunc->getName(), it.first->getName()});
    externalFuncs.insert(it.first->getName());
  }

  // Replace call to thunks in jitted code with direct calls to functions
  for (auto &&fun : newModule.functions()) {
    iterateFuncInstructions(fun, [&](llvm::Instruction &instr) -> bool {
      if (auto call = llvm::dyn_cast<llvm::CallInst>(&instr)) {
        auto callee = call->getCalledValue();
        assert(nullptr != callee);
        auto it = thunkFun2func.find(callee->getName());
        if (thunkFun2func.end() != it) {
          auto realFunc = newModule.getFunction(it->second);
          assert(nullptr != realFunc);
          call->setCalledFunction(realFunc);
        }
      }
      return false;
    });
  }

  int objectsFixed = 0;
  for (auto &&obj : newModule.globals()) {
    auto it = thunkVar2func.find(obj.getName());
    if (thunkVar2func.end() != it) {
      if (obj.hasInitializer()) {
        auto func = newModule.getFunction(it->second);
        assert(nullptr != func);
        obj.setConstant(true);
        obj.setInitializer(func);
      }
      ++objectsFixed;
    }
  }
  for (auto &&obj : newModule.functions()) {
    if (contains(externalFuncs, obj.getName())) {
      obj.setLinkage(llvm::GlobalValue::ExternalLinkage);
      obj.setVisibility(llvm::GlobalValue::DefaultVisibility);
      ++objectsFixed;
    }
  }
  assert((thunkVar2func.size() + externalFuncs.size()) == objectsFixed);
}

llvm::Function *createGlobalVarLoadFun(llvm::Module &module,
                                       llvm::GlobalVariable *var,
                                       const llvm::Twine &funcName) {
  assert(nullptr != var);
  auto &context = module.getContext();
  auto varType = var->getType();
  auto funcType = llvm::FunctionType::get(varType, false);
  auto func = llvm::Function::Create(
      funcType, llvm::GlobalValue::WeakODRLinkage, funcName, &module);
  auto bb = llvm::BasicBlock::Create(context, "", func);

  llvm::IRBuilder<> builder(context);
  builder.SetInsertPoint(bb);
  builder.CreateRet(var);

  return func;
}

void replaceDynamicThreadLocals(llvm::Module &oldModule,
                                llvm::Module &newModule,
                                GlobalValsMap &valsMap) {
  // Wrap all thread locals access in dynamic code by function calls
  // to 'normal' code
  std::unordered_map<llvm::GlobalVariable *, llvm::Function *>
      threadLocalAccessors;

  auto getAccessor = [&](llvm::GlobalVariable *var) {
    assert(nullptr != var);
    auto it = threadLocalAccessors.find(var);
    if (threadLocalAccessors.end() != it) {
      return it->second;
    }

    auto srcVar = oldModule.getGlobalVariable(var->getName());
    assert(nullptr != srcVar);
    auto srcFunc = createGlobalVarLoadFun(oldModule, srcVar,
                                          "." + var->getName() + "_accessor");
    srcFunc->addFnAttr(llvm::Attribute::NoInline);
    auto dstFunc = llvm::Function::Create(srcFunc->getFunctionType(),
                                          llvm::GlobalValue::ExternalLinkage,
                                          srcFunc->getName(), &newModule);
    threadLocalAccessors.insert({var, dstFunc});
    valsMap.insert({srcFunc, GlobalValVisibility::Declaration});
    return dstFunc;
  };

  for (auto &&fun : newModule.functions()) {
    iterateFuncInstructions(fun, [&](llvm::Instruction &instr) -> bool {
      bool changed = false;
      for (unsigned int i = 0; i < instr.getNumOperands(); ++i) {
        auto op = instr.getOperand(i);
        if (auto globalVar = llvm::dyn_cast<llvm::GlobalVariable>(op)) {
          if (globalVar->isThreadLocal()) {
            auto accessor = getAccessor(globalVar);
            assert(nullptr != accessor);
            auto callResult = llvm::CallInst::Create(accessor);
            callResult->insertBefore(&instr);
            instr.setOperand(i, callResult);
            changed = true;
          }
        }
      }
      return changed;
    });
  } // for (auto &&fun : newModule.functions())

  for (auto &&it : threadLocalAccessors) {
    it.first->eraseFromParent();
  }
}

llvm::Constant *getArrayPtr(llvm::Constant *array) {
  assert(nullptr != array);
  llvm::ConstantInt *zero = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(array->getContext()), 0, false);
  llvm::Constant *idxs[] = {zero, zero};
  return llvm::ConstantExpr::getGetElementPtr(nullptr, array, idxs, true);
}

llvm::Constant *getI8Ptr(llvm::GlobalValue *val) {
  assert(nullptr != val);
  return llvm::ConstantExpr::getBitCast(
      val, llvm::IntegerType::getInt8PtrTy(val->getContext()));
}

std::pair<llvm::Constant *, llvm::Constant *>
getArrayAndSize(llvm::Module &module, llvm::Type *elemType,
                llvm::ArrayRef<llvm::Constant *> elements) {
  assert(nullptr != elemType);
  auto arrayType = llvm::ArrayType::get(elemType, elements.size());
  auto arrVar = new llvm::GlobalVariable(
      module, arrayType, true, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantArray::get(arrayType, elements), ".str");
  return std::make_pair(
      getArrayPtr(arrVar),
      llvm::ConstantInt::get(module.getContext(), APInt(32, elements.size())));
}

template <typename T>
void createStaticArray(llvm::Module &mod, llvm::GlobalVariable *var,
                       llvm::GlobalVariable *varLen, // can be null
                       llvm::ArrayRef<T> arr) {
  assert(nullptr != var);
  const auto dataLen = arr.size();
  auto gvar = new llvm::GlobalVariable(
      mod,
      llvm::ArrayType::get(llvm::TypeBuilder<T, false>::get(mod.getContext()),
                           dataLen),
      true, llvm::GlobalValue::InternalLinkage,
      llvm::ConstantDataArray::get(mod.getContext(), arr), ".str");
  var->setInitializer(getArrayPtr(gvar));
  if (nullptr != varLen) {
    varLen->setInitializer(
        llvm::ConstantInt::get(mod.getContext(), APInt(32, dataLen)));
  }
}

llvm::Constant *createStringInitializer(llvm::Module &mod,
                                        llvm::StringRef str) {
  auto nameVar = new llvm::GlobalVariable(
      mod,
      llvm::ArrayType::get(llvm::Type::getInt8Ty(mod.getContext()),
                           str.size() + 1),
      true, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantDataArray::getString(mod.getContext(), str, true), ".str");
  return llvm::ConstantExpr::getBitCast(
      nameVar, llvm::Type::getInt8PtrTy(mod.getContext()));
}

// void createStaticString(llvm::Module& mod,
//                        llvm::GlobalVariable* var,
//                        llvm::GlobalVariable* varLen, //can be null
//                        llvm::StringRef str) {
//  assert(nullptr != var);
//  const auto dataLen = str.size() + 1;
//  auto gvar = new llvm::GlobalVariable(
//                mod,
//                llvm::ArrayType::get(llvm::Type::getInt8Ty(mod.getContext()),
//                dataLen), true, llvm::GlobalValue::InternalLinkage,
//                llvm::ConstantDataArray::getString(mod.getContext(), str,
//                true),
//                ".str");
//  var->setInitializer(getArrayPtr(gvar));
//  if (nullptr != varLen) {
//    varLen->setInitializer(llvm::ConstantInt::get(mod.getContext(), APInt(32,
//    dataLen)));
//  }
//}

// struct RtCompileVarList
// {
//   i8* name;
//   i8* ptr;
// }

llvm::StructType *getVarListElemType(llvm::LLVMContext &context) {
  llvm::Type *elements[] = {
      llvm::IntegerType::getInt8PtrTy(context),
      llvm::IntegerType::getInt8PtrTy(context),
  };
  return llvm::StructType::create(context, elements, /*"RtCompileVarList"*/ "",
                                  true);
}

// struct RtCompileSymList
// {
//   i8* name;
//   i8* sym;
// };

llvm::StructType *getSymListElemType(llvm::LLVMContext &context) {
  llvm::Type *elements[] = {
      llvm::IntegerType::getInt8PtrTy(context),
      llvm::IntegerType::getInt8PtrTy(context),
  };
  return llvm::StructType::create(context, elements, /*"RtCompileSymList"*/ "",
                                  true);
}

// struct RtCompileFuncList
// {
//   i8* name;
//   i8* func;
// };

llvm::StructType *getFuncListElemType(llvm::LLVMContext &context) {
  llvm::Type *elements[] = {
      llvm::IntegerType::getInt8PtrTy(context),
      llvm::IntegerType::getInt8PtrTy(context),
  };
  return llvm::StructType::create(context, elements, /*"RtCompileFuncList"*/ "",
                                  true);
}

// struct RtCompileModuleList
// {
//   RtCompileModuleList* next;
//   i8* irData;
//   i32 irDataSize;
//   RtCompileFuncList* funcList;
//   i32 funcListSize;
//   RtCompileSymList* symList;
//   i32 symListSize;
// };

llvm::StructType *getModuleListElemType(llvm::LLVMContext &context,
                                        llvm::StructType *funcListElemType,
                                        llvm::StructType *symListElemType,
                                        llvm::StructType *varListElemType) {
  assert(nullptr != funcListElemType);
  assert(nullptr != symListElemType);
  assert(nullptr != varListElemType);
  llvm::StructType *ret =
      llvm::StructType::create(context /*, "RtCompileModuleList"*/); // fwddecl
  llvm::Type *elements[] = {
      llvm::PointerType::getUnqual(ret),
      llvm::IntegerType::getInt8PtrTy(context),
      llvm::IntegerType::get(context, 32),
      llvm::PointerType::getUnqual(funcListElemType),
      llvm::IntegerType::get(context, 32),
      llvm::PointerType::getUnqual(symListElemType),
      llvm::IntegerType::get(context, 32),
      llvm::PointerType::getUnqual(varListElemType),
      llvm::IntegerType::get(context, 32),
  };
  ret->setBody(elements, true);
  return ret;
}

struct Types {
  llvm::StructType *funcListElemType;
  llvm::StructType *symListElemType;
  llvm::StructType *varListElemType;
  llvm::StructType *modListElemType;

  Types(llvm::LLVMContext &context)
      : funcListElemType(getFuncListElemType(context)),
        symListElemType(getSymListElemType(context)),
        varListElemType(getVarListElemType(context)),
        modListElemType(getModuleListElemType(
            context, funcListElemType, symListElemType, varListElemType)) {}
};

std::pair<llvm::Constant *, llvm::Constant *>
generateFuncList(IRState *irs, const Types &types) {
  assert(nullptr != irs);
  std::vector<llvm::Constant *> elements;
  for (auto &&it : irs->runtimeCompiledFunctions) {
    assert(nullptr != it.first);
    assert(nullptr != it.second.thunkVar);
    assert(nullptr != it.second.thunkFunc);
    auto name = it.first->getName();
    llvm::Constant *fields[] = {
        createStringInitializer(irs->module, name),
        getI8Ptr(it.second.thunkVar),
    };
    elements.push_back(
        llvm::ConstantStruct::get(types.funcListElemType, fields));
  }
  return getArrayAndSize(irs->module, types.funcListElemType, elements);
}

std::pair<llvm::Constant *, llvm::Constant *>
generateSymList(IRState *irs, const Types &types,
                const GlobalValsMap &globalVals) {
  assert(nullptr != irs);
  std::vector<llvm::Constant *> elements;
  for (auto &&it : globalVals) {
    if (it.second == GlobalValVisibility::Declaration) {
      auto val = it.first;
      if (auto fun = llvm::dyn_cast<llvm::Function>(val)) {
        if (fun->isIntrinsic())
          continue;
      }
      auto name = val->getName();
      llvm::Constant *fields[] = {
          createStringInitializer(irs->module, name),
          getI8Ptr(val),
      };
      elements.push_back(
          llvm::ConstantStruct::get(types.symListElemType, fields));
    }
  }
  return getArrayAndSize(irs->module, types.symListElemType, elements);
}

std::pair<llvm::Constant *, llvm::Constant *>
generateVarList(IRState *irs, const Types &types) {
  assert(nullptr != irs);
  std::vector<llvm::Constant *> elements;
  for (auto &&val : irs->runtimeCompiledVars) {
    auto gvar = llvm::cast<llvm::GlobalVariable>(val->value);
    auto name = gvar->getName();
    llvm::Constant *fields[] = {
        createStringInitializer(irs->module, name),
        getI8Ptr(gvar),
    };
    elements.push_back(
        llvm::ConstantStruct::get(types.varListElemType, fields));
  }
  return getArrayAndSize(irs->module, types.varListElemType, elements);
}

llvm::GlobalVariable *generateModuleListElem(IRState *irs, const Types &types,
                                             llvm::GlobalVariable *irData,
                                             llvm::GlobalVariable *irDataLen,
                                             const GlobalValsMap &globalVals) {
  assert(nullptr != irs);
  auto elem_type = types.modListElemType;
  auto funcListInit = generateFuncList(irs, types);
  auto symListInit = generateSymList(irs, types, globalVals);
  auto varlistInit = generateVarList(irs, types);
  llvm::Constant *fields[] = {
      llvm::ConstantPointerNull::get(llvm::dyn_cast<llvm::PointerType>(
          elem_type->getElementType(0))), // next
      irData->getInitializer(),           // irdata
      irDataLen->getInitializer(),        // irdata len
      funcListInit.first,                 // funclist
      funcListInit.second,                // funclist len
      symListInit.first,                  // symlist
      symListInit.second,                 // symlist len
      varlistInit.first,                  // varlist
      varlistInit.second,                 // varlist len
  };

  auto init = llvm::ConstantStruct::get(elem_type, fields);

  return new llvm::GlobalVariable(irs->module, elem_type, false,
                                  llvm::GlobalValue::PrivateLinkage, init,
                                  ".rtcompile_modlist_elem");
}

llvm::PointerType *getModListHeadType(llvm::LLVMContext &context,
                                      const Types &types) {
  (void)types;
  return llvm::IntegerType::getInt8PtrTy(context);
}

llvm::GlobalVariable *declareModListHead(llvm::Module &module,
                                         const Types &types) {
  auto type = getModListHeadType(module.getContext(), types);
  //  auto existingVar =
  //  module.getGlobalVariable(RuntimeCompileModulesHeadName); if (nullptr !=
  //  existingVar) {
  //    if (type != existingVar->getType()) {
  //      error(Loc(), "Invalid RuntimeCompileModulesHeadName type");
  //      fatal();
  //    }
  //    return existingVar;
  //  }
  return new llvm::GlobalVariable(module, type, false,
                                  llvm::GlobalValue::ExternalLinkage, nullptr,
                                  RuntimeCompileModulesHeadName);
}

void generateCtorBody(IRState *irs, const Types &types, llvm::Function *func,
                      llvm::Value *modListElem) {
  assert(nullptr != irs);
  assert(nullptr != func);
  assert(nullptr != modListElem);

  auto bb = llvm::BasicBlock::Create(irs->context(), "", func);

  llvm::IRBuilder<> builder(irs->context());
  builder.SetInsertPoint(bb);

  auto zero64 = llvm::ConstantInt::get(irs->context(), APInt(64, 0));
  auto zero32 = llvm::ConstantInt::get(irs->context(), APInt(32, 0));
  auto modListHeadPtr = declareModListHead(irs->module, types);
  llvm::Value *gepVals[] = {zero64, zero32};
  auto elemNextPtr = builder.CreateGEP(modListElem, gepVals);
  auto prevHeadVal = builder.CreateLoad(builder.CreateBitOrPointerCast(
      modListHeadPtr, types.modListElemType->getPointerTo()->getPointerTo()));
  auto voidPtr = builder.CreateBitOrPointerCast(
      modListElem, llvm::IntegerType::getInt8PtrTy(irs->context()));
  builder.CreateStore(voidPtr, modListHeadPtr);
  builder.CreateStore(prevHeadVal, elemNextPtr);

  builder.CreateRetVoid();
}

void setupModuleCtor(IRState *irs, llvm::GlobalVariable *irData,
                     llvm::GlobalVariable *irDataLen,
                     const GlobalValsMap &globalVals) {
  assert(nullptr != irs);
  assert(nullptr != irData);
  assert(nullptr != irDataLen);
  Types types(irs->context());
  auto modListElem =
      generateModuleListElem(irs, types, irData, irDataLen, globalVals);
  auto runtimeCompiledCtor = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(irs->context()), false),
      llvm::GlobalValue::InternalLinkage, ".rtcompile_ctor", &irs->module);
  generateCtorBody(irs, types, runtimeCompiledCtor, modListElem);
  llvm::appendToGlobalCtors(irs->module, runtimeCompiledCtor, 0);
}

void setupModuleBitcodeData(const llvm::Module &srcModule, IRState *irs,
                            const GlobalValsMap &globalVals) {
  assert(nullptr != irs);

  llvm::SmallString<1024> str;
  llvm::raw_svector_ostream os(str);
  llvm::WriteBitcodeToFile(&srcModule, os);

  auto runtimeCompiledIr = new llvm::GlobalVariable(
      irs->module, llvm::Type::getInt8PtrTy(irs->context()), true,
      llvm::GlobalValue::PrivateLinkage, nullptr, ".rtcompile_ir");

  auto runtimeCompiledIrSize = new llvm::GlobalVariable(
      irs->module, llvm::IntegerType::get(irs->context(), 32), true,
      llvm::GlobalValue::PrivateLinkage, nullptr, ".rtcompile_irsize");

  createStaticArray(irs->module, runtimeCompiledIr, runtimeCompiledIrSize,
                    llvm::ArrayRef<uint8_t>(
                        reinterpret_cast<uint8_t *>(str.data()), str.size()));

  setupModuleCtor(irs, runtimeCompiledIr, runtimeCompiledIrSize, globalVals);
}

void copyFuncAttributes(llvm::Function &dstFunc,
                        const llvm::Function &srcFunc) {
  dstFunc.setCallingConv(srcFunc.getCallingConv());
  dstFunc.setAttributes(srcFunc.getAttributes());
  dstFunc.setDLLStorageClass(srcFunc.getDLLStorageClass());
  dstFunc.setLinkage(srcFunc.getLinkage());
}

llvm::Function *duplicateFunc(llvm::Module &module, const llvm::Function *src) {
  assert(nullptr != src);
  auto ret = llvm::Function::Create(
      src->getFunctionType(), llvm::GlobalObject::ExternalLinkage,
      src->getName() + "__rtcomp_thunk__", &module);
  copyFuncAttributes(*ret, *src);
  return ret;
}

void createThunkFunc(llvm::Module &module, const llvm::Function *src,
                     llvm::Function *dst, llvm::GlobalVariable *thunkVar) {
  assert(nullptr != src);
  assert(nullptr != dst);
  assert(nullptr != thunkVar);

  auto bb = llvm::BasicBlock::Create(module.getContext(), "", dst);
  llvm::IRBuilder<> builder(module.getContext());
  builder.SetInsertPoint(bb);
  auto thunkPtr = builder.CreateLoad(thunkVar);
  llvm::SmallVector<llvm::Value *, 6> args;
  for (auto &arg : dst->args()) {
    args.push_back(&arg);
  }
  auto ret = builder.CreateCall(thunkPtr, args);
  if (dst->getReturnType()->isVoidTy()) {
    builder.CreateRetVoid();
  } else {
    builder.CreateRet(ret);
  }
}

} // anon namespace

void generateBitcodeForRuntimeCompile(IRState *irs) {
  assert(nullptr != irs);
  if (irs->runtimeCompiledFunctions.empty()) {
    return;
  }
  auto filter = createGlobalValsFilter(irs);

  llvm::ValueToValueMapTy unused;
  auto newModule = llvm::CloneModule(
      &irs->module, unused, [&](const llvm::GlobalValue *val) -> bool {
        // We don't dereference here, so const_cast should be safe
        auto it = filter.find(const_cast<llvm::GlobalValue *>(val));
        return filter.end() != it &&
               it->second != GlobalValVisibility::Declaration;
      });
  if (opts::runtimeCompileTlsWorkaround) {
    replaceDynamicThreadLocals(irs->module, *newModule, filter);
  }
  fixRtModule(*newModule, irs->runtimeCompiledFunctions);

  setupModuleBitcodeData(*newModule, irs, filter);
}

void declareRuntimeCompiledFunction(IRState *irs, IrFunction *func) {
  assert(nullptr != irs);
  assert(nullptr != func);
  assert(nullptr != func->getLLVMFunc());
  if (!opts::enableRuntimeCompile) {
    return;
  }
  auto srcFunc = func->getLLVMFunc();
  auto thunkFunc = duplicateFunc(irs->module, srcFunc);
  func->rtCompileFunc = thunkFunc;
  assert(!contains(irs->runtimeCompiledFunctions, srcFunc));
  irs->runtimeCompiledFunctions.insert(
      std::make_pair(srcFunc, IRState::RtCompiledFuncDesc{nullptr, thunkFunc}));
}

void defineRuntimeCompiledFunction(IRState *irs, IrFunction *func) {
  assert(nullptr != irs);
  assert(nullptr != func);
  assert(nullptr != func->getLLVMFunc());
  assert(nullptr != func->rtCompileFunc);
  if (!opts::enableRuntimeCompile) {
    return;
  }
  auto srcFunc = func->getLLVMFunc();
  auto it = irs->runtimeCompiledFunctions.find(srcFunc);
  assert(irs->runtimeCompiledFunctions.end() != it);
  auto thunkVarType = srcFunc->getFunctionType()->getPointerTo();
  auto thunkVar = new llvm::GlobalVariable(
      irs->module, thunkVarType, false, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantPointerNull::get(thunkVarType),
      ".rtcompile_thunkvar_" + srcFunc->getName());
  auto dstFunc = it->second.thunkFunc;
  createThunkFunc(irs->module, srcFunc, dstFunc, thunkVar);
  it->second.thunkVar = thunkVar;
}

void addRuntimeCompiledVar(IRState *irs, IrGlobal *var) {
  assert(nullptr != irs);
  assert(nullptr != var);
  assert(nullptr != var->value);
  assert(nullptr != var->V);
  if (!opts::enableRuntimeCompile) {
    return;
  }

  if (var->V->isThreadlocal()) {
    error(Loc(), "Runtime compiled variable \"%s\" cannot be thread local",
          var->V->toChars());
    fatal();
  }

  irs->runtimeCompiledVars.insert(var);
}

#else // defined(LDC_RUNTIME_COMPILE)
void generateBitcodeForRuntimeCompile(IRState *) {
  // nothing
}

void declareRuntimeCompiledFunction(IRState *, IrFunction *) {
  // nothing
}

void defineRuntimeCompiledFunction(IRState *, IrFunction *) {
  // nothing
}

void addRuntimeCompiledVar(IRState *, IrGlobal *) {
  // nothing
}

#endif
