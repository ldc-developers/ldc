//===-- iraggr.cpp --------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/iraggr.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/expression.h"
#include "dmd/identifier.h"
#include "dmd/init.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
#include "gen/pragma.h"
#include "gen/tollvm.h"
#include "ir/irdsymbol.h"
#include "ir/irtypeclass.h"
#include "ir/irtypestruct.h"
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////

llvm::StructType *IrAggr::getLLStructType() {
  if (llStructType)
    return llStructType;

  LLType *llType = DtoType(type);
  if (auto irClassType = getIrType(type)->isClass())
    llType = irClassType->getMemoryLLType();

  llStructType = llvm::dyn_cast<LLStructType>(llType);

  return llStructType;
}

//////////////////////////////////////////////////////////////////////////////

bool IrAggr::suppressTypeInfo() const {
  return !global.params.useTypeInfo || !Type::dtypeinfo ||
         aggrdecl->llvmInternal == LLVMno_typeinfo;
}

//////////////////////////////////////////////////////////////////////////////

bool IrAggr::useDLLImport() const {
  return dllimportDataSymbol(aggrdecl);
}

//////////////////////////////////////////////////////////////////////////////

LLConstant *IrAggr::getInitSymbol(bool define) {
  if (!init) {
    const auto irMangle = getIRMangledInitSymbolName(aggrdecl);

    // Init symbols of built-in TypeInfo classes (in rt.util.typeinfo) are
    // special.
    auto cd = aggrdecl->isClassDeclaration();
    const bool isBuiltinTypeInfo =
        cd && llvm::StringRef(cd->ident->toChars()).startswith("TypeInfo_");

    // Only declare the symbol if it isn't yet, otherwise the init symbol of
    // built-in TypeInfos may clash with an existing base-typed forward
    // declaration when compiling the rt.util.typeinfo unittests.
    auto initGlobal = gIR->module.getGlobalVariable(irMangle);
    if (initGlobal) {
      assert(!initGlobal->hasInitializer() &&
             "existing init symbol not expected to be defined");
      assert((isBuiltinTypeInfo ||
              initGlobal->getValueType() == getLLStructType()) &&
             "type of existing init symbol declaration doesn't match");
    } else {
      // Init symbols of built-in TypeInfos need to be kept mutable as the type
      // is not declared as immutable on the D side, and e.g. synchronized() can
      // be used on the implicit monitor.
      const bool isConstant = !isBuiltinTypeInfo;
      initGlobal = declareGlobal(aggrdecl->loc, gIR->module, getLLStructType(),
                                 irMangle, isConstant, false, useDLLImport());
    }

    initGlobal->setAlignment(llvm::MaybeAlign(DtoAlignment(type)));

    init = initGlobal;

    if (!define)
      define = defineOnDeclare(aggrdecl, /*isFunction=*/false);
  }

  if (define) {
    auto initConstant = getDefaultInit();
    auto initGlobal = llvm::dyn_cast<LLGlobalVariable>(init);
    if (initGlobal // NOT a bitcast pointer to helper global
        && !initGlobal->hasInitializer()) {
      init = gIR->setGlobalVarInitializer(initGlobal, initConstant, aggrdecl);
    }
  }

  return init;
}

//////////////////////////////////////////////////////////////////////////////

llvm::Constant *IrAggr::getDefaultInit() {
  if (constInit) {
    return constInit;
  }

  IF_LOG Logger::println("Building default initializer for %s",
                         aggrdecl->toPrettyChars());
  LOG_SCOPE;

  VarInitMap noExplicitInitializers;
  constInit = createInitializerConstant(noExplicitInitializers);
  return constInit;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// helper function that adds zero bytes to a vector of constants
// FIXME A similar function is in ir/irtypeaggr.cpp
static inline size_t
add_zeros(llvm::SmallVectorImpl<llvm::Constant *> &constants,
          size_t startOffset, size_t endOffset) {
  assert(startOffset <= endOffset);
  const size_t paddingSize = endOffset - startOffset;
  if (paddingSize) {
    llvm::ArrayType *pad = llvm::ArrayType::get(
        llvm::Type::getInt8Ty(gIR->context()), paddingSize);
    constants.push_back(llvm::Constant::getNullValue(pad));
  }
  return paddingSize ? 1 : 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

LLConstant *IrAggr::getDefaultInitializer(VarDeclaration *field) {
  // Issue 9057 workaround caused by issue 14666 fix, see DMD upstream
  // commit 069f570005.
  if (field->_init && field->semanticRun < PASS::semantic2done &&
      field->_scope) {
    semantic2(field, field->_scope);
  }

  return DtoConstInitializer(field->_init ? field->_init->loc : field->loc,
                             field->type, field->_init, field->isCsymbol());
}

// return a constant array of type arrTypeD initialized with a constant value,
// or that constant value
static llvm::Constant *FillSArrayDims(Type *arrTypeD, llvm::Constant *init) {
  // Check whether we actually need to expand anything.
  // KLUDGE: We don't have the initializer type here, so we can only check
  // the size without doing an expensive recursive D <-> LLVM type comparison.
  // The better way to solve this would be to just fix the initializer
  // codegen in any place where a scalar initializer might still be generated.
  if (gDataLayout->getTypeStoreSize(init->getType()) >= arrTypeD->size()) {
    return init;
  }

  if (arrTypeD->ty == TY::Tsarray) {
    init = FillSArrayDims(arrTypeD->nextOf(), init);
    size_t dim = static_cast<TypeSArray *>(arrTypeD)->dim->toUInteger();
    llvm::ArrayType *arrty = llvm::ArrayType::get(init->getType(), dim);
    return llvm::ConstantArray::get(arrty,
                                    std::vector<llvm::Constant *>(dim, init));
  }
  return init;
}

llvm::Constant *
IrAggr::createInitializerConstant(const VarInitMap &explicitInitializers) {
  IF_LOG Logger::println("Creating initializer constant for %s",
                         aggrdecl->toChars());
  LOG_SCOPE;

  llvm::SmallVector<llvm::Constant *, 16> constants;
  unsigned offset = 0;

  auto cd = aggrdecl->isClassDeclaration();
  IrClass *irClass = cd ? static_cast<IrClass *>(this) : nullptr;
  if (irClass) {
    // add vtbl
    constants.push_back(irClass->getVtblSymbol());
    offset += target.ptrsize;

    // add monitor (except for C++ classes)
    if (!cd->isCPPclass()) {
      constants.push_back(getNullValue(getVoidPtrType()));
      offset += target.ptrsize;
    }
  }

  // Add the initializers for the member fields.
  unsigned dummy = 0;
  bool isPacked = false;
  addFieldInitializers(constants, explicitInitializers, aggrdecl, offset, dummy,
                       isPacked);

  // tail padding?
  const size_t structsize = aggrdecl->size(Loc());
  if (offset < structsize)
    add_zeros(constants, offset, structsize);

  // get LL field types
  llvm::SmallVector<llvm::Type *, 16> types;
  types.reserve(constants.size());
  for (auto c : constants)
    types.push_back(c->getType());

  const auto llStructType = getLLStructType();
  bool isCompatible = (types.size() == llStructType->getNumElements());
  if (isCompatible) {
    for (size_t i = 0; i < types.size(); i++) {
      if (types[i] != llStructType->getElementType(i)) {
        isCompatible = false;
        break;
      }
    }
  }

  // build constant
  LLStructType *llType = llStructType;
  if (!isCompatible) {
    // Note: isPacked only reflects whether there are misaligned IR fields,
    // not checking whether the aggregate contains appropriate tail padding.
    // So the resulting constant might contain more (implicit) tail padding than
    // the actual type, which should be harmless.
    llType = LLStructType::get(gIR->context(), types, isPacked);
    assert(getTypeAllocSize(llType) >= structsize);
  }

  llvm::Constant *c = LLConstantStruct::get(llType, constants);
  IF_LOG Logger::cout() << "final initializer: " << *c << std::endl;
  return c;
}

void IrAggr::addFieldInitializers(
    llvm::SmallVectorImpl<llvm::Constant *> &constants,
    const VarInitMap &explicitInitializers, AggregateDeclaration *decl,
    unsigned &offset, unsigned &interfaceVtblIndex, bool &isPacked) {

  if (ClassDeclaration *cd = decl->isClassDeclaration()) {
    if (cd->baseClass) {
      addFieldInitializers(constants, explicitInitializers, cd->baseClass,
                           offset, interfaceVtblIndex, isPacked);
    }

    // has interface vtbls?
    if (cd->vtblInterfaces && cd->vtblInterfaces->length > 0) {
      // Align interface infos to pointer size.
      unsigned aligned =
          (offset + target.ptrsize - 1) & ~(target.ptrsize - 1);
      if (offset < aligned) {
        add_zeros(constants, offset, aligned);
        offset = aligned;
      }

      IrClass *irClass = static_cast<IrClass *>(this);
      for (auto bc : *cd->vtblInterfaces) {
        constants.push_back(
            irClass->getInterfaceVtblSymbol(bc, interfaceVtblIndex));
        offset += target.ptrsize;
        ++interfaceVtblIndex;
      }
    }
  }

  AggrTypeBuilder b(offset);
  b.addAggregate(decl,
                 explicitInitializers.empty() ? nullptr : &explicitInitializers,
                 AggrTypeBuilder::Aliases::Skip);
  offset = b.currentOffset();
  if (!isPacked)
    isPacked = b.isPacked();

  const size_t baseLLFieldIndex = constants.size();
  const size_t numNewLLFields = b.defaultTypes().size();
  constants.resize(constants.size() + numNewLLFields, nullptr);

  const auto getFieldInit = [&explicitInitializers](VarDeclaration *field) {
    const auto explicitIt = explicitInitializers.find(field);
    return explicitIt != explicitInitializers.end()
               ? explicitIt->second
               : getDefaultInitializer(field);
  };

  const auto addToBitFieldGroup = [&](BitFieldDeclaration *bf,
                                      unsigned fieldIndex, unsigned bitOffset) {
    LLConstant *init = getFieldInit(bf);
    if (init->isNullValue()) {
      // no bits to set
      return;
    }

    LLConstant *&constant = constants[baseLLFieldIndex + fieldIndex];

    using llvm::APInt;
    const auto fieldType = b.defaultTypes()[fieldIndex];
    const auto intSizeInBits = fieldType->getIntegerBitWidth();
    const APInt oldVal =
        constant ? constant->getUniqueInteger() : APInt(intSizeInBits, 0);
    const APInt bfVal = init->getUniqueInteger().zextOrTrunc(intSizeInBits);
    const APInt mask = APInt::getLowBitsSet(intSizeInBits, bf->fieldWidth)
                       << bitOffset;
    assert(!oldVal.intersects(mask) && "has masked bits set already");
    const APInt newVal = oldVal | ((bfVal << bitOffset) & mask);

    constant = LLConstant::getIntegerValue(fieldType, newVal);
  };

  // add explicit and non-overlapping implicit initializers
  const auto &gepIndices = b.varGEPIndices();
  for (const auto &pair : gepIndices) {
    const auto field = pair.first;
    const auto fieldIndex = pair.second;

    if (auto bf = field->isBitFieldDeclaration()) {
      // multiple bit fields can map to a single IR field (of integer type)
      addToBitFieldGroup(bf, fieldIndex, bf->bitOffset);
    } else {
      LLConstant *&constant = constants[baseLLFieldIndex + fieldIndex];
      assert(!constant);
      constant = FillSArrayDims(field->type, getFieldInit(field));
    }
  }

  // add extra bit field members (greater byte offset than the group's first
  // member)
  for (const auto &pair : b.extraBitFieldMembers()) {
    const auto bf = pair.first;
    const auto primary = pair.second;

    const auto fieldIndexIt = gepIndices.find(primary);
    assert(fieldIndexIt != gepIndices.end());

    assert(bf->offset > primary->offset);
    addToBitFieldGroup(bf, fieldIndexIt->second,
                       (bf->offset - primary->offset) * 8 + bf->bitOffset);
  }

  // TODO: sanity check that all explicit initializers have been dealt with?
  //       (potential issue for bitfields in unions...)

  // zero out remaining fields (padding and/or zero-initialized bit fields)
  for (size_t i = 0; i < numNewLLFields; i++) {
    auto &init = constants[baseLLFieldIndex + i];
    if (!init)
      init = llvm::Constant::getNullValue(b.defaultTypes()[i]);
  }
}

IrAggr *getIrAggr(AggregateDeclaration *decl, bool create) {
  if (!isIrAggrCreated(decl) && create) {
    assert(decl->ir->irAggr == nullptr);
    if (auto cd = decl->isClassDeclaration()) {
      decl->ir->irAggr = new IrClass(cd);
    } else {
      decl->ir->irAggr = new IrStruct(decl->isStructDeclaration());
    }
    decl->ir->m_type = IrDsymbol::AggrType;
  }
  assert(decl->ir->irAggr != nullptr);
  return decl->ir->irAggr;
}

IrStruct *getIrAggr(StructDeclaration *sd, bool create) {
  return static_cast<IrStruct *>(
      getIrAggr(static_cast<AggregateDeclaration *>(sd), create));
}

IrClass *getIrAggr(ClassDeclaration *cd, bool create) {
  return static_cast<IrClass *>(
      getIrAggr(static_cast<AggregateDeclaration *>(cd), create));
}

bool isIrAggrCreated(AggregateDeclaration *decl) {
  int t = decl->ir->type();
  assert(t == IrDsymbol::AggrType || t == IrDsymbol::NotSet);
  return t == IrDsymbol::AggrType;
}
