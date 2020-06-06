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
#include "dmd/init.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/mangling.h"
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
  if (auto irClassType = type->ctype->isClass())
    llType = irClassType->getMemoryLLType();

  llStructType = llvm::dyn_cast<LLStructType>(llType);

  return llStructType;
}

//////////////////////////////////////////////////////////////////////////////

LLConstant *&IrAggr::getInitSymbol(bool define) {
  if (!init) {
    // create the initZ symbol
    const auto irMangle = getIRMangledInitSymbolName(aggrdecl);

    auto initGlobal =
        declareGlobal(aggrdecl->loc, gIR->module, getLLStructType(), irMangle,
                      /*isConstant=*/true);
    initGlobal->setAlignment(LLMaybeAlign(DtoAlignment(type)));

    init = initGlobal;

    if (DtoIsTemplateInstance(aggrdecl))
      define = true;
  }

  if (define) {
    auto initConstant = getDefaultInit();
    auto initGlobal = llvm::dyn_cast<LLGlobalVariable>(init);
    if (initGlobal && !initGlobal->hasInitializer()) {
      init = gIR->setGlobalVarInitializer(initGlobal, initConstant);
      setLinkageAndVisibility(aggrdecl, initGlobal);
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
  if (field->_init) {
    // Issue 9057 workaround caused by issue 14666 fix, see DMD upstream
    // commit 069f570005.
    if (field->semanticRun < PASSsemantic2done && field->_scope) {
      semantic2(field, field->_scope);
    }
    return DtoConstInitializer(field->_init->loc, field->type, field->_init);
  }

  return DtoConstInitializer(field->loc, field->type);
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

  if (arrTypeD->ty == Tsarray) {
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
  if (type->ty == Tclass) {
    // add vtbl
    constants.push_back(getVtblSymbol());
    offset += target.ptrsize;

    // add monitor (except for C++ classes)
    if (!aggrdecl->isClassDeclaration()->isCPPclass()) {
      constants.push_back(getNullValue(getVoidPtrType()));
      offset += target.ptrsize;
    }
  }

  // Add the initializers for the member fields. While we are traversing the
  // class hierarchy, use the opportunity to populate interfacesWithVtbls if
  // we haven't done so previously (due to e.g. ClassReferenceExp, we can
  // have multiple initializer constants for a single class).
  addFieldInitializers(constants, explicitInitializers, aggrdecl, offset,
                       interfacesWithVtbls.empty());

  // tail padding?
  const size_t structsize = aggrdecl->size(Loc());
  if (offset < structsize)
    add_zeros(constants, offset, structsize);

  assert(!constants.empty());

  // get LL field types
  llvm::SmallVector<llvm::Type *, 16> types;
  types.reserve(constants.size());
  for (auto c : constants)
    types.push_back(c->getType());

  auto llStructType = getLLStructType();
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
  LLStructType *llType =
      isCompatible ? llStructType
                   : LLStructType::get(gIR->context(), types, isPacked());
  llvm::Constant *c = LLConstantStruct::get(llType, constants);
  IF_LOG Logger::cout() << "final initializer: " << *c << std::endl;
  return c;
}

void IrAggr::addFieldInitializers(
    llvm::SmallVectorImpl<llvm::Constant *> &constants,
    const VarInitMap &explicitInitializers, AggregateDeclaration *decl,
    unsigned &offset, bool populateInterfacesWithVtbls) {

  if (ClassDeclaration *cd = decl->isClassDeclaration()) {
    if (cd->baseClass) {
      addFieldInitializers(constants, explicitInitializers, cd->baseClass,
                           offset, populateInterfacesWithVtbls);
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

      size_t inter_idx = interfacesWithVtbls.size();
      for (auto bc : *cd->vtblInterfaces) {
        constants.push_back(getInterfaceVtblSymbol(bc, inter_idx));
        offset += target.ptrsize;
        inter_idx++;

        if (populateInterfacesWithVtbls)
          interfacesWithVtbls.push_back(bc);
      }
    }
  }

  AggrTypeBuilder b(false, offset);
  b.addAggregate(decl, &explicitInitializers, AggrTypeBuilder::Aliases::Skip);
  offset = b.currentOffset();

  const size_t baseLLFieldIndex = constants.size();
  const size_t numNewLLFields = b.defaultTypes().size();
  constants.resize(constants.size() + numNewLLFields, nullptr);

  // add explicit and non-overlapping implicit initializers
  for (const auto &pair : b.varGEPIndices()) {
    const auto field = pair.first;
    const size_t fieldIndex = pair.second;

    const auto explicitIt = explicitInitializers.find(field);
    llvm::Constant *init = (explicitIt != explicitInitializers.end()
                                ? explicitIt->second
                                : getDefaultInitializer(field));

    constants[baseLLFieldIndex + fieldIndex] =
        FillSArrayDims(field->type, init);
  }

  // zero out remaining padding fields
  for (size_t i = 0; i < numNewLLFields; i++) {
    auto &init = constants[baseLLFieldIndex + i];
    if (!init)
      init = llvm::Constant::getNullValue(b.defaultTypes()[i]);
  }
}

IrAggr *getIrAggr(AggregateDeclaration *decl, bool create) {
  if (!isIrAggrCreated(decl) && create) {
    assert(decl->ir->irAggr == NULL);
    decl->ir->irAggr = new IrAggr(decl);
    decl->ir->m_type = IrDsymbol::AggrType;
  }
  assert(decl->ir->irAggr != NULL);
  return decl->ir->irAggr;
}

bool isIrAggrCreated(AggregateDeclaration *decl) {
  int t = decl->ir->type();
  assert(t == IrDsymbol::AggrType || t == IrDsymbol::NotSet);
  return t == IrDsymbol::AggrType;
}
