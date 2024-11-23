//===-- irtypeclass.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "ir/irtypeclass.h"

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/dsymbol.h"
#include "dmd/errors.h"
#include "dmd/mtype.h"
#include "dmd/target.h"
#include "dmd/template.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "llvm/IR/DerivedTypes.h"

IrTypeClass::IrTypeClass(ClassDeclaration *cd)
    : IrTypeAggr(cd), cd(cd), tc(static_cast<TypeClass *>(cd->type)) {
  vtbl_type = LLArrayType::get(getOpaquePtrType(), cd->vtbl.length);
}

void IrTypeClass::addClassData(AggrTypeBuilder &builder,
                               ClassDeclaration *currCd) {
  // First, recursively add the fields for our base class and interfaces, if
  // any.
  if (currCd->baseClass) {
    addClassData(builder, currCd->baseClass);
  }

  if (currCd->vtblInterfaces && currCd->vtblInterfaces->length > 0) {
    // KLUDGE: The first pointer in the vtbl will be of type object.Interface;
    // extract that from the "well-known" object.TypeInfo_Class definition.
    // For C++ interfaces, this vtbl entry has to be omitted

    builder.alignCurrentOffset(target.ptrsize);

    for (auto b : *currCd->vtblInterfaces) {
      IF_LOG Logger::println("Adding interface vtbl for %s",
                             b->sym->toPrettyChars());

      // add to the interface map
      addInterfaceToMap(b->sym, builder.currentFieldIndex());
      auto vtblTy = LLArrayType::get(getOpaquePtrType(), b->sym->vtbl.length);
      builder.addType(llvm::PointerType::get(vtblTy, 0), target.ptrsize);

      ++num_interface_vtbls;
    }
  }

  // Finally, the data members for this class.
  builder.addAggregate(currCd);
}

IrTypeClass *IrTypeClass::get(ClassDeclaration *cd) {
  const auto t = new IrTypeClass(cd);
  getIrType(cd->type) = t;
  return t;
}

llvm::Type *IrTypeClass::getLLType() { return getOpaquePtrType(); }

// Lazily build the actual IR struct type when needed.
// Note that it is this function that initializes most fields!
llvm::Type *IrTypeClass::getMemoryLLType() {
  if (!isaStruct(type)->isOpaque())
    return type;

  IF_LOG Logger::println("Building class type %s @ %s", cd->toPrettyChars(),
                         cd->loc.toChars());
  LOG_SCOPE;

  const unsigned instanceSize = cd->structsize;
  IF_LOG Logger::println("Instance size: %u", instanceSize);

  AggrTypeBuilder builder;

  // Objective-C just has an ISA pointer, so just
  // throw that in there.
  if (cd->classKind == ClassKind::objc) {
    builder.addType(getOpaquePtrType(), target.ptrsize);
    isaStruct(type)->setBody(builder.defaultTypes(), builder.isPacked());
    return type;
  }

  // add vtbl
  builder.addType(llvm::PointerType::get(vtbl_type, 0), target.ptrsize);

  if (cd->isInterfaceDeclaration()) {
    // interfaces are just a vtable
    num_interface_vtbls =
        cd->vtblInterfaces ? cd->vtblInterfaces->length : 0;
  } else {
    // classes have monitor and fields
    if (!cd->isCPPclass() && !cd->isCPPinterface()) {
      // add monitor
      builder.addType(getOpaquePtrType(), target.ptrsize);
    }

    // add data members recursively
    addClassData(builder, cd);

    // add tail padding
    if (instanceSize) // can be 0 for opaque classes
      builder.addTailPadding(instanceSize);
  }

  // set struct body and copy GEP indices
  isaStruct(type)->setBody(builder.defaultTypes(), builder.isPacked());
  varGEPIndices = builder.varGEPIndices();

  if (!cd->isInterfaceDeclaration() && instanceSize &&
      getTypeAllocSize(type) != instanceSize) {
    error(cd->loc, "ICE: class IR size does not match the frontend size");
    fatal();
  }

  IF_LOG Logger::cout() << "class type: " << *type << std::endl;

  return type;
}

size_t IrTypeClass::getInterfaceIndex(ClassDeclaration *inter) {
  getMemoryLLType(); // lazily resolve

  auto it = interfaceMap.find(inter);
  if (it == interfaceMap.end()) {
    return ~0UL;
  }
  return it->second;
}

unsigned IrTypeClass::getNumInterfaceVtbls() {
  getMemoryLLType(); // lazily resolve
  return num_interface_vtbls;
}

const VarGEPIndices &IrTypeClass::getVarGEPIndices() {
  getMemoryLLType(); // lazily resolve
  return varGEPIndices;
}

void IrTypeClass::addInterfaceToMap(ClassDeclaration *inter, size_t index) {
  // don't duplicate work or overwrite indices
  if (interfaceMap.find(inter) != interfaceMap.end()) {
    return;
  }

  // add this interface
  interfaceMap.insert(std::make_pair(inter, index));

  // add the direct base interfaces recursively - they
  // are accessed through the same index
  if (inter->interfaces.length > 0) {
    BaseClass *b = inter->interfaces.ptr[0];
    addInterfaceToMap(b->sym, index);
  }
}
