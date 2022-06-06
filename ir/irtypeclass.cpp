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
  vtbl_type = LLArrayType::get(getVoidPtrType(), cd->vtbl.length);
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
      auto vtblTy = LLArrayType::get(getVoidPtrType(), b->sym->vtbl.length);
      builder.addType(llvm::PointerType::get(vtblTy, 0), target.ptrsize);

      ++num_interface_vtbls;
    }
  }

  // Finally, the data members for this class.
  builder.addAggregate(currCd);
}

IrTypeClass *IrTypeClass::get(ClassDeclaration *cd) {
  IF_LOG Logger::println("Building class type %s @ %s", cd->toPrettyChars(),
                         cd->loc.toChars());
  LOG_SCOPE;

  const auto t = new IrTypeClass(cd);
  getIrType(cd->type) = t;

  IF_LOG Logger::println("Instance size: %u", cd->structsize);

  // This class may contain an align declaration. See GitHub #726.
  t->packed = false;
  for (auto base = cd; base != nullptr && !t->packed; base = base->baseClass) {
    t->packed = isPacked(base);
  }

  AggrTypeBuilder builder(t->packed);

  // add vtbl
  builder.addType(llvm::PointerType::get(t->vtbl_type, 0), target.ptrsize);

  if (cd->isInterfaceDeclaration()) {
    // interfaces are just a vtable
    t->num_interface_vtbls =
        cd->vtblInterfaces ? cd->vtblInterfaces->length : 0;
  } else {
    // classes have monitor and fields
    if (!cd->isCPPclass() && !cd->isCPPinterface()) {
      // add monitor
      builder.addType(
          llvm::PointerType::get(llvm::Type::getInt8Ty(gIR->context()), 0),
          target.ptrsize);
    }

    // add data members recursively
    t->addClassData(builder, cd);

    // add tail padding
    builder.addTailPadding(cd->structsize);
  }

  // set struct body and copy GEP indices
  isaStruct(t->type)->setBody(builder.defaultTypes(), t->packed);
  t->varGEPIndices = builder.varGEPIndices();

  IF_LOG Logger::cout() << "class type: " << *t->type << std::endl;

  return t;
}

llvm::Type *IrTypeClass::getLLType() { return llvm::PointerType::get(type, 0); }

llvm::Type *IrTypeClass::getMemoryLLType() { return type; }

size_t IrTypeClass::getInterfaceIndex(ClassDeclaration *inter) {
  auto it = interfaceMap.find(inter);
  if (it == interfaceMap.end()) {
    return ~0UL;
  }
  return it->second;
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
