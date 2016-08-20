//===-- irtypeclass.cpp ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DerivedTypes.h"

#include "aggregate.h"
#include "declaration.h"
#include "dsymbol.h"
#include "mtype.h"
#include "target.h"
#include "template.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/functions.h"
#include "ir/irtypeclass.h"

IrTypeClass::IrTypeClass(ClassDeclaration *cd)
    : IrTypeAggr(cd), cd(cd), tc(static_cast<TypeClass *>(cd->type)) {
  std::string vtbl_name(cd->toPrettyChars());
  vtbl_name.append(".__vtbl");
  vtbl_type = LLStructType::create(gIR->context(), vtbl_name);
  vtbl_size = cd->vtbl.dim;
}

void IrTypeClass::addClassData(AggrTypeBuilder &builder,
                               ClassDeclaration *currCd) {
  // First, recursively add the fields for our base class and interfaces, if
  // any.
  if (currCd->baseClass) {
    addClassData(builder, currCd->baseClass);
  }

  if (currCd->vtblInterfaces && currCd->vtblInterfaces->dim > 0) {
    // KLUDGE: The first pointer in the vtbl will be of type object.Interface;
    // extract that from the "well-known" object.TypeInfo_Class definition.
    // For C++ interfaces, this vtbl entry has to be omitted
    const auto interfaceArrayType = Type::typeinfoclass->fields[3]->type;
    const auto interfacePtrType = interfaceArrayType->nextOf()->pointerTo();

    builder.alignCurrentOffset(Target::ptrsize);

    for (auto b : *currCd->vtblInterfaces) {
      IF_LOG Logger::println("Adding interface vtbl for %s",
                             b->sym->toPrettyChars());

      FuncDeclarations arr;
      b->fillVtbl(cd, &arr, currCd == cd);

      // add to the interface map
      addInterfaceToMap(b->sym, builder.currentFieldIndex());
      Type* first = b->sym->isCPPinterface() ? nullptr : interfacePtrType;
      const auto ivtblType =
          llvm::StructType::get(gIR->context(), buildVtblType(first, &arr));
      builder.addType(llvm::PointerType::get(ivtblType, 0), Target::ptrsize);

      ++num_interface_vtbls;
    }
  }

  // Finally, the data members for this class.
  builder.addAggregate(currCd);
}

IrTypeClass *IrTypeClass::get(ClassDeclaration *cd) {
  const auto t = new IrTypeClass(cd);
  cd->type->ctype = t;

  IF_LOG Logger::println("Building class type %s @ %s", cd->toPrettyChars(),
                         cd->loc.toChars());
  LOG_SCOPE;
  IF_LOG Logger::println("Instance size: %u", cd->structsize);

  // This class may contain an align declaration. See GitHub #726.
  t->packed = false;
  for (auto base = cd; base != nullptr && !t->packed; base = base->baseClass) {
    t->packed = isPacked(base);
  }

  AggrTypeBuilder builder(t->packed);

  // add vtbl
  builder.addType(llvm::PointerType::get(t->vtbl_type, 0), Target::ptrsize);

  if (cd->isInterfaceDeclaration()) {
    // interfaces are just a vtable
    t->num_interface_vtbls = cd->vtblInterfaces ? cd->vtblInterfaces->dim : 0;
  } else {
    // classes have monitor and fields
    if (!cd->isCPPclass() && !cd->isCPPinterface()) {
      // add monitor
      builder.addType(
          llvm::PointerType::get(llvm::Type::getInt8Ty(gIR->context()), 0),
          Target::ptrsize);
    }

    // add data members recursively
    t->addClassData(builder, cd);

    // add tail padding
    builder.addTailPadding(cd->structsize);
  }

  if (global.errors) {
    fatal();
  }

  // set struct body and copy GEP indices
  isaStruct(t->type)->setBody(builder.defaultTypes(), t->packed);
  t->varGEPIndices = builder.varGEPIndices();

  // set vtbl type body
  FuncDeclarations vtbl;
  vtbl.reserve(cd->vtbl.dim);
  if (!cd->isCPPclass())
    vtbl.push(nullptr);
  for (size_t i = cd->vtblOffset(); i < cd->vtbl.dim; ++i) {
    FuncDeclaration *fd = cd->vtbl[i]->isFuncDeclaration();
    assert(fd);
    vtbl.push(fd);
  }
  Type* first = cd->isCPPclass() ? nullptr : Type::typeinfoclass->type;
  t->vtbl_type->setBody(t->buildVtblType(first, &vtbl));

  IF_LOG Logger::cout() << "class type: " << *t->type << std::endl;

  return t;
}

std::vector<llvm::Type *>
IrTypeClass::buildVtblType(Type *first, FuncDeclarations *vtbl_array) {
  IF_LOG Logger::println("Building vtbl type for class %s",
                         cd->toPrettyChars());
  LOG_SCOPE;

  std::vector<llvm::Type *> types;
  types.reserve(vtbl_array->dim);

  auto I = vtbl_array->begin();
  // first comes the classinfo for D interfaces
  if (first) {
    types.push_back(DtoType(first));
    ++I;
  }

  // then come the functions
  for (auto E = vtbl_array->end(); I != E; ++I) {
    FuncDeclaration *fd = *I;
    if (fd == nullptr) {
      // FIXME: This stems from the ancient D1 days – can it still happen?
      types.push_back(getVoidPtrType());
      continue;
    }

    IF_LOG Logger::println("Adding type of %s", fd->toPrettyChars());

    // If inferring return type and semantic3 has not been run, do it now.
    // This pops up in some other places in the frontend as well, however
    // it is probably a bug that it still occurs that late.
    if (!fd->type->nextOf() && fd->inferRetType) {
      Logger::println("Running late functionSemantic to infer return type.");
      TemplateInstance *spec = fd->isSpeculative();
      unsigned int olderrs = global.errors;
      fd->functionSemantic();
      if (spec && global.errors != olderrs) {
        spec->errors = global.errors - olderrs;
      }
    }

    if (!fd->type->nextOf()) {
      // Return type of the function has not been inferred. This seems to
      // happen with virtual functions and is probably a frontend bug.
      IF_LOG Logger::println("Broken function type, semanticRun: %d",
                             fd->semanticRun);
      types.push_back(getVoidPtrType());
      continue;
    }

    types.push_back(getPtrToType(DtoFunctionType(fd)));
  }

  return types;
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
