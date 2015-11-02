//===-- declarations.cpp --------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "aggregate.h"
#include "declaration.h"
#include "enum.h"
#include "id.h"
#include "init.h"
#include "rmem.h"
#include "template.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "ir/irtype.h"
#include "ir/irvar.h"
#include "llvm/ADT/SmallString.h"

//////////////////////////////////////////////////////////////////////////////
// FIXME: Integrate these functions
void TypeInfoDeclaration_codegen(TypeInfoDeclaration *decl, IRState *p);
void TypeInfoClassDeclaration_codegen(TypeInfoDeclaration *decl, IRState *p);

//////////////////////////////////////////////////////////////////////////////

namespace {
// from dmd/src/typinf.c
bool isSpeculativeType(Type *t) {
  class SpeculativeTypeVisitor : public Visitor {
  public:
    bool result;

    SpeculativeTypeVisitor() : result(false) {}

    using Visitor::visit;
    void visit(Type *t) override {
      Type *tb = t->toBasetype();
      if (tb != t) {
        tb->accept(this);
      }
    }
    void visit(TypeNext *t) override {
      if (t->next) {
        t->next->accept(this);
      }
    }
    void visit(TypeBasic *t) override {}
    void visit(TypeVector *t) override { t->basetype->accept(this); }
    void visit(TypeAArray *t) override {
      t->index->accept(this);
      visit((TypeNext *)t);
    }
    void visit(TypeFunction *t) override {
      visit((TypeNext *)t);
      // Currently TypeInfo_Function doesn't store parameter types.
    }
    void visit(TypeStruct *t) override {
      StructDeclaration *sd = t->sym;
      if (TemplateInstance *ti = sd->isInstantiated()) {
        if (!ti->needsCodegen()) {
          if (ti->minst || sd->requestTypeInfo) {
            return;
          }

          /* Bugzilla 14425: TypeInfo_Struct would refer the members of
           * struct (e.g. opEquals via xopEquals field), so if it's instantiated
           * in speculative context, TypeInfo creation should also be
           * stopped to avoid 'unresolved symbol' linker errors.
           */
          /* When -debug/-unittest is specified, all of non-root instances are
           * automatically changed to speculative, and here is always reached
           * from those instantiated non-root structs.
           * Therefore, if the TypeInfo is not auctually requested,
           * we have to elide its codegen.
           */
          result |= true;
          return;
        }
      } else {
        // assert(!sd->inNonRoot() || sd->requestTypeInfo);  // valid?
      }
    }
    void visit(TypeClass *t) override {}
    void visit(TypeTuple *t) override {
      if (t->arguments) {
        for (size_t i = 0; i < t->arguments->dim; i++) {
          Type *tprm = (*t->arguments)[i]->type;
          if (tprm) {
            tprm->accept(this);
          }
          if (result) {
            return;
          }
        }
      }
    }
  };
  SpeculativeTypeVisitor v;
  t->accept(&v);
  return v.result;
}
}

//////////////////////////////////////////////////////////////////////////////

class CodegenVisitor : public Visitor {
  IRState *irs;

public:
  explicit CodegenVisitor(IRState *irs) : irs(irs) {}

  //////////////////////////////////////////////////////////////////////////

  // Import all functions from class Visitor
  using Visitor::visit;

  //////////////////////////////////////////////////////////////////////////

  void visit(Dsymbol *sym) LLVM_OVERRIDE {
    IF_LOG Logger::println("Ignoring Dsymbol::codegen for %s",
                           sym->toPrettyChars());
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(InterfaceDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("InterfaceDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir.isDefined()) {
      return;
    }

    if (decl->type->ty == Terror) {
      error(decl->loc, "had semantic errors when compiling");
      decl->ir.setDefined();
      return;
    }

    if (decl->members && decl->symtab) {
      DtoResolveClass(decl);
      decl->ir.setDefined();

      // Emit any members (e.g. final functions).
      for (auto m : *decl->members) {
        m->accept(this);
      }

      // Emit TypeInfo.
      DtoTypeInfoOf(decl->type);

      // Define __InterfaceZ.
      IrAggr *ir = getIrAggr(decl);
      llvm::GlobalVariable *interfaceZ = ir->getClassInfoSymbol();
      interfaceZ->setInitializer(ir->getClassInfoInit());
      LinkageWithCOMDAT lwc = DtoLinkage(decl);
      interfaceZ->setLinkage(lwc.first);
      if (lwc.second) {
        SET_COMDAT(interfaceZ, gIR->module);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(StructDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("StructDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir.isDefined()) {
      return;
    }

    if (decl->type->ty == Terror) {
      error(decl->loc, "had semantic errors when compiling");
      decl->ir.setDefined();
      return;
    }

    if (decl->members && decl->symtab) {
      DtoResolveStruct(decl);
      decl->ir.setDefined();

      for (auto m : *decl->members) {
        m->accept(this);
      }

      // Define the __initZ symbol.
      IrAggr *ir = getIrAggr(decl);
      llvm::GlobalVariable *initZ = ir->getInitSymbol();
      initZ->setInitializer(ir->getDefaultInit());
      LinkageWithCOMDAT lwc = DtoLinkage(decl);
      initZ->setLinkage(lwc.first);
      if (lwc.second) {
        SET_COMDAT(initZ, gIR->module);
      }

      // emit typeinfo
      DtoTypeInfoOf(decl->type);

      // Emit __xopEquals/__xopCmp/__xtoHash.
      if (decl->xeq && decl->xeq != decl->xerreq) {
        decl->xeq->accept(this);
      }
      if (decl->xcmp && decl->xcmp != decl->xerrcmp) {
        decl->xcmp->accept(this);
      }
      if (decl->xhash) {
        decl->xhash->accept(this);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ClassDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("ClassDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir.isDefined()) {
      return;
    }

    if (decl->type->ty == Terror) {
      error(decl->loc, "had semantic errors when compiling");
      decl->ir.setDefined();
      return;
    }

    if (decl->members && decl->symtab) {
      DtoResolveClass(decl);
      decl->ir.setDefined();

      for (auto m : *decl->members) {
        m->accept(this);
      }

      IrAggr *ir = getIrAggr(decl);
      const LinkageWithCOMDAT lwc = DtoLinkage(decl);

      llvm::GlobalVariable *initZ = ir->getInitSymbol();
      initZ->setInitializer(ir->getDefaultInit());
      initZ->setLinkage(lwc.first);
      if (lwc.second) {
        SET_COMDAT(initZ, gIR->module);
      }

      llvm::GlobalVariable *vtbl = ir->getVtblSymbol();
      vtbl->setInitializer(ir->getVtblInit());
      vtbl->setLinkage(lwc.first);
      if (lwc.second) {
        SET_COMDAT(vtbl, gIR->module);
      }

      llvm::GlobalVariable *classZ = ir->getClassInfoSymbol();
      classZ->setInitializer(ir->getClassInfoInit());
      classZ->setLinkage(lwc.first);
      if (lwc.second) {
        SET_COMDAT(classZ, gIR->module);
      }

      // No need to do TypeInfo here, it is <name>__classZ for classes in D2.
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TupleDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("TupleDeclaration::codegen(): '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir.isDefined()) {
      return;
    }
    decl->ir.setDefined();

    assert(decl->isexp);
    assert(decl->objects);

    for (auto o : *decl->objects) {
      DsymbolExp *exp = static_cast<DsymbolExp *>(o);
      assert(exp->op == TOKdsymbol);
      exp->s->accept(this);
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(VarDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("VarDeclaration::codegen(): '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE;

    if (decl->ir.isDefined()) {
      return;
    }

    if (decl->type->ty == Terror) {
      error(decl->loc, "had semantic errors when compiling");
      decl->ir.setDefined();
      return;
    }

    DtoResolveVariable(decl);
    decl->ir.setDefined();

    // just forward aliases
    if (decl->aliassym) {
      Logger::println("alias sym");
      decl->toAlias()->accept(this);
      return;
    }

    // global variable
    if (decl->isDataseg()) {
      Logger::println("data segment");

      assert(!(decl->storage_class & STCmanifest) &&
             "manifest constant being codegen'd!");

      IrGlobal *irGlobal = getIrGlobal(decl);
      llvm::GlobalVariable *gvar =
          llvm::cast<llvm::GlobalVariable>(irGlobal->value);
      assert(gvar && "DtoResolveVariable should have created value");

      const LinkageWithCOMDAT lwc = DtoLinkage(decl);

      // Check if we are defining or just declaring the global in this module.
      if (!(decl->storage_class & STCextern)) {
        // Build the initializer. Might use this->ir.irGlobal->value!
        LLConstant *initVal =
            DtoConstInitializer(decl->loc, decl->type, decl->init);

        // In case of type mismatch, swap out the variable.
        if (initVal->getType() != gvar->getType()->getElementType()) {
          llvm::GlobalVariable *newGvar = getOrCreateGlobal(
              decl->loc, irs->module, initVal->getType(), gvar->isConstant(),
              lwc.first, nullptr,
              "", // We take on the name of the old global below.
              gvar->isThreadLocal());
          if (lwc.second) {
            SET_COMDAT(newGvar, gIR->module);
          }

          newGvar->setAlignment(gvar->getAlignment());
          newGvar->takeName(gvar);

          llvm::Constant *newValue =
              llvm::ConstantExpr::getBitCast(newGvar, gvar->getType());
          gvar->replaceAllUsesWith(newValue);

          gvar->eraseFromParent();
          gvar = newGvar;
          irGlobal->value = newGvar;
        }

        // Now, set the initializer.
        assert(!irGlobal->constInit);
        irGlobal->constInit = initVal;
        gvar->setInitializer(initVal);
        gvar->setLinkage(lwc.first);
        if (lwc.second) {
          SET_COMDAT(gvar, gIR->module);
        }

        // Also set up the debug info.
        irs->DBuilder.EmitGlobalVariable(gvar, decl);
      }

      // If this global is used from a naked function, we need to create an
      // artificial "use" for it, or it could be removed by the optimizer if
      // the only reference to it is in inline asm.
      if (irGlobal->nakedUse) {
        irs->usedArray.push_back(DtoBitCast(gvar, getVoidPtrType()));
      }

      IF_LOG Logger::cout() << *gvar << '\n';
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(EnumDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("Ignoring EnumDeclaration::codegen: '%s'",
                           decl->toPrettyChars());

    if (decl->type->ty == Terror) {
      error(decl->loc, "had semantic errors when compiling");
      return;
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(FuncDeclaration *decl) LLVM_OVERRIDE {
    DtoDefineFunction(decl);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TemplateInstance *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("TemplateInstance::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir.isDefined()) {
      return;
    }
    decl->ir.setDefined();

    // FIXME: This is #673 all over again.
    if (!decl->needsCodegen()) {
      return;
    }

    if (!isError(decl) && decl->members) {
      for (auto m : *decl->members) {
        m->accept(this);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TemplateMixin *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("TemplateInstance::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir.isDefined()) {
      return;
    }
    decl->ir.setDefined();

    if (!isError(decl) && decl->members) {
      for (auto m : *decl->members) {
        m->accept(this);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(AttribDeclaration *decl) LLVM_OVERRIDE {
    Dsymbols *d = decl->include(nullptr, nullptr);

    if (d) {
      for (auto s : *d) {
        s->accept(this);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(PragmaDeclaration *decl) LLVM_OVERRIDE {
    if (decl->ident == Id::lib) {
      assert(decl->args && decl->args->dim == 1);

      Expression *e = static_cast<Expression *>(decl->args->data[0]);

      assert(e->op == TOKstring);
      StringExp *se = static_cast<StringExp *>(e);

      size_t nameLen = se->len;
      if (global.params.targetTriple.isWindowsGNUEnvironment()) {
        if (nameLen > 4 &&
            !memcmp(static_cast<char *>(se->string) + nameLen - 4, ".lib", 4)) {
          // On MinGW, strip the .lib suffix, if any, to improve
          // compatibility with code written for DMD (we pass the name to GCC
          // via -l, just as on Posix).
          nameLen -= 4;
        }

        if (nameLen >= 7 && !memcmp(se->string, "shell32", 7)) {
          // Another DMD compatibility kludge: Ignore
          // pragma(lib, "shell32.lib"), it is implicitly provided by
          // MinGW.
          return;
        }
      }

      // With LLVM 3.3 or later we can place the library name in the object
      // file. This seems to be supported only on Windows.
      if (global.params.targetTriple.isWindowsMSVCEnvironment()) {
        llvm::SmallString<24> LibName(
            llvm::StringRef(static_cast<const char *>(se->string), nameLen));

        // Win32: /DEFAULTLIB:"curl"
        if (LibName.endswith(".a")) {
          LibName = LibName.substr(0, LibName.size() - 2);
        }
        if (LibName.endswith(".lib")) {
          LibName = LibName.substr(0, LibName.size() - 4);
        }
        llvm::SmallString<24> tmp("/DEFAULTLIB:\"");
        tmp.append(LibName);
        tmp.append("\"");
        LibName = tmp;

// Embedd library name as linker option in object file
#if LDC_LLVM_VER >= 306
        llvm::Metadata *Value = llvm::MDString::get(gIR->context(), LibName);
        gIR->LinkerMetadataArgs.push_back(
            llvm::MDNode::get(gIR->context(), Value));
#else
        llvm::Value *Value = llvm::MDString::get(gIR->context(), LibName);
        gIR->LinkerMetadataArgs.push_back(
            llvm::MDNode::get(gIR->context(), Value));
#endif
      } else {
        size_t const n = nameLen + 3;
        char *arg = static_cast<char *>(mem.xmalloc(n));
        arg[0] = '-';
        arg[1] = 'l';
        memcpy(arg + 2, se->string, nameLen);
        arg[n - 1] = 0;
        global.params.linkswitches->push(arg);
      }
    }
    visit(static_cast<AttribDeclaration *>(decl));
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TypeInfoDeclaration *decl) LLVM_OVERRIDE {
    if (isSpeculativeType(decl->tinfo)) {
      return;
    }
    TypeInfoDeclaration_codegen(decl, irs);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TypeInfoClassDeclaration *decl) LLVM_OVERRIDE {
    if (isSpeculativeType(decl->tinfo)) {
      return;
    }
    TypeInfoClassDeclaration_codegen(decl, irs);
  }
};

//////////////////////////////////////////////////////////////////////////////

void Declaration_codegen(Dsymbol *decl) {
  CodegenVisitor v(gIR);
  decl->accept(&v);
}

void Declaration_codegen(Dsymbol *decl, IRState *irs) {
  CodegenVisitor v(irs);
  decl->accept(&v);
}

//////////////////////////////////////////////////////////////////////////////
