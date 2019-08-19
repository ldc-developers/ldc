//===-- declarations.cpp --------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/aggregate.h"
#include "dmd/declaration.h"
#include "dmd/enum.h"
#include "dmd/errors.h"
#include "dmd/expression.h"
#include "dmd/id.h"
#include "dmd/import.h"
#include "dmd/init.h"
#include "dmd/nspace.h"
#include "dmd/root/rmem.h"
#include "dmd/template.h"
#include "driver/cl_options.h"
#include "gen/classes.h"
#include "gen/functions.h"
#include "gen/irstate.h"
#include "gen/llvm.h"
#include "gen/llvmhelpers.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/typinf.h"
#include "gen/uda.h"
#include "ir/irtype.h"
#include "ir/irvar.h"
#include "llvm/ADT/SmallString.h"

//////////////////////////////////////////////////////////////////////////////

class CodegenVisitor : public Visitor {
  IRState *irs;

public:
  explicit CodegenVisitor(IRState *irs) : irs(irs) {}

  //////////////////////////////////////////////////////////////////////////

  // Import all functions from class Visitor
  using Visitor::visit;

  //////////////////////////////////////////////////////////////////////////

  void visit(Dsymbol *sym) override {
    IF_LOG Logger::println("Ignoring Dsymbol::codegen for %s",
                           sym->toPrettyChars());
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(Import *im) override {
    IF_LOG Logger::println("Import::codegen for %s", im->toPrettyChars());
    LOG_SCOPE

    irs->DBuilder.EmitImport(im);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(Nspace *ns) override {
    IF_LOG Logger::println("Nspace::codegen for %s", ns->toPrettyChars());
    LOG_SCOPE

    if (!isError(ns) && ns->members) {
      for (auto sym : *ns->members)
        sym->accept(this);
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(InterfaceDeclaration *decl) override {
    IF_LOG Logger::println("InterfaceDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    assert(!irs->dcomputetarget);

    if (decl->ir->isDefined()) {
      return;
    }

    if (decl->type->ty == Terror) {
      decl->error("had semantic errors when compiling");
      decl->ir->setDefined();
      return;
    }

    if (decl->members && decl->symtab) {
      DtoResolveClass(decl);
      decl->ir->setDefined();

      // Emit any members (e.g. final functions).
      for (auto m : *decl->members) {
        m->accept(this);
      }

      // Emit TypeInfo.
      IrAggr *ir = getIrAggr(decl);
      if (!ir->suppressTypeInfo() && !isSpeculativeType(decl->type)) {
        llvm::GlobalVariable *interfaceZ = ir->getClassInfoSymbol();
        defineGlobal(interfaceZ, ir->getClassInfoInit(), decl);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(StructDeclaration *decl) override {
    IF_LOG Logger::println("StructDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir->isDefined()) {
      return;
    }

    if (decl->type->ty == Terror) {
      decl->error("had semantic errors when compiling");
      decl->ir->setDefined();
      return;
    }

    if (!(decl->members && decl->symtab)) {
      return;
    }

    DtoResolveStruct(decl);
    decl->ir->setDefined();

    for (auto m : *decl->members) {
      m->accept(this);
    }

    // Skip __initZ and typeinfo for @compute device code.
    // TODO: support global variables and thus __initZ
    if (!irs->dcomputetarget) {
      // Define the __initZ symbol.
      IrAggr *ir = getIrAggr(decl);
      auto &initZ = ir->getInitSymbol();
      auto initGlobal = llvm::cast<LLGlobalVariable>(initZ);
      initZ = irs->setGlobalVarInitializer(initGlobal, ir->getDefaultInit());
      setLinkageAndVisibility(decl, initGlobal);

      // emit typeinfo
      if (!ir->suppressTypeInfo()) {
        DtoTypeInfoOf(decl->type, /*base=*/false);

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
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(ClassDeclaration *decl) override {
    IF_LOG Logger::println("ClassDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    assert(!irs->dcomputetarget);

    if (decl->ir->isDefined()) {
      return;
    }

    if (decl->type->ty == Terror) {
      decl->error("had semantic errors when compiling");
      decl->ir->setDefined();
      return;
    }

    if (decl->members && decl->symtab) {
      DtoResolveClass(decl);
      decl->ir->setDefined();

      for (auto m : *decl->members) {
        m->accept(this);
      }

      IrAggr *ir = getIrAggr(decl);

      auto &initZ = ir->getInitSymbol();
      auto initGlobal = llvm::cast<LLGlobalVariable>(initZ);
      initZ = irs->setGlobalVarInitializer(initGlobal, ir->getDefaultInit());
      setLinkageAndVisibility(decl, initGlobal);

      llvm::GlobalVariable *vtbl = ir->getVtblSymbol();
      defineGlobal(vtbl, ir->getVtblInit(), decl);

      ir->defineInterfaceVtbls();

      // Emit TypeInfo.
      if (!ir->suppressTypeInfo() && !isSpeculativeType(decl->type)) {
        llvm::GlobalVariable *classZ = ir->getClassInfoSymbol();
        defineGlobal(classZ, ir->getClassInfoInit(), decl);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TupleDeclaration *decl) override {
    IF_LOG Logger::println("TupleDeclaration::codegen(): '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir->isDefined()) {
      return;
    }
    decl->ir->setDefined();

    assert(decl->isexp);
    assert(decl->objects);

    for (auto o : *decl->objects) {
      DsymbolExp *exp = static_cast<DsymbolExp *>(o);
      assert(exp->op == TOKdsymbol);
      exp->s->accept(this);
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(VarDeclaration *decl) override {
    IF_LOG Logger::println("VarDeclaration::codegen(): '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE;

    if (decl->ir->isDefined()) {
      return;
    }

    if (decl->type->ty == Terror) {
      decl->error("had semantic errors when compiling");
      decl->ir->setDefined();
      return;
    }

    DtoResolveVariable(decl);
    decl->ir->setDefined();

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
      assert(!irs->dcomputetarget);

      IrGlobal *irGlobal = getIrGlobal(decl);
      LLGlobalVariable *gvar = llvm::cast<LLGlobalVariable>(irGlobal->value);
      assert(gvar && "DtoResolveVariable should have created value");

      if (global.params.vtls && gvar->isThreadLocal() &&
          !(decl->storage_class & STCtemp)) {
        const char *p = decl->loc.toChars();
        message("%s: `%s` is thread local", p, decl->toChars());
      }

      // Check if we are defining or just declaring the global in this module.
      // If we reach here during codegen of an available_externally function,
      // new variable declarations should stay external and therefore must not
      // have an initializer.
      if (!(decl->storage_class & STCextern) && !decl->inNonRoot()) {
        // Build the initializer. Might use irGlobal->value!
        LLConstant *initVal =
            DtoConstInitializer(decl->loc, decl->type, decl->_init);

        // Cache it.
        assert(!irGlobal->constInit);
        irGlobal->constInit = initVal;

        // Set the initializer, swapping out the variable if the types do not
        // match.
        irGlobal->value = irs->setGlobalVarInitializer(gvar, initVal);

        // Finalize linkage & DLL storage class.
        const auto lwc = DtoLinkage(decl);
        setLinkage(lwc, gvar);
        if (gvar->hasDLLImportStorageClass()) {
          gvar->setDLLStorageClass(LLGlobalValue::DLLExportStorageClass);
        }

        // Hide non-exported symbols
        if (opts::defaultToHiddenVisibility && !decl->isExport()) {
          gvar->setVisibility(LLGlobalValue::HiddenVisibility);
        }

        // Also set up the debug info.
        irs->DBuilder.EmitGlobalVariable(gvar, decl);
      }

      // If this global is used from a naked function, we need to create an
      // artificial "use" for it, or it could be removed by the optimizer if
      // the only reference to it is in inline asm.
      if (irGlobal->nakedUse) {
        irs->usedArray.push_back(gvar);
      }

      IF_LOG Logger::cout() << *gvar << '\n';
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(EnumDeclaration *decl) override {
    IF_LOG Logger::println("Ignoring EnumDeclaration::codegen: '%s'",
                           decl->toPrettyChars());

    if (decl->type->ty == Terror) {
      decl->error("had semantic errors when compiling");
      return;
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(FuncDeclaration *decl) override {
    // don't touch function aliases, they don't contribute any new symbols
    if (!decl->isFuncAliasDeclaration()) {
      DtoDefineFunction(decl);
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TemplateInstance *decl) override {
    IF_LOG Logger::println("TemplateInstance::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir->isDefined()) {
      Logger::println("Already defined, skipping.");
      return;
    }
    decl->ir->setDefined();

    if (isError(decl)) {
      Logger::println("Has errors, skipping.");
      return;
    }

    if (!decl->members) {
      Logger::println("Has no members, skipping.");
      return;
    }

    // Force codegen if this is a templated function with pragma(inline, true).
    if ((decl->members->dim == 1) &&
        ((*decl->members)[0]->isFuncDeclaration()) &&
        ((*decl->members)[0]->isFuncDeclaration()->inlining == PINLINEalways)) {
      Logger::println("needsCodegen() == false, but function is marked with "
                      "pragma(inline, true), so it really does need "
                      "codegen.");
    } else {
      // FIXME: This is #673 all over again.
      if (!decl->needsCodegen()) {
        Logger::println("Does not need codegen, skipping.");
        return;
      }

      if (irs->dcomputetarget && (decl->tempdecl == Type::rtinfo ||
                                  decl->tempdecl == Type::rtinfoImpl)) {
        // Emitting object.RTInfo(Impl) template instantiations in dcompute
        // modules would require dcompute support for global variables.
        Logger::println("Skipping object.RTInfo(Impl) template instantiations "
                        "in dcompute modules.");
        return;
      }
    }

    for (auto &m : *decl->members) {
      m->accept(this);
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TemplateMixin *decl) override {
    IF_LOG Logger::println("TemplateInstance::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE

    if (decl->ir->isDefined()) {
      return;
    }
    decl->ir->setDefined();

    if (!isError(decl) && decl->members) {
      for (auto m : *decl->members) {
        m->accept(this);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(AttribDeclaration *decl) override {
    Dsymbols *d = decl->include(nullptr);

    if (d) {
      for (auto s : *d) {
        s->accept(this);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////

  static std::string getPragmaStringArg(PragmaDeclaration *decl) {
    assert(decl->args && decl->args->dim == 1);
    Expression *e = static_cast<Expression *>((*decl->args)[0]);
    assert(e->op == TOKstring);
    StringExp *se = static_cast<StringExp *>(e);
    return std::string(se->toPtr(), se->numberOfCodeUnits());
  }

  void visit(PragmaDeclaration *decl) override {
    if (decl->ident == Id::lib) {
      assert(!irs->dcomputetarget);
      const std::string name = getPragmaStringArg(decl);
      auto nameLen = name.size();

      if (global.params.targetTriple->isWindowsGNUEnvironment()) {
        if (nameLen > 4 && !memcmp(&name[nameLen - 4], ".lib", 4)) {
          // On MinGW, strip the .lib suffix, if any, to improve
          // compatibility with code written for DMD (we pass the name to GCC
          // via -l, just as on Posix).
          nameLen -= 4;
        }

        if (nameLen >= 7 && !memcmp(name.data(), "shell32", 7)) {
          // Another DMD compatibility kludge: Ignore
          // pragma(lib, "shell32.lib"), it is implicitly provided by
          // MinGW.
          return;
        }
      }

      // With LLVM 3.3 or later we can place the library name in the object
      // file. This seems to be supported only on Windows.
      if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
        llvm::SmallString<24> LibName(name);

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

        // Embed library name as linker option in object file
        auto Value = llvm::MDString::get(gIR->context(), LibName);
        gIR->LinkerMetadataArgs.push_back(
            llvm::MDNode::get(gIR->context(), Value));
      } else {
        size_t const n = nameLen + 3;
        char *arg = static_cast<char *>(mem.xmalloc(n));
        arg[0] = '-';
        arg[1] = 'l';
        memcpy(arg + 2, name.data(), nameLen);
        arg[n - 1] = 0;
        global.params.linkswitches.push(arg);
      }
    } else if (decl->ident == Id::linkerDirective) {
      if (global.params.targetTriple->isWindowsMSVCEnvironment()) {
        // Embed directly as linker option in object file
        const std::string directive = getPragmaStringArg(decl);
        auto Value = llvm::MDString::get(gIR->context(), directive);
        gIR->LinkerMetadataArgs.push_back(
            llvm::MDNode::get(gIR->context(), Value));
      }
    }
    visit(static_cast<AttribDeclaration *>(decl));
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TypeInfoDeclaration *decl) override {
    if (!irs->dcomputetarget)
      TypeInfoDeclaration_codegen(decl, irs);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TypeInfoClassDeclaration *decl) override {
    if (!irs->dcomputetarget)
      TypeInfoClassDeclaration_codegen(decl, irs);
  }
};

//////////////////////////////////////////////////////////////////////////////

void Declaration_codegen(Dsymbol *decl) {
  Declaration_codegen(decl, gIR);
}

void Declaration_codegen(Dsymbol *decl, IRState *irs) {
  CodegenVisitor v(irs);
  decl->accept(&v);
}

//////////////////////////////////////////////////////////////////////////////
