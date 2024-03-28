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
#include "dmd/target.h"
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
#include "ir/irdsymbol.h"
#include "ir/irtype.h"
#include "ir/irvar.h"
#include "llvm/ADT/SmallString.h"

using namespace dmd;

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

    if (decl->type->ty == TY::Terror) {
      error(decl->loc, "%s `%s` had semantic errors when compiling",
            decl->kind(), decl->toPrettyChars());
      decl->ir->setDefined();
      return;
    }

    if (!(decl->members && decl->symtab)) {
      return;
    }

    DtoResolveClass(decl);
    decl->ir->setDefined();

    // Emit any members (e.g. final functions).
    for (auto m : *decl->members) {
      m->accept(this);
    }

    // Emit TypeInfo.
    IrClass *ir = getIrAggr(decl);
    if (!ir->suppressTypeInfo()) {
      ir->getClassInfoSymbol(/*define=*/true);
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

    if (decl->type->ty == TY::Terror) {
      error(decl->loc, "%s `%s` had semantic errors when compiling",
            decl->kind(), decl->toPrettyChars());
      decl->ir->setDefined();
      return;
    }

    if (!(decl->members && decl->symtab)) {
      // nothing to do for opaque structs anymore
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
      IrStruct *ir = getIrAggr(decl);

      // Define the __initZ symbol.
      if (!decl->zeroInit()) {
        ir->getInitSymbol(/*define=*/true);
      }

      // Emit special __xopEquals/__xopCmp/__xtoHash member functions required
      // for the TypeInfo.
      if (!ir->suppressTypeInfo()) {
        if (decl->xeq && decl->xeq != decl->xerreq) {
          decl->xeq->accept(this);
        }
        if (decl->xcmp && decl->xcmp != decl->xerrcmp) {
          decl->xcmp->accept(this);
        }
        if (decl->xhash) {
          decl->xhash->accept(this);
        }

        // the TypeInfo itself is emitted into each referencing CU
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

    if (decl->type->ty == TY::Terror) {
      error(decl->loc, "%s `%s` had semantic errors when compiling",
            decl->kind(), decl->toPrettyChars());
      decl->ir->setDefined();
      return;
    }

    if (!(decl->members && decl->symtab)) {
      return;
    }

    DtoResolveClass(decl);
    decl->ir->setDefined();

    for (auto m : *decl->members) {
      m->accept(this);
    }

    IrClass *ir = getIrAggr(decl);

    ir->getInitSymbol(/*define=*/true);

    ir->getVtblSymbol(/*define*/true);

    ir->defineInterfaceVtbls();

    // Emit TypeInfo.
    if (!ir->suppressTypeInfo()) {
      ir->getClassInfoSymbol(/*define=*/true);
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

    decl->foreachVar(this);
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(VarDeclaration *decl) override {
    IF_LOG Logger::println("VarDeclaration::codegen(): '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE;

    if (decl->ir->isDefined()) {
      return;
    }

    if (decl->type->ty == TY::Terror) {
      error(decl->loc, "%s `%s` had semantic errors when compiling",
            decl->kind(), decl->toPrettyChars());
      decl->ir->setDefined();
      return;
    }

    DtoResolveVariable(decl);
    decl->ir->setDefined();

    // just forward aliases
    if (decl->aliasTuple) {
      Logger::println("aliasTuple");
      decl->toAlias()->accept(this);
      return;
    }

    // global variable
    if (decl->isDataseg()) {
      Logger::println("data segment");

      assert(!(decl->storage_class & STCmanifest) &&
             "manifest constant being codegen'd!");
      assert(!irs->dcomputetarget);

      getIrGlobal(decl)->getValue(/*define=*/true);
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(EnumDeclaration *decl) override {
    IF_LOG Logger::println("Ignoring EnumDeclaration::codegen: '%s'",
                           decl->toPrettyChars());

    if (decl->type->ty == TY::Terror) {
      error(decl->loc, "%s `%s` had semantic errors when compiling",
            decl->kind(), decl->toPrettyChars());
      return;
    }
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(FuncDeclaration *decl) override {
    // don't touch function aliases, they don't contribute any new symbols
    if (!decl->skipCodegen() && !decl->isFuncAliasDeclaration() &&
        // skip fwd declarations (IR-declared lazily)
        decl->fbody) {
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

    // With -linkonce-templates-aggressive, only non-speculative instances make
    // it to module members (see `TemplateInstance.appendToModuleMember()`), and
    // we don't need full needsCodegen() culling in that case; isDiscardable()
    // is sufficient. Speculative ones are lazily emitted if actually referenced
    // during codegen - per IR module.
    if ((global.params.linkonceTemplates == LinkonceTemplates::aggressive &&
         decl->isDiscardable()) ||
        (global.params.linkonceTemplates != LinkonceTemplates::aggressive &&
         !decl->needsCodegen())) {
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

  static llvm::StringRef getPragmaStringArg(PragmaDeclaration *decl,
                                            d_size_t i = 0) {
    assert(decl->args && decl->args->length > i);
    auto se = (*decl->args)[i]->isStringExp();
    assert(se);
    DString str = se->peekString();
    return {str.ptr, str.length};
  }

  void visit(PragmaDeclaration *decl) override {
    const auto &triple = *global.params.targetTriple;

#if LDC_LLVM_VER >= 1800
    #define endswith ends_with
    #define startswith starts_with
#endif

    if (decl->ident == Id::lib) {
      assert(!irs->dcomputetarget);
      llvm::StringRef name = getPragmaStringArg(decl);

      if (triple.isWindowsGNUEnvironment()) {
        if (name.endswith(".lib")) {
          // On MinGW, strip the .lib suffix, if any, to improve compatibility
          // with code written for DMD (we pass the name to GCC via -l, just as
          // on Posix).
          name = name.drop_back(4);
        }

        if (name.startswith("shell32")) {
          // Another DMD compatibility kludge: Ignore
          // pragma(lib, "shell32.lib"), it is implicitly provided by
          // MinGW.
          return;
        }
      }

      if (triple.isWindowsMSVCEnvironment()) {
        if (name.endswith(".a")) {
          name = name.drop_back(2);
        }
        if (name.endswith(".lib")) {
          name = name.drop_back(4);
        }

        // embed linker directive in COFF object file; don't push to
        // global.params.linkswitches
        std::string arg = ("/DEFAULTLIB:\"" + name + "\"").str();
        gIR->addLinkerOption(llvm::StringRef(arg));
      } else {
        const bool isStaticLib = name.endswith(".a");
        const size_t nameLen = name.size();

        char *arg = nullptr;
        if (!isStaticLib) { // name => -lname
          const size_t n = nameLen + 3;
          arg = static_cast<char *>(mem.xmalloc(n));
          arg[0] = '-';
          arg[1] = 'l';
          memcpy(arg + 2, name.data(), nameLen);
          arg[n - 1] = 0;
        } else {
          arg = static_cast<char *>((mem.xmalloc(nameLen + 1)));
          memcpy(arg, name.data(), nameLen);
          arg[nameLen] = 0;
        }

        global.params.linkswitches.push(arg);

        if (triple.isOSBinFormatMachO()) {
          // embed linker directive in Mach-O object file too
          gIR->addLinkerOption(llvm::StringRef(arg));
        } else if (triple.isOSBinFormatELF()) {
          // embed library name as dependent library in ELF object file too
          // (supported by LLD v9+)
          gIR->addLinkerDependentLib(name);
        }
      }
    } else if (decl->ident == Id::linkerDirective) {
      // embed in object file (if supported)
      if (target.supportsLinkerDirective()) {
        assert(decl->args);
        llvm::SmallVector<llvm::StringRef, 2> args;
        args.reserve(decl->args->length);
        for (d_size_t i = 0; i < decl->args->length; ++i)
          args.push_back(getPragmaStringArg(decl, i));
        gIR->addLinkerOption(args);
      }
    }
    visit(static_cast<AttribDeclaration *>(decl));

#if LDC_LLVM_VER >= 1800
    #undef endswith
    #undef startswith
#endif
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(TypeInfoDeclaration *decl) override {
    llvm_unreachable("Should be emitted from codegen layer only");
  }

  //////////////////////////////////////////////////////////////////////////

  void visit(CAsmDeclaration *ad) override {
    auto se = ad->code->isStringExp();
    assert(se);

    DString str = se->peekString();
    if (str.length)
      irs->module.appendModuleInlineAsm({str.ptr, str.length});
  }
};

//////////////////////////////////////////////////////////////////////////////

void Declaration_codegen(Dsymbol *decl) { Declaration_codegen(decl, gIR); }

void Declaration_codegen(Dsymbol *decl, IRState *irs) {
  CodegenVisitor v(irs);
  decl->accept(&v);
}

//////////////////////////////////////////////////////////////////////////////
