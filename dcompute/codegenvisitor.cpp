//===-- dcompute/codegenvisitor.cpp ---------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//


#include "codegenvisitor.h"

#include "aggregate.h"
#include "declaration.h"
#include "enum.h"
#include "id.h"
#include "init.h"
#include "nspace.h"
#include "rmem.h"
#include "template.h"
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
#include "module.h"
#include "identifier.h"
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

class DComputeCodegenVisitor : public Visitor {
    IRState *irs;
    DComputeTarget &dct;
    
public:
    explicit DComputeCodegenVisitor(IRState *irs,DComputeTarget &_dct) : irs(irs),dct(_dct)
    {
    }
    


void visit(Dsymbol *sym)  {
    IF_LOG Logger::println("Ignoring @compute Dsymbol::codegen for %s",
                           sym->toPrettyChars());
}

//////////////////////////////////////////////////////////////////////////

void visit(Nspace *ns) LLVM_OVERRIDE {
    IF_LOG Logger::println("@compute Nspace::codegen for %s", ns->toPrettyChars());
    LOG_SCOPE
    
    if (!isError(ns) && ns->members) {
        for (auto sym : *ns->members)
            sym->accept(this);
    }
}

void visit(InterfaceDeclaration *decl) LLVM_OVERRIDE {
    decl->error("Interfaces not allowed in @compute code");
}

void visit(StructDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("@compute StructDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE
    
    if (decl->ir->isDefined()) {
        return;
    }
    
    if (decl->type->ty == Terror) {
        error(decl->loc, "had semantic errors when compiling");
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
    //TODO: Reqire a zeroinitialser and dont emit an __initZ symbol
    // Define the __initZ symbol.
    /*IrAggr *ir = getIrAggr(decl);
    llvm::GlobalVariable *initZ = ir->getInitSymbol();
    initZ->setInitializer(ir->getDefaultInit());
    setLinkage(decl, initZ);
    */
    // emit typeinfo
    //DtoTypeInfoOf(decl->type);
    
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

void visit(ClassDeclaration *decl) LLVM_OVERRIDE {
    decl->error("Classes not allowed in @compute code");
}

void visit(TupleDeclaration *decl) LLVM_OVERRIDE {
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

void visit(VarDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("VarDeclaration::codegen(): '%s'",
                           decl->toPrettyChars());
    LOG_SCOPE;
    
    if (decl->ir->isDefined()) {
        return;
    }
    
    if (decl->type->ty == Terror) {
        error(decl->loc, "had semantic errors when compiling");
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
        decl->error("global variables currently not allowed in @compute code");
    }
}


void visit(EnumDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("Ignoring EnumDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
    
    if (decl->type->ty == Terror) {
        error(decl->loc, "had semantic errors when compiling");
        return;
    }
}

void visit(FuncDeclaration *decl) LLVM_OVERRIDE {
    // don't touch function aliases, they don't contribute any new symbols
    if (!decl->isFuncAliasDeclaration()) {
        //Part I of a hack to replace things in dcompute.types with their correct types
        //we emit their definition to the module directly
        
        //Part II is in statementvisitor.cpp where the returned value is altered to be the correct type

        DtoDefineFunction(decl);
        if (hasKernelAttr(decl)) {
            auto fn = irs->module.getFunction(decl->mangleString);
            IF_LOG Logger::println("Fn = %p",fn);
            dct.handleKernelFunc(decl,fn);
        }
        else
            dct.handleNonKernelFunc(decl,irs->module.getFunction(decl->mangleString));
    }
}

void visit(TemplateInstance *decl) LLVM_OVERRIDE {
    //TODO: resolve struct Pointer(uint n, T) instances to addrspace(n) T*
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
    if (!strcmp(decl->getModule()->ident->string,"object")) {
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
    }
    
    for (auto &m : *decl->members) {
        m->accept(this);
    }
}

void visit(TemplateMixin *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("TemplateMixin::codegen: '%s'",
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

void visit(AttribDeclaration *decl) LLVM_OVERRIDE {

    Dsymbols *d = decl->include(nullptr, nullptr);
    
    if (d) {
        for (auto s : *d) {
            s->accept(this);
        }
    }
}

void visit(PragmaDeclaration *decl) LLVM_OVERRIDE {
    if (decl->ident == Id::lib)
        decl->error("pragma(lib, \"...\" not allowed in @compute code");
    visit(static_cast<AttribDeclaration *>(decl));
}

void visit(TypeInfoDeclaration *decl) LLVM_OVERRIDE {
    if (isSpeculativeType(decl->tinfo)) {
        return;
    }
    IF_LOG Logger::println("@compute ignoring TypeInfoDeclaration::codegen: '%s'",
                           decl->toPrettyChars());
}

void visit(TypeInfoClassDeclaration *decl) LLVM_OVERRIDE {
    IF_LOG Logger::println("@compute ignoring TypeInfoDeclaration::codegen: '%s' should have already errored",
                           decl->toPrettyChars());
}
};
void DcomputeDeclaration_codegen(Dsymbol *decl, IRState *irs, DComputeTarget &dct)
{
    DComputeCodegenVisitor v(irs,dct);
    decl->accept(&v);
}


