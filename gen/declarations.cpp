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
void TypeInfoDeclaration_codegen(TypeInfoDeclaration *decl, IRState* p);
void TypeInfoClassDeclaration_codegen(TypeInfoDeclaration *decl, IRState* p);

//////////////////////////////////////////////////////////////////////////////

class CodegenVisitor : public Visitor {
    IRState *irs;
public:

    CodegenVisitor(IRState *irs) : irs(irs) { }

    //////////////////////////////////////////////////////////////////////////

    // Import all functions from class Visitor
    using Visitor::visit;

    //////////////////////////////////////////////////////////////////////////

    void visit(Dsymbol *sym) LLVM_OVERRIDE {
        IF_LOG Logger::println("Ignoring Dsymbol::codegen for %s", sym->toPrettyChars());
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(InterfaceDeclaration *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("InterfaceDeclaration::codegen: '%s'", decl->toPrettyChars());
        LOG_SCOPE

        if (decl->ir.defined) return;
        decl->ir.defined = true;

        if (decl->type->ty == Terror)
        {   error(decl->loc, "had semantic errors when compiling");
            return;
        }

        if (decl->members && decl->symtab)
        {
            DtoResolveClass(decl);

            // Emit any members (e.g. final functions).
            for (Dsymbols::iterator I = decl->members->begin(),
                                    E = decl->members->end();
                                    I != E; ++I)
            {
                (*I)->accept(this);
            }

            // Emit TypeInfo.
            DtoTypeInfoOf(decl->type);

            // Define __InterfaceZ.
            llvm::GlobalVariable *interfaceZ = decl->ir.irAggr->getClassInfoSymbol();
            interfaceZ->setInitializer(decl->ir.irAggr->getClassInfoInit());
            interfaceZ->setLinkage(DtoExternalLinkage(decl));
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(StructDeclaration *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("StructDeclaration::codegen: '%s'", decl->toPrettyChars());
        LOG_SCOPE

        IrDsymbol &ir = decl->ir;
        if (ir.defined) return;
        ir.defined = true;

        if (decl->type->ty == Terror)
        {   error(decl->loc, "had semantic errors when compiling");
            return;
        }

        if (!decl->isAnonymous() && decl->members)
        {
            DtoResolveStruct(decl);

            for (Dsymbols::iterator I = decl->members->begin(),
                                    E = decl->members->end();
                                    I != E; ++I)
            {
                (*I)->accept(this);
            }

            // Define the __initZ symbol.
            llvm::GlobalVariable *initZ = ir.irAggr->getInitSymbol();
            initZ->setInitializer(ir.irAggr->getDefaultInit());
            initZ->setLinkage(DtoExternalLinkage(decl));

            // emit typeinfo
            DtoTypeInfoOf(decl->type);

            // Emit __xopEquals/__xopCmp.
            if (decl->xeq && decl->xeq != decl->xerreq)
                decl->xeq->accept(this);
            if (decl->xcmp && decl->xcmp != decl->xerrcmp)
                decl->xcmp->accept(this);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(ClassDeclaration *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("ClassDeclaration::codegen: '%s'", decl->toPrettyChars());
        LOG_SCOPE

        IrDsymbol &ir = decl->ir;
        if (ir.defined) return;
        ir.defined = true;

        if (decl->type->ty == Terror)
        {   error(decl->loc, "had semantic errors when compiling");
            return;
        }

        if (decl->members && decl->symtab)
        {
            DtoResolveClass(decl);

            for (Dsymbols::iterator I = decl->members->begin(),
                                    E = decl->members->end();
                                    I != E; ++I)
            {
                (*I)->accept(this);
            }

            llvm::GlobalValue::LinkageTypes const linkage = DtoExternalLinkage(decl);

            llvm::GlobalVariable *initZ = ir.irAggr->getInitSymbol();
            initZ->setInitializer(ir.irAggr->getDefaultInit());
            initZ->setLinkage(linkage);

            llvm::GlobalVariable *vtbl = ir.irAggr->getVtblSymbol();
            vtbl->setInitializer(ir.irAggr->getVtblInit());
            vtbl->setLinkage(linkage);

            llvm::GlobalVariable *classZ = ir.irAggr->getClassInfoSymbol();
            classZ->setInitializer(ir.irAggr->getClassInfoInit());
            classZ->setLinkage(linkage);

            // No need to do TypeInfo here, it is <name>__classZ for classes in D2.
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TupleDeclaration *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("TupleDeclaration::codegen(): '%s'", decl->toPrettyChars());
        LOG_SCOPE

        if (decl->ir.defined) return;
        decl->ir.defined = true;

        assert(decl->isexp);
        assert(decl->objects);

        for (Objects::iterator I = decl->objects->begin(),
                               E = decl->objects->end();
                               I != E; ++I)
        {
            DsymbolExp *exp = static_cast<DsymbolExp *>(*I);
            assert(exp->op == TOKdsymbol);
            exp->s->accept(this);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(VarDeclaration *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("VarDeclaration::codegen(): '%s'", decl->toPrettyChars());
        LOG_SCOPE;

        if (decl->ir.defined) return;
        decl->ir.defined = true;

        if (decl->type->ty == Terror)
        {   error(decl->loc, "had semantic errors when compiling");
            return;
        }

        DtoResolveVariable(decl);

        // just forward aliases
        if (decl->aliassym)
        {
            Logger::println("alias sym");
            decl->toAlias()->accept(this);
            return;
        }

        // global variable
        if (decl->isDataseg() || (decl->storage_class & (STCconst | STCimmutable) && decl->init))
        {
            Logger::println("data segment");

        #if 0 // TODO:
            assert(!(decl->storage_class & STCmanifest) &&
                "manifest constant being codegen'd!");
        #endif

            llvm::GlobalVariable *gvar = llvm::cast<llvm::GlobalVariable>(
                decl->ir.irGlobal->value);
            assert(gvar && "DtoResolveVariable should have created value");

            const llvm::GlobalValue::LinkageTypes llLinkage = DtoLinkage(decl);

            // Check if we are defining or just declaring the global in this module.
            if (!(decl->storage_class & STCextern))
            {
                // Build the initializer. Might use this->ir.irGlobal->value!
                LLConstant *initVal = DtoConstInitializer(decl->loc, decl->type, decl->init);

                // In case of type mismatch, swap out the variable.
                if (initVal->getType() != gvar->getType()->getElementType())
                {
                    llvm::GlobalVariable* newGvar = getOrCreateGlobal(decl->loc,
                        *irs->module, initVal->getType(), gvar->isConstant(),
                        llLinkage, 0,
                        "", // We take on the name of the old global below.
                        gvar->isThreadLocal());

                    newGvar->setAlignment(gvar->getAlignment());
                    newGvar->takeName(gvar);

                    llvm::Constant* newValue =
                        llvm::ConstantExpr::getBitCast(newGvar, gvar->getType());
                    gvar->replaceAllUsesWith(newValue);

                    gvar->eraseFromParent();
                    gvar = newGvar;
                    decl->ir.irGlobal->value = newGvar;
                }

                // Now, set the initializer.
                assert(!decl->ir.irGlobal->constInit);
                decl->ir.irGlobal->constInit = initVal;
                gvar->setInitializer(initVal);
                gvar->setLinkage(llLinkage);

                // Also set up the edbug info.
                irs->DBuilder.EmitGlobalVariable(gvar, decl);
            }

            // If this global is used from a naked function, we need to create an
            // artificial "use" for it, or it could be removed by the optimizer if
            // the only reference to it is in inline asm.
            if (decl->nakedUse)
                irs->usedArray.push_back(DtoBitCast(gvar, getVoidPtrType()));

            if (Logger::enabled())
                Logger::cout() << *gvar << '\n';
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TypedefDeclaration *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("TypedefDeclaration::codegen: '%s'", decl->toPrettyChars());
        LOG_SCOPE;

        if (decl->ir.defined) return;
        decl->ir.defined = true;

        if (decl->type->ty == Terror)
        {   error(decl->loc, "had semantic errors when compiling");
            return;
        }

        // generate typeinfo
        DtoTypeInfoOf(decl->type, false);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(EnumDeclaration *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("Ignoring EnumDeclaration::codegen: '%s'", decl->toPrettyChars());

        if (decl->type->ty == Terror)
        {   error(decl->loc, "had semantic errors when compiling");
            return;
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(FuncDeclaration *decl) LLVM_OVERRIDE {
        // don't touch function aliases, they don't contribute any new symbols
        if (!decl->isFuncAliasDeclaration())
        {
            DtoDefineFunction(decl);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TemplateInstance *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("TemplateInstance::codegen: '%s'", decl->toPrettyChars());
        LOG_SCOPE

        if (decl->ir.defined) return;
        decl->ir.defined = true;

        if (!decl->errors && decl->members)
        {
            for (Dsymbols::iterator I = decl->members->begin(),
                                    E = decl->members->end();
                                    I != E; ++I)
            {
                (*I)->accept(this);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TemplateMixin *decl) LLVM_OVERRIDE {
        IF_LOG Logger::println("TemplateInstance::codegen: '%s'", decl->toPrettyChars());
        LOG_SCOPE

        if (decl->ir.defined) return;
        decl->ir.defined = true;

        if (!decl->errors && decl->members)
        {
            for (Dsymbols::iterator I = decl->members->begin(),
                                    E = decl->members->end();
                                    I != E; ++I)
            {
                (*I)->accept(this);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(AttribDeclaration *decl) LLVM_OVERRIDE {
        Dsymbols *d = decl->include(NULL, NULL);

        if (d)
        {
            for (Dsymbols::iterator I = d->begin(),
                                    E = d->end();
                                    I != E; ++I)
            {
                (*I)->accept(this);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(PragmaDeclaration *decl) LLVM_OVERRIDE {
        if (decl->ident == Id::lib)
        {
            assert(decl->args && decl->args->dim == 1);

            Expression *e = static_cast<Expression *>(decl->args->data[0]);

            assert(e->op == TOKstring);
            StringExp *se = static_cast<StringExp *>(e);

            size_t nameLen = se->len;
            if (global.params.targetTriple.getOS() == llvm::Triple::MinGW32)
            {
                if (nameLen > 4 &&
                    !memcmp(static_cast<char*>(se->string) + nameLen - 4, ".lib", 4))
                {
                    // On MinGW, strip the .lib suffix, if any, to improve
                    // compatibility with code written for DMD (we pass the name to GCC
                    // via -l, just as on Posix).
                    nameLen -= 4;
                }

                if (nameLen >= 7 && !memcmp(se->string, "shell32", 7))
                {
                    // Another DMD compatibility kludge: Ignore
                    // pragma(lib, "shell32.lib"), it is implicitly provided by
                    // MinGW.
                    return;
                }
            }

    #if LDC_LLVM_VER >= 303
            // With LLVM 3.3 or later we can place the library name in the object
            // file. This seems to be supported only on Windows.
            if (global.params.targetTriple.getOS() == llvm::Triple::Win32)
            {
                llvm::SmallString<24> LibName(llvm::StringRef(static_cast<const char *>(se->string), nameLen));

                // Win32: /DEFAULTLIB:"curl"
                if (LibName.endswith(".a"))
                    LibName = LibName.substr(0, LibName.size()-2);
                if (LibName.endswith(".lib"))
                    LibName = LibName.substr(0, LibName.size()-4);
                llvm::SmallString<24> tmp("/DEFAULTLIB:\""); 
                tmp.append(LibName);
                tmp.append("\"");
                LibName = tmp;

                // Embedd library name as linker option in object file
                llvm::Value *Value = llvm::MDString::get(gIR->context(), LibName);
                gIR->LinkerMetadataArgs.push_back(llvm::MDNode::get(gIR->context(), Value));
            }
            else
    #endif
            {
                size_t const n = nameLen + 3;
                char *arg = static_cast<char *>(mem.malloc(n));
                arg[0] = '-';
                arg[1] = 'l';
                memcpy(arg + 2, se->string, nameLen);
                arg[n-1] = 0;
                global.params.linkswitches->push(arg);
            }
        }
        visit(static_cast<AttribDeclaration *>(decl));
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TypeInfoDeclaration *decl) LLVM_OVERRIDE {
        TypeInfoDeclaration_codegen(decl, irs);
    }

    //////////////////////////////////////////////////////////////////////////

    void visit(TypeInfoClassDeclaration *decl) LLVM_OVERRIDE {
        TypeInfoClassDeclaration_codegen(decl, irs);
    }
};

//////////////////////////////////////////////////////////////////////////////

void Declaration_codegen(Dsymbol *decl)
{
    CodegenVisitor v(gIR);
    decl->accept(&v);
}

void Declaration_codegen(Dsymbol *decl, IRState *irs)
{
    CodegenVisitor v(irs);
    decl->accept(&v);
}

//////////////////////////////////////////////////////////////////////////////
