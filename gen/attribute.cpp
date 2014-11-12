//===-- gen/attribute.h - Handling of @ldc.attribute ------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Contains helpers for handling of @ldc.attribute.
//
//===----------------------------------------------------------------------===//

#include "gen/attribute.h"
#include "llvm/ADT/StringSwitch.h"
#if LDC_LLVM_VER >= 303
#include "llvm/IR/Function.h"
#else
#include "llvm/Function.h"
#endif
#include "aggregate.h"
#include "attrib.h"
#include "declaration.h"
#include "expression.h"
#include "module.h"
#include "root.h"

#if LDC_LLVM_VER >= 303
typedef llvm::Attribute::AttrKind AttrVal;
#elif LDC_LLVM_VER == 302
typedef llvm::Attributes::AttrVal AttrVal;
namespace llvm
{
    typedef llvm::Attributes Attribute;
}
#else
typedef llvm::Attribute::AttrConst AttrVal;
#endif

////////////////////////////////////////////////////////////////////////////////

// FIXME: merge with pragma.cpp#parseStringExp
static bool parseStringExp(Expression* e, std::string& res)
{
    StringExp *s = NULL;

    e = e->optimize(WANTexpand);
    if (e->op == TOKstring && (s = static_cast<StringExp *>(e)))
    {
        char* str = static_cast<char*>(s->string);
        res = str;
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////

// FIXME: merge with pragma.cpp#parseStringExp
static bool parseIntExp(Expression* e, dinteger_t& res)
{
    IntegerExp *i = NULL;

    e = e->optimize(WANTexpand);
    if (e->op == TOKint64 && (i = static_cast<IntegerExp *>(e)))
    {
        res = i->getInteger();
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////

static bool isLdcAttribute(Expression *attr)
{
    if (!attr || attr->op != TOKcall) return false;
    if (attr->type->ty != Tstruct) return false;
    TypeStruct *ts = static_cast<TypeStruct *>(attr->type);
    StructDeclaration *sym = ts->sym;
    if (strcmp("Attribute", sym->ident->string)) return false;
    Module *module = sym->getModule();
    if (strcmp("attribute", module->md->id->string)) return false;
    if (module->md->packages->dim != 1 ||
        strcmp("ldc", (*module->md->packages)[0]->string)) return false;
    return true;
}

////////////////////////////////////////////////////////////////////////////////

static inline void checkArgNum(Loc &loc, size_t argc, size_t reqArgc,
                               const std::string &name)
{
    if (argc != reqArgc)
    {
        error(loc, "Attribute %s requires %d argument(s)", name.c_str(), reqArgc);
    }
}

////////////////////////////////////////////////////////////////////////////////

void DtoFuncDeclarationAttribute(FuncDeclaration* fdecl, llvm::Function *func)
{
    if (!fdecl->userAttribDecl) return;
    for (Expressions::iterator I = fdecl->userAttribDecl->getAttributes()->begin(),
                               E = fdecl->userAttribDecl->getAttributes()->end();
                               I != E; ++I)
    //for (ArrayIter<Expression> it(fdecl->userAttribDecl); !it.done(); it.next())
    {
        Expression *attr = (*I)->optimize(WANTexpand);
        if (!isLdcAttribute(attr)) continue;

        Expressions *exps = static_cast<CallExp *>(attr)->arguments;
        assert(exps && exps->dim >= 1);

        Expression *exp = (*exps)[0];
        Loc &loc = exp->loc;
        std::string name;
        if (!parseStringExp(exp, name))
            error(loc, "First argument of @ldc.attribute must be of type string");

        AttrVal attrVal = llvm::StringSwitch<AttrVal>(name)
                                .Case("alignstack", llvm::Attribute::StackAlignment)
                                .Case("alwaysinline", llvm::Attribute::AlwaysInline)
#if LDC_LLVM_VER >= 304
                                .Case("builtin", llvm::Attribute::Builtin)
                                .Case("cold", llvm::Attribute::Cold)
#endif
                                .Case("inlinehint", llvm::Attribute::InlineHint)
#if LDC_LLVM_VER >= 302
                                .Case("minsize", llvm::Attribute::MinSize)
#endif
                                .Case("naked", llvm::Attribute::Naked)
#if LDC_LLVM_VER >= 304
                                .Case("nobuiltin", llvm::Attribute::NoBuiltin)
                                .Case("noduplicate", llvm::Attribute::NoDuplicate)
#endif
                                .Case("noimplicitfloat", llvm::Attribute::NoImplicitFloat)
                                .Case("noinline", llvm::Attribute::NoInline)
                                .Case("nonlazybind", llvm::Attribute::NonLazyBind)
                                .Case("noredzone", llvm::Attribute::NoRedZone)
                                .Case("noreturn", llvm::Attribute::NoReturn)
                                .Case("nounwind", llvm::Attribute::NoUnwind)
#if LDC_LLVM_VER >= 304
                                .Case("optnone", llvm::Attribute::OptimizeNone)
#endif
                                .Case("optsize", llvm::Attribute::OptimizeForSize)
                                .Case("readnone", llvm::Attribute::ReadNone)
                                .Case("readonly", llvm::Attribute::ReadOnly)
                                .Case("returns_twice", llvm::Attribute::ReturnsTwice)
#if LDC_LLVM_VER >= 303
                                .Case("sanitize_address", llvm::Attribute::SanitizeAddress)
                                .Case("sanitize_memory", llvm::Attribute::SanitizeMemory)
                                .Case("sanitize_thread", llvm::Attribute::SanitizeThread)
#else
                                .Case("sanitize_address", llvm::Attribute::AddressSafety)
#endif
                                .Case("ssp", llvm::Attribute::StackProtect)
                                .Case("sspreq", llvm::Attribute::StackProtectReq)
#if LDC_LLVM_VER >= 303
                                .Case("sspstrong", llvm::Attribute::StackProtectStrong)
#endif
                                .Case("uwtable", llvm::Attribute::UWTable)
                                .Default(llvm::Attribute::None);

        if (attrVal != llvm::Attribute::None) {
            if (attrVal != llvm::Attribute::StackAlignment) {
                checkArgNum(loc, exps->dim, 1, name);
            } else {
                checkArgNum(loc, exps->dim, 2, name);
            }
        }

        switch (attrVal) {
        case llvm::Attribute::None:
            if (name == "section") {
                std::string section;
                checkArgNum(loc, exps->dim, 2, name);
                if (parseStringExp((*exps)[1], section)) {
                    func->setSection(section);
                } else
                    error(loc, "Attribute %s requires 1 string argument", name.c_str());
            } else
                error(loc, "Unknown attribute %s\n", name.c_str());
            break;
        case llvm::Attribute::StackAlignment:
            dinteger_t align;
            if (parseIntExp((*exps)[1], align)) {
                std::string val(std::to_string(align));
                func->addFnAttr(name, val);
            } else
                error(loc, "Attribute %s requires 1 integer argument", name.c_str());
            break;
        case llvm::Attribute::AlwaysInline:
            func->addFnAttr(attrVal);
            func->removeFnAttr(llvm::Attribute::NoInline);
            func->removeFnAttr(llvm::Attribute::InlineHint);
            break;
        case llvm::Attribute::NoInline:
            func->addFnAttr(attrVal);
            func->removeFnAttr(llvm::Attribute::AlwaysInline);
            func->removeFnAttr(llvm::Attribute::InlineHint);
            break;
        case llvm::Attribute::InlineHint:
            func->addFnAttr(attrVal);
            func->removeFnAttr(llvm::Attribute::AlwaysInline);
            func->removeFnAttr(llvm::Attribute::NoInline);
            break;
        default:
            func->addFnAttr(attrVal);
        }
    }
}
