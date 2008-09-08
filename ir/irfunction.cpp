
#include "gen/llvm.h"
#include "gen/tollvm.h"
#include "ir/irfunction.h"

#include <sstream>

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

IrFunction::IrFunction(FuncDeclaration* fd)
{
    decl = fd;

    Type* t = fd->type->toBasetype();
    assert(t->ty == Tfunction);
    type = (TypeFunction*)t;
    func = NULL;
    allocapoint = NULL;

    queued = false;
    defined = false;

    retArg = NULL;
    thisArg = NULL;
    nestArg = NULL;

    nestedVar = NULL;
    
    _arguments = NULL;
    _argptr = NULL;
    
    dwarfSubProg = NULL;

    srcfileArg = NULL;
    msgArg = NULL;

    nextUnique.push(0);
}

std::string IrFunction::getScopedLabelName(const char* ident)
{
    if(labelScopes.empty())
        return std::string(ident);

    std::string result = "__";
    for(unsigned int i = 0; i < labelScopes.size(); ++i)
        result += labelScopes[i] + "_";
    return result + ident;
}

void IrFunction::pushUniqueLabelScope(const char* name)
{
    std::ostringstream uniquename;
    uniquename << name << nextUnique.top()++;
    nextUnique.push(0);
    labelScopes.push_back(uniquename.str());
}

void IrFunction::popLabelScope()
{
    labelScopes.pop_back();
    nextUnique.pop();
}

void IrFunction::setNeverInline()
{
    llvm::FunctionNotes cur = func->getNotes();
    assert(!(cur & llvm::FN_NOTE_AlwaysInline) && "function can't be never- and always-inline at the same time");
    func->setNotes(cur | llvm::FN_NOTE_NoInline);
}

void IrFunction::setAlwaysInline()
{
    llvm::FunctionNotes cur = func->getNotes();
    assert(!(cur & llvm::FN_NOTE_NoInline) && "function can't be never- and always-inline at the same time");
    func->setNotes(cur | llvm::FN_NOTE_AlwaysInline);
}
