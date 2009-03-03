#ifndef LDC_IR_IRFUNCTION_H
#define LDC_IR_IRFUNCTION_H

#include "gen/llvm.h"
#include "ir/ir.h"
#include "ir/irlandingpad.h"

#include <vector>
#include <stack>
#include <map>

struct ABIRewrite;

// represents a function type argument
// both explicit and implicit as well as return values
struct IrFuncTyArg : IrBase
{
    Type* type;
    const llvm::Type* ltype;
    unsigned attrs;
    bool byref;

    ABIRewrite* rewrite;

    bool isInReg() const { return (attrs & llvm::Attribute::InReg) != 0; }
    bool isSRet() const  { return (attrs & llvm::Attribute::StructRet) != 0; }
    bool isByVal() const { return (attrs & llvm::Attribute::ByVal) != 0; }

    IrFuncTyArg(Type* t, bool byref, unsigned a = 0);
};

// represents a function type
struct IrFuncTy : IrBase
{
    // return value
    IrFuncTyArg* ret;

    // null if not applicable
    IrFuncTyArg* arg_sret;
    IrFuncTyArg* arg_this;
    IrFuncTyArg* arg_nest;
    IrFuncTyArg* arg_arguments;
    IrFuncTyArg* arg_argptr;

    // normal explicit arguments
    LLSmallVector<IrFuncTyArg*, 4> args;

    // C varargs
    bool c_vararg;

    // range of normal parameters to reverse
    bool reverseParams;

    IrFuncTy()
    :   ret(NULL),
        arg_sret(NULL),
        arg_this(NULL),
        arg_nest(NULL),
        arg_arguments(NULL),
        arg_argptr(NULL),
        c_vararg(false),
        reverseParams(false)
    {}

    llvm::Value* putRet(Type* dty, llvm::Value* val);
    llvm::Value* getRet(Type* dty, llvm::Value* val);

    llvm::Value* getParam(Type* dty, int idx, llvm::Value* val);
    llvm::Value* putParam(Type* dty, int idx, llvm::Value* val);
};

// represents a function
struct IrFunction : IrBase
{
    llvm::Function* func;
    llvm::Instruction* allocapoint;
    FuncDeclaration* decl;
    TypeFunction* type;

    bool queued;
    bool defined;
    
    llvm::Value* retArg; // return in ptr arg
    llvm::Value* thisArg; // class/struct 'this' arg
    llvm::Value* nestArg; // nested function 'this' arg
    
    llvm::Value* nestedVar; // nested var alloca
    
    llvm::Value* _arguments;
    llvm::Value* _argptr;
    
    llvm::DISubprogram diSubprogram;

    // pushes a unique label scope of the given name
    void pushUniqueLabelScope(const char* name);
    // pops a label scope
    void popLabelScope();

    // gets the string under which the label's BB
    // is stored in the labelToBB map.
    // essentially prefixes ident by the strings in labelScopes
    std::string getScopedLabelName(const char* ident);

    // label to basic block lookup
    typedef std::map<std::string, llvm::BasicBlock*> LabelToBBMap;
    LabelToBBMap labelToBB;

    // landing pads for try statements
    IRLandingPad landingPad;

    IrFunction(FuncDeclaration* fd);

    // annotations
    void setNeverInline();
    void setAlwaysInline();

private:
    // prefix for labels and gotos
    // used for allowing labels to be emitted twice
    std::vector<std::string> labelScopes;

    // next unique id stack
    std::stack<int> nextUnique;
};

#endif
