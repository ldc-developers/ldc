#include <cassert>

#include "gen/llvm.h"
#include "llvm/Module.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"

#include "root.h"
#include "mars.h"

#include "gen/runtime.h"
#include "gen/logger.h"

static llvm::Module* M = NULL;
static bool runtime_failed = false;

//////////////////////////////////////////////////////////////////////////////////////////////////

bool LLVM_D_InitRuntime()
{
    Logger::println("*** Loading D runtime ***");
    LOG_SCOPE;

    if (!global.params.runtimeImppath) {
        error("You must set the runtime import path with -E");
        fatal();
    }
    std::string filename(global.params.runtimeImppath);
    filename.append("/llvmdcore.bc");
    llvm::MemoryBuffer* buffer = llvm::MemoryBuffer::getFile(filename.c_str(), filename.length());
    if (!buffer) {
        Logger::println("Failed to load runtime library from disk");
        runtime_failed = true;
        return false;
    }

    std::string errstr;
    bool retval = false;
    M = llvm::ParseBitcodeFile(buffer, &errstr);
    if (M) {
        retval = true;
    }
    else {
        Logger::println("Failed to load runtime: %s", errstr.c_str());
        runtime_failed = true;
    }

    delete buffer;
    return retval;
}

void LLVM_D_FreeRuntime()
{
    if (M) {
        Logger::println("*** Freeing D runtime ***");
        delete M;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::Function* LLVM_D_GetRuntimeFunction(llvm::Module* target, const char* name)
{
    // TODO maybe check the target module first, to allow overriding the runtime on a pre module basis?
    // could be done and seems like it could be neat too :)

    if (global.params.noruntime) {
        error("No implicit runtime calls allowed with -noruntime option enabled");
        fatal();
    }

    if (!M) {
        assert(!runtime_failed);
        LLVM_D_InitRuntime();
    }

    llvm::Function* fn = M->getFunction(name);
    if (!fn) {
        error("Runtime function '%s' was not found", name);
        fatal();
        //return NULL;
    }

    const llvm::FunctionType* fnty = fn->getFunctionType();
    return llvm::cast<llvm::Function>(target->getOrInsertFunction(name, fnty));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable* LLVM_D_GetRuntimeGlobal(llvm::Module* target, const char* name)
{
    // TODO maybe check the target module first, to allow overriding the runtime on a pre module basis?
    // could be done and seems like it could be neat too :)

    llvm::GlobalVariable* gv = target->getNamedGlobal(name);
    if (gv) {
        return gv;
    }

    if (global.params.noruntime) {
        error("No implicit runtime calls allowed with -noruntime option enabled");
        fatal();
    }

    if (!M) {
        assert(!runtime_failed);
        LLVM_D_InitRuntime();
    }

    llvm::GlobalVariable* g = M->getNamedGlobal(name);
    if (!g) {
        error("Runtime global '%s' was not found", name);
        fatal();
        //return NULL;
    }

    const llvm::PointerType* t = g->getType();
    return new llvm::GlobalVariable(t->getElementType(),g->isConstant(),g->getLinkage(),NULL,g->getName(),target);
}
