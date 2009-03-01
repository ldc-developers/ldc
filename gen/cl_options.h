#ifndef LDC_CL_OPTIONS_H
#define LDC_CL_OPTIONS_H

#include "mars.h"

#include <deque>
#include <vector>

#include "llvm/Support/RegistryParser.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Support/CommandLine.h"

namespace opts {
    namespace cl = llvm::cl;

    /* Mostly generated with the following command:
       egrep -e '^(cl::|#if|#e)' gen/cl_options.cpp \
        | sed -re 's/^(cl::.*)\(.*$/    extern \1;/'
     */
    extern cl::list<std::string> fileList;
    extern cl::list<std::string> runargs;
    extern cl::opt<bool> compileOnly;
    extern cl::opt<bool> noAsm;
    extern cl::opt<bool> dontWriteObj;
    extern cl::opt<std::string> objectFile;
    extern cl::opt<std::string> objectDir;
    extern cl::opt<bool> output_bc;
    extern cl::opt<bool> output_ll;
    extern cl::opt<bool> output_s;
    extern cl::opt<cl::boolOrDefault> output_o;
    extern cl::opt<std::string> ddocDir;
    extern cl::opt<std::string> ddocFile;
#ifdef _DH
    extern cl::opt<std::string> hdrDir;
    extern cl::opt<std::string> hdrFile;
#endif
    extern cl::list<std::string> versions;

    extern cl::opt<const llvm::TargetMachineRegistry::entry*, false,
                    llvm::RegistryParser<llvm::TargetMachine> > mArch;
    extern cl::opt<bool> m32bits;
    extern cl::opt<bool> m64bits;
    extern cl::opt<std::string> mCPU;
    extern cl::list<std::string> mAttrs;
    extern cl::opt<std::string> mTargetTriple;

    // Arguments to -d-debug
    extern std::vector<std::string> debugArgs;
    // Arguments to -run
}
#endif
