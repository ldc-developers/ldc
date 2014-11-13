//===-- driver/cl_options.h - LDC command line options ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the LDC command line options as handled using the LLVM command
// line parsing library.
//
//===----------------------------------------------------------------------===//

#ifndef LDC_DRIVER_CL_OPTIONS_H
#define LDC_DRIVER_CL_OPTIONS_H

#include "driver/targetmachine.h"
#include "gen/cl_helpers.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include <deque>
#include <vector>

namespace opts {
    namespace cl = llvm::cl;

    enum BoundsCheck {
        BC_Off,
        BC_SafeOnly,
        BC_On,
        BC_Default
    };

    /* Mostly generated with the following command:
       egrep -e '^(cl::|#if|#e)' gen/cl_options.cpp \
        | sed -re 's/^(cl::.*)\(.*$/    extern \1;/'
     */
    extern cl::list<std::string> fileList;
    extern cl::list<std::string> runargs;
    extern cl::opt<bool> compileOnly;
    extern cl::opt<bool, true> enforcePropertySyntax;
    extern cl::opt<bool> createStaticLib;
    extern cl::opt<bool> createSharedLib;
    extern cl::opt<bool> noAsm;
    extern cl::opt<bool> dontWriteObj;
    extern cl::opt<std::string> objectFile;
    extern cl::opt<std::string> objectDir;
    extern cl::opt<std::string> soname;
    extern cl::opt<bool> output_bc;
    extern cl::opt<bool> output_ll;
    extern cl::opt<bool> output_s;
    extern cl::opt<cl::boolOrDefault> output_o;
    extern cl::opt<bool, true> disableRedZone;
    extern cl::opt<std::string> ddocDir;
    extern cl::opt<std::string> ddocFile;
    extern cl::opt<std::string> jsonFile;
    extern cl::opt<std::string> hdrDir;
    extern cl::opt<std::string> hdrFile;
    extern cl::list<std::string> versions;
    extern cl::opt<std::string> moduleDepsFile;

    extern cl::opt<std::string> mArch;
    extern cl::opt<bool> m32bits;
    extern cl::opt<bool> m64bits;
    extern cl::opt<std::string> mCPU;
    extern cl::list<std::string> mAttrs;
    extern cl::opt<std::string> mTargetTriple;
    extern cl::opt<llvm::Reloc::Model> mRelocModel;
    extern cl::opt<llvm::CodeModel::Model> mCodeModel;
    extern cl::opt<bool> disableFpElim;
    extern cl::opt<FloatABI::Type> mFloatABI;
    extern cl::opt<bool> enableFPMAD;
    extern cl::opt<bool> enableUnsafeFPMath;
    extern cl::opt<bool> enableNoInfsFPMath;
    extern cl::opt<bool> enableNoNaNsFPMath;
    extern cl::opt<bool> enableHonorSignDependentRoundingFPMath;
    extern cl::opt<bool, true> singleObj;
    extern cl::opt<bool> linkonceTemplates;
    extern cl::opt<bool> disableLinkerStripDead;

    extern BoundsCheck boundsCheck;
    extern bool nonSafeBoundsChecks;

    extern cl::opt<unsigned, true> nestedTemplateDepth;

    // Arguments to -d-debug
    extern std::vector<std::string> debugArgs;
    // Arguments to -run
}
#endif
