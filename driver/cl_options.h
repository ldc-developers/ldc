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

#pragma once

#include "driver/cl_options-llvm.h"
#include "driver/targetmachine.h"
#include "gen/cl_helpers.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include <deque>
#include <vector>

namespace llvm {
class FastMathFlags;
class TargetMachine;
}

namespace opts {
namespace cl = llvm::cl;

/// Stores the commandline arguments list, including the ones specified by the
/// config and response files.
extern llvm::SmallVector<const char *, 32> allArguments;

extern cl::OptionCategory linkingCategory;

/* Mostly generated with the following command:
   egrep -e '^(cl::|#if|#e)' gen/cl_options.cpp \
    | sed -re 's/^(cl::.*)\(.*$/    extern \1;/'
 */
extern cl::list<std::string> fileList;
extern cl::list<std::string> runargs;
extern cl::opt<bool> invokedByLDMD;
extern cl::opt<bool> compileOnly;
extern cl::opt<bool> noAsm;
extern cl::opt<bool> dontWriteObj;
extern cl::opt<std::string> objectFile;
extern cl::opt<std::string> objectDir;
extern cl::opt<std::string> soname;
extern cl::opt<bool> output_bc;
extern cl::opt<bool> output_ll;
extern cl::opt<bool> output_mlir;
extern cl::opt<bool> output_s;
extern cl::opt<cl::boolOrDefault> output_o;
extern cl::opt<std::string> ddocDir;
extern cl::opt<std::string> ddocFile;
extern cl::opt<std::string> jsonFile;
extern cl::list<std::string> jsonFields;
extern cl::opt<std::string> hdrDir;
extern cl::opt<std::string> hdrFile;
extern cl::opt<bool> hdrKeepAllBodies;
extern cl::opt<std::string> cxxHdrDir;
extern cl::opt<std::string> cxxHdrFile;
extern cl::opt<std::string> mixinFile;
extern cl::list<std::string> versions;
extern cl::list<std::string> transitions;
extern cl::list<std::string> previews;
extern cl::list<std::string> reverts;
extern cl::opt<std::string> moduleDeps;
extern cl::opt<std::string> cacheDir;
extern cl::list<std::string> linkerSwitches;
extern cl::list<std::string> ccSwitches;
extern cl::list<std::string> includeModulePatterns;

extern cl::opt<bool> m32bits;
extern cl::opt<bool> m64bits;
extern cl::opt<std::string> mTargetTriple;
extern cl::opt<std::string> mABI;
extern FloatABI::Type floatABI;
extern cl::opt<bool> linkonceTemplates;
extern cl::opt<bool> disableLinkerStripDead;
extern cl::opt<unsigned char> defaultToHiddenVisibility;
extern cl::opt<bool> noPLT;

// Math options
extern bool fFastMath;
extern llvm::FastMathFlags defaultFMF;
void setDefaultMathOptions(llvm::TargetOptions &targetOptions);

// Arguments to -d-debug
extern std::vector<std::string> debugArgs;
// Arguments to -run

void createClashingOptions();
void hideLLVMOptions();

// LTO options
enum LTOKind {
  LTO_None,
  LTO_Full,
  LTO_Thin,
};
extern cl::opt<LTOKind> ltoMode;
inline bool isUsingLTO() { return ltoMode != LTO_None; }
inline bool isUsingThinLTO() { return ltoMode == LTO_Thin; }

#if LDC_LLVM_VER >= 400
extern cl::opt<std::string> saveOptimizationRecord;
#endif
#if LDC_LLVM_SUPPORTED_TARGET_SPIRV || LDC_LLVM_SUPPORTED_TARGET_NVPTX
extern cl::list<std::string> dcomputeTargets;
extern cl::opt<std::string> dcomputeFilePrefix;
#endif

#if defined(LDC_DYNAMIC_COMPILE)
extern cl::opt<bool> enableDynamicCompile;
extern cl::opt<bool> dynamicCompileTlsWorkaround;
#else
constexpr bool enableDynamicCompile = false;
#endif
}
