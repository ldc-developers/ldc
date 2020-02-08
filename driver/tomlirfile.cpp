//
// Created by Roberto Rosmaninho on 09/10/19.
//

#include "tomlirfile.h"

#include "dmd/errors.h"
#include "driver/cl_options.h"
#include "driver/cache.h"
#include "driver/targetmachine.h"
#include "driver/tool.h"
#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/optimizer.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#if LDC_LLVM_VER >= 400
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#else
#include "llvm/Bitcode/ReaderWriter.h"
#endif
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#if LDC_LLVM_VER >= 600
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#else
#include "llvm/Target/TargetSubtargetInfo.h"
#endif
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Module.h"
#include <cstddef>
#include <fstream>
#include <mlir/IR/Module.h>

#ifdef LDC_LLVM_SUPPORTED_TARGET_SPIRV
namespace llvm {
    ModulePass *createSPIRVWriterPass(llvm::raw_ostream &Str);
}
#endif

#include "gen/logger.h"
#include "mlir/IR/Module.h"
#include "dmd/globals.h"
#include "gen/MLIR/MLIRGen.h"
#include "dmd/expression.h"

void writeMLIRModule(Module *m, mlir::MLIRContext &mlirContext,
                     const char *filename, IRState *irs){
  const auto outputFlags = {global.params.output_o, global.params.output_bc,
                            global.params.output_ll, global.params.output_s,
                            global.params.output_mlir};
  const auto numOutputFiles =
      std::count_if(outputFlags.begin(), outputFlags.end(),
                    [](OUTPUTFLAG flag) { return flag != 0; });

  const auto replaceExtensionWith =
      [=](const DArray<const char> &ext) -> std::string {
        if (numOutputFiles == 1)
          return filename;
        llvm::SmallString<128> buffer(filename);
        llvm::sys::path::replace_extension(buffer,
                                           llvm::StringRef(ext.ptr, ext.length));
        return buffer.str();
      };

  //Write MLIR
  if(global.params.output_mlir) {
    const auto llpath = replaceExtensionWith(global.mlir_ext);
    Logger::println("Writting MLIR to %s\n", llpath.c_str());
    std::error_code errinfo;
    llvm::raw_fd_ostream aos(llpath.c_str(), errinfo, llvm::sys::fs::F_None);
    if(aos.has_error()){
      error(Loc(), "Cannot write MLIR file '%s':%s", llpath.c_str(),
            errinfo.message().c_str());
      fatal();
    }
    mlir::OwningModuleRef module = ldc_mlir::mlirGen(mlirContext, m, irs);
    if(!module){
      IF_LOG Logger::println("Cannot write MLIR file to '%s'", llpath.c_str());
      fatal();
    }
    module->print(aos);
  }
}