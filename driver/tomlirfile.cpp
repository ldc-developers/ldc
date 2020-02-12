//===-- tomlirfile.cpp-----------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_MLIR_ENABLED

#include "tomlirfile.h"
#include "dmd/errors.h"

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
    Logger::println("Writing MLIR to %s\n", llpath.c_str());
    std::error_code errinfo;
    llvm::raw_fd_ostream aos(llpath, errinfo, llvm::sys::fs::F_None);
    if(aos.has_error()){
      error(Loc(), "Cannot write MLIR file '%s': %s", llpath.c_str(),
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

#endif //LDC_MLIR_ENABLED

