#ifndef LLVMDC_GEN_LINKER_H
#define LLVMDC_GEN_LINKER_H

/**
 * Links the modules given in MV in to dst.
 * @param dst Destination module.
 * @param MV Vector of modules to link in to destination.
 */
void linkModules(llvm::Module* dst, const std::vector<llvm::Module*>& MV);

#endif // LLVMDC_GEN_LINKER_H
