#ifndef LLVMDC_GEN_LINKER_H
#define LLVMDC_GEN_LINKER_H

#include <vector>

namespace llvm
{
    class Module;
}

/**
 * Links the modules given in MV in to dst.
 * @param dst Destination module.
 * @param MV Vector of modules to link in to destination.
 */
void linkModules(llvm::Module* dst, const std::vector<llvm::Module*>& MV);

/**
 * Link an executable.
 * @return 0 on success.
 */
int linkExecutable();

/**
 * Delete the executable that was previously linked with linkExecutable.
 */
void deleteExecutable();

/**
 * Runs the executable that was previously linked with linkExecutable.
 * @return the return status of the executable.
 */
int runExectuable();

#endif // LLVMDC_GEN_LINKER_H
