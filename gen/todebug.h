#ifndef LLVMDC_GEN_TODEBUG_H
#define LLVMDC_GEN_TODEBUG_H

void RegisterDwarfSymbols(llvm::Module* mod);

llvm::GlobalVariable* DtoDwarfCompileUnit(Module* m);
llvm::GlobalVariable* DtoDwarfSubProgram(FuncDeclaration* fd, llvm::GlobalVariable* compileUnit);

void DtoDwarfFuncStart(FuncDeclaration* fd);
void DtoDwarfFuncEnd(FuncDeclaration* fd);

void DtoDwarfStopPoint(unsigned ln);

/**
 * Emits all things necessary for making debug info for a local variable vd.
 * @param ll LLVM Value of the variable.
 * @param vd Variable declaration to emit debug info for.
 */
void DtoDwarfLocalVariable(LLValue* ll, VarDeclaration* vd);

/**
 * Emits all things necessary for making debug info for a global variable vd.
 * @param ll 
 * @param vd 
 * @return 
 */
LLGlobalVariable* DtoDwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd);

#endif // LLVMDC_GEN_TODEBUG_H


