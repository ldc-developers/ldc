#ifndef LLVMDC_GEN_TODEBUG_H
#define LLVMDC_GEN_TODEBUG_H

void RegisterDwarfSymbols(llvm::Module* mod);

const llvm::StructType* GetDwarfAnchorType();
const llvm::StructType* GetDwarfCompileUnitType();
const llvm::StructType* GetDwarfSubProgramType();

llvm::GlobalVariable* DtoDwarfCompileUnit(Module* m);
llvm::GlobalVariable* DtoDwarfSubProgram(FuncDeclaration* fd, llvm::GlobalVariable* compileUnit);

void DtoDwarfFuncStart(FuncDeclaration* fd);
void DtoDwarfFuncEnd(FuncDeclaration* fd);

void DtoDwarfStopPoint(unsigned ln);

#endif // LLVMDC_GEN_TODEBUG_H


