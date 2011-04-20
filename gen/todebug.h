#ifndef LDC_GEN_TODEBUG_H
#define LDC_GEN_TODEBUG_H

#ifndef DISABLE_DEBUG_INFO

void RegisterDwarfSymbols(llvm::Module* mod);

/**
 * Emit the Dwarf compile_unit global for a Module m.
 * @param m
 */
void DtoDwarfCompileUnit(Module* m);

/**
 * Emit the Dwarf subprogram global for a function declaration fd.
 * @param fd
 * @return the Dwarf subprogram global.
 */
llvm::DISubprogram DtoDwarfSubProgram(FuncDeclaration* fd);

/**
 * Emit the Dwarf subprogram global for a internal function.
 * This is used for generated functions like moduleinfoctors,
 * module ctors/dtors and unittests.
 * @return the Dwarf subprogram global.
 */
llvm::DISubprogram DtoDwarfSubProgramInternal(const char* prettyname, const char* mangledname);

void DtoDwarfFuncStart(FuncDeclaration* fd);
void DtoDwarfFuncEnd(FuncDeclaration* fd);

void DtoDwarfStopPoint(unsigned ln);

void DtoDwarfValue(LLValue* var, VarDeclaration* vd);

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
llvm::DIGlobalVariable DtoDwarfGlobalVariable(LLGlobalVariable* ll, VarDeclaration* vd);


#endif // DISABLE_DEBUG_INFO

#endif // LDC_GEN_TODEBUG_H
