#ifndef RUNTIMECOMPILE_H
#define RUNTIMECOMPILE_H

struct IRState;
struct IrFunction;
struct IrGlobal;

void generateBitcodeForRuntimeCompile(IRState *irs);
void declareRuntimeCompiledFunction(IRState *irs, IrFunction *func);
void defineRuntimeCompiledFunction(IRState *irs, IrFunction *func);
void addRuntimeCompiledVar(IRState *irs, IrGlobal *var);

#endif // RUNTIMECOMPILE_H
