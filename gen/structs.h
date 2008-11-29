#ifndef LLVMD_GEN_STRUCTS_H
#define LLVMD_GEN_STRUCTS_H

struct StructInitializer;

LLConstant* DtoConstStructInitializer(StructInitializer* si);

/**
 * Resolves the llvm type for a struct
 */
void DtoResolveStruct(StructDeclaration* sd);

/**
 * Provides the llvm declaration for a struct
 */
void DtoDeclareStruct(StructDeclaration* sd);

/**
 * Constructs the constant default initializer a struct
 */
void DtoConstInitStruct(StructDeclaration* sd);

/**
 * Provides the llvm definition for a struct
 */
void DtoDefineStruct(StructDeclaration* sd);

/**
 * Returns a boolean=true if the two structs are equal
 */
LLValue* DtoStructEquals(TOK op, DValue* lhs, DValue* rhs);

// index a struct one level
LLValue* DtoIndexStruct(LLValue* src, StructDeclaration* sd, VarDeclaration* vd);

#endif
