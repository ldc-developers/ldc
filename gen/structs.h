#ifndef LLVMD_GEN_STRUCTS_H
#define LLVMD_GEN_STRUCTS_H

struct StructInitializer;

/// Generate code for the struct.
void DtoResolveStruct(StructDeclaration* sd);

/// Build constant struct initializer.
LLConstant* DtoConstStructInitializer(StructInitializer* si);

/// Build values for a struct literal.
std::vector<llvm::Value*> DtoStructLiteralValues(const StructDeclaration* sd, const std::vector<llvm::Value*>& inits);

/// Returns a boolean=true if the two structs are equal.
LLValue* DtoStructEquals(TOK op, DValue* lhs, DValue* rhs);

/// index a struct one level
LLValue* DtoIndexStruct(LLValue* src, StructDeclaration* sd, VarDeclaration* vd);

#endif
