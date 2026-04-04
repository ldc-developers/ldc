// Regression test: importing a module with a recursive template via -I and using
// __traits(compiles, ...) on it should not crash the compiler during dcompute
// semantic analysis. Previously, the VarDeclaration's type field could be null
// for error'd template instantiations, causing a segfault in
// DComputeSemanticAnalyser::visit(VarDeclaration*).

// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 -o- -I%S/inputs %s

@compute(CompileFor.deviceOnly) module tests.compilable.dcompute_template_import;
import ldc.dcompute;
import dcompute_testmod;

static assert(!__traits(compiles, { imported!"dcompute_testmod".hasIndirections!(int[]); }));
