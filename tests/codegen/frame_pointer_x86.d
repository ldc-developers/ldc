// REQUIRES: target_X86

// RUN: %ldc -c -mtriple=x86_64 -output-s -of=%t.s %s
// RUN: FileCheck %s --check-prefixes=COMMON,FP < %t.s
// RUN: %ldc -c -mtriple=x86_64 -output-s -of=%t.s %s -O2
// RUN: FileCheck %s --check-prefixes=COMMON,NO_FP < %t.s
// RUN: %ldc -c -mtriple=x86_64 -output-s -of=%t.s %s -O2 -frame-pointer=all
// RUN: FileCheck %s --check-prefixes=COMMON,FP < %t.s
// RUN: %ldc -c -mtriple=x86_64 -output-s -of=%t.s %s -frame-pointer=none
// RUN: FileCheck %s --check-prefixes=COMMON,NO_FP < %t.s

// COMMON-LABEL: _D17frame_pointer_x8613inlineAsmLeafFZv:
// COMMON:       pushq %rbp
// COMMON-LABEL: _D17frame_pointer_x8616inlineAsmNonLeafFZv:
// COMMON:       pushq %rbp

// COMMON-LABEL: _D17frame_pointer_x863fooFZv:
// FP:           pushq %rbp
// NO_FP-NOT:    pushq %rbp

void externalFunc();
void inlineAsmLeaf() { asm { nop; } }
void inlineAsmNonLeaf() { asm { nop; } externalFunc(); }

void foo() {}
