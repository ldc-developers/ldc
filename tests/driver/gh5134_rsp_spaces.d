// https://github.com/ldc-developers/ldc/issues/5134
// Dub writes LF-only response files on Windows; LLVM's Windows tokenizer
// expects CRLF, which used to glue the next flag onto a quoted .lib path
// (Error: unrecognized file extension lib).
//
// REQUIRES: Windows
// RUN: mkdir "%t dir with spaces"
// RUN: echo void main(){} > "%t dir with spaces/t.d"
// RUN: python %S/inputs/write_lf_rsp.py %t.rsp "\"%t dir with spaces/t.d\"" -c -o- -vcolumns
// RUN: %ldc @%t.rsp
// RUN: python %S/inputs/write_lf_rsp.py %t2.rsp "\"%t dir with spaces/dummy.lib\"" -vcolumns
// RUN: not %ldc @%t2.rsp 2>&1 | FileCheck %s
// CHECK-NOT: unrecognized file extension
