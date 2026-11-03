// RUN: %ldc -mtriple=x86_64-apple-darwin -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// REQUIRES: target_X86

// CHECK: !llvm.linker.options = !{![[OPT0:[0-9]+]], ![[OPT1:[0-9]+]], ![[OPT2:[0-9]+]]}

// CHECK: ![[OPT0]] = !{!"-lmylib"}
pragma(lib, "mylib");

// CHECK: ![[OPT1]] = !{!"-myflag"}
pragma(linkerDirective, "-myflag");
// CHECK: ![[OPT2]] = !{!"-framework", !"CoreFoundation"}
pragma(linkerDirective, "-framework", "CoreFoundation");
