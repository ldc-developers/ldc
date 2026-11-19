// RUN: %ldc -mtriple=x86_64-pc-windows-msvc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// REQUIRES: target_X86

// CHECK: !llvm.linker.options = !{![[OPT0:[0-9]+]], ![[OPT1:[0-9]+]], ![[OPT2:[0-9]+]]}

// CHECK: ![[OPT0]] = !{!"/DEFAULTLIB:\22mylib\22"}
pragma(lib, "mylib");

// CHECK: ![[OPT1]] = !{!"-myflag"}
pragma(linkerDirective, "-myflag");
// CHECK: ![[OPT2]] = !{!"-framework", !"CoreFoundation"}
pragma(linkerDirective, "-framework", "CoreFoundation");
