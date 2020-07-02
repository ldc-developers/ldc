// RUN: %ldc -mtriple=x86_64-apple-darwin -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// REQUIRES: atleast_llvm500, target_X86

// CHECK: !llvm.linker.options = !{!{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}

// CHECK: !{{[0-9]+}} = !{!"-lmylib"}
pragma(lib, "mylib");

// CHECK: !{{[0-9]+}} = !{!"-myflag"}
pragma(linkerDirective, "-myflag");
// CHECK: !{{[0-9]+}} = !{!"-framework", !"CoreFoundation"}
pragma(linkerDirective, "-framework", "CoreFoundation");
