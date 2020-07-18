// RUN: %ldc -mtriple=x86_64-pc-windows-msvc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// REQUIRES: target_X86

// CHECK: !llvm.linker.options = !{!0, !1, !2}

// CHECK: !0 = !{!"/DEFAULTLIB:\22mylib\22"}
pragma(lib, "mylib");

// CHECK: !1 = !{!"-myflag"}
pragma(linkerDirective, "-myflag");
// CHECK: !2 = !{!"-framework", !"CoreFoundation"}
pragma(linkerDirective, "-framework", "CoreFoundation");
