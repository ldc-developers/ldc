// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// REQUIRES: target_X86

// CHECK: !llvm.dependent-libraries = !{!0}
// CHECK: !0 = !{!"mylib"}
pragma(lib, "mylib");

// silently ignored because not (yet?) embeddable in ELF object file:
pragma(linkerDirective, "-myflag");
pragma(linkerDirective, "-framework", "CoreFoundation");
