// RUN: %ldc -mtriple=x86_64-linux-gnu --relocation-model=static --output-ll -of=%t.ll %s && FileCheck --check-prefix=ELF_STATIC_RELOC %s < %t.ll
// RUN: %ldc -mtriple=x86_64-linux-gnu --relocation-model=pic --output-ll -of=%t.ll %s && FileCheck --check-prefix=ELF_PIC_DEFAULT %s < %t.ll
// RUN: %ldc -mtriple=x86_64-linux-gnu -fvisibility=hidden --relocation-model=pic --output-ll -of=%t.ll %s && FileCheck --check-prefix=ELF_PIC_HIDDEN %s < %t.ll
// RUN: %ldc -mtriple=x86_64-apple-darwin --relocation-model=static --output-ll -of=%t.ll %s && FileCheck --check-prefix=MACHO_STATIC_RELOC %s < %t.ll
// RUN: %ldc -mtriple=x86_64-apple-darwin --relocation-model=pic --output-ll -of=%t.ll %s && FileCheck --check-prefix=MACHO_PIC %s < %t.ll
// RUN: %ldc -mtriple=x86_64-windows-coff -output-ll -of=%t.ll %s && FileCheck --check-prefix=COFF %s < %t.ll

import ldc.attributes : weak;

// ELF_STATIC_RELOC: define{{( dso_local)?}} i32 @{{.*}}3foo
// ELF_PIC_DEFAULT: define i32 @{{.*}}3foo
// ELF_PIC_HIDDEN: define{{( dso_local)?}} hidden i32 @{{.*}}3foo
// MACHO_STATIC_RELOC: define dso_local i32 @{{.*}}3foo
// MACHO_PIC: define dso_local i32 @{{.*}}3foo
// COFF: define dso_local {{.*}} i32 @{{.*}}3foo
private int foo() { return 42; }


// ELF_STATIC_RELOC: define{{( dso_local)?}} i32 @{{.*}}3bar
// ELF_PIC_DEFAULT: define i32 @{{.*}}3bar
// ELF_PIC_HIDDEN: define hidden i32 @{{.*}}3bar
// MACHO_STATIC_RELOC: define dso_local i32 @{{.*}}3bar
// MACHO_PIC: define dso_local i32 @{{.*}}3bar
// COFF: define dso_local {{.*}} i32 @{{.*}}3bar
public int bar() { return foo(); }


// ELF_STATIC_RELOC: define weak{{( dso_local)?}} i32 @{{.*}}3baz
// ELF_PIC_DEFAULT: define weak i32 @{{.*}}3baz
// ELF_PIC_HIDDEN: define weak hidden i32 @{{.*}}3baz
// MACHO_STATIC_RELOC: define weak i32 @{{.*}}3baz
// MACHO_PIC: define weak i32 @{{.*}}3baz
// COFF: define x86_vectorcallcc i32 @{{.*}}3baz
@weak int baz() { return 42; }


version(Windows)
{
// COFF: declare extern_weak void @weakreffunction
pragma(LDC_extern_weak) extern(C) void weakreffunction();

void piyo()
{
    auto a = &weakreffunction;
}
}
