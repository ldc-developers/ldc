// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

// CHECK: define void @_D7asm_gcc5cpuidFZv
void cpuid()
{
    uint max_extended_cpuid;
    // CHECK:      %1 = call i32 asm sideeffect "cpuid", "={eax},{eax},~{ebx},~{ecx},~{edx}"(i32 -2147483648), !srcloc
    // CHECK-NEXT: store i32 %1, ptr %max_extended_cpuid
    asm { "cpuid" : "=eax" (max_extended_cpuid) : "eax" (0x8000_0000) : "ebx", "ecx", "edx"; }
}

// CHECK: define void @_D7asm_gcc14multipleOutputFZv
void multipleOutput()
{
    // CHECK-NEXT: %r = alloca [4 x i32]
    uint[4] r = void;
    // CHECK-NEXT: %1 = getelementptr {{.*}} %r, i32 0, i64 0
    // CHECK-NEXT: %2 = getelementptr {{.*}} %r, i32 0, i64 1
    // CHECK-NEXT: %3 = getelementptr {{.*}} %r, i32 0, i64 2
    // CHECK-NEXT: %4 = getelementptr {{.*}} %r, i32 0, i64 3
    // CHECK-NEXT: %5 = call { i32, i32, i32, i32 } asm sideeffect "cpuid", "={eax},={ebx},={ecx},={edx},{eax}"(i32 2), !srcloc
    // CHECK-NEXT: %6 = extractvalue { i32, i32, i32, i32 } %5, 0
    // CHECK-NEXT: store i32 %6, ptr %1
    // CHECK-NEXT: %7 = extractvalue { i32, i32, i32, i32 } %5, 1
    // CHECK-NEXT: store i32 %7, ptr %2
    // CHECK-NEXT: %8 = extractvalue { i32, i32, i32, i32 } %5, 2
    // CHECK-NEXT: store i32 %8, ptr %3
    // CHECK-NEXT: %9 = extractvalue { i32, i32, i32, i32 } %5, 3
    // CHECK-NEXT: store i32 %9, ptr %4
    asm { "cpuid" : "=eax" (r[0]), "=ebx" (r[1]), "=ecx" (r[2]), "=edx" (r[3]) : "eax" (2); }
}

// CHECK: define void @_D7asm_gcc14indirectOutputFkZv
void indirectOutput(uint eax)
{
    // CHECK-NEXT: %eax = alloca i32
    // CHECK-NEXT: %r = alloca [4 x i32]
    // CHECK-NEXT: store i32 %eax_arg, ptr %eax
    uint[4] r = void;
    // CHECK-NEXT: %1 = load i32, ptr %eax
    // CHECK-NEXT: call void asm sideeffect "cpuid
    // CHECK-SAME: "=*m,{eax},~{eax},~{ebx},~{ecx},~{edx}"(ptr elementtype([4 x i32]) %r, i32 %1), !srcloc
    asm
    {
        `cpuid
         movl %%eax,   %0
         movl %%ebx,  4%0
         movl %%ecx,  8%0
         movl %%edx, 12%0`
        : "=m" (r)
        : "eax" (eax)
        : "eax", "ebx", "ecx", "edx";
    }
}

// CHECK: define void @_D7asm_gcc13indirectInputFkZv
void indirectInput(uint eax)
{
    // CHECK-NEXT: %eax = alloca i32
    // CHECK-NEXT: store i32 %eax_arg, ptr %eax
    // CHECK-NEXT: call void asm sideeffect "movl %eax, $0", "*m,~{eax}"(ptr elementtype(i32) %eax), !srcloc
    asm { "movl %%eax, %0" : : "m" (eax) : "eax"; }
}

// CHECK: define void @_D7asm_gcc15specialNamesX86FZv
void specialNamesX86()
{
    byte b;
    short s;
    int i;
    long l;
    // CHECK: = call { i8, i16, i32, i64 } asm sideeffect "nop", "={ax},={bx},={cx},={dx},{si},{di}"(i16 %1, i64 2), !srcloc
    asm { "nop" : "=a" (b), "=b" (s), "=c" (i), "=d" (l) : "S" (short(1)), "D" (2L); }
}
