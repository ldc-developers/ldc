// Test @ldc.attributes.llvmFastMathFlag UDA

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll
// RUN: not %ldc -c -w -d-version=WARNING %s 2>&1 | FileCheck %s --check-prefix WARNING

import ldc.attributes;

version(WARNING)
{
    // WARNING: attr_llvmFMF.d(11): Warning: ignoring unrecognized flag parameter `unrecognized` for `@ldc.attributes.llvmFastMathFlag`
    @llvmFastMathFlag("unrecognized")
    void foo() {}
}

// LLVM-LABEL: define{{.*}} @notfast
// LLVM-SAME: #[[ATTR_NOTFAST:[0-9]+]]
extern (C) double notfast(double a, double b)
{
    @llvmFastMathFlag("fast")
    double nested_fast(double a, double b)
    {
        return a * b;
    }

// LLVM: fmul double
    return a * b;
}
// LLVM-LABEL: define{{.*}} @{{.*}}nested_fast
// LLVM: fmul fast double

// LLVM-LABEL: define{{.*}} @{{.*}}nnan_arcp
@llvmFastMathFlag("nnan")
@llvmFastMathFlag("arcp")
double nnan_arcp(double a, double b)
{
// LLVM: fmul nnan arcp double
    return a * b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}ninf_nsz
@llvmFastMathFlag("ninf")
@llvmFastMathFlag("nsz")
double ninf_nsz(double a, double b)
{
// LLVM: fmul ninf nsz double
    return a * b;
}

// LLVM-LABEL: define{{.*}} @{{.*}}cleared
@llvmFastMathFlag("ninf")
@llvmFastMathFlag("clear")
double cleared(double a, double b)
{
// LLVM: fmul double
    return a * b;
}

