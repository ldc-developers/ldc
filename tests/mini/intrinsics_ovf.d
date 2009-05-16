module intrinsics_ovf;


//version = PRINTF;

version(PRINTF)
    extern(C) int printf(char*, ...);


import ldc.intrinsics;


int saddo(int a, int b, out bool overflow) {
    auto Result = llvm_sadd_with_overflow(a, b);
    overflow = Result.overflow;
    return Result.result;
}

int uaddo(int a, int b, out bool overflow) {
    auto Result = llvm_uadd_with_overflow(a, b);
    overflow = Result.overflow;
    return Result.result;
}

int smulo(int a, int b, out bool overflow) {
    auto Result = llvm_smul_with_overflow(a, b);
    overflow = Result.overflow;
    return Result.result;
}

/*
uint umulo(uint a, uint b, out bool overflow) {
    auto Result = llvm_umul_with_overflow(a, b);
    overflow = Result.overflow;
    return Result.result;
}
*/

void test(int function(int, int, out bool) fn,
          int a, int b, int result_e, bool ovf_e) {
    version(PRINTF)
        printf("%8x :: %8x :: %8x :: %.*s\n", a, b, result_e,
                (ovf_e ? "true" : "false"));
    
    bool ovf;
    int result = fn(a, b, ovf);
    
    version(PRINTF)
        printf("____________________    %8x :: %.*s\n", result,
                (ovf ? "true" : "false"));
    
    assert(ovf == ovf_e);
    assert(result == result_e);
}

void main() {
    test(&saddo, int.min, int.min, int.min + int.min, true);
    test(&saddo, int.min, int.max, int.min + int.max, false);
    test(&saddo, 1, int.max, 1 + int.max, true);
    test(&saddo, 1, 2, 3, false);
    test(&saddo, -1, -2, -3, false);
    
    test(&uaddo, 0, uint.max, 0 + uint.max, false);
    test(&uaddo, 1, uint.max, 1 + uint.max, true);
    test(&uaddo, 1, 2, 3, false);

    test(&smulo, int.min, int.min, int.min * int.min, true);
    test(&smulo, int.min, int.max, int.min * int.max, true);
    test(&smulo, int.max, int.max, int.max * int.max, true);
    test(&smulo, 1, int.max, 1 * int.max, false);
    test(&smulo, 2, int.max/2, 2 * (int.max/2), false);
    test(&smulo, 2, int.max/2 + 1, 2 * (int.max/2 + 1), true);
    test(&smulo, 2, int.min/2, 2 * (int.min/2), false);
    test(&smulo, 2, int.min/2 - 1, 2 * (int.min/2 - 1), true);
    test(&smulo, 1, 2, 2, false);
    test(&smulo, -1, -2, 2, false);
}
