// Tests that naked lambda functions don't get dllexport with internal linkage.
//
// Lambda functions get internal linkage, and internal linkage is incompatible
// with dllexport storage class. This caused an assertion failure:
// "local linkage requires DefaultStorageClass"
//
// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-windows-msvc -fvisibility=public -c %s --output-ll -of=%t.ll

// The lambda should have internal linkage but NOT dllexport.
// Bug: "define internal dllexport" - invalid IR that fails LLVM verification.
// Use LLVM opt verifier to check for invalid IR.
// RUN: opt -passes=verify -S %t.ll -o /dev/null

module naked_lambda_no_dllexport;

void caller() {
    // Function literal (lambda) with naked asm gets internal linkage.
    // With -fvisibility=public, this must NOT get dllexport.
    auto nakedLambda = () {
        asm {
            naked;
            xor EAX, EAX;
            ret;
        }
    };

    nakedLambda();
}
