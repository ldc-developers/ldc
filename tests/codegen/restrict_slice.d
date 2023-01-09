// RUN: %ldc %s -c -output-ll | FileCheck
import ldc.attributes;

//CHECK: define void @_D4testQfFAiZv({{i(64|32)}} %a_arg_len, {{i32\*|ptr}} noalias %a_arg_ptr)
//CHECK-NEXT: %a = alloca {{{i(64|32)}}, {{i32\*|ptr}}} align 8
void test(@restrict int[] a) {}

void use()
{
    int[4] x;
    test(x[]);
}
