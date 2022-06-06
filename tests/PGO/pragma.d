// Test pragma to turn on/off PGO instrumentation

// RUN: %ldc -w -c -o- %s
// RUN: %ldc -w -c -output-ll -fprofile-instr-generate -of=%t.ll %s && FileCheck %s < %t.ll

// disable name mangling
extern(C):

// CHECK-NOT: @__llvm_profile{{.*}}not_instrumented
// CHECK-NOT: @__prof{{.*}}not_instrumented

// CHECK-DAG: @__{{(llvm_profile_counters|profc)}}_instrumented ={{.*}} global [1 x i64] zeroinitializer
void instrumented() {}

void not_instrumented() {
    pragma(LDC_profile_instr, false);
}


pragma(LDC_profile_instr, false) {

    void not_instrumented_2() {}

    // CHECK-DAG: @__{{(llvm_profile_counters|profc)}}_instrumented2_override ={{.*}} global [1 x i64] zeroinitializer
    void instrumented2_override() {
        pragma(LDC_profile_instr, true);
    }


    // CHECK-DAG: @__{{(llvm_profile_counters|profc).*}}_{{.*}}instrumented_template{{.*}} global [1 x i64] zeroinitializer
    void instrumented_template(T)(T i) {
        pragma(LDC_profile_instr, true);
    }
    void not_instrumented_template(T)(T i) {}

    pragma(LDC_profile_instr, true):
    // CHECK-DAG: @__{{(llvm_profile_counters|profc)}}_{{.*}}instantiate_templates{{.*}} global [1 x i64] zeroinitializer
    void instantiate_templates() {
        not_instrumented_template(1);
        instrumented_template(1);
    }

} // pragma(LDC_profile_instr, false)

// CHECK-DAG: @__{{(llvm_profile_counters|profc)}}_instrumented_two ={{.*}} global [1 x i64] zeroinitializer
void instrumented_two() {}

pragma(LDC_profile_instr, false)
struct Strukt {
    void not_instrumented() {}

    // CHECK-DAG: @__{{(llvm_profile_counters|profc).*}}_{{.*}}Strukt{{.*}}instrumented_method{{.*}} global [1 x i64] zeroinitializer
    void instrumented_method() {
        pragma(LDC_profile_instr, true);
    }
}

// CHECK-DAG: @__{{(llvm_profile_counters|profc)}}_instrumented_three ={{.*}} global [1 x i64] zeroinitializer
void instrumented_three() {}
