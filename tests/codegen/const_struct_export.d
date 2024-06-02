// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK:     @.immutablearray{{.*}} = internal constant [2 x ptr] {{.*}}exportedFunction
// CHECK-NOT: @.immutablearray{{.*}} [2 x ptr] {{.*}}importedFunction
// CHECK:     @.immutablearray{{.*}} = internal constant [2 x ptr] {{.*}}exportedVariable
// CHECK-NOT: @.immutablearray{{.*}} [2 x ptr] {{.*}}importedVariable

export void exportedFunction() {}
export void importedFunction();
export immutable int exportedVariable = 1;
export extern immutable int importedVariable;

void foo () {
    immutable auto exportedFuncs = [ &exportedFunction, &exportedFunction ];
    immutable auto importedFuncs = [ &importedFunction, &importedFunction ];
    // CHECK: store ptr @{{.*}}D19const_struct_export16importedFunctionFZv
    immutable auto exportedVars = [ &exportedVariable, &exportedVariable ];
    immutable auto importedVars = [ &importedVariable, &importedVariable ];
    // CHECK: store ptr @{{.*}}D19const_struct_export16importedVariable
}
