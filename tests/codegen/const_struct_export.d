// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK:     @.immutablearray{{.*}} = internal constant [2 x void ()*] {{.*}}exportedFunction
// CHECK-NOT: @.immutablearray{{.*}} [2 x void ()*] {{.*}}importedFunction
// CHECK:     @.immutablearray{{.*}} = internal constant [2 x i32*] {{.*}}exportedVariable
// CHECK-NOT: @.immutablearray{{.*}} [2 x i32*] {{.*}}importedVariable

export void exportedFunction() {}
export void importedFunction();
export immutable int exportedVariable = 1;
export immutable int importedVariable;

void foo () {
    immutable auto exportedFuncs = [ &exportedFunction, &exportedFunction ];
    immutable auto importedFuncs = [ &importedFunction, &importedFunction ];
    // CHECK: store void ()* @{{.*}}D19const_struct_export16importedFunctionFZv
    immutable auto exportedVars = [ &exportedVariable, &exportedVariable ];
    immutable auto importedVars = [ &importedVariable, &importedVariable ];
    // CHECK: store i32* @{{.*}}D19const_struct_export16importedVariable
}
