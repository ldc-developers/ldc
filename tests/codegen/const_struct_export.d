// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK:     @.immutablearray{{.*}} = internal constant [2 x {{void \(\)\*|ptr}}] {{.*}}exportedFunction
// CHECK-NOT: @.immutablearray{{.*}} [2 x {{void \(\)\*|ptr}}] {{.*}}importedFunction
// CHECK:     @.immutablearray{{.*}} = internal constant [2 x {{i32\*|ptr}}] {{.*}}exportedVariable
// CHECK-NOT: @.immutablearray{{.*}} [2 x {{i32\*|ptr}}] {{.*}}importedVariable

export void exportedFunction() {}
export void importedFunction();
export immutable int exportedVariable = 1;
export extern immutable int importedVariable;

void foo () {
    immutable auto exportedFuncs = [ &exportedFunction, &exportedFunction ];
    immutable auto importedFuncs = [ &importedFunction, &importedFunction ];
    // CHECK: store {{void \(\)\*|ptr}} @{{.*}}D19const_struct_export16importedFunctionFZv
    immutable auto exportedVars = [ &exportedVariable, &exportedVariable ];
    immutable auto importedVars = [ &importedVariable, &importedVariable ];
    // CHECK: store {{i32\*|ptr}} @{{.*}}D19const_struct_export16importedVariable
}
