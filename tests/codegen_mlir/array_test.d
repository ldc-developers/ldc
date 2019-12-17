// REQUIRES: atleast_llvm1000
// RUN: %ldc -output-mlir -of=%t.mlir %s &&  FileCheck %s < %t.mlir
int main(){
  int[5] array = [1,2,3,4,5];
  float[] arr2 = [5.9, 2.6, 1.4, 10.4];
  double[] arr3 = [4.76, 34, 243.6, 918.908];
return 0;
}

// CHECK-LABEL: func @_Dmain()
// CHECK: [[VAL_0:%.*]] = "D.int"() {value = dense<[1, 2, 3, 4, 5]> : tensor<5xi32>} : () -> tensor<5xi32>
// CHECK-NEXT: [[VAL_1:%.*]] = "D.float"() {value = dense<[5.900000e+00, 2.600000e+00, 1.400000e+00, 1.040000e+01]> : tensor<4xf32>} : () -> tensor<4xf32> 
// CHECK-NEXT: [[VAL_2:%.*]] = "D.double"() {value = dense<[4.760000e+00, 3.400000e+01, 2.436000e+02, 9.189080e+02]> : tensor<4xf64>} : () -> tensor<4xf64>
// CHECK-NEXT: [[VAL_3:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_4:%.*]] = "std.return"([[VAL_3]]) : (i32) -> i32
