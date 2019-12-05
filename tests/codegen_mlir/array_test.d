// RUN: ldc2 -output-mlir -of=%t.mlir %s &&  FileCheck %s < %t.mlir
int main(){
  int[5] array = [1,2,3,4,5];
  int[] arr2 = [5, 2, 1, 10];
return 0;
}

// CHECK-LABEL: func @_Dmain()
// CHECK: [[VAL_0:%.*]] = "ldc.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf64>} : () -> tensor<5xf64>
// CHECK-NEXT: [[VAL_1:%.*]] = "ldc.constant"() {value = dense<[5.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+01]> : tensor<4xf64>} : () -> tensor<4xf64>
// CHECK-NEXT:  [[VAL_2:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: "ldc.return"([[VAL_2]]) : (i32) -> ()

