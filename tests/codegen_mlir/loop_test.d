// REQUIRES: atleast_llvm1000
// RUN: %ldc -output-mlir -of=%t.mlir %s &&  FileCheck %s < %t.mlir

int call(){
  return 0;
}

int main(){
  for(int i = 0; i < 10; i++){
     call();
  }

return 0;
}

// CHECK-LABEL: func @_D4loop4callFZi()
// CHECK: [[VAL_0:%.*]] =  "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: "ldc.return"([[VAL_0]]) : (i32) -> ()

// CHECK-LABEL: func @_Dmain()
// CHECK: [[VAL_0:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: "ldc.br"()[^bb4] : () -> ()
// CHECK-NEXT: ^bb1: // pred: ^bb2
// CHECK-NEXT: "ldc.br"()[^bb4] : () -> ()
// CHECK-NEXT: ^bb2: // pred: ^bb4
// CHECK-NEXT: [[VAL_1:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_2:%.*]] = "ldc.call"() {callee = @_D4loop4callFZi} : () -> i32
// CHECK-NEXT:"ldc.br"()[^bb1] : () -> ()
// CHECK-NEXT: ^bb3: // pred: ^bb4
// CHECK-NEXT: [[VAL_3:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: "ldc.return"(%3) : (i32) -> ()
// CHECK-NEXT: [[VAL_4:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: "ldc.return"(%4) : (i32) -> ()
// CHECK-NEXT: ^bb4: // 2 preds: ^bb0, ^bb1
// CHECK-NEXT: [[VAL_5:%.*]] = "ldc.IntegerExp"() {value = 10 : i32} : () -> i32
// CHECK-NEXT: [[VAL_6:%.*]] = "icmp"(%0, %5) {Type = "slt"} : (i32, i32) -> i1
// CHECK-NEXT: "ldc.br"(%6)[^bb2, ^bb3] : (i1) -> ()
