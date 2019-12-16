 // REQUIRES: atleast_llvm1000
 // RUN: %ldc -output-mlir -of=%t.mlir %s &&  FileCheck %s < %t.mlir

int main(){
  int a = 10;
  int b = 5;
  int c = 0;
  if(a > b){
    int g = b + a;
  }else{
    int h = c + a;
  }

  if(a == b)
    a++;
  else if(a != 0)
    b--;

  if(a == b)
    if(a == 0)
      a++;
    b++;

return 0;
}

// CHECK-LABEL: func @_Dmain()
// CHECK: [[VAL_0:%.*]] = "ldc.IntegerExp"() {value = 10 : i32} : () -> i32
// CHECK-NEXT: [[VAL_1:%.*]] = "ldc.IntegerExp"() {value = 5 : i32} : () -> i32
// CHECK-NEXT: [[VAL_2:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_3:%.*]] = "icmp"([[VAL_0]], [[VAL_1]]) {Type = "sgt"} : (i32, i32) -> i1
// CHECK-NEXT: "ldc.if"([[VAL_3]])[^bb1, ^bb2] : (i1) -> ()
// CHECK-NEXT: ^bb1: // pred: ^bb0
// CHECK-NEXT: [[VAL_4:%.*]] = "D.addi"([[VAL_1]], [[VAL_0]]) : (i32, i32) -> i32
// CHECK-NEXT: "ldc.br"()[^bb3] : () -> ()
// CHECK-NEXT: ^bb2: // pred: ^bb0
// CHECK-NEXT: [[VAL_5:%.*]] = "D.addi"([[VAL_2]], [[VAL_0]]) : (i32, i32) -> i32
// CHECK-NEXT: "ldc.br"()[^bb3] : () -> ()
// CHECK-NEXT: ^bb3: // 2 preds: ^bb1, ^bb2
// CHECK-NEXT: [[VAL_6:%.*]] = "icmp"([[VAL_0]], [[VAL_1]]) {Type = "eq"} : (i32, i32) -> i1
// CHECK-NEXT: "ldc.if"([[VAL_6]])[^bb4, ^bb5] : (i1) -> ()
// CHECK-NEXT: ^bb4: // pred: ^bb3
// CHECK-NEXT: [[VAL_7:%.*]] = "ldc.Plusplus"([[VAL_0]]) {value = 1 : i16} : (i32) -> i32
// CHECK-NEXT: "ldc.br"()[^bb6] : () -> ()
// CHECK-NEXT: ^bb5: // pred: ^bb3
// CHECK-NEXT: [[VAL_8:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_9:%.*]] = "icmp"([[VAL_0]], [[VAL_8]]) {Type = "neq"} : (i32, i32) -> i1
// CHECK-NEXT: "ldc.if"([[VAL_9]])[^bb7, ^bb8] : (i1) -> ()
// CHECK-NEXT: ^bb6: // 2 preds: ^bb4, ^bb8
// CHECK-NEXT: [[VAL_10:%.*]] = "icmp"([[VAL_0]], [[VAL_1]]) {Type = "eq"} : (i32, i32) -> i1
// CHECK-NEXT: "ldc.if"([[VAL_10]])[^bb9, ^bb10] : (i1) -> ()
// CHECK-NEXT: ^bb7: // pred: ^bb5
// CHECK-NEXT: [[VAL_11:%.*]] = "ldc.MinusMinus"([[VAL_1]]) {value = 1 : i16} : (i32) -> i32
// CHECK-NEXT: "ldc.br"()[^bb8] : () -> ()
// CHECK-NEXT: ^bb8: // 2 preds: ^bb5, ^bb7
// CHECK-NEXT: "ldc.br"()[^bb6] : () -> ()
// CHECK-NEXT: ^bb9: // pred: ^bb6
// CHECK-NEXT: "ldc.br"()[^bb10] : () -> ()
// CHECK-NEXT: ^bb10:  // 2 preds: ^bb6, ^bb9
// CHECK-NEXT: [[VAL_12:%.*]] = "ldc.Plusplus"([[VAL_1]]) {value = 1 : i16} : (i32) -> i32
// CHECK-NEXT: [[VAL_13:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: "ldc.return"([[VAL_13]]) : (i32) -> ()
