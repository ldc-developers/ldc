// REQUIRES: atleast_llvm1000
// RUN: %ldc -output-mlir -of=%t.mlir %s &&  FileCheck %s < %t.mlir

// CHECK-LABEL: func @_Dmain()
// CHECK: [[VAL_0:%.*]] = "D.int"() {value = 10 : i32} : () -> i32
// CHECK-NEXT: [[VAL_1:%.*]] = "D.int"() {value = 5 : i32} : () -> i32
// CHECK-NEXT: [[VAL_2:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_3:%.*]] = "std.cmpi"([[VAL_0]], [[VAL_1]]) {predicate = 4 : i64} : (i32, i32) -> i1
// CHECK-NEXT: "std.cond_br"([[VAL_3]])[^bb1, ^bb2] : (i1) -> ()
int main(){
  int a = 10;
  int b = 5;
  int c = 0;

// CHECK-NEXT: ^bb1: // pred: ^bb0
// CHECK-NEXT: [[VAL_4:%.*]] = "D.add"([[VAL_1]], [[VAL_0]]) : (i32, i32) -> i32
// CHECK-NEXT: "std.br"()[^bb3] : () -> ()
// CHECK-NEXT: ^bb2: // pred: ^bb0
// CHECK-NEXT: [[VAL_5:%.*]] = "D.add"([[VAL_2]], [[VAL_0]]) : (i32, i32) -> i32
// CHECK-NEXT: "std.br"()[^bb3] : () -> ()
  if(a > b){
    int g = b + a;
  }else{
    int h = c + a;
  }

// CHECK-NEXT: ^bb3: // 2 preds: ^bb1, ^bb2
// CHECK-NEXT: [[VAL_6:%.*]] = "std.cmpi"([[VAL_0]], [[VAL_1]]) {predicate = 0 : i64} : (i32, i32) -> i1
// CHECK-NEXT: "std.cond_br"([[VAL_6]])[^bb4, ^bb5] : (i1) -> ()
// CHECK-NEXT: ^bb4: // pred: ^bb3
// CHECK-NEXT: [[VAL_7:%.*]] = "D.int"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_8:%.*]] = "D.add"([[VAL_0]], [[VAL_7]]) : (i32, i32) -> i32
// CHECK-NEXT: "std.br"()[^bb6] : () -> ()
// CHECK-NEXT: ^bb5: // pred: ^bb3
// CHECK-NEXT: [[VAL_9:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_10:%.*]] = "std.cmpi"([[VAL_0]], [[VAL_9]]) {predicate = 1 : i64} : (i32, i32) -> i1
// CHECK-NEXT: "std.cond_br"([[VAL_10]])[^bb7, ^bb8] : (i1) -> ()
  if(a == b)
    a++;
  else if(a != 0)
    b--;

// CHECK-NEXT: ^bb6: // 2 preds: ^bb4, ^bb8
// CHECK-NEXT: [[VAL_11:%.*]] = "std.cmpi"([[VAL_0]], [[VAL_1]]) {predicate = 0 : i64} : (i32, i32) -> i1
// CHECK-NEXT: "std.cond_br"([[VAL_11]])[^bb9, ^bb10] : (i1) -> ()
// CHECK-NEXT: ^bb7: // pred: ^bb5
// CHECK-NEXT: [[VAL_12:%.*]] = "D.int"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_13:%.*]] = "D.sub"([[VAL_1]], [[VAL_12]]) : (i32, i32) -> i32
// CHECK-NEXT: "std.br"()[^bb8] : () -> ()
// CHECK-NEXT: ^bb8: // 2 preds: ^bb5, ^bb7
// CHECK-NEXT: "std.br"()[^bb6] : () -> ()
// CHECK-NEXT: ^bb9: // pred: ^bb6
// CHECK-NEXT: [[VAL_14:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_15:%.*]] = "std.cmpi"([[VAL_0]], [[VAL_14]]) {predicate = 0 : i64} : (i32, i32) -> i1
// CHECK-NEXT: "std.cond_br"([[VAL_15]])[^bb11, ^bb12] : (i1) -> ()
// CHECK-NEXT: ^bb10:  // 2 preds: ^bb6, ^bb12
// CHECK-NEXT: [[VAL_16:%.*]] = "D.int"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_17:%.*]] = "D.add"([[VAL_1]], [[VAL_16]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_18:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_19:%.*]] = "std.return"([[VAL_18]]) : (i32) -> i32
// CHECK-NEXT: ^bb11:  // pred: ^bb9
// CHECK-NEXT: [[VAL_20:%.*]] = "D.int"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_21:%.*]] = "D.add"([[VAL_0]], [[VAL_20]]) : (i32, i32) -> i32
// CHECK-NEXT: "std.br"()[^bb12] : () -> ()
// CHECK-NEXT: ^bb12:  // 2 preds: ^bb9, ^bb11
// CHECK-NEXT: "std.br"()[^bb10] : () -> ()
  if(a == b)
    if(a == 0)
      a++;
    b++;

return 0;
}
