// REQUIRES: atleast_llvm1000
// RUN: %ldc -output-mlir -of=%t.mlir %s &&  FileCheck %s < %t.mlir


// CHECK-LABEL: func @_D9loop_test4callFZi() 
// CHECK: [[VAL_0:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_1:%.*]] = "std.return"([[VAL_0]]) : (i32) -> i32
int call(){
  return 0;
}

int main(){
  for(int i = 0; i < 10; i++){
    call();
  }
return 0;
}

// CHECK-LABEL: func @_Dmain()
// CHECK: [[VAL_0:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: "std.br"()[^bb4] : () -> () 
// CHECK-NEXT: ^bb1: // pred: ^bb2
// CHECK-NEXT: [[VAL_1:%.*]] = "D.int"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_2:%.*]] = "D.add"([[VAL_0]], [[VAL_1]]) : (i32, i32) -> i32
// CHECK-NEXT: "std.br"()[^bb4] : () -> ()
// CHECK-NEXT: ^bb2: // pred: ^bb4
// CHECK-NEXT: [[VAL_3:%.*]] = "D.call"() {callee = @_D9loop_test4callFZi} : () -> i32
// CHECK-NEXT:"std.br"()[^bb1] : () -> ()
// CHECK-NEXT: ^bb3: // pred: ^bb4
// CHECK-NEXT: [[VAL_4:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_5:%.*]] = "std.return"([[VAL_4]]) : (i32) -> i32
// CHECK-NEXT: ^bb4: // 2 preds: ^bb0, ^bb1
// CHECK-NEXT: [[VAL_6:%.*]] = "D.int"() {value = 10 : i32} : () -> i32
// CHECK-NEXT: [[VAL_7:%.*]] = "std.cmpi"([[VAL_0]], [[VAL_6]]) {predicate = 2 : i64} : (i32, i32) -> i1
// CHECK-NEXT: "std.cond_br"([[VAL_7]])[^bb2, ^bb3] : (i1) -> ()
