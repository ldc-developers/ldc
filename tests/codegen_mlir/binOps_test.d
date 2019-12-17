// REQUIRES: atleast_llvm1000
// RUN: %ldc -output-mlir -of=%t.mlir %s &&  FileCheck %s < %t.mlir 
int main(){
  int a = 10;
  float b = 5;
  bool c = 1;
  double d = 9.924809859032987496;
  a += b;
  d += a;
  a = a * cast(int)b;
  b = b*d;
  a = a / cast(int)b;
  b = d / d;
  int e = a - 1;
  int f = a % e;
  int g = 0;
  int h = a & g;
  a++;
  int i = cast(int)b | g;
  bool k = !1;
  int j = c ^ g;
  b--;
  a+=1;
  b-=2;
  d/=4;
  e&=g;
  f|=g;
  g^=g;
  d = d / b;

// CHECK-LABEL: func @_Dmain() 
// CHECK: [[VAL_0:%.*]] = "D.int"() {value = 10 : i32} : () -> i32
// CHECK-NEXT: [[VAL_1:%.*]] = "D.float"() {value = 5.000000e+00 : f32} : () -> f32
// CHECK-NEXT: [[VAL_2:%.*]] = "D.int"() {value = 1 : i1} : () -> i1
// CHECK-NEXT: [[VAL_3:%.*]] = "D.double"() {value = 9.9248098590329867 : f64} : () -> f64
// CHECK-NEXT: [[VAL_4:%.*]] = "D.cast"([[VAL_0]]) : (i32) -> f32
// CHECK-NEXT: [[VAL_5:%.*]] = "D.fadd"([[VAL_4]], [[VAL_1]]) : (f32, f32) -> f32
// CHECK-NEXT: [[VAL_6:%.*]] = "D.cast"([[VAL_0]]) : (i32) -> f64
// CHECK-NEXT: [[VAL_7:%.*]] = "D.fadd"([[VAL_3]], [[VAL_6]]) : (f64, f64) -> f64
// CHECK-NEXT: [[VAL_8:%.*]] = "D.cast"([[VAL_1]]) : (f32) -> i32
// CHECK-NEXT: [[VAL_9:%.*]] = "D.mul"([[VAL_0]], [[VAL_8]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_10:%.*]] = "D.cast"([[VAL_1]]) : (f32) -> f64
// CHECK-NEXT: [[VAL_11:%.*]] = "D.fmul"([[VAL_10]], [[VAL_3]]) : (f64, f64) -> f64
// CHECK-NEXT: [[VAL_12:%.*]] = "D.cast"([[VAL_11]]) : (f64) -> f32
// CHECK-NEXT: [[VAL_13:%.*]] = "D.cast"([[VAL_1]]) : (f32) -> i32
// CHECK-NEXT: [[VAL_14:%.*]] = "D.sdiv"([[VAL_0]], [[VAL_13]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_15:%.*]] = "D.fdiv"([[VAL_3]], [[VAL_3]]) : (f64, f64) -> f64
// CHECK-NEXT: [[VAL_16:%.*]] = "D.cast"([[VAL_15]]) : (f64) -> f32
// CHECK-NEXT: [[VAL_17:%.*]] = "D.int"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_18:%.*]] = "D.sub"([[VAL_0]], [[VAL_17]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_19:%.*]] = "D.srem"([[VAL_0]], [[VAL_18]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_20:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_21:%.*]] = "D.and"([[VAL_0]], [[VAL_20]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_22:%.*]] = "D.int"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_23:%.*]] = "D.add"([[VAL_0]], [[VAL_22]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_24:%.*]] = "D.cast"([[VAL_1]]) : (f32) -> i32
// CHECK-NEXT: [[VAL_25:%.*]] = "D.or"([[VAL_24]], [[VAL_20]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_26:%.*]] = "D.int"() {value = 0 : i1} : () -> i1
// CHECK-NEXT: [[VAL_27:%.*]] = "D.cast"([[VAL_2]]) : (i1) -> i32
// CHECK-NEXT: [[VAL_28:%.*]] = "D.xor"([[VAL_27]], [[VAL_20]]) : (i32, i32) -> i32 
// CHECK-NEXT: [[VAL_29:%.*]] = "D.float"() {value = 1.000000e+00 : f32} : () -> f32
// CHECK-NEXT: [[VAL_30:%.*]] = "D.fsub"([[VAL_1]], [[VAL_29]]) : (f32, f32) -> f32
// CHECK-NEXT: [[VAL_31:%.*]] = "D.int"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_32:%.*]] = "D.add"([[VAL_0]], [[VAL_31]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_33:%.*]] = "D.float"() {value = 2.000000e+00 : f32} : () -> f32
// CHECK-NEXT: [[VAL_34:%.*]] = "D.fsub"([[VAL_1]], [[VAL_33]]) : (f32, f32) -> f32 
// CHECK-NEXT: [[VAL_35:%.*]] = "D.double"() {value = 4.000000e+00 : f64} : () -> f64
// CHECK-NEXT: [[VAL_36:%.*]] = "D.fdiv"([[VAL_3]], [[VAL_35]]) : (f64, f64) -> f64
// CHECK-NEXT: [[VAL_37:%.*]] = "D.and"([[VAL_18]], [[VAL_20]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_38:%.*]] = "D.or"([[VAL_19]], [[VAL_20]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_39:%.*]] = "D.xor"([[VAL_20]], [[VAL_20]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_40:%.*]] = "D.cast"([[VAL_1]]) : (f32) -> f64
// CHECK-NEXT: [[VAL_41:%.*]] = "D.fdiv"([[VAL_3]], [[VAL_40]]) : (f64, f64) -> f64
// CHECK-NEXT: [[VAL_42:%.*]] = "D.int"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_43:%.*]] = "std.return"([[VAL_42]]) : (i32) -> i32
return 0;
}
