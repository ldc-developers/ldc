// RUN: ldc2 -output-mlir -of=%t.mlir %s &&  FileCheck %s < %t.mlir
int main(){
  int a = 10;
  float b = 5;
  bool c = 1;
  double d = 9.924809859032987496;
  a += b;
  d += a;
  a = a * cast(int)b;
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
// CHECK: [[VAL_0:%.*]] = "ldc.IntegerExp"() {value = 10 : i32} : () -> i32
// CHECK-NEXT: [[VAL_1:%.*]] = "ldc.RealExp"() {value = 5.000000e+00 : f32} : () -> f32
// CHECK-NEXT: [[VAL_2:%.*]] = "ldc.bool"() {value = 1 : i1} : () -> i1
// CHECK-NEXT: [[VAL_3:%.*]] = "ldc.RealExp"() {value = 9.9248098590329867 : f64} : () -> f64
// CHECK-NEXT: [[VAL_4:%.*]] = "ldc.CastOp"([[VAL_0]]) : (i32) -> f32
// CHECK-NEXT: [[VAL_5:%.*]] = "D.addf"([[VAL_4]], [[VAL_1]]) : (f32, f32) -> f32
// CHECK-NEXT: [[VAL_6:%.*]] = "ldc.CastOp"([[VAL_0]]) : (i32) -> f64
// CHECK-NEXT: [[VAL_7:%.*]] = "D.addf"([[VAL_3]], [[VAL_6]]) : (f64, f64) -> f64
// CHECK-NEXT: [[VAL_8:%.*]] = "ldc.CastOp"([[VAL_1]]) : (f32) -> i32
// CHECK-NEXT: [[VAL_9:%.*]] = "ldc.mul"([[VAL_0]], [[VAL_8]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_10:%.*]] = "ldc.IntegerExp"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_11:%.*]] = "ldc.neg"([[VAL_0]], [[VAL_10]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_12:%.*]] = "ldc.mod"([[VAL_0]], [[VAL_11]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_13:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK-NEXT: [[VAL_14:%.*]] = "ldc.and"([[VAL_0]], [[VAL_13]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_15:%.*]] = "ldc.Plusplus"([[VAL_0]]) {value = 1 : i16} : (i32) -> i32
// CHECK-NEXT: [[VAL_16:%.*]] = "ldc.CastOp"([[VAL_1]]) : (f32) -> i32
// CHECK-NEXT: [[VAL_17:%.*]] = "ldc.or"([[VAL_16]], [[VAL_13]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_18:%.*]] = "ldc.bool"() {value = 0 : i1} : () -> i1
// CHECK-NEXT: [[VAL_19:%.*]] = "ldc.CastOp"([[VAL_2]]) : (i1) -> i32
// CHECK-NEXT: [[VAL_20:%.*]] = "ldc.xor"([[VAL_19]], [[VAL_13]]) : (i32, i32) -> i32 
// CHECK-NEXT: [[VAL_21:%.*]] = "ldc.MinusMinus"([[VAL_1]]) {value = 1 : i16} : (f32) -> f32
// CHECK-NEXT: [[VAL_22:%.*]] = "ldc.IntegerExp"() {value = 1 : i32} : () -> i32
// CHECK-NEXT: [[VAL_23:%.*]] = "D.addi"([[VAL_0]], [[VAL_22]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_24:%.*]] = "ldc.RealExp"() {value = 2.000000e+00 : f32} : () -> f32
// CHECK-NEXT: [[VAL_25:%.*]] = "ldc.neg"([[VAL_1]], [[VAL_24]]) : (f32, f32) -> f32 
// CHECK-NEXT: [[VAL_26:%.*]] = "ldc.RealExp"() {value = 4.000000e+00 : f64} : () -> f64
// CHECK-NEXT: [[VAL_27:%.*]] = "ldc.div"([[VAL_3]], [[VAL_26]]) : (f64, f64) -> f64
// CHECK-NEXT: [[VAL_28:%.*]] = "ldc.and"([[VAL_11]], [[VAL_13]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_29:%.*]] = "ldc.or"([[VAL_12]], [[VAL_13]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_30:%.*]] = "ldc.xor"([[VAL_13]], [[VAL_13]]) : (i32, i32) -> i32
// CHECK-NEXT: [[VAL_31:%.*]] = "ldc.CastOp"([[VAL_1]]) : (f32) -> f64
// CHECK-NEXT: [[VAL_32:%.*]] = "ldc.div"([[VAL_3]], [[VAL_31]]) : (f64, f64) -> f64
// CHECK-NEXT: [[VAL_33:%.*]] = "ldc.IntegerExp"() {value = 0 : i32} : () -> i32
// CHECK: "ldc.return"([[VAL_33]]) : (i32) -> () 
return 0;
}
