// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -run %s

struct S0 { uint  x; }
struct S1 { S0    y; this(this) { y.x = 1; } }
struct S2 { S1[3] z; }

struct C0 { int  *x; }

void testNested() {
  int x;
  // The 'x' here is accessed via the nested context pointer
  struct N1 { ~this() { ++x; } }
  struct N0 { N1[3] x; }
  { N0 n; }
  assert(x == 3);
}

// CHECK: @.immutablearray{{.*}} = internal constant [4 x i32]
// CHECK: @.immutablearray{{.*}} = internal constant [2 x float]
// CHECK: @.immutablearray{{.*}} = internal constant [2 x double]
// CHECK: @.immutablearray{{.*}} = internal constant [2 x { i{{32|64}}, ptr }]
// CHECK: @.immutablearray{{.*}} = internal constant [1 x %const_struct.S2]
// CHECK: @.immutablearray{{.*}} = internal constant [2 x ptr] {{.*}}globVar
// CHECK: @.immutablearray{{.*}} = internal constant [2 x ptr] {{.*}}Dmain

void main () {
    // Simple types
    immutable int[] aA     = [ 1, 2, 3, 4 ];
    immutable float[] aB   = [ 3.14, 3.33 ];
    immutable double[] aC  = [ 3.14, 3.33 ];
    immutable string[] aD  = [ "one", "two" ];

    // Complex type
    immutable S2[] aE = [ { [ { { 42 } }, { { 43 } }, { { 44 } } ] } ];
    // Complex type with non-constant initializer
    // CHECK: %.gc_mem = call { i{{32|64}}, ptr } @_d_newarrayU
    // CHECK-SAME: @{{.*}}_D29TypeInfo_yAS12const_struct2C06__initZ
    immutable C0[] aF = [ { new int(42) }, { new int(24) } ];

    // Pointer types
    static immutable int globVar;
    immutable auto globalVariables = [ &globVar, &globVar ];
    immutable auto functionPointers = [ &main, &main ];
    // Pointer arrays with non-const initializer
    immutable int localVar;
    immutable auto locA = [ &localVar, &localVar ];
    // CHECK: %.gc_mem{{.*}} = call { i{{32|64}}, ptr } @_d_newarrayU
    // CHECK-SAME: @{{.*}}_D13TypeInfo_yAPi6__initZ

    testNested();
}
