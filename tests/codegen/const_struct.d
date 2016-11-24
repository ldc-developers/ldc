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

// CHECK: @.immutablearray = internal constant [1 x %const_struct.S2] [%const_struct.S2 { [3 x %const_struct.S1] [%const_struct.S1 { %const_struct.S0 { i32 42 } }, %const_struct.S1 { %const_struct.S0 { i32 43 } }, %const_struct.S1 { %const_struct.S0 { i32 44 } }] }] ;
void main () {
    // CHECK: store %const_struct.S2* getelementptr inbounds ({{.*}}[1 x %const_struct.S2]* @.immutablearray, i32 0, i32 0), %const_struct.S2** %2
    immutable S2[] xyz = [ { [ { { 42 } }, { { 43 } }, { { 44 } } ] } ];
    // CHECK: %.gc_mem = call {{{.*}}} @_d_newarrayU(%object.TypeInfo* bitcast (%"typeid(immutable(C0[]))"* @{{.*}} to %object.TypeInfo*), i{{32|64}} 3)
    immutable C0[] zyx = [ { new int(42) }, { null }, { null } ];

    testNested();
}
