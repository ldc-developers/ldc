// RUN: %ldc -O2 -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -O2 -disable-gc2stack -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix NOOPT < %t.ll

// CHECK: define{{.*foo1}}
int foo1()
{
  // NOOPT: call{{.*}}_d_newarrayT
  // CHECK-NOT: _d_newarrayT
  int[] i = new int[5];
  i[3] = 42;
  // CHECK: ret
  return i[3];
}

// CHECK: define{{.*foo2}}
int foo2()
{
  // NOOPT: call{{.*}}_d_allocmemoryT
  // CHECK-NOT: _d_allocmemoryT
  int* i = new int;
  *i = 42;
  // CHECK: ret
  return *i;
}

// CHECK: define{{.*bar1}}
int[] bar1()
{
  // CHECK: _d_newarrayT
  int[] i = new int[5];
  // CHECK: ret
  return i;
}

// CHECK: define{{.*bar2}}
int* bar2()
{
  // CHECK: _d_allocmemoryT
  int* i = new int;
  // CHECK: ret
  return i;
}

extern void fun(int[]);
extern void fun(int*);

// CHECK: define{{.*baz1}}
int baz1()
{
  // CHECK: _d_newarrayT
  int[] i = new int[5];
  fun(i);
  i[3] = 42;
  // CHECK: ret
  return i[3];
}

// CHECK: define{{.*baz2}}
int baz2()
{
  // CHECK: _d_allocmemoryT
  int* i = new int;
  fun(i);
  *i = 42;
  // CHECK: ret
  return *i;
}

__gshared int[] p1;
__gshared int* p2;

// CHECK: define{{.*bzz1}}
int bzz1()
{
  // CHECK: _d_newarrayT
  int[] i = new int[5];
  p1 = i;
  i[3] = 42;
  // CHECK: ret
  return i[3];
}

// CHECK: define{{.*bzz2}}
int bzz2()
{
  // CHECK: _d_allocmemoryT
  int* i = new int;
  p2 = i;
  *i = 42;
  // CHECK: ret
  return *i;
}
