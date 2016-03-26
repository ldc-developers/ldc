// RUN: not %ldc %s -c -o- 2>&1 | FileCheck %s

class A {}

void foo()
{
  {
    A a;
// CHECK-NOT: immutable_synchronize.d(10):
    synchronized(a) {}
  }
  {
    const A a;
// CHECK: immutable_synchronize.d(15): Error: {{.*}} synchronize on a mutable object, not on 'a' of type 'const(A)'
    synchronized(a) {}
  }
  {
    immutable A a;
// CHECK: immutable_synchronize.d(20): Error: {{.*}} synchronize on a mutable object, not on 'a' of type 'immutable(A)'
    synchronized(a) {}
  }
}
