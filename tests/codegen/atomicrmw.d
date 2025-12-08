// RUN: %ldc -c -de -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import core.atomic;

void main() {
  shared ubyte x = 3;
  ubyte r;

  r = atomicOp!"+="(x, uint(257));
  assert(x == r);
  // CHECK: = atomicrmw add ptr

  r = atomicOp!"+="(x, int(-263));
  assert(x == r);
  // CHECK: = atomicrmw add ptr

  r = atomicOp!"-="(x, ushort(257));
  assert(x == r);
  // CHECK: = atomicrmw sub ptr

  r = atomicOp!"-="(x, short(-263));
  assert(x == r);
  // CHECK: = atomicrmw sub ptr

  r = atomicOp!"&="(x, ubyte(255));
  assert(x == r);
  // CHECK: = atomicrmw and ptr

  r = atomicOp!"|="(x, short(3));
  assert(x == r);
  // CHECK: = atomicrmw or ptr

  r = atomicOp!"^="(x, int(3));
  assert(x == r);
  // CHECK: = atomicrmw xor ptr

  {
    shared float x2 = 2.5f;
    const r2 = atomicOp!"+="(x2, 1.0f);
    assert(x2 == r2);
    // CHECK: = cmpxchg weak ptr
  }
}
