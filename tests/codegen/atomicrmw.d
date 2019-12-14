// RUN: %ldc -c -de -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import core.atomic;

void main() {
  shared ubyte x = 3;
  ubyte r;

  r = atomicOp!"+="(x, uint(257));
  assert(x == r);
  // CHECK: = atomicrmw add i8*

  r = atomicOp!"+="(x, int(-263));
  assert(x == r);
  // CHECK: = atomicrmw add i8*

  r = atomicOp!"-="(x, ushort(257));
  assert(x == r);
  // CHECK: = atomicrmw sub i8*

  r = atomicOp!"-="(x, short(-263));
  assert(x == r);
  // CHECK: = atomicrmw sub i8*

  r = atomicOp!"&="(x, ubyte(255));
  assert(x == r);
  // CHECK: = atomicrmw and i8*

  r = atomicOp!"|="(x, short(3));
  assert(x == r);
  // CHECK: = atomicrmw or i8*

  r = atomicOp!"^="(x, int(3));
  assert(x == r);
  // CHECK: = atomicrmw xor i8*

  r = atomicOp!"+="(x, 1.0f);
  assert(x == r);
  // CHECK: = cmpxchg weak i8*
}
