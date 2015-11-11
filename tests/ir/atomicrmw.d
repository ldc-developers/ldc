// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s --check-prefix LLVM < %t.ll

import core.atomic;

void main() {
  shared ubyte x = 3;
  ubyte r;

  r = atomicOp!"+="(x, uint(257));
  assert(x == r);
  // LLVM: = atomicrmw add i8*

  r = atomicOp!"+="(x, int(-263));
  assert(x == r);
  // LLVM: = atomicrmw add i8*

  r = atomicOp!"-="(x, ushort(257));
  assert(x == r);
  // LLVM: = atomicrmw sub i8*

  r = atomicOp!"-="(x, short(-263));
  assert(x == r);
  // LLVM: = atomicrmw sub i8*

  r = atomicOp!"&="(x, ubyte(255));
  assert(x == r);
  // LLVM: = atomicrmw and i8*

  r = atomicOp!"|="(x, short(3));
  assert(x == r);
  // LLVM: = atomicrmw or i8*

  r = atomicOp!"^="(x, int(3));
  assert(x == r);
  // LLVM: = atomicrmw xor i8*

  r = atomicOp!"+="(x, 1.0f);
  assert(x == r);
  // LLVM: = cmpxchg i8*
}
