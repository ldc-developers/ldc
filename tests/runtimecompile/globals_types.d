
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

struct Foo
{
  int i;
  float f;
  void* p;
  int[3] a;
  int[] da;
}

struct Bar
{
  Foo f;
  Foo* pf;
  Bar* pb;
}

@dynamicCompileConst
{
__gshared byte  i8  = 42 + 1;
__gshared short i16 = 42 + 2;
__gshared int   i32 = 42 + 3;
__gshared long  i64 = 42 + 4;

__gshared ubyte  u8  = 42 + 5;
__gshared ushort u16 = 42 + 6;
__gshared uint   u32 = 42 + 7;
__gshared ulong  u64 = 42 + 8;

__gshared float  f32 = 42 + 9;
__gshared double f64 = 42 + 10;

__gshared void* ptr = cast(void*)(42 + 11);

__gshared int[3] arr = [42 + 12,42 + 13,42 + 14];
__gshared int[] darr = [42 + 15,42 + 16,42 + 17,42 + 18];

__gshared Foo foo = Foo(42 + 19,42 + 20,cast(void*)(42 + 21),[42 + 22,42 + 23,42 + 24],[42 + 25,42 + 26,42 + 27,42 + 28]);
__gshared Bar bar = Bar(Foo(42 + 19,42 + 20,cast(void*)(42 + 21),[42 + 22,42 + 23,42 + 24],[42 + 25,42 + 26,42 + 27,42 + 28]), cast(Foo*)(42 + 29), cast(Bar*)(42 + 30));
}

@dynamicCompile
{
byte  foo_i8()  { return i8; }
short foo_i16() { return i16; }
int   foo_i32() { return i32; }
long  foo_i64() { return i64; }

ubyte  foo_u8()  { return u8; }
ushort foo_u16() { return u16; }
uint   foo_u32() { return u32; }
ulong  foo_u64() { return u64; }

float  foo_f32() { return f32; }
double foo_f64() { return f64; }

void* foo_ptr() { return ptr; }

int[3]  foo_arr()  { return arr; }
int[]   foo_darr() { return darr; }

Foo foo_foo() { return foo; }
Bar foo_bar() { return bar; }
}

void test(T,F)(ref T val, F fun)
{
  assert(val == fun());
}

void main(string[] args)
{
  compileDynamicCode();

  test(i8,  &foo_i8);
  test(i16, &foo_i16);
  test(i32, &foo_i32);
  test(i64, &foo_i64);

  test(u8,  &foo_u8);
  test(u16, &foo_u16);
  test(u32, &foo_u32);
  test(u64, &foo_u64);

  test(f32, &foo_f32);
  test(f64, &foo_f64);

  assert(ptr is foo_ptr());

  test(arr, &foo_arr);
  test(darr, &foo_darr);

  test(foo, &foo_foo);
  test(bar, &foo_bar);
}
