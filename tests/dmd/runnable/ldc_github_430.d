import std.conv;

align(1) struct Foo {
align(1):
  ushort b;
  uint c;
}

void main() {
  ubyte[6] arr = [0x01, 0x01, 0x01, 0x00, 0x00, 0x01];
  Foo f = cast(Foo) arr;

  assert(to!string(f) == "Foo(257, 16777217)");
  assert(f.sizeof == 6);
}
