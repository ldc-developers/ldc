// RUN: %ldc -c %s

struct Foo {}

void main() {
    Foo f;
    auto x = cast(ubyte[Foo.sizeof])f;
}
