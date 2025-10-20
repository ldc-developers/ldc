// RUN: %ldc -c %s

struct Foo {}

struct Bar
{
    int[4] v;
}

void main() {
    Foo f;
    auto x = cast(ubyte[Foo.sizeof])f;

    Bar b;
    auto y = cast(__vector(int[4]))b;
}
