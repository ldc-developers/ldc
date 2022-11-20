struct Foo {
    this(int a) {
        val = a;
    }
    int val;
}

void main() {
    auto a = cast(void*)(new Foo(1));
    auto b = cast(Foo*)a;
    assert(b.val == 1);
}
