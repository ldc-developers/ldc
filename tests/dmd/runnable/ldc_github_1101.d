struct Foo {
    Foo[] bar = [];
    this(const int x) {}
}

void main() {
	Foo f;
	assert(f.bar.length == 0);
}
