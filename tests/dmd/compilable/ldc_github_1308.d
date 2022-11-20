interface Foo {
    package @property bool baz() { return true; }
}

bool consumer(Foo f) {
    return f.baz();
}
