class Baz {
    this(Bar[] a) {}
}
class Foo {
    Bar[] foo(){ return []; }
}
class Bar {
    Foo bar(){ return null; }
}
