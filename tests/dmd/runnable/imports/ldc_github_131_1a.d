module imports.ldc_github_131_1a;

struct FilterResult(alias pred, Range) {
    Range _input;
    this(Range r) {
        auto a = pred();
    }
}

struct RefCounted(T) {
    T payload;
    ~this() {}
}

struct DirIteratorImpl {}

struct DirIterator {
    RefCounted!(DirIteratorImpl) impl;
    @property int front(){ return 0; }
}

auto dirEntries() {
    static bool f() { return false; }
    return FilterResult!(f, DirIterator)(DirIterator());
}
