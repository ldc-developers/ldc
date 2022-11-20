import std.algorithm : all, canFind;

struct Foo {
    float a, b, c;

    auto opDispatch(string swiz)() if (swiz.all!(x => "abc".canFind(x))) {
        return mixin("Foo(" ~ swiz[0] ~ "," ~ swiz[1] ~ "," ~ swiz[2] ~ ")");
    }
}

auto f(Foo f) {
    return f.cab;
}
