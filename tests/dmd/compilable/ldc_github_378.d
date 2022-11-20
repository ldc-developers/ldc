string find(alias pred)(string haystack) {
    for (; !pred(haystack); ) {}
    return haystack;
}

bool any(alias pred)() if (is(typeof(find!pred("")))) {
    return !find!pred("");
}

struct StaticRegex(Char) {
    bool function(BacktrackingMatcher!Char) MatchFn;
}

struct BacktrackingMatcher(Char) {
    StaticRegex!Char re;
    size_t lastState = 0;
}

auto match(RegEx)(RegEx ) {
    return "";
}

auto match(RegEx)(RegEx ) if(is(RegEx == StaticRegex!(typeof({})))) {}

void applyNoRemoveRegex() {
    if (any!((a){return !match(a);})()){}
}
