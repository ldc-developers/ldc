module imports.ldc_github_131_2a;

auto filter(alias pred, Range)(Range rs) {
    return FilterResult!(pred, Range)(rs);
}

struct FilterResult(alias pred, Range) {
    this(Range r) {}
}
