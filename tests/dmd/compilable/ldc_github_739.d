// EXTRA_SOURCES: imports/ldc_github_739b.d
module ldc_github_739;

import imports.ldc_github_739a;

template map(fun...) {
    auto map(Range)(Range) {
        return MapResult!(fun, Range)();
    }
}

struct MapResult(alias fun, R) {
    R _input;
    MapResult opSlice() {
        return MapResult();
    }
}
