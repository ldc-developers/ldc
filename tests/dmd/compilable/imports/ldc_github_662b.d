module imports.ldc_github_662b;

import imports.ldc_github_662c;
import imports.ldc_github_662d;

import std.range;

class Font {
    mixin RCounted;

    auto makeTextData(string s) {
        // split text by spaces
        auto arr = s.splitter.array;
    }
}

class Engine {
    RC!Font font;
}
