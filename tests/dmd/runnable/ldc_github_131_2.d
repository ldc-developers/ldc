// EXTRA_SOURCES: imports/ldc_github_131_2a.d
// Note: Crash was dependent on the command line source file order.
import imports.ldc_github_131_2a;

struct DirIterator {}

auto dirEntries() {
    void f() {}
    return filter!f(DirIterator());
}

void main() {}
