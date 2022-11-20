module imports.ldc_github_217a;

struct B(alias pred) {
    this(int r) {
        pred(0);
    }
}
