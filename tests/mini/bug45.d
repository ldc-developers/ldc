module bug45;

void foo() {
    int bar;
    scope(exit) { bar++; }
    if (bar) return;
}

void main() {
    foo();
}
