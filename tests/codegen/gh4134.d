// https://github.com/ldc-developers/ldc/issues/4134
// RUN: %ldc -run %s

int i;

string getString() {
    ++i;
    return "Abc";
}

void main() {
    const r = getString() ~ getString();
    assert(i == 2);
}
