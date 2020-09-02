// https://github.com/ldc-developers/ldc/issues/3553
// RUN: %ldc -run %s

auto makeDelegate(alias fn)(long l) {
    auto callIt() {
        return fn() * l;
    }
    return &callIt;
}

void main()
{
    int i = 7;
    auto dg = makeDelegate!(() => i)(3);
    assert(dg() == 21);
}
