module bug6;
class wrong { }
void bark(ref wrong s) { s = new wrong; }
void foo(wrong tree) {
    auto old = tree;
    bark(tree);
    assert(old !is tree);
}
void main()
{
    auto w = new wrong;
    auto old = w;
    foo(w);
    assert(w is old);
}
