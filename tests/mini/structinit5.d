struct Vertex {
    uint[1] c;
}

void main() {
    uint[1] c = 0xffffffff;

    auto v = Vertex(c);

    assert(v.c[0] == 0xffffffff);  // fails in LDC
}
