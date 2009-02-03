// testcase from bug #199

struct Color {
    uint c;

}

struct Vertex {
    Color c;
}

void main() {
    Color c = {0xffffffff};

    auto v = Vertex(c);

    assert(v.c.c == 0xffffffff);  // fails in LDC
}
