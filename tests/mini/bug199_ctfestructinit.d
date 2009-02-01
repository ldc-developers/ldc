struct Color {
    uint c;

}

struct Vertex {
    double x, y;
    Color c;
    static Vertex opCall(double x, double y, Color c) {
        Vertex ret;
        ret.x = x;
        ret.y = y;
        ret.c = c;
        return ret;
    }
}

void main() {
    Color c = {0xffffffff};

    auto v = Vertex(1, 5, c);

    assert(v.x == 1 && v.y == 5); // passes
    assert(v.c.c == 0xffffffff);  // fails in LDC
}