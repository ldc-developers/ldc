void main() {
    union TestUnion {
        ubyte[20] small;
        ubyte[28] large;
    }

    struct Container {
        TestUnion u;
        byte b;
    }

    Container c;
    c.b = 123;
    assert(*((cast(ubyte*)cast(void*)&c) + Container.b.offsetof) == 123);
}
