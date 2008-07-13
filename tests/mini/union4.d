module union4;

pragma(LLVM_internal, "notypeinfo")
union U {
    struct { float x,y,z; }
    float[3] xyz;
}

void main() {
    const float[3] a = [1f,2,3];
    U u = U(1,2,3);
    assert(u.xyz == a);
    assert(u.x == 1);
    assert(u.y == 2);
    assert(u.z == 3);
}
