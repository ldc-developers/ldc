bool foo(void delegate() a, void delegate() b) {
    return a < b;
}
