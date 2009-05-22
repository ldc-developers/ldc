module scope_exit_foreach;

void bar(size_t);

long foo(ubyte[] arr) {
    scope(exit) {
        foreach (ref b; arr) {
            bar(b);
        }
    }
    if (arr.length == 3)
        return 0;
    return arr.length;
}
