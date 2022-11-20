// ICE with unreachable code after multiple returns on MSVC.
void dummy(T)(T t) {}

void foo() {
    try {
        return;
        return;
        dummy("foo bar");
    } catch (Throwable) {}
}
