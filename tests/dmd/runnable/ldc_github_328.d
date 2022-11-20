void bar() {
    scope(exit) { }
    throw new Exception("Enforcement failed");
}

void main() {
    try
        bar();
    catch (Exception)
        {}
}
