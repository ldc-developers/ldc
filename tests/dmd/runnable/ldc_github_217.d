import imports.ldc_github_217a;

struct A {
    bool foo()  {
        return true;
    }

    void bar()  {
        B!((i) { return foo(); })(0);
    }
}

void main() {
}
