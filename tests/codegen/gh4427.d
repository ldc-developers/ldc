// RUN: %ldc -run %s

alias AliasSeq(TList...) = TList;

int i = 0;
struct A {
    ~this() {
        i++;
    }
}

void main() {
    {
        AliasSeq!(A, A) params;
    }

    assert(i == 2);
}
