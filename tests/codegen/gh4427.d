// RUN: %ldc -run %s

import core.stdc.stdio: printf;
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
    printf("%d\n", i);
    assert(i == 2);
}
