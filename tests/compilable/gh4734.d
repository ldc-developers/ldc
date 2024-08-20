// RUN: %ldc -c %s

align(1) struct Item {
    KV v;
    uint i;
}

struct KV {
    align(1) S* s;
    uint k;
}

struct S {
    Table table;
}

struct Table {
    char a;
    Item v;
}
