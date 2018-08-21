module inputs.import_a;

static __gshared int a_Glob = 4213;

struct a_sA
{
    static char statChar = 'B';
}

void bar() {
    a_sA.statChar = 'C';
}
