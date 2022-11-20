/*
TEST_OUTPUT:
---
fail_compilation/ldc_github_1864.d(16): Error: function `ldc_github_1864.templateDecl!(Aggr!(var)).templateDecl` cannot access frame of function `ldc_github_1864.foo`
---
*/

struct Aggr(alias V) {
    alias Value = V;
}

template templateDecl( Vs... ) {
    void templateDecl() {
        foreach (v; Vs)
            static if ( is( v : Aggr!A, alias A ) )
                assert(A.length == 3);
    }
}

alias templateInst(alias V) = templateDecl!( Aggr!V );

void foo() {
    string var = "var";
    templateInst!var();
}
