// Tests that enum members are correctly handled when they show up as VarExp in the AST.

/+
 + Consider `type == EnumerationOfStructs.Colon, type == EnumerationOfStructs.Comma;`
 + This expression currently (front-end v2.071) results in a VarExp for the enum member
 + in the LHS.
 +
 + See DMD issues 16022 and 16100.
 +
 + The problem appears to be that in a comma expression, an enum member appearing in the
 + LHS is not constant folded and a VarExp remains in the AST.
 + The AST for the LHS of that expression is an ExpStatement with AndAndExp's of
 + EqualExp's for every struct field.
 + Because of the missing constant folding, the generated code is verbose: a
 + new struct temporary is created for every struct field comparison, because the EqualExp
 + will still have a DotVarExp into a VarExp (the enumeration member). And we create a new
 + enumeration member temporary for each VarExp.
 + With -O3 it all disappears.
 +/

// RUN: %ldc %s -c -output-ll -of=%t.ll

bool test16022()
{
    enum Type
    {
        Colon,
        Comma
    }

    Type type;
    return type == Type.Colon, type == Type.Comma;
}

bool foo()
{
    struct A
    {
        int i;
        string s;
    }

    enum Foo
    {
        Colon = A(0, "hoi"),
        Comma = A(1, "yoyo")
    }

    Foo type;
    return type == Foo.Colon, type == Foo.Comma;
}
