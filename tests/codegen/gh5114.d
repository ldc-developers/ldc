module tests.codegen.gh5114;

// During interface contract context setup, a qualifier-only cast for the same
// interface symbol (e.g., `const(I)` -> `I`) must be handled as a repaint/
// bitcast and must not route through dynamic interface cast lowering.
//
// True dynamic interface casts (different interface symbols) are still valid
// and are covered below.
//
// RUN: %ldc -unittest -main -run %s

extern (D):

interface I {
    void fn() const
    out {
        // Positive case: contract body performs a true interface -> interface cast.
        auto b = cast(const(B)) this;
        assert(b !is null);
        assert(b.b() == 22);
    };
}

interface A {
    int a();
}

interface B {
    int b() const;
}

interface J : I {
}

class C : I, A, B {
    override void fn() const
    out (; true)
    {
    }

    override int a() {
        return 11;
    }

    override int b() const {
        return 22;
    }
}

class D : J, B {
    override void fn() const
    out (; true)
    {
    }

    override int b() const {
        return 22;
    }
}

unittest {
    A a = new C();

    // True dynamic cast: interface -> interface, resolved via druntime cast hook.
    B b = cast(B) a;

    assert(b !is null);
    assert(b.b() == 22);

    // Ensure the interface contract executes in extern(D) call flow.
    I i = cast(I) a;
    i.fn();

    // Derived-interface call flow for I's contract; this may route through a
    // non-same-interface contract context conversion.
    J j = new D();
    j.fn();
}
