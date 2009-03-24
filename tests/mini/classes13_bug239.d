extern(C) int printf(char*, ...);

class A {
    bool Afoo = false;
    void foo() { Afoo = true; }
}

class B : A {}

class C : B {
    bool Cfoo = false;
    void foo() { Cfoo = true; }
}

void main()
{
        scope c1 = new C();
        c1.foo();
	assert(c1.Cfoo && !c1.Afoo);
	
	scope c2 = new C();
	c2.B.foo();
	assert(!c2.Cfoo && c2.Afoo);

	scope c3 = new C();
	c3.A.foo();
	assert(!c3.Cfoo && c3.Afoo);
}
