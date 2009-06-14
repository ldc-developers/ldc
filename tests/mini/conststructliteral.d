struct S { int i; }

const S s1;
static this() { s1 = S(5); }
const S s2 = { 5 };
const S s3 = S(5);
S foo() { S t; t.i = 5; return t; }
const S s4 = foo();

const ps1 = &s1;
const ps2 = &s2;
//const ps3 = &s3; // these could be made to work
//const ps4 = &s4;

extern(C) int printf(char*,...);
void main() {
  printf("%p %p\n", ps1, ps2);
  printf("%p %p %p %p\n", &s1, &s2, &s3, &s4);
  
  assert(ps1 == ps1);
  assert(ps2 == ps2);
  assert(&s1 == &s1);
  assert(&s2 == &s2);
  assert(&s3 == &s3);
  assert(&s4 == &s4);
}

