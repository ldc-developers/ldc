template t9578(alias f) { void tf()() { f(); } }

void g9578a(alias f)()  { f(); }        // Error -> OK
void g9578b(alias ti)() { ti.tf(); }    // Error -> OK

void test9578()
{
    int i = 0;
    int m() { return i; }

    g9578a!(t9578!m.tf)();
    g9578b!(t9578!m)();
}
void main() { test9578(); }