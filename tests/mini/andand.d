bool ok = false;
void f(){ ok = true; } void main() { bool b=true; b && f(); assert(ok); }

