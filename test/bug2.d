module bug2;
struct Vec { Vec barf() { return Vec(); } }
class test { this(Vec whee) { } }
void main() { Vec whee; new test(whee.barf()); }
