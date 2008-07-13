module bug52;
struct Vec { double x,y,z; }
struct Pair(T, U) { T first; U second; }
typedef Pair!(double, Vec) Hit;
void main() {}
