module mini.foreach9;
extern(C) int printf(char* str, ...);

struct array2d(T) {
  int test() {
    printf("%p\n", cast(void*) this);
    foreach (x; *this) {
      printf("%p\n", cast(void*) this);
    }
    return true;
  }
  int opApply(int delegate(ref int) dg) {
    int x;
    return dg(x), 0;
  }
}

unittest {
  array2d!(int) test;
  test.test();
  //int i = 0; i /= i;
}

void main() { }
