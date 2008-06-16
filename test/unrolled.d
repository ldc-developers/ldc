module unrolled;

extern(C) int printf(char*, ...);

void test(T...)(T t) {
    foreach (value; t) {
        printf("%d\n", value);
        if (value == 2)
            break;
    }
}

void main() {
    test(1,4,3);
}
