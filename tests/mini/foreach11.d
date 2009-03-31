module foreach11;

extern(C) int printf(char*, ...);

void main() {
    char* last = null;
    printf("The addresses should remain constant:\n");
    foreach (c; "bar") {
        auto a = {
            printf("%x '%c'\n", c, c);
            printf("ptr = %p\n", &c);
            if (last)
                assert(last == &c);
        };
        a();
        last = &c;
    }
}
