module foreach10;

extern(C) int printf(char*, ...);

void main() {
    char* last = null;
    printf("The addresses should increment:\n");
    foreach (ref c; "bar") {
        auto a = {
            printf("%x '%c'\n", c, c);
            return &c;
        };
        auto nw = a();
        printf("ptr = %p\n", nw);
        if (last != null)
            assert(nw == last+1);
        last = nw;
    }
}
