module inputs.link_bitcode_import;

extern(C)
struct SomeStrukt {
    int i;
}

extern(C) void takeStrukt(SomeStrukt*) {};
