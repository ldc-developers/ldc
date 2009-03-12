module asm10;

struct S {
    ushort first;
    ushort second;
    int unaccessed;
}

void main() {
    auto s = S(512, 42, -1);
    ushort x = 0;
    version(D_InlineAsm_X86) {
        asm {
            lea EAX, s;
            mov CX, S.second[EAX];
            mov x, CX;
            mov S.first[EAX], 640;
        }
    } else version(D_InlineAsm_X86_64) {
        asm {
            lea RAX, s;
            mov CX, S.second[RAX];
            mov x, CX;
            mov S.first[RAX], 640;
        }
    }
    assert(x == 42);
    assert(s.first == 640);
    assert(s.second == 42);
    assert(s.unaccessed == -1);
}
