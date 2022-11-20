version(D_InlineAsm_X86_64) version = DMD_InlineAsm;
version(D_InlineAsm_X86) version = DMD_InlineAsm;

version(InlineAsm)
{

void fooNormal()() {
    asm {
        jmp Llabel;
Llabel:
        nop;
    }
}

void fooNaked()() {
    asm {
        naked;
        jmp Llabel;
Llabel:
        ret;
    }
}

void main() {
    fooNormal();
    fooNaked();
}

}
else
{
void main() {}
}
