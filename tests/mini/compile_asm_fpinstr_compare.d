void main() {
asm {
    fmul;
    fmul ST, ST(1);
    fmul ST(1), ST;
    fmulp;
    fmulp ST(1), ST;

    fdiv;
    fdiv ST, ST(1);
    fdiv ST(1), ST;
    fdivp;
    fdivp ST(1), ST;
    fdivr;
    fdivr ST, ST(1);
    fdivr ST(1), ST;
    fdivrp;
    fdivrp ST(1), ST;

    fsub;
    fsub ST, ST(1);
    fsub ST(1), ST;
    fsubp;
    fsubp ST(1), ST;
    fsubr;
    fsubr ST, ST(1);
    fsubr ST(1), ST;
    fsubrp;
    fsubrp ST(1), ST;    
}
}