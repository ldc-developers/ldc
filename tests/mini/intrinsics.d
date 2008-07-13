import llvm.intrinsic;

extern(C) int printf(char*,...);
extern(C) int scanf(char*,...);

void main()
{
    {
    float f;
    printf("Enter float: ");
    scanf("%f", &f);
    float sf = llvm_sqrt(f);
    printf("sqrt(%f) = %f\n", f, sf);
    }
    
    {
    double d;
    printf("Enter double: ");
    scanf("%lf", &d);
    double sd = llvm_sqrt(d);
    printf("sqrt(%lf) = %lf\n", d, sd);
    }
    
    {
    real d;
    printf("Enter real: ");
    scanf("%lf", &d);
    real sd = llvm_sqrt(d);
    printf("sqrt(%lf) = %lf\n", d, sd);
    }
}
