module tangotests.ina1;

import tango.stdc.stdio;

void main()
{
    int alder;
    printf("Hvor gammel er du?\n");
    scanf("%d", &alder);
    printf("om 10 år er du %d\n", alder + 10);
}
