module tangotests.mem1;

import tango.stdc.stdio;

void main()
{
    printf("new int;\n");
    int* i = new int;
    assert(*i == 0);

    printf("new int[3];\n");
    int[] ar = new int[3];
    ar[0] = 1;
    ar[1] = 56;
    assert(ar.length == 3);
    assert(ar[0] == 1);
    assert(ar[1] == 56);
    assert(ar[2] == 0);

    printf("array ~= elem;\n");
    int[] ar2;
    ar2 ~= 22;
    assert(ar2.length == 1);
    assert(ar2[0] == 22);

    printf("array ~= array;\n");
    ar2 ~= ar;
    assert(ar2.length == 4);
    assert(ar2[0] == 22);
    assert(ar2[1] == 1);
    printf("%d %d %d %d\n", ar2[0], ar2[1], ar2[2], ar2[3]);
    assert(ar2[2] == 56);
    assert(ar2[3] == 0);

    printf("array ~ array;\n");
    int[] ar5 = ar ~ ar2;
    assert(ar5.length == 7);
    assert(ar5[0] == 1);
    assert(ar5[1] == 56);
    assert(ar5[2] == 0);
    assert(ar5[3] == 22);
    assert(ar5[4] == 1);
    assert(ar5[5] == 56);
    assert(ar5[6] == 0);

    printf("array ~ elem;\n");
    int[] ar4 = ar2 ~ 123;
    assert(ar4.length == 5);
    assert(ar4[0] == 22);
    assert(ar4[1] == 1);
    assert(ar4[2] == 56);
    assert(ar4[3] == 0);
    assert(ar4[4] == 123);

    printf("elem ~ array;\n");
    int[] ar3 = 123 ~ ar2;
    assert(ar3.length == 5);
    assert(ar3[0] == 123);
    assert(ar3[1] == 22);
    assert(ar3[2] == 1);
    assert(ar3[3] == 56);
    assert(ar3[4] == 0);
}
