// REQUIRES: target_MSP430

// RUN: %ldc -mtriple=msp430 -c -output-ll %s

void test()
{
    int[1] array;

    ushort i = 0;
    auto value = array[i];

    short j = 0;
    value = array[j];

    uint k = 0;
    value = array[k];

    int l = 0;
    value = array[l];
}
