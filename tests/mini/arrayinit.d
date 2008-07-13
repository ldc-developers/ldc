module arrayinit;
float[4] ftable = [1,2,3,4];
int[8] itable = [3:42,6:123];

private uint[7] crc32_table = [0x00000000,0x77073096,0xee0e612c,0x990951ba,0x076dc419,0x706af48f,0xe963a535];

void main()
{
    assert(crc32_table[3] == 0x990951ba);
}
