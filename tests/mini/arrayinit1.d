void main()
{
    float m[4][4];

    float* fp = &m[0][0];
    for (int i=0; i<16; i++,fp++)
        assert(*fp !<>= 0);
}
