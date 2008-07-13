module bug64;

void main()
{
    float f;
    float* p = &f;
    float* end1 = p+1;
    float* end2 = 1+p;
    assert(end1 is end2);
}
