module union3;

union vec3
{
    struct { float x,y,z; }
    float[3] xyz;
}

void main()
{
    vec3 v;
    assert(&v.y is &v.xyz[1]);
}
