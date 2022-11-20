void main()
{
    mat4[1024] transforms = mat4.identity;
}

struct mat4
{
    float[16] data;

    static @property auto identity()
    {
        return mat4(
            [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1]
        );
    }
}
