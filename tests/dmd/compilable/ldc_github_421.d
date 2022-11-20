int main() {
    import core.simd;
    float[16] a = 1.0;
    float4 t = 0, k = 2;
    auto b = cast(float4[])a;
    for (size_t i = 0; i < b.length; i++)
        t += b[i] * k;
    return cast(int)t.array[2];
}