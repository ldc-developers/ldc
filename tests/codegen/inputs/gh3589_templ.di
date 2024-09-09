import ldc.attributes;
template getInfo(int I) {
    @(section("test_section")) @assumeUsed shared int getInfo = I;
}
