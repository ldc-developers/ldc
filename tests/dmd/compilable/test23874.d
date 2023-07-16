// https://issues.dlang.org/show_bug.cgi?id=23874
// REQUIRED_ARGS: -profile=gc
// DISABLED: LDC // -profile=gc not supported

string myToString()
{
    return "";
}

enum x = myToString ~ "";
immutable x2 = myToString ~ "";
