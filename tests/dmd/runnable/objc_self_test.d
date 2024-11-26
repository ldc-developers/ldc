// EXTRA_OBJC_SOURCES: objc_self_test.m
// REQUIRED_ARGS: -L-framework -LFoundation -L-w

extern (C) int getValue();

void main ()
{
    assert(getValue() == 3);
}
