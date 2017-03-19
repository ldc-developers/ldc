// RUN: %ldc --enable-inlining -of=%t%exe %s

// https://github.com/ldc-developers/ldc/issues/1678

import std.datetime;

// Extra test that fail when a simple frontend change is tried that names __invariant using the line and column number.
class A {
    mixin(genInv("666")); mixin(genInv("777"));
}

string genInv(string a) {
    return "invariant() { }";
}

void main()
{
    auto currentTime = Clock.currTime();
    auto timeString = currentTime.toISOExtString();
    auto restoredTime = SysTime.fromISOExtString(timeString);
}