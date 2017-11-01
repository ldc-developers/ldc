// REQUIRES: Darwin
// RUN: %ldc -c %s

extern (Objective-C) interface NSView
{
    void setWantsLayer(bool value) @selector("setWantsLayer:");
}

void foo(NSView v) { v.setWantsLayer(true); }
