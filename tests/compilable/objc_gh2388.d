// REQUIRES: Darwin
// RUN: %ldc -c -singleobj %s %S/objc_gh2387.d

void alloc()
{
    NSObject o;
    o.alloc();
}

extern (Objective-C):

interface NSObject
{
    NSObject alloc() @selector("alloc");
}
