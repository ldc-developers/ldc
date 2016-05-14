// RUN: %ldc -c %s

extern(C++) interface XUnknown {}
class ComObject : XUnknown {}
class DComObject : ComObject {}
