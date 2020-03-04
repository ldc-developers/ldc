// RUN: %ldc -run %s

import core.thread;

void foo() {}

shared static this()
{
    auto f = new Fiber(&foo); // depends on shared static this in core.thread.osthread.
}

void main() {}
