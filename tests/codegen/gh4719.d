// RUN: %ldc -c -output-ll -of=%t.ll %s

struct TraceBuf {
    align(1) uint args;
}

// Test correct compilation (no error) of the context pointer type for the delegate of `foo`.
void foo() {
    byte[2] fixDescs;
    TraceBuf fixLog;

    auto dlg = delegate() {
        fixDescs[0] = 1;
        fixLog.args = 1;
    };
}

class TraceClass {
    align(1)
    uint args;
}

// Test correct compilation (no error) of the context pointer type for the delegate of `foo2`.
void foo2() {
    byte[2] fixDescs;
    scope TraceClass fixLog;

    auto dlg = delegate() {
        fixDescs[0] = 1;
        fixLog.args = 1;
    };
}
