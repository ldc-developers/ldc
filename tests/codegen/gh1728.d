// RUN: %ldc -run %s

struct OpApply {
    int opApply(int delegate(int) cb) {
        return cb(42);
    }
}

struct Bolinha {
    int a;
    this(ref OpApply moviadao) {
        foreach(int b; moviadao) {
            this.a = b;
            return;
        }
    }
}

void main() {
    OpApply range;
    const s = Bolinha(range);
    assert(s.a == 42);
}
