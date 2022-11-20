module imports.ldc_github_662d;

template RCounted() {
    void release() { this.destroy; }
}

struct RC(T) {
    ~this() { _rc_ptr.release; }
    T _rc_ptr;
}
