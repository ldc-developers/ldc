void foo();

void main() {
    set_stacksize_in_TCB_relative_to_rsp(1_000);

    foo();
}
