// Check that we can generate code for both the host and device in one compiler invocation
// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 -mdcompute-file-prefix=host_and_device -Iinputs %s %S/inputs/kernel.d

import inputs.kernel : foo;

int tlGlobal;
__gshared int gGlobal;

void main(string[] args)
{
    tlGlobal = 0;
    gGlobal  = 0;
    string s = foo.mangleof;
}
