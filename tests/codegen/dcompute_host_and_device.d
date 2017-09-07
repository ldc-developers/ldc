// Check that we can generate code for both the host and device in one compiler invocation
// REQUIRES: atleast_llvm309
// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 -m64 -Iinputs %s %S/inputs/kernel.d

import inputs.kernel : foo;

int tlGlobal;
__gshared int gGlobal;

void main(string[] args)
{
    tlGlobal = 0;
    gGlobal  = 0;
    string s = foo.mangleof;
}
