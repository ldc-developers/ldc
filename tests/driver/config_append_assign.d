// RUN: %ldc -o- -v -conf=%S/inputs/appending_assign.conf %s 2>&1

module object;

version(Section1_1)
static assert(false);
version(Section1_2) {}
else static assert(false);

version(Section2) {}
else static assert(false);
