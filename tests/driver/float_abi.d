// REQUIRES: target_ARM

// RUN: %ldc -c -o- %s -mtriple=armv7-linux-android -float-abi=soft   -d-version=SOFT
// RUN: %ldc -c -o- %s -mtriple=armv7-linux-android -float-abi=softfp -d-version=SOFTFP
// RUN: %ldc -c -o- %s -mtriple=armv7-linux-gnueabihf                 -d-version=HARD

version (SOFT)
{
    version (ARM_SoftFloat) {} else static assert(0);
    version (ARM_SoftFP) static assert(0);
    version (ARM_HardFloat) static assert(0);

    version (D_SoftFloat) {} else static assert(0);
    version (D_HardFloat) static assert(0);
}
else version (SOFTFP)
{
    version (ARM_SoftFloat) static assert(0);
    version (ARM_SoftFP) {} else static assert(0);
    version (ARM_HardFloat) static assert(0);

    version (D_SoftFloat) static assert(0);
    version (D_HardFloat) {} else static assert(0);
}
else version (HARD)
{
    version (ARM_SoftFloat) static assert(0);
    version (ARM_SoftFP) static assert(0);
    version (ARM_HardFloat) {} else static assert(0);

    version (D_SoftFloat) static assert(0);
    version (D_HardFloat) {} else static assert(0);
}
else
    static assert(0);
