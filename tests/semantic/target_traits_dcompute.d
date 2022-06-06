// Tests LDC-specific target __traits

// RUN: %ldc -c %s -mdcompute-targets=cuda-350 -mdcompute-file-prefix=testing
// REQUIRES: target_NVPTX

static assert([ __traits(getTargetInfo, "dcomputeTargets") ] == ["cuda-350"]);
static assert(__traits(getTargetInfo, "dcomputeFilePrefix") == "testing");
