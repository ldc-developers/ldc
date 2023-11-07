// REQUIRED_ARGS: -fIBT
// DISABLED: LDC_not_x86

// Test for Intel CET IBT (branch) protection

static assert(__traits(getTargetInfo, "CET") == 1);
