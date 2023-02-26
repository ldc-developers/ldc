// DISABLED: win

// OS X note: ld complains extern_weak symbols are undefined unless ld options
// -undefined dynamic_lookup or -U __D15ldc_extern_weak11nonExistenti are
// provided.  extern_weak not really needed on OS X though.
// REQUIRED_ARGS(osx): -L-undefined -Ldynamic_lookup

// the linker on macOS-arm64 emits a warning
// TRANSFORM_OUTPUT(osx): remove_lines("warning: -undefined dynamic_lookup may not work with chained fixups")

extern __gshared pragma(LDC_extern_weak) int nonExistent;

bool doesNonExistentExist() {
    return &nonExistent !is null;
}

void main() {
    // Make sure that the frontend does not statically fold the address check
    // to 'true' for weak symbols.
    assert(!doesNonExistentExist());
}
