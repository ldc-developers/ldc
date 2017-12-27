// Test that `-cov` does not lead to harmless import cycle errors.
// Github issue 2177

// RUN: %ldc -cov=100 -Iinputs %s %S/inputs/coverage_cycle_input.d -of=%t%exe
// RUN: %t%exe

module coverage_cycle_gh2177;

import inputs.coverage_cycle_input;

static this() {
    int i;
}

int main () {
    return 0;
}
