// Test importing a @weak function with LTO

// REQUIRES: LTO

// RUN: %ldc -c %S/inputs/attr_weak_external_input.d -of=%t.thin%obj -flto=thin
// RUN: %ldc -I%S -of=%t.thin%exe -flto=thin %s %t.thin%obj
// RUN: %t.thin%exe

// RUN: %ldc -c %S/inputs/attr_weak_external_input.d -of=%t.full%obj -flto=full
// RUN: %ldc -I%S -of=%t.full%exe -flto=full %s %t.full%obj
// RUN: %t.full%exe

import inputs.attr_weak_external_input: weak_definition_seven;

// Forward declaration intentionally without `@weak`
extern(C) int weak_definition_four();

void main()
{
    assert(&weak_definition_four != null);
    assert(&weak_definition_seven != null);
}
