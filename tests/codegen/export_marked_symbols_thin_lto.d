// Tests that mismatching symbol visibilities between declarations and definitions
// work with thin LTO.

// REQUIRES: LTO

// RUN: %ldc %S/inputs/export_marked_symbols_lib.d -c -fvisibility=hidden -flto=thin -of=%t_lib%obj
// RUN: %ldc %s -I%S/inputs -flto=thin -of=%t%exe %t_lib%obj

import export_marked_symbols_lib;

void main()
{
    exportedGlobal = 1;
    normalGlobal = 2; // declared in this module with default visibility, defined as hidden

    exportedFoo();
    normalFoo(); // ditto
}
