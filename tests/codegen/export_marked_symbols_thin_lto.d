// Test if -export-marked-symbols works with thin LTO

// REQUIRES: LTO

// RUN: ldc2 %S/inputs/export_marked_symbols_thin_lto_lib.d -c -export-marked-symbols -flto=thin -of=%t1%obj
// RUN: ldc2 %s -I%S/inputs -c -flto=thin -of=%t2%obj
// RUN: ldc2 %t1%obj %t2%obj -flto=thin

import export_marked_symbols_thin_lto_lib;

void main()
{
    exportedFoo();
    normalFoo();
}
