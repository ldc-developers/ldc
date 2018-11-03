// Test if -export-marked-symbols works with thin LTO

// REQUIRES: LTO

// RUN: ldc2 %S/inputs/export_marked_symbols_thin_lto_lib.d -c -export-marked-symbols -flto=thin -of=%t1.o
// RUN: ldc2 %s -I%S/inputs -c -flto=thin -of=%t2.o
// RUN: ldc2 %t1.o %t2.o -flto=thin

import export_marked_symbols_thin_lto_lib;

void main()
{
    exportedFoo();
    normalFoo();
}
