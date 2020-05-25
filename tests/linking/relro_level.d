// REQUIRES: Linux
// RUN: %ldc --gcc=echo --relro-level=full %s | grep -e \\-Wl,-z,relro -e \\-Wl,-z,now
// RUN: %ldc --gcc=echo --relro-level=partial %s | grep \\-Wl,-z,relro
// RUN: %ldc --gcc=echo --relro-level=off %s | grep \\-Wl,-z,norelro

void main() {}
