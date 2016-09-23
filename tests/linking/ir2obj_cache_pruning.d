// Test recognition of -ir2obj-cache-prune-* commandline flags

// RUN: %ldc %s -ir2obj-cache=%T/prunecache1 -ir2obj-cache-prune
// RUN: %ldc %s -ir2obj-cache=%T/prunecache1 -ir2obj-cache-prune-interval=10 -ir2obj-cache-prune-maxbytes=10000
// RUN: %ldc %s -ir2obj-cache=%T/prunecache1 -ir2obj-cache-prune -ir2obj-cache-prune-interval=0
// RUN: %ldc %s -ir2obj-cache=%T/prunecache1 -ir2obj-cache-prune -ir2obj-cache-prune-maxbytes=10000
// RUN: %ldc %s -ir2obj-cache=%T/prunecache1 -ir2obj-cache-prune -ir2obj-cache-prune-expiration=10000
// RUN: %ldc %s -ir2obj-cache=%T/prunecache1 -ir2obj-cache-prune-maxpercentage=50
// RUN: %ldc %s -ir2obj-cache=%T/prunecache1 -ir2obj-cache-prune -ir2obj-cache-prune-maxpercentage=150

void main()
{
}
