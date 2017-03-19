// Test recognition of -cache-prune-* commandline flags

// RUN: %ldc %s -cache=%T/prunecache1 -cache-prune
// RUN: %ldc %s -cache=%T/prunecache1 -cache-prune-interval=10 -cache-prune-maxbytes=10000
// RUN: %ldc %s -cache=%T/prunecache1 -cache-prune -cache-prune-interval=0
// RUN: %ldc %s -cache=%T/prunecache1 -cache-prune -cache-prune-maxbytes=10000
// RUN: %ldc %s -cache=%T/prunecache1 -cache-prune -cache-prune-expiration=10000
// RUN: %ldc %s -cache=%T/prunecache1 -cache-prune-maxpercentage=50
// RUN: %ldc %s -cache=%T/prunecache1 -cache-prune -cache-prune-maxpercentage=150

void main()
{
}
