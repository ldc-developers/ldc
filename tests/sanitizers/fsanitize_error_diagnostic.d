// Test sanitizer errors on unrecognized values

// RUN: not %ldc -c -fsanitize=poiuyt -fsanitize-coverage=aqswdefr %s 2>&1 | FileCheck %s

// CHECK-DAG: Unrecognized -fsanitize value 'poiuyt'
// CHECK-DAG: Unrecognized -fsanitize-coverage option 'aqswdefr'
