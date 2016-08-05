// Test -cov-merge commandline flag.

// RUN: %ldc -cov -cov-merge -c -output-ll -of=%t.ll     %s && FileCheck %s --check-prefix MERGE   < %t.ll
// RUN: %ldc -cov            -c -output-ll -of=%t.not.ll %s && FileCheck %s --check-prefix NOMERGE < %t.not.ll

// Also test linking.
// RUN: %ldc -cov -cov-merge -of=%t%exe %s

void main()
{
}

// MERGE: call{{.*}} void @dmd_coverSetMerge(i1 true)
// NOMERGE-NOT: dmd_coverSetMerge
