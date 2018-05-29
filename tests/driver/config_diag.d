// RUN: not %ldc -conf=%S/inputs/noswitches.conf %s 2>&1 | FileCheck %s --check-prefix=NOSWITCHES
// NOSWITCHES: Could not look up switches in {{.*}}noswitches.conf

// RUN: not %ldc -conf=%S/inputs/section_aaa.conf %s 2>&1 | FileCheck %s --check-prefix=NO_SEC
// NO_SEC: No matching section for triple '{{.*}}' in {{.*}}section_aaa.conf


void foo()
{
}
