// RUN: %ldc -o- -conf=%S/inputs/noswitches.conf %s 2>&1 | FileCheck %s --check-prefix=NOSWITCHES
// NOSWITCHES: Error while reading config file: {{.*}}noswitches.conf
// NOSWITCHES-NEXT: Could not look up switches

// RUN: %ldc -o- -conf=%S/inputs/section_aaa.conf %s 2>&1 | FileCheck %s --check-prefix=NO_SEC
// NO_SEC: Error while reading config file: {{.*}}section_aaa.conf
// NO_SEC-NEXT: No matching section for triple '{{.*}}'

// RUN: %ldc -o- -conf=%S/inputs/invalid_append.conf %s 2>&1 | FileCheck %s --check-prefix=APP
// APP: Error while reading config file: {{.*}}invalid_append.conf
// APP-NEXT: line 3: '~=' is not supported with scalar values

module object;
