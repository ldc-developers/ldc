// RUN: %ldc -c -I%S/inputs %s

import gh2422a;

void main() { auto aa = ["": new Empty]; }
