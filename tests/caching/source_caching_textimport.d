// Test that caching works correctly with text file importing.

// RUN: copyfile %S/inputs/textimport/text1.txt %T/textimport.txt \
// RUN: && %ldc -c -cache=%T/textimp %s -J%T -of=%t%obj \
// RUN: && %prunecache -f --max-bytes=1 %T/textimp  \
// RUN: && %ldc -c -cache=%T/textimp -cache-sourcefiles %s -J%T -of=%t%obj -vv \
// RUN: && %ldc -c -cache=%T/textimp -cache-sourcefiles %s -J%T -of=%t%obj -vv | FileCheck --check-prefix=HIT %s \
// RUN: && removefile %T/textimport.txt \
// RUN: && copyfile %S/inputs/textimport/text2.txt %T/textimport.txt \
// RUN: && %ldc -c -cache=%T/textimp -cache-sourcefiles %s -J%T -of=%t%obj -vv | FileCheck --check-prefix=NO_HIT %s

// HIT: Cache manifest checks out
// HIT: Recovering outputs from cache
// NO_HIT: Manifest hash failure for {{.*}}textimport.txt

string txt = import("textimport.txt");
