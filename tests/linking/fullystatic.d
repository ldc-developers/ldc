/* Make sure -static overrides -link-defaultlib-shared.
 * We only care about the default libs in the linker command line;
 * make sure linking fails in all cases (no main()) as linking would
 * fail if there are no static default libs (BUILD_SHARED_LIBS=ON).
 */

// RUN: not %ldc -v -static -link-defaultlib-shared %s | FileCheck %s
// CHECK-NOT: druntime-ldc-shared
// CHECK-NOT: phobos2-ldc-shared
