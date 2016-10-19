// RUN: %ldc -c -output-ll -finstrument-functions -of=%t.ll %s && FileCheck %s < %t.ll

void fun0 () {
  // CHECK-LABEL: define{{.*}} @{{.*}}4fun0FZv
  // CHECK: call void @__cyg_profile_func_enter
  // CHECK: call void @__cyg_profile_func_exit
  // CHECK-NEXT: ret
  return;
}

pragma(LDC_profile_instr, false)
int fun1 (int x) {
  // CHECK-LABEL: define{{.*}} @{{.*}}4fun1FiZi
  // CHECK-NOT: call void @__cyg_profile_func_enter
  // CHECK-NOT: call void @__cyg_profile_func_exit
  return 42;
}

bool fun2 (int x) {
  // CHECK-LABEL: define{{.*}} @{{.*}}4fun2FiZb
  if (x < 10)
    // CHECK: call void @__cyg_profile_func_exit
    // CHECK-NEXT: ret
    return true;

  // CHECK: call void @__cyg_profile_func_exit
  // CHECK-NEXT: ret
  return false;
}
