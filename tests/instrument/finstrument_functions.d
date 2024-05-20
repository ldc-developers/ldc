// RUN: %ldc -c -output-ll -finstrument-functions -of=%t.ll %s && FileCheck %s < %t.ll

void fun0 () {
  // CHECK-LABEL: define{{.*}} @{{.*}}4fun0FZv
  // CHECK: [[RET1:%[0-9]]] = call ptr @llvm.returnaddress(i32 0)
  // CHECK: call void @__cyg_profile_func_enter{{.*}}4fun0FZv{{.*}}[[RET1]]
  // CHECK: [[RET2:%[0-9]]] = call ptr @llvm.returnaddress(i32 0)
  // CHECK: call void @__cyg_profile_func_exit{{.*}}4fun0FZv{{.*}}[[RET2]]
  // CHECK-NEXT: ret
  return;
}

pragma(LDC_profile_instr, false)
int fun1 (int x) {
  // CHECK-LABEL: define{{.*}} @{{.*}}4fun1FiZi
  // CHECK-NOT: __cyg_profile_func_enter
  // CHECK-NOT: __cyg_profile_func_exit
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
