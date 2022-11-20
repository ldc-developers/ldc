  /**
 * Contains forward references to LLVM's sanitizers interface.
 * See compiler-rt/include/sanitizer/common_interface_defs.h
 *
 * Copyright: Authors 2019-2019
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   LDC Developers
 */
module ldc.sanitizer_common;

@system:
@nogc:
nothrow:
extern (C):

// Fiber annotation interface.
// Before switching to a different stack, one must call
// __sanitizer_start_switch_fiber with a pointer to the bottom of the
// destination stack and its size. When code starts running on the new stack,
// it must call __sanitizer_finish_switch_fiber to finalize the switch.
// The start_switch function takes a void** to store the current fake stack if
// there is one (it is needed when detect_stack_use_after_return is enabled).
// When restoring a stack, this pointer must be given to the finish_switch
// function. In most cases, this void* can be stored on the stack just before
// switching.  When leaving a fiber definitely, null must be passed as first
// argument to the start_switch function so that the fake stack is destroyed.
// If you do not want support for stack use-after-return detection, you can
// always pass null to these two functions.
// Note that the fake stack mechanism is disabled during fiber switch, so if a
// signal callback runs during the switch, it will not benefit from the stack
// use-after-return detection.
void __sanitizer_start_switch_fiber(void** fake_stack_save,
                                    const(void)* bottom, size_t size);
void __sanitizer_finish_switch_fiber(void* fake_stack_save,
                                     const(void)** bottom_old,
                                     size_t* size_old);
