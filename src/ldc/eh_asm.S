/**
 * Exception handling support code that is best written in assembly
 * goes here.
 *
 * Copyright: Copyright The LDC Developers 2016
 * License:   <a href="http://www.boost.org/LICENSE_1_0.txt">Boost License 1.0</a>.
 */

/*
 * Mark the resulting object file as not requiring execution
 * permissions on stack memory. The absence of this section would mark
 * the whole resulting library as requiring an executable stack,
 * making it impossible to dynamically load druntime on several Linux
 * platforms where this is forbidden due to security policies.
 */

#if (defined(__linux__) || defined(__FreeBSD__)) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
.previous
#endif

/*
 * Called by our compiler-generate code to resume unwinding after a
 * finally block (or dtor destruction block) has been run.  'ptr' (r0)
 * is to a _d_exception.
 *
 *    void _d_eh_resume_unwind(void* ptr)
 *
 * _Unwind_Resume for ARM_EABI expects registers not to be clobbered
 * by our cleanup routine, so need this wrapper to preserve scratch
 * registers (7.4 [6 Note]) before entering it.
 *
 * Note: Current codegen of D catch landing pads are incompatible with
 * GCC provided _Unwind_Resume because the LLVM inliner can create
 * landing pads that advertise to catch more exceptions than are
 * handled, falling into _d_eh_resume_unwind to find the real handler.
 * _Unwind_Resume ignores the saved IP and resets it to the original
 * call site in this frame, but we need the callsite of
 * _d_eh_resume_unwind to find the next landing pad.  Workaround is to
 * capture it, passing to _d_arm_eabi_end_cleanup as second arg.
 */
#ifdef __ARM_EABI__
	// say we will preseve 8-byte stack when we push
        .eabi_attribute 25, 1
        .text
	.global	_d_eh_resume_unwind
	.align	2
_d_eh_resume_unwind:
	push	{r1-r3,lr}      // end_cleanup may trash these
	mov	r1,lr           // callsite IP
	bl	_d_arm_eabi_end_cleanup
	pop	{r1-r3,lr}      // restore regs to state at entry
	b	_Unwind_Resume  // r0 has returned ucb
#endif //__ARM_EABI
