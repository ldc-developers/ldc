@compute(CompileFor.deviceOnly)
module inputs.kernel;

import ldc.dcompute;
@kernel void k_foo(GlobalPointer!float x_in)
{
	SharedPointer!float shared_x;
	PrivatePointer!float private_x;
	ConstantPointer!float const_x;
	*shared_x = *x_in;
	*private_x = *x_in;
	*x_in = *const_x;

	*x_in = *shared_x;
	*x_in = *private_x;
}
