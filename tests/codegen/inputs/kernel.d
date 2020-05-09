@compute(CompileFor.deviceOnly)
module inputs.kernel;

import ldc.dcompute;
@kernel void k_foo(GlobalPointer!float x_in)
{
	SharedPointer!float shared_x;
	PrivatePointer!float private_x;
	ConstantPointer!float const_x;
	shared_x[0] = x_in[0];
	private_x[0] = x_in[0];
	x_in[0] = const_x[0];

	x_in[0] = shared_x[0];
	x_in[0] = private_x[0];
}
