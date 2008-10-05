module foo;

template ParameterTupleOf( Fn )
{
    static if( is( Fn Params == function ) )
        alias Params ParameterTupleOf;
    else static if( is( Fn Params == delegate ) )
        alias ParameterTupleOf!(Params) ParameterTupleOf;
    else static if( is( Fn Params == Params* ) )
        alias ParameterTupleOf!(Params) ParameterTupleOf;
    else
        static assert( false, "Argument has no parameters." );
}

struct S
{
	int opApply(T)(T dg)
	{
		alias ParameterTupleOf!(T) U;
		U u;
		u[0] = 1;
		u[1] = 2;
		return 0;
	}
}

void main()
{
	foreach(int x, int y; S()){}
}
