import tango.core.Exception;
import cn.kuehne.flectioned;


TracedExceptionInfo traceHandler( void* ptr = null )
{
    class FlectionedTrace :
        TracedExceptionInfo
    {
        this( void* ptr = null )
        {
            if( ptr )
                m_trace = Trace.getTrace( cast(size_t) ptr );
            else
                m_trace = Trace.getTrace();
        }

        int opApply( int delegate( inout char[] ) dg )
        {
            int ret = 0;
            foreach( t; m_trace )
            {
                char[] buf = t.toString;
                ret = dg( buf );
                if( ret != 0 )
                    break;
            }
            return ret;
        }

    private:
        Trace[] m_trace;
    }

    return new FlectionedTrace( ptr );
}


static this()
{
    setTraceHandler( &traceHandler );
}
