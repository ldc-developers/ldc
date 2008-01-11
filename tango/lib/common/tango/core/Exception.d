/**
 * The exception module defines all system-level exceptions and provides a
 * mechanism to alter system-level error handling.
 *
 * Copyright: Copyright (C) 2005-2006 Sean Kelly, Kris Bell.  All rights reserved.
 * License:   BSD style: $(LICENSE)
 * Authors:   Sean Kelly, Kris Bell
 */
module tango.core.Exception;


private
{
    alias void  function( char[] file, size_t line, char[] msg = null ) assertHandlerType;
    alias TracedExceptionInfo function( void* ptr = null ) traceHandlerType;

    assertHandlerType   assertHandler   = null;
    traceHandlerType    traceHandler    = null;
}


interface TracedExceptionInfo
{
    int opApply( int delegate( inout char[] ) );
}


////////////////////////////////////////////////////////////////////////////////
/*
- Exception
  - OutOfMemoryException

  - TracedException
    - SwitchException
    - AssertException
    - ArrayBoundsException
    - FinalizeException

    - PlatformException
      - ProcessException
      - ThreadException
        - FiberException
      - SyncException
      - IOException
        - SocketException
          - SocketAcceptException
        - AddressException
        - HostException
        - VfsException
        - ClusterException

    - NoSuchElementException
      - CorruptedIteratorException

    - IllegalArgumentException
      - IllegalElementException

    - TextException
      - RegexException
      - LocaleException
      - UnicodeException

    - PayloadException
*/
////////////////////////////////////////////////////////////////////////////////


/**
 * Thrown on an out of memory error.
 */
class OutOfMemoryException : Exception
{
    this( char[] file, size_t line )
    {
        super( "Memory allocation failed", file, line );
    }

    char[] toString()
    {
        return msg ? super.toString() : "Memory allocation failed";
    }
}


/**
 * Stores a stack trace when thrown.
 */
class TracedException : Exception
{
    this( char[] msg )
    {
        super( msg );
        m_info = traceContext();
    }

    this( char[] msg, Exception e )
    {
        super( msg, e );
        m_info = traceContext();
    }

    this( char[] msg, char[] file, size_t line )
    {
        super( msg, file, line );
        m_info = traceContext();
    }

    char[] toString()
    {
        if( m_info is null )
            return super.toString();
        char[] buf = super.toString();
        buf ~= "\n----------------";
        foreach( line; m_info )
            buf ~= "\n" ~ line;
        return buf;
    }

    int opApply( int delegate( inout char[] buf ) dg )
    {
        if( m_info is null )
            return 0;
        return m_info.opApply( dg );
    }

private:
    TracedExceptionInfo m_info;
}


/**
 * Base class for operating system or library exceptions.
 */
class PlatformException : TracedException
{
    this( char[] msg )
    {
        super( msg );
    }
}

/**
 * Thrown on an assert error.
 */
class AssertException : TracedException
{
    this( char[] file, size_t line )
    {
        super( "Assertion failure", file, line );
    }

    this( char[] msg, char[] file, size_t line )
    {
        super( msg, file, line );
    }
}


/**
 * Thrown on an array bounds error.
 */
class ArrayBoundsException : TracedException
{
    this( char[] file, size_t line )
    {
        super( "Array index out of bounds", file, line );
    }
}


/**
 * Thrown on finalize error.
 */
class FinalizeException : TracedException
{
    ClassInfo   info;

    this( ClassInfo c, Exception e = null )
    {
        super( "Finalization error", e );
        info = c;
    }

    char[] toString()
    {
        //return "An exception was thrown while finalizing an instance of class " ~ info.name;
        assert(0);
    }
}


/**
 * Thrown on a switch error.
 */
class SwitchException : TracedException
{
    this( char[] file, size_t line )
    {
        super( "No appropriate switch clause found", file, line );
    }
}


/**
 * Represents a text processing error.
 */
class TextException : TracedException
{
    this( char[] msg )
    {
        super( msg );
    }
}

/**
 * Thrown on a unicode conversion error.
 */
class UnicodeException : TextException
{
    size_t idx;

    this( char[] msg, size_t idx )
    {
        super( msg );
        this.idx = idx;
    }
}


/**
 * Base class for thread exceptions.
 */
class ThreadException : PlatformException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Base class for fiber exceptions.
 */
class FiberException : ThreadException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Base class for synchronization exceptions.
 */
class SyncException : PlatformException
{
    this( char[] msg )
    {
        super( msg );
    }
}



/**
 * The basic exception thrown by the tango.io package. One should try to ensure
 * that all Tango exceptions related to IO are derived from this one.
 */
class IOException : PlatformException
{
    this( char[] msg )
    {
        super( msg );
    }
}

/**
 * The basic exception thrown by the tango.io.vfs package. 
 */
private class VfsException : IOException
{
    this( char[] msg )
    {
        super( msg );
    }
}

/**
 * The basic exception thrown by the tango.io.cluster package. 
 */
private class ClusterException : IOException
{
    this( char[] msg )
    {
        super( msg );
    }
}

/**
 * Base class for socket exceptions.
 */
class SocketException : IOException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Base class for exception thrown by an InternetHost.
 */
class HostException : IOException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Base class for exceptiond thrown by an Address.
 */
class AddressException : IOException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Thrown when a socket failed to accept an incoming connection.
 */
class SocketAcceptException : SocketException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Thrown on a process error.
 */
class ProcessException : PlatformException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Base class for regluar expression exceptions.
 */
class RegexException : TextException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Base class for locale exceptions.
 */
class LocaleException : TextException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * RegistryException is thrown when the NetworkRegistry encounters a
 * problem during proxy registration, or when it sees an unregistered
 * guid.
 */
class RegistryException : TracedException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Thrown when an illegal argument is encountered.
 */
class IllegalArgumentException : TracedException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 *
 * IllegalElementException is thrown by Collection methods
 * that add (or replace) elements (and/or keys) when their
 * arguments are null or do not pass screeners.
 *
 */
class IllegalElementException : IllegalArgumentException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Thrown on past-the-end errors by iterators and containers.
 */
class NoSuchElementException : TracedException
{
    this( char[] msg )
    {
        super( msg );
    }
}


/**
 * Thrown when a corrupt iterator is detected.
 */
class CorruptedIteratorException : NoSuchElementException
{
    this( char[] msg )
    {
        super( msg );
    }
}


////////////////////////////////////////////////////////////////////////////////
// Overrides
////////////////////////////////////////////////////////////////////////////////


/**
 * Overrides the default assert hander with a user-supplied version.
 *
 * Params:
 *  h = The new assert handler.  Set to null to use the default handler.
 */
void setAssertHandler( assertHandlerType h )
{
    assertHandler = h;
}


/**
 * Overrides the default trace hander with a user-supplied version.
 *
 * Params:
 *  h = The new trace handler.  Set to null to use the default handler.
 */
void setTraceHandler( traceHandlerType h )
{
    traceHandler = h;
}


////////////////////////////////////////////////////////////////////////////////
// Overridable Callbacks
////////////////////////////////////////////////////////////////////////////////


/**
 * A callback for assert errors in D.  The user-supplied assert handler will
 * be called if one has been supplied, otherwise an AssertException will be
 * thrown.
 *
 * Params:
 *  file = The name of the file that signaled this error.
 *  line = The line number on which this error occurred.
 */
extern (C) void onAssertError( char[] file, size_t line )
{
    if( assertHandler is null )
        throw new AssertException( file, line );
    assertHandler( file, line );
}


/**
 * A callback for assert errors in D.  The user-supplied assert handler will
 * be called if one has been supplied, otherwise an AssertException will be
 * thrown.
 *
 * Params:
 *  file = The name of the file that signaled this error.
 *  line = The line number on which this error occurred.
 *  msg  = An error message supplied by the user.
 */
extern (C) void onAssertErrorMsg( char[] file, size_t line, char[] msg )
{
    if( assertHandler is null )
        throw new AssertException( msg, file, line );
    assertHandler( file, line, msg );
}


/**
 * This function will be called when a TracedException is constructed.  The
 * user-supplied trace handler will be called if one has been supplied,
 * otherwise no trace will be generated.
 *
 * Params:
 *  ptr = A pointer to the location from which to generate the trace, or null
 *        if the trace should be generated from within the trace handler
 *        itself.
 *
 * Returns:
 *  An object describing the current calling context or null if no handler is
 *  supplied.
 */
TracedExceptionInfo traceContext( void* ptr = null )
{
    if( traceHandler is null )
        return null;
    return traceHandler( ptr );
}

////////////////////////////////////////////////////////////////////////////////
// Internal Error Callbacks
////////////////////////////////////////////////////////////////////////////////


/**
 * A callback for array bounds errors in D.  An ArrayBoundsException will be
 * thrown.
 *
 * Params:
 *  file = The name of the file that signaled this error.
 *  line = The line number on which this error occurred.
 *
 * Throws:
 *  ArrayBoundsException.
 */
extern (C) void onArrayBoundsError( char[] file, size_t line )
{
    throw new ArrayBoundsException( file, line );
}


/**
 * A callback for finalize errors in D.  A FinalizeException will be thrown.
 *
 * Params:
 *  e = The exception thrown during finalization.
 *
 * Throws:
 *  FinalizeException.
 */
extern (C) void onFinalizeError( ClassInfo info, Exception ex )
{
    throw new FinalizeException( info, ex );
}


/**
 * A callback for out of memory errors in D.  An OutOfMemoryException will be
 * thrown.
 *
 * Throws:
 *  OutOfMemoryException.
 */
extern (C) void onOutOfMemoryError()
{
    // NOTE: Since an out of memory condition exists, no allocation must occur
    //       while generating this object.
    throw cast(OutOfMemoryException) cast(void*) OutOfMemoryException.classinfo.init;
}


/**
 * A callback for switch errors in D.  A SwitchException will be thrown.
 *
 * Params:
 *  file = The name of the file that signaled this error.
 *  line = The line number on which this error occurred.
 *
 * Throws:
 *  SwitchException.
 */
extern (C) void onSwitchError( char[] file, size_t line )
{
    throw new SwitchException( file, line );
}


/**
 * A callback for unicode errors in D.  A UnicodeException will be thrown.
 *
 * Params:
 *  msg = Information about the error.
 *  idx = String index where this error was detected.
 *
 * Throws:
 *  UnicodeException.
 */
extern (C) void onUnicodeError( char[] msg, size_t idx )
{
    throw new UnicodeException( msg, idx );
}
