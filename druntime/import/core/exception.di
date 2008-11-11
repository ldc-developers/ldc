// D import file generated from 'core/exception.d'
module core.exception;
private
{
    alias void function(string file, size_t line, string msg = null) assertHandlerType;
    assertHandlerType assertHandler = null;
}
class ArrayBoundsException : Exception
{
    this(string file, size_t line)
{
super("Array index out of bounds",file,line);
}
}
class AssertException : Exception
{
    this(string file, size_t line)
{
super("Assertion failure",file,line);
}
    this(string msg, string file, size_t line)
{
super(msg,file,line);
}
}
class FinalizeException : Exception
{
    ClassInfo info;
    this(ClassInfo c, Exception e = null)
{
super("Finalization error",e);
info = c;
}
    override
{
    string toString()
{
return "An exception was thrown while finalizing an instance of class " ~ info.name;
}
}
}
class HiddenFuncException : Exception
{
    this(ClassInfo ci)
{
super("Hidden method called for " ~ ci.name);
}
}
class OutOfMemoryException : Exception
{
    this(string file, size_t line)
{
super("Memory allocation failed",file,line);
}
    override
{
    string toString()
{
return msg ? super.toString() : "Memory allocation failed";
}
}
}
class SwitchException : Exception
{
    this(string file, size_t line)
{
super("No appropriate switch clause found",file,line);
}
}
class UnicodeException : Exception
{
    size_t idx;
    this(string msg, size_t idx)
{
super(msg);
this.idx = idx;
}
}
void setAssertHandler(assertHandlerType h)
{
assertHandler = h;
}
extern (C) 
{
    void onAssertError(string file, size_t line);
}
extern (C) 
{
    void onAssertErrorMsg(string file, size_t line, string msg);
}
extern (C) 
{
    void onArrayBoundsError(string file, size_t line);
}
extern (C) 
{
    void onFinalizeError(ClassInfo info, Exception ex);
}
extern (C) 
{
    void onHiddenFuncError(Object o);
}
extern (C) 
{
    void onOutOfMemoryError();
}
extern (C) 
{
    void onSwitchError(string file, size_t line);
}
extern (C) 
{
    void onUnicodeError(string msg, size_t idx);
}
