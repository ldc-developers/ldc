// Written in the D programming language.

/**
 * Interface to C++ <typeinfo>
 *
 * Copyright: Copyright (c) 2016 D Language Foundation
 * License:   $(HTTP boost.org/LICENSE_1_0.txt, Boost License 1.0).
 * Authors:   $(HTTP digitalmars.com, Walter Bright)
 * Source:    $(DRUNTIMESRC core/stdcpp/_typeinfo.d)
 */

module core.stdcpp.typeinfo;

version (LDC) import ldc.attributes : weak;
else          private enum weak = null;

version (CppRuntime_DigitalMars)
{
    import core.stdcpp.exception;

    extern (C++, "std"):

    class type_info
    {
        void* pdata;

    public:
        //virtual ~this();
        void dtor() { }     // reserve slot in vtbl[]

        //bool operator==(const type_info rhs) const;
        //bool operator!=(const type_info rhs) const;
        final bool before(const type_info rhs) const;
        final const(char)* name() const;
    protected:
        //type_info();
    private:
        //this(const type_info rhs);
        //type_info operator=(const type_info rhs);
    }

    class bad_cast : exception
    {
        this() nothrow { }
        this(const bad_cast) nothrow { }
        //bad_cast operator=(const bad_cast) nothrow { return this; }
        //virtual ~this() nothrow;
        override const(char)* what() const nothrow;
    }

    class bad_typeid : exception
    {
        this() nothrow { }
        this(const bad_typeid) nothrow { }
        //bad_typeid operator=(const bad_typeid) nothrow { return this; }
        //virtual ~this() nothrow;
        override const (char)* what() const nothrow;
    }
}
else version (CppRuntime_Microsoft)
{
    import core.stdcpp.exception;

    extern (C++, "std"):

    struct __type_info_node
    {
        void* _MemPtr;
        __type_info_node* _Next;
    }

    extern __gshared __type_info_node __type_info_root_node;

    class type_info
    {
    @nogc:
        @weak // LDC
        ~this() nothrow {}
        //bool operator==(const type_info rhs) const;
        //bool operator!=(const type_info rhs) const;
        final bool before(const type_info rhs) const nothrow;
        final const(char)* name(__type_info_node* p = &__type_info_root_node) const nothrow;

    private:
        void* pdata;
        char[1] _name;
        //type_info operator=(const type_info rhs);
    }

    class bad_cast : exception
    {
        extern(D) this(const(char)* msg = "bad cast") @nogc nothrow { super(msg); }
    }

    class bad_typeid : exception
    {
        extern(D) this(const(char)* msg = "bad typeid") @nogc nothrow { super(msg); }
    }
}
else version (CppRuntime_Gcc)
{
    import core.stdcpp.exception;

    extern (C++, "__cxxabiv1")
    {
        class __class_type_info;
    }

    extern (C++, "std"):

    abstract // LDC
    class type_info
    {
    @nogc:
        @weak // LDC
        ~this() {}
        @weak // LDC
        final const(char)* name() const nothrow
        {
            return _name[0] == '*' ? _name + 1 : _name;
        }
        @weak // LDC
        final bool before(const type_info _arg) const nothrow
        {
            import core.stdc.string : strcmp;
            return (_name[0] == '*' && _arg._name[0] == '*')
                ? _name < _arg._name
                : strcmp(_name, _arg._name) < 0;
        }
        //bool operator==(const type_info) const;
        bool __is_pointer_p() const;
        bool __is_function_p() const;
        bool __do_catch(const type_info, void**, uint) const;
        bool __do_upcast(const __class_type_info, void**) const;

    protected:
        const(char)* _name;

        extern(D) this(const(char)* name) { _name = name; }
    }

    class bad_cast : exception
    {
    @nogc:
        extern(D) this() nothrow {}
        @weak // LDC
        override const(char)* what() const nothrow { return "bad cast"; }
    }

    class bad_typeid : exception
    {
    @nogc:
        extern(D) this() nothrow {}
        @weak // LDC
        override const(char)* what() const nothrow { return "bad typeid"; }
    }
}
else
{
    version (LDC) { /* empty module for unsupported C++ runtimes */ } else
    static assert(0, "Missing std::type_info binding for this platform");
}
