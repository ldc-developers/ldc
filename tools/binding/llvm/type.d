// Written in the D programming language by Tomas Lindquist Olsen 2008
// Binding of llvm.c.Core types for D.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
module llvm.type;

import llvm.c.Core;
import llvm.c.Ext;

import llvm.util;

// we need free
version(Tango) {
    import tango.stdc.stdlib;
}
else {
    import std.c.stdlib;
}


/**
 * Each value in the IR has a type, an instance of [lltype]. See the
 * llvm::Type class.
 */
class Type
{
    /// global registry for 1:1 mapping of LLVMTypeRef's -> Type's
    private static Type[LLVMTypeRef] registry;
    ///
    alias LLVMTypeKind Kind;
    ///
    private LLVMTypeRef type;
    // used to detect if the kind of type has changed after refinement
    private const Kind cached_kind;
    ///
    private this(LLVMTypeRef t) {
        assert(t !is null);
        type = t;
        cached_kind = kind;
        assert((t in registry) is null, "Duplicate type");
        registry[t] = this;
        if (isAbstract())
            registerAbstractType();
    }
    ///
    void registerAbstractType() {
        static extern(C) void onTypeRefine(void* old, LLVMTypeRef newTypeRef) {
            Type T = cast(Type) old;
            registry.remove(T.type);
            
            if (LLVMGetTypeKind(newTypeRef) == T.cached_kind) {
                // The kind of type didn't change, so try to update and
                // recycle the Type by updating the LLVMTypeRef.
                T.type = newTypeRef;
                if (newTypeRef in registry) {
                    // We can't update the Type if it already exists
                    // but is abstract since doing so requires we pass
                    // a pointer to an object that will be stored where
                    // the GC can't see it. (If it's not in the registry
                    // it's safe because we'll put a reference in there
                    // for the GC to find, and if it's not abstract
                    // there's no need for the pointer to get out)
                    if (T.isAbstract())
                        T.type = null;
                } else {
                    registry[newTypeRef] = T;
                    // This callback only gets called once per type. If
                    // we recycle the old Type object for another
                    // abstract type we need to re-register it with the
                    // new LLVMTypeRef.
                    if (T.isAbstract())
                        T.registerAbstractType();
                }
            } else {
                // Kind of type has changed, invalidate the old object.
                T.type = null;
                // The new Type will get entered into the registry when
                // it's first needed.
            }
        }
        
        // Make sure we there's a reference to the passed object that
        // the GC can see in the registry.
        auto p = this.type in registry;
        assert((p !is null) && (*p is this), "Can't safely register an abstract type that isn't in the registry");
        
        LLVMRegisterAbstractTypeCallback(this.type,
                                         cast(void*) *p,
                                         &onTypeRefine);
    }
    ///
    char[] toString() {
        auto cstr = LLVMTypeToString(type);
        auto result = from_stringz(cstr).dup;
        free(cstr);
        return result;
    }
    ///
    Kind kind() {
        return LLVMGetTypeKind(type);
    }
    ///
    bool isAbstract() {
        return LLVMIsTypeAbstract(type) != 0;
    }
    /** Note: may invalidate the current object. Returns the refined Type
     *  if it can, or null otherwise.
     */
    Type refineAbstractType(Type to) {
        assert(isAbstract());
        
        LLVMRefineType(type, to.type);
        
        // Either type will do. Go through the registry to try to use the
        // "canonical" Type object for the type.
        if (type != null && to.type != null) {
            assert(type == to.type, "After refinement they should be equal, right?");
            return registry[type];
        } else if (type != null) {
            return registry[type];
        } else if (to.type != null) {
            return registry[to.type];
        }
        // Both types were invalidated. Is this even possible?
        return null;
    }
    ///
    static IntegerType IntType(uint bits) {
        return IntegerType.Get(bits);
    }
    ///
    static const Type Void,Label;
    ///
    static const IntegerType Int1, Int8, Int16, Int32, Int64, Size_t;
    ///
    static const RealType Float,Double,X86_FP80, FP128, PPC_FP128;
    ///
    static this()
    {
        Void = new Type(LLVMVoidType());
        Label = new Type(LLVMLabelType());

        Int1  = new IntegerType(LLVMInt1Type());
        Int8  = new IntegerType(LLVMInt8Type());
        Int16 = new IntegerType(LLVMInt16Type());
        Int32 = new IntegerType(LLVMInt32Type());
        Int64 = new IntegerType(LLVMInt64Type());
        if (size_t.sizeof == 4)
            Size_t = Int32;
        else
            Size_t = Int64;

        Float     = new RealType(LLVMFloatType());
        Double    = new RealType(LLVMDoubleType());
        X86_FP80  = new RealType(LLVMX86FP80Type());
        FP128     = new RealType(LLVMFP128Type());
        PPC_FP128 = new RealType(LLVMPPCFP128Type());
    }
    ///
    LLVMTypeRef ll()
    {
        return type;
    }
    ///
    void dump()
    {
        LLVMDumpType(type);
    }
    ///
    bool isBasic()
    {
        auto k = kind;
        if (k == Kind.Struct || k == Kind.Array || k == Kind.Function)
            return false;
        return true;
    }
}

///
class IntegerType : Type
{
    ///
    private this(LLVMTypeRef t)
    {
        super(t);
    }
    ///
    static IntegerType Get(uint nbits)
    {
        if (nbits == 1)
            return Type.Int1;
        else if (nbits == 8)
            return Type.Int8;
        else if (nbits == 16)
            return Type.Int16;
        else if (nbits == 32)
            return Type.Int32;
        else if (nbits == 64)
            return Type.Int64;
        else
        {
            auto t = LLVMIntType(nbits);
            auto ptr = t in registry;
            if (ptr !is null)
                return cast(IntegerType)*ptr;
            auto it = new IntegerType(t);
            return it;
        }
    }
    ///
    uint numBits()
    {
        return LLVMGetIntTypeWidth(type);
    }
}

///
class RealType : Type
{
    ///
    private this(LLVMTypeRef t)
    {
        super(t);
    }
}

///
class FunctionType : Type
{
    ///
    private Type ret;
    private const Type[] params;
    ///
    protected this(LLVMTypeRef t, Type r, Type[] pars)
    {
        super(t);
        ret = r;
        params = pars;
    }
    ///
    static FunctionType Get(Type r, Type[] pars, bool vararg=false)
    {
        auto p = new LLVMTypeRef[pars.length];
        foreach(i,v; pars) p[i] = v.ll;
        auto t = LLVMFunctionType(r.ll, p.ptr, p.length, vararg);
        auto ptr = t in registry;
        if (ptr !is null)
            return cast(FunctionType)*ptr;
        auto ft = new FunctionType(t, r, pars);
        return ft;
    }
    ///
    bool isVarArg()
    {
        return (LLVMIsFunctionVarArg(type) != 0);
    }
    ///
    Type returnType()
    {
        if (!ret.type)
            ret = getTypeOf(LLVMGetReturnType(type));
        
        return ret;
    }
    ///
    Type[] paramTypes()
    {
        foreach (par ; params) {
            if (!par.type) {
                updateParams();
            }
        }
        return params;
    }
    ///
    Type getParamType(uint idx)
    {
        auto par = params[idx];
        if (!par.type) {
            updateParams();
            par = params[idx];
        }
        return params[idx];
    }
    ///
    uint numParams()
    {
        return params.length;
    }
    /** Called when one or more of the parameter types have been
     *  invalidated.
     */
    private void updateParams() {
        assert (LLVMCountParamTypes(type) == params.length);
        auto llparams = new LLVMTypeRef[params.length];
        LLVMGetParamTypes(type, llparams.ptr);
        foreach (idx, llpar ; llparams) {
            params[idx] = getTypeOf(llpar);
        }
    }
}

///
class StructType : Type
{
    ///
    private this(LLVMTypeRef t)
    {
        super(t);
    }
    ///
    static StructType Get(Type[] elems, bool packed=false)
    {
        auto tys = new LLVMTypeRef[elems.length];
        foreach(i,e; elems) tys[i] = e.ll;
        auto t = LLVMStructType(tys.ptr, tys.length, packed);
        auto ptr = t in registry;
        if (ptr !is null)
            return cast(StructType)*ptr;
        auto st = new StructType(t);
        return st;
    }
    ///
    bool packed()
    {
        return (LLVMIsPackedStruct(type) != 0);
    }
    ///
    uint numElements()
    {
        return LLVMCountStructElementTypes(type);
    }
    ///
    Type[] elementTypes()
    {
        auto n = numElements();
        auto dst = new LLVMTypeRef[n];
        LLVMGetStructElementTypes(type, dst.ptr);
        auto e = new Type[n];
        for(auto i=0; i<n; i++)
            e[i] = getTypeOf(dst[i]);
        return e;
    }
}

///
abstract class SequenceType : Type
{
    ///
    private Type elemty;
    ///
    private this(LLVMTypeRef t, Type elemty)
    {
        super(t);
        this.elemty = elemty;
    }
    ///
    Type elementType()
    {
        if (!elemty.type)
            elemty = getTypeOf(LLVMGetElementType(type));
        return elemty;
    }
}

///
class PointerType : SequenceType
{
    ///
    private const uint addrSpace;
    ///
    protected this(LLVMTypeRef t, Type e, uint as)
    {
        super(t, e);
        addrSpace = as;
    }
    ///
    static PointerType Get(Type e, uint as=0)
    {
        auto t = LLVMPointerType(e.ll, as);
        auto ptr = t in registry;
        if (ptr !is null)
            return cast(PointerType)*ptr;
        auto pt = new PointerType(t, e, as);
        return pt;
    }
    ///
    uint addressSpace()
    {
        return addrSpace;
    }
}

///
class ArrayType : SequenceType
{
    ///
    private const uint arrlen;
    ///
    protected this(LLVMTypeRef t, Type e, uint l)
    {
        super(t, e);
        arrlen = l;
    }
    ///
    static ArrayType Get(Type e, uint l)
    {
        auto t = LLVMArrayType(e.ll, l);
        auto ptr = t in registry;
        if (ptr !is null) return cast(ArrayType)*ptr;
        auto at = new ArrayType(t, e, l);
        return at;
    }
    ///
    uint length()
    {
        return arrlen;
    }
}

///
class VectorType : SequenceType
{
    ///
    private const uint vecsz;
    ///
    protected this(LLVMTypeRef t, Type e, uint s)
    {
        super(t, e);
        vecsz = s;
    }
    ///
    static VectorType Get(Type e, uint s)
    {
        auto t = LLVMVectorType(e.ll, s);
        auto ptr = t in registry;
        if (ptr !is null) return cast(VectorType)*ptr;
        auto at = new VectorType(t, e, s);
        return at;
    }
    ///
    uint vectorSize()
    {
        return vecsz;
    }
}

///
class OpaqueType : Type
{
    ///
    private this(LLVMTypeRef t)
    {
        super(t);
    }
    ///
    static OpaqueType Get()
    {
        auto t = LLVMOpaqueType();
        auto ot = new OpaqueType(t);
        return ot;
    }
    ///
    private static OpaqueType Get(LLVMTypeRef t)
    {
        auto ptr = t in registry;
        if (ptr !is null)
            return cast(OpaqueType)*ptr;
        auto ot = new OpaqueType(t);
        return ot;
    }
}

///
class TypeHandle
{
    ///
    private LLVMTypeHandleRef handle;
    ///
    this()
    {
        handle = LLVMCreateTypeHandle(LLVMOpaqueType());
    }
    ///
    Type resolve()
    {
        assert(handle !is null);
        auto t = LLVMResolveTypeHandle(handle);
        return getTypeOf(t);
    }
    ///
    void refine(Type to)
    {
        assert(handle !is null);
        auto t = LLVMResolveTypeHandle(handle);
        LLVMRefineType(t, to.ll);
    }
    ///
    void dispose()
    {
        assert(handle !is null);
        LLVMDisposeTypeHandle(handle);
        handle = null;
    }
    ///
    ~this()
    {
        if (handle)
        {
            // Safe because handle isn't on the GC heap and isn't exposed.
            dispose();
        }
    }
}

///
Type getTypeOf(LLVMValueRef v)
{
    return getTypeOf(LLVMTypeOf(v));
}

///
Type getTypeOf(LLVMTypeRef ty)
{
    // first check the registry
    auto ptr = ty in Type.registry;
    if (ptr !is null) return *ptr;

    // reconstruct D type from C type and query it
    auto kind = LLVMGetTypeKind(ty);
    switch(kind)
    {
    case Type.Kind.Integer:
        auto bw = LLVMGetIntTypeWidth(ty);
        return Type.IntType(bw);

    case Type.Kind.Pointer:
        auto e = LLVMGetElementType(ty);
        auto a = LLVMGetPointerAddressSpace(ty);
        return PointerType.Get(getTypeOf(e), a);

    case Type.Kind.Struct:
        auto t = new StructType(ty);
        return t;
        // was broken for recursive types ...
        /*auto n = LLVMCountStructElementTypes(ty);
        auto e = new LLVMTypeRef[n];
        LLVMGetStructElementTypes(ty, e.ptr);
        auto p = LLVMIsPackedStruct(ty);
        auto t = new Type[n];
        foreach(i,et; e) t[i] = getTypeOf(et);
        return StructType.Get(t,p!=0);*/

    case Type.Kind.Opaque:
        return OpaqueType.Get(ty);

    case Type.Kind.Function:
        auto llr = LLVMGetReturnType(ty);
        auto lla = new LLVMTypeRef[LLVMCountParamTypes(ty)];
        LLVMGetParamTypes(ty, lla.ptr);
        auto args = new Type[lla.length];
        foreach(i,a; lla) args[i] = getTypeOf(a);
        int isvararg = LLVMIsFunctionVarArg(ty);
        return FunctionType.Get(getTypeOf(llr), args, isvararg!=0);

    case Type.Kind.Array:
        auto lle = LLVMGetElementType(ty);
        auto len = LLVMGetArrayLength(ty);
        return ArrayType.Get(getTypeOf(lle), len);

    case Type.Kind.Vector:
        auto lle = LLVMGetElementType(ty);
        auto sz = LLVMGetVectorSize(ty);
        return VectorType.Get(getTypeOf(lle), sz);

    case Type.Kind.Void:
    case Type.Kind.Float:
    case Type.Kind.Double:
    case Type.Kind.X86_FP80:
    case Type.Kind.FP128:
    case Type.Kind.PPC_FP128:
    case Type.Kind.Label:
        assert(0, "basic type not in registry");
    }
}
