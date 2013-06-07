//===-- irclass.cpp -------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_VER >= 303
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#else
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#endif
#include "llvm/ADT/SmallString.h"

#include "aggregate.h"
#include "declaration.h"
#include "mtype.h"

#include "gen/irstate.h"
#include "gen/logger.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/utils.h"
#include "gen/arrays.h"
#include "gen/metadata.h"
#include "gen/runtime.h"

#include "ir/irstruct.h"
#include "ir/irtypeclass.h"

//////////////////////////////////////////////////////////////////////////////

extern LLConstant* get_default_initializer(VarDeclaration* vd, Initializer* init);
extern size_t add_zeros(std::vector<llvm::Constant*>& constants, size_t diff);

extern LLConstant* DtoDefineClassInfo(ClassDeclaration* cd);

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable * IrStruct::getVtblSymbol()
{
    if (vtbl)
        return vtbl;

    // create the initZ symbol
    std::string initname("_D");
    initname.append(aggrdecl->mangle());
    initname.append("6__vtblZ");

    llvm::GlobalValue::LinkageTypes _linkage = DtoExternalLinkage(aggrdecl);

    LLType* vtblTy = stripModifiers(type)->irtype->isClass()->getVtbl();

    vtbl = new llvm::GlobalVariable(
        *gIR->module, vtblTy, true, _linkage, NULL, initname);

    return vtbl;
}

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable * IrStruct::getClassInfoSymbol()
{
    if (classInfo)
        return classInfo;

    // create the initZ symbol
    std::string initname("_D");
    initname.append(aggrdecl->mangle());

    if (aggrdecl->isInterfaceDeclaration())
        initname.append("11__InterfaceZ");
    else
        initname.append("7__ClassZ");

    llvm::GlobalValue::LinkageTypes _linkage = DtoExternalLinkage(aggrdecl);

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    DtoType(cinfo->type);
    IrTypeClass* tc = stripModifiers(cinfo->type)->irtype->isClass();
    assert(tc && "invalid ClassInfo type");

    // classinfos cannot be constants since they're used as locks for synchronized
    classInfo = new llvm::GlobalVariable(
        *gIR->module, tc->getMemoryLLType(), false, _linkage, NULL, initname);

    // Generate some metadata on this ClassInfo if it's for a class.
    ClassDeclaration* classdecl = aggrdecl->isClassDeclaration();
    if (classdecl && !aggrdecl->isInterfaceDeclaration()) {
        // Gather information
        LLType* type = DtoType(aggrdecl->type);
        LLType* bodyType = llvm::cast<LLPointerType>(type)->getElementType();
        bool hasDestructor = (classdecl->dtor != NULL);
        bool hasCustomDelete = (classdecl->aggDelete != NULL);
        // Construct the fields
        MDNodeField* mdVals[CD_NumFields];
        mdVals[CD_BodyType] = llvm::UndefValue::get(bodyType);
        mdVals[CD_Finalize] = LLConstantInt::get(LLType::getInt1Ty(gIR->context()), hasDestructor);
        mdVals[CD_CustomDelete] = LLConstantInt::get(LLType::getInt1Ty(gIR->context()), hasCustomDelete);
        // Construct the metadata and insert it into the module.
        llvm::SmallString<64> name;
        llvm::NamedMDNode* node = gIR->module->getOrInsertNamedMetadata(
            llvm::Twine(CD_PREFIX, initname).toStringRef(name));
        node->addOperand(llvm::MDNode::get(gIR->context(),
            llvm::makeArrayRef(mdVals, CD_NumFields)));
    }

    return classInfo;
}

//////////////////////////////////////////////////////////////////////////////

LLGlobalVariable * IrStruct::getInterfaceArraySymbol()
{
    if (classInterfacesArray)
        return classInterfacesArray;

    ClassDeclaration* cd = aggrdecl->isClassDeclaration();

    size_t n = stripModifiers(type)->irtype->isClass()->getNumInterfaceVtbls();
    assert(n > 0 && "getting ClassInfo.interfaces storage symbol, but we "
                    "don't implement any interfaces");

    VarDeclarationIter idx(ClassDeclaration::classinfo->fields, 3);
    LLType* InterfaceTy = DtoType(idx->type->nextOf());

    // create Interface[N]
    LLArrayType* array_type = llvm::ArrayType::get(InterfaceTy,n);

    // put it in a global
    std::string name("_D");
    name.append(cd->mangle());
    name.append("16__interfaceInfosZ");

    llvm::GlobalValue::LinkageTypes _linkage = DtoExternalLinkage(aggrdecl);
    classInterfacesArray = new llvm::GlobalVariable(*gIR->module,
        array_type, true, _linkage, NULL, name);

    return classInterfacesArray;
}

//////////////////////////////////////////////////////////////////////////////

LLConstant * IrStruct::getVtblInit()
{
    if (constVtbl)
        return constVtbl;

    IF_LOG Logger::println("Building vtbl initializer");
    LOG_SCOPE;

    ClassDeclaration* cd = aggrdecl->isClassDeclaration();
    assert(cd && "not class");

    std::vector<llvm::Constant*> constants;
    constants.reserve(cd->vtbl.dim);

    // start with the classinfo
    llvm::Constant* c = getClassInfoSymbol();
    c = DtoBitCast(c, DtoType(ClassDeclaration::classinfo->type));
    constants.push_back(c);

    // add virtual function pointers
    size_t n = cd->vtbl.dim;
    for (size_t i = 1; i < n; i++)
    {
        Dsymbol* dsym = static_cast<Dsymbol*>(cd->vtbl.data[i]);
        assert(dsym && "null vtbl member");

        FuncDeclaration* fd = dsym->isFuncDeclaration();
        assert(fd && "vtbl entry not a function");

        if (cd->isAbstract() || (fd->isAbstract() && !fd->fbody))
        {
            c = getNullValue(DtoType(fd->type->pointerTo()));
        }
        else
        {
            fd->codegen(Type::sir);
            assert(fd->ir.irFunc && "invalid vtbl function");
            c = fd->ir.irFunc->func;
            if (cd->isFuncHidden(fd))
            {   /* fd is hidden from the view of this class.
                 * If fd overlaps with any function in the vtbl[], then
                 * issue 'hidden' error.
                 */
                for (size_t j = 1; j < n; j++)
                {   if (j == i)
                        continue;
                    FuncDeclaration *fd2 = static_cast<Dsymbol *>(cd->vtbl.data[j])->isFuncDeclaration();
                    if (!fd2->ident->equals(fd->ident))
                        continue;
                    if (fd->leastAsSpecialized(fd2) || fd2->leastAsSpecialized(fd))
                    {
                        TypeFunction *tf = static_cast<TypeFunction *>(fd->type);
                        if (tf->ty == Tfunction)
                            cd->deprecation(
                                "use of %s%s hidden by %s is deprecated. Use 'alias %s.%s %s;' to introduce base class overload set.",
                                fd->toPrettyChars(),
                                Parameter::argsTypesToChars(tf->parameters, tf->varargs),
                                cd->toChars(),
                                fd->parent->toChars(),
                                fd->toChars(),
                                fd->toChars()
                           );
                        else
                            cd->deprecation("use of %s hidden by %s is deprecated", fd->toPrettyChars(), cd->toChars());

                        c = DtoBitCast(LLVM_D_GetRuntimeFunction(gIR->module, "_d_hidden_func"), c->getType());
                        break;
                    }
                }
            }
        }
        constants.push_back(c);
    }

    // build the constant struct
    LLType* vtblTy = stripModifiers(type)->irtype->isClass()->getVtbl();
    constVtbl = LLConstantStruct::get(isaStruct(vtblTy), constants);

#if 0
   IF_LOG Logger::cout() << "constVtbl type: " << *constVtbl->getType() << std::endl;
   IF_LOG Logger::cout() << "vtbl type: " << *stripModifiers(type)->irtype->isClass()->getVtbl() << std::endl;
#endif

#if 0

    size_t nc = constants.size();

    for (size_t i = 0; i < nc; ++i)
    {
        if (constVtbl->getOperand(i)->getType() != vtblTy->getContainedType(i))
        {
            Logger::cout() << "type mismatch for entry # " << i << " in vtbl initializer" << std::endl;

            constVtbl->getOperand(i)->dump();
            vtblTy->getContainedType(i)->dump();
        }
    }

#endif

    assert(constVtbl->getType() == stripModifiers(type)->irtype->isClass()->getVtbl() &&
        "vtbl initializer type mismatch");

    return constVtbl;
}

//////////////////////////////////////////////////////////////////////////////

LLConstant * IrStruct::getClassInfoInit()
{
    if (constClassInfo)
        return constClassInfo;
    constClassInfo = DtoDefineClassInfo(aggrdecl->isClassDeclaration());
    return constClassInfo;
}

//////////////////////////////////////////////////////////////////////////////

void IrStruct::addBaseClassInits(
    std::vector<llvm::Constant*>& constants,
    ClassDeclaration* base,
    size_t& offset,
    size_t& field_index)
{
    if (base->baseClass)
    {
        addBaseClassInits(constants, base->baseClass, offset, field_index);
    }

    IrTypeClass* tc = stripModifiers(base->type)->irtype->isClass();
    assert(tc);

    // go through fields
    IrTypeAggr::iterator it;
    for (it = tc->def_begin(); it != tc->def_end(); ++it)
    {
        VarDeclaration* vd = *it;

        IF_LOG Logger::println("Adding default field %s %s (+%u)", vd->type->toChars(), vd->toChars(), vd->offset);
        LOG_SCOPE;

        assert(vd->offset >= offset && "default fields not sorted by offset");

        // get next aligned offset for this type
        size_t alignedoffset = realignOffset(offset, vd->type);

        // insert explicit padding?
        if (alignedoffset < vd->offset)
        {
            add_zeros(constants, vd->offset - alignedoffset);
        }

        // add default type
        constants.push_back(get_default_initializer(vd, vd->init));

        // advance offset to right past this field
        offset = vd->offset + vd->type->size();
    }

    // has interface vtbls?
    if (base->vtblInterfaces && base->vtblInterfaces->dim > 0)
    {
        // false when it's not okay to use functions from super classes
        bool newinsts = (base == aggrdecl->isClassDeclaration());

        size_t inter_idx = interfacesWithVtbls.size();

        offset = (offset + PTRSIZE - 1) & ~(PTRSIZE - 1);

        ArrayIter<BaseClass> it2(*base->vtblInterfaces);
        for (; !it2.done(); it2.next())
        {
            BaseClass* b = it2.get();
            constants.push_back(getInterfaceVtbl(b, newinsts, inter_idx));
            offset += PTRSIZE;

            // add to the interface list
            interfacesWithVtbls.push_back(b);
            inter_idx++;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

std::vector<llvm::Constant*> IrStruct::createClassDefaultInitializer()
{
    ClassDeclaration* cd = aggrdecl->isClassDeclaration();
    assert(cd && "invalid class aggregate");

    IF_LOG Logger::println("Building class default initializer %s @ %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;
    IF_LOG Logger::println("Instance size: %u", cd->structsize);

    // find the fields that contribute to the default initializer.
    // these will define the default type.

    std::vector<llvm::Constant*> constants;
    constants.reserve(32);

    // add vtbl
    constants.push_back(getVtblSymbol());
    // add monitor
    constants.push_back(getNullValue(DtoType(Type::tvoid->pointerTo())));

    // we start right after the vtbl and monitor
    size_t offset = PTRSIZE * 2;
    size_t field_index = 2;

    // add data members recursively
    addBaseClassInits(constants, cd, offset, field_index);

    // tail padding?
    if (offset < cd->structsize)
        add_zeros(constants, cd->structsize - offset);

    return constants;
}

//////////////////////////////////////////////////////////////////////////////

llvm::GlobalVariable * IrStruct::getInterfaceVtbl(BaseClass * b, bool new_instance, size_t interfaces_index)
{
    ClassGlobalMap::iterator it = interfaceVtblMap.find(b->base);
    if (it != interfaceVtblMap.end())
        return it->second;

    IF_LOG Logger::println("Building vtbl for implementation of interface %s in class %s",
        b->base->toPrettyChars(), aggrdecl->toPrettyChars());
    LOG_SCOPE;

    ClassDeclaration* cd = aggrdecl->isClassDeclaration();
    assert(cd && "not a class aggregate");

    FuncDeclarations vtbl_array;
    b->fillVtbl(cd, &vtbl_array, new_instance);

    std::vector<llvm::Constant*> constants;
    constants.reserve(vtbl_array.dim);

    // start with the interface info
    VarDeclarationIter interfaces_idx(ClassDeclaration::classinfo->fields, 3);

    // index into the interfaces array
    llvm::Constant* idxs[2] = {
        DtoConstSize_t(0),
        DtoConstSize_t(interfaces_index)
    };

    llvm::Constant* c = llvm::ConstantExpr::getGetElementPtr(
        getInterfaceArraySymbol(), idxs, true);

    constants.push_back(c);

    // add virtual function pointers
    size_t n = vtbl_array.dim;
    for (size_t i = 1; i < n; i++)
    {
        Dsymbol* dsym = static_cast<Dsymbol*>(vtbl_array.data[i]);
        if (dsym == NULL)
        {
            // FIXME
            // why is this null?
            // happens for mini/s.d
            constants.push_back(getNullValue(getVoidPtrType()));
            continue;
        }

        FuncDeclaration* fd = dsym->isFuncDeclaration();
        assert(fd && "vtbl entry not a function");

        assert((!fd->isAbstract() || fd->fbody) &&
            "null symbol in interface implementation vtable");

        fd->codegen(Type::sir);
        assert(fd->ir.irFunc && "invalid vtbl function");

        constants.push_back(fd->ir.irFunc->func);
    }

    // build the vtbl constant
    llvm::Constant* vtbl_constant = LLConstantStruct::getAnon(gIR->context(), constants, false);

    // create the global variable to hold it
    llvm::GlobalValue::LinkageTypes _linkage = DtoExternalLinkage(aggrdecl);

    std::string mangle("_D");
    mangle.append(cd->mangle());
    mangle.append("11__interface");
    mangle.append(b->base->mangle());
    mangle.append("6__vtblZ");

    llvm::GlobalVariable* GV = new llvm::GlobalVariable(
        *gIR->module,
        vtbl_constant->getType(),
        true,
        _linkage,
        vtbl_constant,
        mangle
    );

    // insert into the vtbl map
    interfaceVtblMap.insert(std::make_pair(b->base, GV));

    return GV;
}

//////////////////////////////////////////////////////////////////////////////

LLConstant * IrStruct::getClassInfoInterfaces()
{
    IF_LOG Logger::println("Building ClassInfo.interfaces");
    LOG_SCOPE;

    ClassDeclaration* cd = aggrdecl->isClassDeclaration();
    assert(cd);

    size_t n = interfacesWithVtbls.size();
    assert(stripModifiers(type)->irtype->isClass()->getNumInterfaceVtbls() == n &&
        "inconsistent number of interface vtables in this class");

    VarDeclarationIter interfaces_idx(ClassDeclaration::classinfo->fields, 3);

    if (n == 0)
        return getNullValue(DtoType(interfaces_idx->type));

// Build array of:
//
//     struct Interface
//     {
//         ClassInfo   classinfo;
//         void*[]     vtbl;
//         ptrdiff_t   offset;
//     }

    LLSmallVector<LLConstant*, 6> constants;
    constants.reserve(cd->vtblInterfaces->dim);

    LLType* classinfo_type = DtoType(ClassDeclaration::classinfo->type);
    LLType* voidptrptr_type = DtoType(
        Type::tvoid->pointerTo()->pointerTo());
    VarDeclarationIter idx(ClassDeclaration::classinfo->fields, 3);
    LLStructType* interface_type = isaStruct(DtoType(idx->type->nextOf()));
    assert(interface_type);

    for (size_t i = 0; i < n; ++i)
    {
        BaseClass* it = interfacesWithVtbls[i];

        IF_LOG Logger::println("Adding interface %s", it->base->toPrettyChars());

        IrStruct* irinter = it->base->ir.irStruct;
        assert(irinter && "interface has null IrStruct");
        IrTypeClass* itc = stripModifiers(irinter->type)->irtype->isClass();
        assert(itc && "null interface IrTypeClass");

        // classinfo
        LLConstant* ci = irinter->getClassInfoSymbol();
        ci = DtoBitCast(ci, classinfo_type);

        // vtbl
        LLConstant* vtb;
        // interface get a null
        if (cd->isInterfaceDeclaration())
        {
            vtb = DtoConstSlice(DtoConstSize_t(0), getNullValue(voidptrptr_type));
        }
        else
        {
            ClassGlobalMap::iterator itv = interfaceVtblMap.find(it->base);
            assert(itv != interfaceVtblMap.end() && "interface vtbl not found");
            vtb = itv->second;
            vtb = DtoBitCast(vtb, voidptrptr_type);
            vtb = DtoConstSlice(DtoConstSize_t(itc->getVtblSize()), vtb);
        }

        // offset
        LLConstant* off = DtoConstSize_t(it->offset);

        // create Interface struct
        LLConstant* inits[3] = { ci, vtb, off };
        LLConstant* entry = LLConstantStruct::get(interface_type, llvm::makeArrayRef(inits, 3));
        constants.push_back(entry);
    }

    // create Interface[N]
    LLArrayType* array_type = llvm::ArrayType::get(interface_type, n);

    // create and apply initializer
    LLConstant* arr = LLConstantArray::get(array_type, constants);
    classInterfacesArray->setInitializer(arr);

    // return null, only baseclass provide interfaces
    if (cd->vtblInterfaces->dim == 0)
    {
        return getNullValue(DtoType(interfaces_idx->type));
    }

    // only the interface explicitly implemented by this class
    // (not super classes) should show in ClassInfo
    LLConstant* idxs[2] = {
        DtoConstSize_t(0),
        DtoConstSize_t(n - cd->vtblInterfaces->dim)
    };

    LLConstant* ptr = llvm::ConstantExpr::getGetElementPtr(
        classInterfacesArray, idxs, true);

    // return as a slice
    return DtoConstSlice( DtoConstSize_t(cd->vtblInterfaces->dim), ptr );
}

//////////////////////////////////////////////////////////////////////////////

void IrStruct::initializeInterface()
{
    InterfaceDeclaration* base = aggrdecl->isInterfaceDeclaration();
    assert(base && "not interface");

    // has interface vtbls?
    if (!base->vtblInterfaces)
        return;

    ArrayIter<BaseClass> it(*base->vtblInterfaces);
    for (; !it.done(); it.next())
    {
        // add to the interface list
        interfacesWithVtbls.push_back(it.get());
    }
}

//////////////////////////////////////////////////////////////////////////////
