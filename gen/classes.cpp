#include <sstream>
#include "gen/llvm.h"

#include "mtype.h"
#include "aggregate.h"
#include "init.h"
#include "declaration.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/llvmhelpers.h"
#include "gen/arrays.h"
#include "gen/logger.h"
#include "gen/classes.h"
#include "gen/structs.h"
#include "gen/functions.h"
#include "gen/runtime.h"
#include "gen/dvalue.h"

#include "ir/irstruct.h"

//////////////////////////////////////////////////////////////////////////////////////////

static void LLVM_AddBaseClassInterfaces(ClassDeclaration* target, BaseClasses* bcs)
{
    // add base class data members first
    for (int j=0; j<bcs->dim; j++)
    {
        BaseClass* bc = (BaseClass*)(bcs->data[j]);

        // base *classes* might add more interfaces?
        DtoResolveClass(bc->base);
        LLVM_AddBaseClassInterfaces(target, &bc->base->baseclasses);

        // resolve interfaces while we're at it
        if (bc->base->isInterfaceDeclaration())
        {
            // don't add twice
            if (target->ir.irStruct->interfaceMap.find(bc->base) == target->ir.irStruct->interfaceMap.end())
            {
                Logger::println("adding interface '%s'", bc->base->toPrettyChars());
                IrInterface* iri = new IrInterface(bc);

                // add to map
                target->ir.irStruct->interfaceMap.insert(std::make_pair(bc->base, iri));
                // add to ordered list
                target->ir.irStruct->interfaceVec.push_back(iri);

                // Fill in vtbl[]
                if (!target->isAbstract()) {
                    bc->fillVtbl(target, &bc->vtbl, 0);
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

static void LLVM_AddBaseClassData(IrStruct* irstruct, BaseClasses* bcs)
{
    // add base class data members first
    for (int j=0; j<bcs->dim; j++)
    {
        BaseClass* bc = (BaseClass*)(bcs->data[j]);

        // interfaces never add data fields
        if (bc->base->isInterfaceDeclaration())
            continue;

        // recursively add baseclass data
        LLVM_AddBaseClassData(irstruct, &bc->base->baseclasses);

        Array* arr = &bc->base->fields;
        if (arr->dim == 0)
            continue;

        Logger::println("Adding base class members of %s", bc->base->toChars());
        LOG_SCOPE;

        for (int k=0; k < arr->dim; k++) {
            VarDeclaration* v = (VarDeclaration*)(arr->data[k]);
            Logger::println("Adding field: %s %s", v->type->toChars(), v->toChars());
            // init fields, used to happen in VarDeclaration::toObjFile
            irstruct->addField(v);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoResolveClass(ClassDeclaration* cd)
{
    if (cd->ir.resolved) return;
    cd->ir.resolved = true;

    Logger::println("DtoResolveClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    //printf("resolve class: %s\n", cd->toPrettyChars());

    // get the TypeClass
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    // make sure the IrStruct is created
    IrStruct* irstruct = cd->ir.irStruct;
    if (!irstruct) {
        irstruct = new IrStruct(ts);
        cd->ir.irStruct = irstruct;
    }

    // resolve the base class
    if (cd->baseClass) {
        DtoResolveClass(cd->baseClass);
    }

    // resolve interface vtables
    /*if (cd->vtblInterfaces) {
        Logger::println("Vtbl interfaces for '%s'", cd->toPrettyChars());
        LOG_SCOPE;
        for (int i=0; i < cd->vtblInterfaces->dim; i++) {
            BaseClass *b = (BaseClass *)cd->vtblInterfaces->data[i];
            ClassDeclaration *id = b->base;
            Logger::println("Vtbl interface: '%s'", id->toPrettyChars());
            DtoResolveClass(id);
            // Fill in vtbl[]
            b->fillVtbl(cd, &b->vtbl, 1);
        }
    }*/

    // push state
    gIR->structs.push_back(irstruct);
    gIR->classes.push_back(cd);

    // vector holding the field types
    std::vector<const LLType*> fieldtypes;

    // add vtable
    ts->ir.vtblType = new llvm::PATypeHolder(llvm::OpaqueType::get());
    const LLType* vtabty = getPtrToType(ts->ir.vtblType->get());
    fieldtypes.push_back(vtabty);

    // add monitor
    fieldtypes.push_back(getVoidPtrType());

    // add base class data fields first
    LLVM_AddBaseClassData(irstruct, &cd->baseclasses);

    // add own fields
    Array* fields = &cd->fields;
    for (int k=0; k < fields->dim; k++)
    {
        VarDeclaration* v = (VarDeclaration*)fields->data[k];
        Logger::println("Adding field: %s %s", v->type->toChars(), v->toChars());
        // init fields, used to happen in VarDeclaration::toObjFile
        irstruct->addField(v);
    }

    // then add other members of us, if any
    if(cd->members) {
        for (int k=0; k < cd->members->dim; k++) {
            Dsymbol* dsym = (Dsymbol*)(cd->members->data[k]);
            dsym->toObjFile(0); // TODO: multiobj
        }
    }

    // resolve class data fields (possibly unions)
    Logger::println("doing class fields");

    if (irstruct->offsets.empty())
    {
        Logger::println("has no fields");
    }
    else
    {
        Logger::println("has fields");
        unsigned prevsize = (unsigned)-1;
        unsigned lastoffset = (unsigned)-1;
        const LLType* fieldtype = NULL;
        VarDeclaration* fieldinit = NULL;
        size_t fieldpad = 0;
        int idx = 0;
        for (IrStruct::OffsetMap::iterator i=irstruct->offsets.begin(); i!=irstruct->offsets.end(); ++i) {
            // first iteration
            if (lastoffset == (unsigned)-1) {
                lastoffset = i->first;
                fieldtype = i->second.type;
                fieldinit = i->second.var;
                prevsize = getABITypeSize(fieldtype);
                i->second.var->ir.irField->index = idx;
            }
            // colliding offset?
            else if (lastoffset == i->first) {
                size_t s = getABITypeSize(i->second.type);
                if (s > prevsize) {
                    fieldpad += s - prevsize;
                    prevsize = s;
                }
                cd->ir.irStruct->hasUnions = true;
                i->second.var->ir.irField->index = idx;
            }
            // intersecting offset?
            else if (i->first < (lastoffset + prevsize)) {
                size_t s = getABITypeSize(i->second.type);
                assert((i->first + s) <= (lastoffset + prevsize)); // this holds because all types are aligned to their size
                cd->ir.irStruct->hasUnions = true;
                i->second.var->ir.irField->index = idx;
                i->second.var->ir.irField->indexOffset = (i->first - lastoffset) / s;
            }
            // fresh offset
            else {
                // commit the field
                fieldtypes.push_back(fieldtype);
                irstruct->defaultFields.push_back(fieldinit);
                if (fieldpad) {
                    fieldtypes.push_back(llvm::ArrayType::get(LLType::Int8Ty, fieldpad));
                    irstruct->defaultFields.push_back(NULL);
                    idx++;
                }

                idx++;

                // start new
                lastoffset = i->first;
                fieldtype = i->second.type;
                fieldinit = i->second.var;
                prevsize = getABITypeSize(fieldtype);
                i->second.var->ir.irField->index = idx;
                fieldpad = 0;
            }
        }
        fieldtypes.push_back(fieldtype);
        irstruct->defaultFields.push_back(fieldinit);
        if (fieldpad) {
            fieldtypes.push_back(llvm::ArrayType::get(LLType::Int8Ty, fieldpad));
            irstruct->defaultFields.push_back(NULL);
        }
    }

    // populate interface map
    {
        Logger::println("Adding interfaces to '%s'", cd->toPrettyChars());
        LOG_SCOPE;
        LLVM_AddBaseClassInterfaces(cd, &cd->baseclasses);
        Logger::println("%d interfaces added", cd->ir.irStruct->interfaceVec.size());
        assert(cd->ir.irStruct->interfaceVec.size() == cd->ir.irStruct->interfaceMap.size());
    }

    // add interface vtables at the end
    int interIdx = (int)fieldtypes.size();
    for (IrStruct::InterfaceVectorIter i=irstruct->interfaceVec.begin(); i!=irstruct->interfaceVec.end(); ++i)
    {
        IrInterface* iri = *i;
        ClassDeclaration* id = iri->decl;

        // set vtbl type
        TypeClass* itc = (TypeClass*)id->type;
        const LLType* ivtblTy = itc->ir.vtblType->get();
        assert(ivtblTy);
        if (Logger::enabled())
            Logger::cout() << "interface vtbl type: " << *ivtblTy << '\n';
        fieldtypes.push_back(getPtrToType(ivtblTy));

        // fix the interface vtable type
        assert(iri->vtblTy == NULL);
        iri->vtblTy = new llvm::PATypeHolder(ivtblTy);

        // set index
        iri->index = interIdx++;
    }
    Logger::println("%d interface vtables added", cd->ir.irStruct->interfaceVec.size());
    assert(cd->ir.irStruct->interfaceVec.size() == cd->ir.irStruct->interfaceMap.size());

    // create type
    const llvm::StructType* structtype = llvm::StructType::get(fieldtypes);

    // refine abstract types for stuff like: class C {C next;}
    assert(irstruct->recty != 0);
    llvm::PATypeHolder& spa = irstruct->recty;
    llvm::cast<llvm::OpaqueType>(spa.get())->refineAbstractTypeTo(structtype);
    structtype = isaStruct(spa.get());

    // make it official
    if (!ts->ir.type)
        ts->ir.type = new llvm::PATypeHolder(structtype);
    else
        *ts->ir.type = structtype;
    spa = *ts->ir.type;

    // name the type
    gIR->module->addTypeName(cd->mangle(), ts->ir.type->get());

    // create vtable type
    llvm::GlobalVariable* svtblVar = 0;
#if OPAQUE_VTBLS
    // void*[vtbl.dim]
    const llvm::ArrayType* svtbl_ty
        = llvm::ArrayType::get(getVoidPtrType(), cd->vtbl.dim);

#else
    std::vector<const LLType*> sinits_ty;

    for (int k=0; k < cd->vtbl.dim; k++)
    {
        Dsymbol* dsym = (Dsymbol*)cd->vtbl.data[k];
        assert(dsym);
        //Logger::cout() << "vtblsym: " << dsym->toChars() << '\n';

        if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
            DtoResolveFunction(fd);
            //assert(fd->type->ty == Tfunction);
            //TypeFunction* tf = (TypeFunction*)fd->type;
            //const LLType* fpty = getPtrToType(tf->ir.type->get());
            const llvm::FunctionType* vfty = DtoBaseFunctionType(fd);
            const LLType* vfpty = getPtrToType(vfty);
            sinits_ty.push_back(vfpty);
        }
        else if (ClassDeclaration* cd2 = dsym->isClassDeclaration()) {
            Logger::println("*** ClassDeclaration in vtable: %s", cd2->toChars());
            const LLType* cinfoty;
            if (cd->isInterfaceDeclaration()) {
                cinfoty = DtoInterfaceInfoType();
            }
            else if (cd != ClassDeclaration::classinfo) {
                cinfoty = ClassDeclaration::classinfo->type->ir.type->get();
            }
            else {
                // this is the ClassInfo class, the type is this type
                cinfoty = ts->ir.type->get();
            }
            const LLType* cty = getPtrToType(cinfoty);
            sinits_ty.push_back(cty);
        }
        else
        assert(0);
    }

    // get type
    assert(!sinits_ty.empty());
    const llvm::StructType* svtbl_ty = llvm::StructType::get(sinits_ty);
#endif

    // refine for final vtable type
    llvm::cast<llvm::OpaqueType>(ts->ir.vtblType->get())->refineAbstractTypeTo(svtbl_ty);

#if !OPAQUE_VTBLS
    // name vtbl type
    std::string styname(cd->mangle());
    styname.append("__vtblType");
    gIR->module->addTypeName(styname, svtbl_ty);
#endif

    // log
    //Logger::cout() << "final class type: " << *ts->ir.type->get() << '\n';

    // pop state
    gIR->classes.pop_back();
    gIR->structs.pop_back();

    // queue declare
    gIR->declareList.push_back(cd);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareClass(ClassDeclaration* cd)
{
    if (cd->ir.declared) return;
    cd->ir.declared = true;

    Logger::println("DtoDeclareClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    //printf("declare class: %s\n", cd->toPrettyChars());

    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    assert(cd->ir.irStruct);
    IrStruct* irstruct = cd->ir.irStruct;

    gIR->structs.push_back(irstruct);
    gIR->classes.push_back(cd);

    bool needs_definition = false;
    if (cd->getModule() == gIR->dmodule || DtoIsTemplateInstance(cd)) {
        needs_definition = true;
    }

    llvm::GlobalValue::LinkageTypes _linkage = DtoLinkage(cd);

    // interfaces have no static initializer
    // same goes for abstract classes
    if (!cd->isInterfaceDeclaration() && !cd->isAbstract()) {
        // vtable
        std::string varname("_D");
        varname.append(cd->mangle());
        varname.append("6__vtblZ");
        cd->ir.irStruct->vtbl = new llvm::GlobalVariable(ts->ir.vtblType->get(), true, _linkage, 0, varname, gIR->module);
    }

    // get interface info type
    const llvm::StructType* infoTy = DtoInterfaceInfoType();

    // interface info array
    if (!cd->ir.irStruct->interfaceVec.empty()) {
        // symbol name
        std::string nam = "_D";
        nam.append(cd->mangle());
        nam.append("16__interfaceInfosZ");
        // resolve array type
        const llvm::ArrayType* arrTy = llvm::ArrayType::get(infoTy, cd->ir.irStruct->interfaceVec.size());
        // declare global
        irstruct->interfaceInfosTy = arrTy;
        irstruct->interfaceInfos = new llvm::GlobalVariable(arrTy, true, _linkage, NULL, nam, gIR->module);
    }

    // interfaces have no static initializer
    // same goes for abstract classes
    if (!cd->isInterfaceDeclaration() && !cd->isAbstract()) {
        // interface vtables
        unsigned idx = 0;
        for (IrStruct::InterfaceVectorIter i=irstruct->interfaceVec.begin(); i!=irstruct->interfaceVec.end(); ++i)
        {
            IrInterface* iri = *i;
            ClassDeclaration* id = iri->decl;

            std::string nam("_D");
            nam.append(cd->mangle());
            nam.append("11__interface");
            nam.append(id->mangle());
            nam.append("6__vtblZ");

            assert(iri->vtblTy);
            iri->vtbl = new llvm::GlobalVariable(iri->vtblTy->get(), true, _linkage, 0, nam, gIR->module);
            LLConstant* idxs[2] = {DtoConstUint(0), DtoConstUint(idx)};
            iri->info = llvm::ConstantExpr::getGetElementPtr(irstruct->interfaceInfos, idxs, 2);
            idx++;
        }

        // init
        std::string initname("_D");
        initname.append(cd->mangle());
        initname.append("6__initZ");

        llvm::GlobalVariable* initvar = new llvm::GlobalVariable(ts->ir.type->get(), true, _linkage, NULL, initname, gIR->module);
        cd->ir.irStruct->init = initvar;
    }

    gIR->classes.pop_back();
    gIR->structs.pop_back();

    gIR->constInitList.push_back(cd);
    if (needs_definition)
        gIR->defineList.push_back(cd);

    // classinfo
    DtoDeclareClassInfo(cd);

    // typeinfo
    if (needs_definition)
        DtoTypeInfoOf(cd->type, false);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitClass(ClassDeclaration* cd)
{
    if (cd->ir.initialized) return;
    cd->ir.initialized = true;

    Logger::println("DtoConstInitClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    IrStruct* irstruct = cd->ir.irStruct;
    gIR->structs.push_back(irstruct);
    gIR->classes.push_back(cd);

    // get the struct (class) type
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;
    const llvm::StructType* structtype = isaStruct(ts->ir.type->get());
#if OPAQUE_VTBLS
    const llvm::ArrayType* vtbltype = isaArray(ts->ir.vtblType->get());
#else
    const llvm::StructType* vtbltype = isaStruct(ts->ir.vtblType->get());
#endif

    // make sure each offset knows its default initializer
    for (IrStruct::OffsetMap::iterator i=irstruct->offsets.begin(); i!=irstruct->offsets.end(); ++i)
    {
        IrStruct::Offset* so = &i->second;
        LLConstant* finit = DtoConstFieldInitializer(so->var->loc, so->var->type, so->var->init);
        so->init = finit;
        so->var->ir.irField->constInit = finit;
    }

    // fill out fieldtypes/inits
    std::vector<LLConstant*> fieldinits;

    // first field is always the vtable
    if (cd->isAbstract() || cd->isInterfaceDeclaration())
    {
        const LLType* ptrTy = getPtrToType(ts->ir.vtblType->get());
        fieldinits.push_back(llvm::Constant::getNullValue(ptrTy));
    }
    else
    {
        assert(cd->ir.irStruct->vtbl != 0);
        fieldinits.push_back(cd->ir.irStruct->vtbl);
    }

    // then comes monitor
    fieldinits.push_back(llvm::ConstantPointerNull::get(getPtrToType(LLType::Int8Ty)));

    // go through the field inits and build the default initializer
    size_t nfi = irstruct->defaultFields.size();
    for (size_t i=0; i<nfi; ++i) {
        LLConstant* c;
        if (irstruct->defaultFields[i]) {
            c = irstruct->defaultFields[i]->ir.irField->constInit;
            assert(c);
        }
        else {
            const llvm::ArrayType* arrty = isaArray(structtype->getElementType(i+2));
            assert(arrty);
            std::vector<LLConstant*> vals(arrty->getNumElements(), llvm::ConstantInt::get(LLType::Int8Ty, 0, false));
            c = llvm::ConstantArray::get(arrty, vals);
        }
        fieldinits.push_back(c);
    }

    // last comes interface vtables
    const llvm::StructType* infoTy = DtoInterfaceInfoType();
    for (IrStruct::InterfaceVectorIter i=irstruct->interfaceVec.begin(); i!=irstruct->interfaceVec.end(); ++i)
    {
        IrInterface* iri = *i;
        iri->infoTy = infoTy;

        if (cd->isAbstract() || cd->isInterfaceDeclaration())
        {
            fieldinits.push_back(llvm::Constant::getNullValue(structtype->getElementType(iri->index)));
        }
        else
        {
            assert(iri->vtbl);
            fieldinits.push_back(iri->vtbl);
        }
    }

    // generate initializer
#if 0
    //Logger::cout() << cd->toPrettyChars() << " | " << *structtype << '\n';
    assert(fieldinits.size() == structtype->getNumElements());
    for(size_t i=0; i<structtype->getNumElements(); ++i) {
        Logger::cout() << "s#" << i << " = " << *structtype->getElementType(i) << '\n';
        Logger::cout() << "i#" << i << " = " << *fieldinits[i] << '\n';
        assert(fieldinits[i]->getType() == structtype->getElementType(i));
    }
#endif

#if 0
    for(size_t i=0; i<fieldinits.size(); ++i) {
        Logger::cout() << "i#" << i << " = " << *fieldinits[i]->getType() << '\n';
    }
#endif

    LLConstant* _init = llvm::ConstantStruct::get(structtype, fieldinits);
    assert(_init);
    cd->ir.irStruct->constInit = _init;

    // abstract classes have no static vtable
    // neither do interfaces (on their own, the implementing class supplies the vtable)
    if (!cd->isInterfaceDeclaration() && !cd->isAbstract())
    {
        // generate vtable initializer
        std::vector<LLConstant*> sinits;

        for (int k=0; k < cd->vtbl.dim; k++)
        {
            Dsymbol* dsym = (Dsymbol*)cd->vtbl.data[k];
            assert(dsym);
            //Logger::cout() << "vtblsym: " << dsym->toChars() << '\n';

        #if OPAQUE_VTBLS
            const LLType* targetTy = getVoidPtrType();
        #else
            const LLType* targetTy = vtbltype->getElementType(k);
        #endif

            LLConstant* c = NULL;
            // virtual method
            if (FuncDeclaration* fd = dsym->isFuncDeclaration()) {
                DtoForceDeclareDsymbol(fd);
                assert(fd->ir.irFunc->func);
                c = llvm::cast<llvm::Constant>(fd->ir.irFunc->func);
            }
            // classinfo
            else if (ClassDeclaration* cd2 = dsym->isClassDeclaration()) {
                assert(cd->ir.irStruct->classInfo);
                c = cd->ir.irStruct->classInfo;
            }
            assert(c != NULL);

            // cast if necessary (overridden method)
            if (c->getType() != targetTy)
                c = llvm::ConstantExpr::getBitCast(c, targetTy);
            sinits.push_back(c);
        }
    #if OPAQUE_VTBLS
        const llvm::ArrayType* svtbl_ty = isaArray(ts->ir.vtblType->get());
        cd->ir.irStruct->constVtbl = llvm::ConstantArray::get(svtbl_ty, sinits);
    #else
        const llvm::StructType* svtbl_ty = isaStruct(ts->ir.vtblType->get());
        LLConstant* cvtblInit = llvm::ConstantStruct::get(svtbl_ty, sinits);
        cd->ir.irStruct->constVtbl = llvm::cast<llvm::ConstantStruct>(cvtblInit);
    #endif

        // create interface vtable const initalizers
        for (IrStruct::InterfaceVectorIter i=irstruct->interfaceVec.begin(); i!=irstruct->interfaceVec.end(); ++i)
        {
            IrInterface* iri = *i;
            BaseClass* b = iri->base;

            ClassDeclaration* id = iri->decl;
            assert(id->type->ty == Tclass);
            TypeClass* its = (TypeClass*)id->type;

        #if OPAQUE_VTBLS
            const llvm::ArrayType* ivtbl_ty = isaArray(its->ir.vtblType->get());
        #else
            const llvm::StructType* ivtbl_ty = isaStruct(its->ir.vtblType->get());
        #endif

            // generate interface info initializer
            std::vector<LLConstant*> infoInits;

            // classinfo
            assert(id->ir.irStruct->classInfo);
            LLConstant* c = id->ir.irStruct->classInfo;
            infoInits.push_back(c);

            // vtbl
            const LLType* byteptrptrty = getPtrToType(getPtrToType(LLType::Int8Ty));
            c = llvm::ConstantExpr::getBitCast(iri->vtbl, byteptrptrty);
            c = DtoConstSlice(DtoConstSize_t(b->vtbl.dim), c);
            infoInits.push_back(c);

            // offset
            assert(iri->index >= 0);
            size_t ioff = gTargetData->getStructLayout(isaStruct(cd->type->ir.type->get()))->getElementOffset(iri->index);
            infoInits.push_back(DtoConstUint(ioff));

            // create interface info initializer constant
            iri->infoInit = llvm::cast<llvm::ConstantStruct>(llvm::ConstantStruct::get(iri->infoTy, infoInits));

            // generate vtable initializer
            std::vector<LLConstant*> iinits;

            // add interface info
        #if OPAQUE_VTBLS
            const LLType* targetTy = getVoidPtrType();
            iinits.push_back(llvm::ConstantExpr::getBitCast(iri->info, targetTy));
        #else
            iinits.push_back(iri->info);
        #endif

            for (int k=1; k < b->vtbl.dim; k++)
            {
//                 Logger::println("interface vtbl const init nr. %d", k);
                Dsymbol* dsym = (Dsymbol*)b->vtbl.data[k];

                // error on unimplemented functions, error was already generated earlier
                if(!dsym)
                    fatal();

                FuncDeclaration* fd = dsym->isFuncDeclaration();
                assert(fd);
                DtoForceDeclareDsymbol(fd);
                assert(fd->ir.irFunc->func);
                LLConstant* c = llvm::cast<llvm::Constant>(fd->ir.irFunc->func);

            #if !OPAQUE_VTBLS
                const LLType* targetTy = iri->vtblTy->getContainedType(k);
            #endif

                // we have to bitcast, as the type created in ResolveClass expects a different this type
                c = llvm::ConstantExpr::getBitCast(c, targetTy);
                iinits.push_back(c);
//                 if (Logger::enabled())
//                     Logger::cout() << "c: " << *c << '\n';
            }

        #if OPAQUE_VTBLS
//             if (Logger::enabled())
//                 Logger::cout() << "n: " << iinits.size() << " ivtbl_ty: " << *ivtbl_ty << '\n';
            LLConstant* civtblInit = llvm::ConstantArray::get(ivtbl_ty, iinits);
            iri->vtblInit = llvm::cast<llvm::ConstantArray>(civtblInit);
        #else
            LLConstant* civtblInit = llvm::ConstantStruct::get(ivtbl_ty, iinits);
            iri->vtblInit = llvm::cast<llvm::ConstantStruct>(civtblInit);
        #endif
        }
    }
    // we always generate interfaceinfos as best we can
    else
    {
        // TODO: this is duplicated code from right above... I'm just too lazy to generalise it right now :/
        // create interface vtable const initalizers
        for (IrStruct::InterfaceVectorIter i=irstruct->interfaceVec.begin(); i!=irstruct->interfaceVec.end(); ++i)
        {
            IrInterface* iri = *i;
            BaseClass* b = iri->base;

            ClassDeclaration* id = iri->decl;
            assert(id->type->ty == Tclass);
            TypeClass* its = (TypeClass*)id->type;

            // generate interface info initializer
            std::vector<LLConstant*> infoInits;

            // classinfo
            assert(id->ir.irStruct->classInfo);
            LLConstant* c = id->ir.irStruct->classInfo;
            infoInits.push_back(c);

            // vtbl
            const LLType* byteptrptrty = getPtrToType(getPtrToType(LLType::Int8Ty));
            c = DtoConstSlice(DtoConstSize_t(0), getNullPtr(byteptrptrty));
            infoInits.push_back(c);

            // offset
            assert(iri->index >= 0);
            size_t ioff = gTargetData->getStructLayout(isaStruct(cd->type->ir.type->get()))->getElementOffset(iri->index);
            infoInits.push_back(DtoConstUint(ioff));

            // create interface info initializer constant
            iri->infoInit = llvm::cast<llvm::ConstantStruct>(llvm::ConstantStruct::get(iri->infoTy, infoInits));
        }
    }

    gIR->classes.pop_back();
    gIR->structs.pop_back();
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineClass(ClassDeclaration* cd)
{
    if (cd->ir.defined) return;
    cd->ir.defined = true;

    Logger::println("DtoDefineClass(%s): %s", cd->toPrettyChars(), cd->loc.toChars());
    LOG_SCOPE;

    // get the struct (class) type
    assert(cd->type->ty == Tclass);
    TypeClass* ts = (TypeClass*)cd->type;

    if (cd->getModule() == gIR->dmodule || DtoIsTemplateInstance(cd)) {

        // interfaces don't have static initializer/vtable
        // neither do abstract classes
        if (!cd->isInterfaceDeclaration() && !cd->isAbstract())
        {
            cd->ir.irStruct->init->setInitializer(cd->ir.irStruct->constInit);
            cd->ir.irStruct->vtbl->setInitializer(cd->ir.irStruct->constVtbl);

            // initialize interface vtables
            IrStruct* irstruct = cd->ir.irStruct;
            for (IrStruct::InterfaceVectorIter i=irstruct->interfaceVec.begin(); i!=irstruct->interfaceVec.end(); ++i)
            {
                IrInterface* iri = *i;
                iri->vtbl->setInitializer(iri->vtblInit);
            }
        }

        // always do interface info array when possible
        IrStruct* irstruct = cd->ir.irStruct;
        std::vector<LLConstant*> infoInits;
        for (IrStruct::InterfaceVectorIter i=irstruct->interfaceVec.begin(); i!=irstruct->interfaceVec.end(); ++i)
        {
            IrInterface* iri = *i;
            infoInits.push_back(iri->infoInit);
        }
        // set initializer
        if (!infoInits.empty())
        {
            LLConstant* arrInit = llvm::ConstantArray::get(irstruct->interfaceInfosTy, infoInits);
            irstruct->interfaceInfos->setInitializer(arrInit);
        }
        else
        {
            assert(irstruct->interfaceInfos == NULL);
        }

        // generate classinfo
        DtoDefineClassInfo(cd);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoNewClass(Loc loc, TypeClass* tc, NewExp* newexp)
{
    // resolve type
    DtoForceDeclareDsymbol(tc->sym);

    // allocate
    LLValue* mem;
    if (newexp->onstack)
    {
        mem = DtoAlloca(DtoType(tc)->getContainedType(0), ".newclass_alloca");
    }
    // custom allocator
    else if (newexp->allocator)
    {
        DtoForceDeclareDsymbol(newexp->allocator);
        DFuncValue dfn(newexp->allocator, newexp->allocator->ir.irFunc->func);
        DValue* res = DtoCallFunction(newexp->loc, NULL, &dfn, newexp->newargs);
        mem = DtoBitCast(res->getRVal(), DtoType(tc), ".newclass_custom");
    }
    // default allocator
    else
    {
        llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_allocclass");
        mem = gIR->CreateCallOrInvoke(fn, tc->sym->ir.irStruct->classInfo, ".newclass_gc_alloc")->get();
        mem = DtoBitCast(mem, DtoType(tc), ".newclass_gc");
    }

    // init
    DtoInitClass(tc, mem);

    // init inner-class outer reference
    if (newexp->thisexp)
    {
        Logger::println("Resolving outer class");
        LOG_SCOPE;
        DValue* thisval = newexp->thisexp->toElem(gIR);
        size_t idx = 2 + tc->sym->vthis->ir.irField->index;
        LLValue* src = thisval->getRVal();
        LLValue* dst = DtoGEPi(mem,0,idx,"tmp");
        if (Logger::enabled())
            Logger::cout() << "dst: " << *dst << "\nsrc: " << *src << '\n';
        DtoStore(src, dst);
    }
    // set the context for nested classes
    else if (tc->sym->isNested() && tc->sym->vthis)
    {
        Logger::println("Resolving nested context");
        LOG_SCOPE;

        // get context
        LLValue* nest = DtoNestedContext(loc, tc->sym);

        // store into right location
        size_t idx = 2 + tc->sym->vthis->ir.irField->index;
        LLValue* gep = DtoGEPi(mem,0,idx,"tmp");
        DtoStore(DtoBitCast(nest, gep->getType()->getContainedType(0)), gep);
    }

    // call constructor
    if (newexp->member)
    {
        assert(newexp->arguments != NULL);
        DtoForceDeclareDsymbol(newexp->member);
        DFuncValue dfn(newexp->member, newexp->member->ir.irFunc->func, mem);
        return DtoCallFunction(newexp->loc, tc, &dfn, newexp->arguments);
    }

    // return default constructed class
    return new DImValue(tc, mem);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoInitClass(TypeClass* tc, LLValue* dst)
{
    size_t presz = 2*getABITypeSize(DtoSize_t());
    uint64_t n = getABITypeSize(tc->ir.type->get()) - presz;

    // set vtable field seperately, this might give better optimization
    assert(tc->sym->ir.irStruct->vtbl);
    DtoStore(tc->sym->ir.irStruct->vtbl, DtoGEPi(dst,0,0,"vtbl"));

    // monitor always defaults to zero
    LLValue* tmp = DtoGEPi(dst,0,1,"monitor");
    DtoStore(llvm::Constant::getNullValue(tmp->getType()->getContainedType(0)), tmp);

    // done?
    if (n == 0)
        return;

    // copy the rest from the static initializer
    assert(tc->sym->ir.irStruct->init);
    assert(dst->getType() == tc->sym->ir.irStruct->init->getType());

    LLValue* dstarr = DtoGEPi(dst,0,2,"tmp");
    LLValue* srcarr = DtoGEPi(tc->sym->ir.irStruct->init,0,2,"tmp");

    DtoMemCpy(dstarr, srcarr, DtoConstSize_t(n));
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoFinalizeClass(LLValue* inst)
{
    // get runtime function
    llvm::Function* fn = LLVM_D_GetRuntimeFunction(gIR->module, "_d_callfinalizer");
    // build args
    LLSmallVector<LLValue*,1> arg;
    arg.push_back(DtoBitCast(inst, fn->getFunctionType()->getParamType(0), ".tmp"));
    // call
    gIR->CreateCallOrInvoke(fn, arg.begin(), arg.end(), "");
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastClass(DValue* val, Type* _to)
{
    Logger::println("DtoCastClass(%s, %s)", val->getType()->toChars(), _to->toChars());
    LOG_SCOPE;

    Type* to = _to->toBasetype();
    if (to->ty == Tpointer) {
        Logger::println("to pointer");
        const LLType* tolltype = DtoType(_to);
        LLValue* rval = DtoBitCast(val->getRVal(), tolltype);
        return new DImValue(_to, rval);
    }
    else if (to->ty == Tbool) {
        Logger::println("to bool");
        LLValue* llval = val->getRVal();
        LLValue* zero = LLConstant::getNullValue(llval->getType());
        return new DImValue(_to, gIR->ir->CreateICmpNE(llval, zero, "tmp"));
    }

    assert(to->ty == Tclass);
    TypeClass* tc = (TypeClass*)to;

    Type* from = val->getType()->toBasetype();
    TypeClass* fc = (TypeClass*)from;

    if (tc->sym->isInterfaceDeclaration()) {
        Logger::println("to interface");
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("from interface");
            return DtoDynamicCastInterface(val, _to);
        }
        else {
            Logger::println("from object");
            return DtoDynamicCastObject(val, _to);
        }
    }
    else {
        Logger::println("to class");
        int poffset;
        if (fc->sym->isInterfaceDeclaration()) {
            Logger::println("interface cast");
            return DtoCastInterfaceToObject(val, _to);
        }
        else if (!tc->sym->isInterfaceDeclaration() && tc->sym->isBaseOf(fc->sym,NULL)) {
            Logger::println("static down cast)");
            const LLType* tolltype = DtoType(_to);
            LLValue* rval = DtoBitCast(val->getRVal(), tolltype);
            return new DImValue(_to, rval);
        }
        else {
            Logger::println("dynamic up cast");
            return DtoDynamicCastObject(val, _to);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoDynamicCastObject(DValue* val, Type* _to)
{
    // call:
    // Object _d_dynamic_cast(Object o, ClassInfo c)

    DtoForceDeclareDsymbol(ClassDeclaration::object);
    DtoForceDeclareDsymbol(ClassDeclaration::classinfo);

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_dynamic_cast");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    std::vector<LLValue*> args;

    // Object o
    LLValue* obj = val->getRVal();
    obj = DtoBitCast(obj, funcTy->getParamType(0));
    assert(funcTy->getParamType(0) == obj->getType());

    // ClassInfo c
    TypeClass* to = (TypeClass*)_to->toBasetype();
    DtoForceDeclareDsymbol(to->sym);
    assert(to->sym->ir.irStruct->classInfo);
    LLValue* cinfo = to->sym->ir.irStruct->classInfo;
    // unfortunately this is needed as the implementation of object differs somehow from the declaration
    // this could happen in user code as well :/
    cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));
    assert(funcTy->getParamType(1) == cinfo->getType());

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke2(func, obj, cinfo, "tmp")->get();

    // cast return value
    ret = DtoBitCast(ret, DtoType(_to));

    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoCastInterfaceToObject(DValue* val, Type* to)
{
    // call:
    // Object _d_toObject(void* p)

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_toObject");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    // void* p
    LLValue* tmp = val->getRVal();
    tmp = DtoBitCast(tmp, funcTy->getParamType(0));

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke(func, tmp, "tmp")->get();

    // cast return value
    if (to != NULL)
        ret = DtoBitCast(ret, DtoType(to));
    else
        to = ClassDeclaration::object->type;

    return new DImValue(to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

DValue* DtoDynamicCastInterface(DValue* val, Type* _to)
{
    // call:
    // Object _d_interface_cast(void* p, ClassInfo c)

    DtoForceDeclareDsymbol(ClassDeclaration::object);
    DtoForceDeclareDsymbol(ClassDeclaration::classinfo);

    llvm::Function* func = LLVM_D_GetRuntimeFunction(gIR->module, "_d_interface_cast");
    const llvm::FunctionType* funcTy = func->getFunctionType();

    std::vector<LLValue*> args;

    // void* p
    LLValue* ptr = val->getRVal();
    ptr = DtoBitCast(ptr, funcTy->getParamType(0));

    // ClassInfo c
    TypeClass* to = (TypeClass*)_to->toBasetype();
    DtoForceDeclareDsymbol(to->sym);
    assert(to->sym->ir.irStruct->classInfo);
    LLValue* cinfo = to->sym->ir.irStruct->classInfo;
    // unfortunately this is needed as the implementation of object differs somehow from the declaration
    // this could happen in user code as well :/
    cinfo = DtoBitCast(cinfo, funcTy->getParamType(1));

    // call it
    LLValue* ret = gIR->CreateCallOrInvoke2(func, ptr, cinfo, "tmp")->get();

    // cast return value
    ret = DtoBitCast(ret, DtoType(_to));

    return new DImValue(_to, ret);
}

//////////////////////////////////////////////////////////////////////////////////////////

static unsigned LLVM_ClassOffsetToIndex(ClassDeclaration* cd, unsigned os, unsigned& idx)
{
    // start at the bottom of the inheritance chain
    if (cd->baseClass != 0) {
        unsigned o = LLVM_ClassOffsetToIndex(cd->baseClass, os, idx);
        if (o != (unsigned)-1)
            return o;
    }

    // check this class
    unsigned i;
    for (i=0; i<cd->fields.dim; ++i) {
        VarDeclaration* vd = (VarDeclaration*)cd->fields.data[i];
        if (os == vd->offset)
            return i+idx;
    }
    idx += i;

    return (unsigned)-1;
}

//////////////////////////////////////////////////////////////////////////////////////////

void ClassDeclaration::offsetToIndex(Type* t, unsigned os, std::vector<unsigned>& result)
{
    unsigned idx = 0;
    unsigned r = LLVM_ClassOffsetToIndex(this, os, idx);
    assert(r != (unsigned)-1 && "Offset not found in any aggregate field");
    // vtable is 0, monitor is 1
    r += 2;
    // the final index was not pushed
    result.push_back(r); 
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoIndexClass(LLValue* src, ClassDeclaration* cd, VarDeclaration* vd)
{
    Logger::println("indexing class field %s:", vd->toPrettyChars());
    LOG_SCOPE;

    if (Logger::enabled())
        Logger::cout() << "src: " << *src << '\n';

    // vd must be a field
    IrField* field = vd->ir.irField;
    assert(field);

    unsigned idx = field->index + 2; // vtbl & monitor
    unsigned off = field->indexOffset;

    const LLType* st = DtoType(cd->type);
    src = DtoBitCast(src, st);

    LLValue* val = DtoGEPi(src, 0,idx);
    val = DtoBitCast(val, getPtrToType(DtoType(vd->type)));

    if (off)
        val = DtoGEPi1(val, off);

    if (Logger::enabled())
        Logger::cout() << "value: " << *val << '\n';

    return val;
}

//////////////////////////////////////////////////////////////////////////////////////////

LLValue* DtoVirtualFunctionPointer(DValue* inst, FuncDeclaration* fdecl)
{
    assert(fdecl->isVirtual());//fdecl->isAbstract() || (!fdecl->isFinal() && fdecl->isVirtual()));
    assert(fdecl->vtblIndex > 0);
    assert(inst->getType()->toBasetype()->ty == Tclass);

    LLValue* vthis = inst->getRVal();
    if (Logger::enabled())
        Logger::cout() << "vthis: " << *vthis << '\n';

    LLValue* funcval;
    funcval = DtoGEPi(vthis, 0, 0, "tmp");
    funcval = DtoLoad(funcval);
    funcval = DtoGEPi(funcval, 0, fdecl->vtblIndex, fdecl->toPrettyChars());
    funcval = DtoLoad(funcval);

    if (Logger::enabled())
        Logger::cout() << "funcval: " << *funcval << '\n';

#if OPAQUE_VTBLS
    funcval = DtoBitCast(funcval, getPtrToType(DtoType(fdecl->type)));
    if (Logger::enabled())
        Logger::cout() << "funcval casted: " << *funcval << '\n';
#endif

    return funcval;
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareClassInfo(ClassDeclaration* cd)
{
    if (cd->ir.irStruct->classDeclared) return;
    cd->ir.irStruct->classDeclared = true;

    Logger::println("DtoDeclareClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    DtoResolveClass(cinfo);

    std::string gname("_D");
    gname.append(cd->mangle());
    if (!cd->isInterfaceDeclaration())
        gname.append("7__ClassZ");
    else
        gname.append("11__InterfaceZ");

    const LLType* st = cinfo->type->ir.type->get();

    cd->ir.irStruct->classInfo = new llvm::GlobalVariable(st, false, DtoLinkage(cd), NULL, gname, gIR->module);
}

static LLConstant* build_offti_entry(ClassDeclaration* cd, VarDeclaration* vd)
{
    std::vector<const LLType*> types;
    std::vector<LLConstant*> inits;

    types.push_back(DtoSize_t());

    assert(vd->ir.irField);
    assert(vd->ir.irField->index >= 0);
    size_t offset = gTargetData->getStructLayout(isaStruct(cd->type->ir.type->get()))->getElementOffset(vd->ir.irField->index+2);
    inits.push_back(DtoConstSize_t(offset));

    LLConstant* c = DtoTypeInfoOf(vd->type, true);
    const LLType* tiTy = c->getType();
    //Logger::cout() << "tiTy = " << *tiTy << '\n';

    types.push_back(tiTy);
    inits.push_back(c);

    const llvm::StructType* sTy = llvm::StructType::get(types);
    return llvm::ConstantStruct::get(sTy, inits);
}

static LLConstant* build_offti_array(ClassDeclaration* cd, LLConstant* init)
{
    const llvm::StructType* initTy = isaStruct(init->getType());
    assert(initTy);

    std::vector<LLConstant*> arrayInits;
    for (ClassDeclaration *cd2 = cd; cd2; cd2 = cd2->baseClass)
    {
    if (cd2->members)
    {
        for (size_t i = 0; i < cd2->members->dim; i++)
        {
        Dsymbol *sm = (Dsymbol *)cd2->members->data[i];
        if (VarDeclaration* vd = sm->isVarDeclaration()) // is this enough?
        {
            if (!vd->isDataseg()) // static members dont have an offset!
            {
                LLConstant* c = build_offti_entry(cd, vd);
                assert(c);
                arrayInits.push_back(c);
            }
        }
        }
    }
    }

    size_t ninits = arrayInits.size();
    LLConstant* size = DtoConstSize_t(ninits);
    LLConstant* ptr;

    if (ninits > 0) {
        // OffsetTypeInfo type
        std::vector<const LLType*> elemtypes;
        elemtypes.push_back(DtoSize_t());
        const LLType* tiTy = getPtrToType(Type::typeinfo->type->ir.type->get());
        elemtypes.push_back(tiTy);
        const llvm::StructType* sTy = llvm::StructType::get(elemtypes);

        // array type
        const llvm::ArrayType* arrTy = llvm::ArrayType::get(sTy, ninits);
        LLConstant* arrInit = llvm::ConstantArray::get(arrTy, arrayInits);

        std::string name(cd->type->vtinfo->toChars());
        name.append("__OffsetTypeInfos");

        llvm::GlobalVariable* gvar = new llvm::GlobalVariable(arrTy,true,DtoInternalLinkage(cd),arrInit,name,gIR->module);
        ptr = llvm::ConstantExpr::getBitCast(gvar, getPtrToType(sTy));
    }
    else {
        ptr = llvm::ConstantPointerNull::get(isaPointer(initTy->getElementType(1)));
    }

    return DtoConstSlice(size, ptr);
}

static LLConstant* build_class_dtor(ClassDeclaration* cd)
{
    FuncDeclaration* dtor = cd->dtor;

    // if no destructor emit a null
    if (!dtor)
        return getNullPtr(getVoidPtrType());

    DtoForceDeclareDsymbol(dtor);
    return llvm::ConstantExpr::getBitCast(dtor->ir.irFunc->func, getPtrToType(LLType::Int8Ty));
}

static unsigned build_classinfo_flags(ClassDeclaration* cd)
{
    // adapted from original dmd code
    unsigned flags = 0;
    //flags |= isCOMclass(); // IUnknown
    bool hasOffTi = false;
    if (cd->ctor) flags |= 8;
    for (ClassDeclaration *cd2 = cd; cd2; cd2 = cd2->baseClass)
    {
    if (cd2->members)
    {
        for (size_t i = 0; i < cd2->members->dim; i++)
        {
        Dsymbol *sm = (Dsymbol *)cd2->members->data[i];
        if (sm->isVarDeclaration() && !sm->isVarDeclaration()->isDataseg()) // is this enough?
            hasOffTi = true;
        //printf("sm = %s %s\n", sm->kind(), sm->toChars());
        if (sm->hasPointers())
            goto L2;
        }
    }
    }
    flags |= 2;         // no pointers
L2:
    if (hasOffTi)
        flags |= 4;
    return flags;
}

void DtoDefineClassInfo(ClassDeclaration* cd)
{
//     The layout is:
//        {
//         void **vptr;
//         monitor_t monitor;
//         byte[] initializer;     // static initialization data
//         char[] name;        // class name
//         void *[] vtbl;
//         Interface[] interfaces;
//         ClassInfo *base;        // base class
//         void *destructor;
//         void *invariant;        // class invariant
//         uint flags;
//         void *deallocator;
//         OffsetTypeInfo[] offTi;
//         void *defaultConstructor;
//        }

    if (cd->ir.irStruct->classDefined) return;
    cd->ir.irStruct->classDefined = true;

    Logger::println("DtoDefineClassInfo(%s)", cd->toChars());
    LOG_SCOPE;

    assert(cd->type->ty == Tclass);
    assert(cd->ir.irStruct->classInfo);

    TypeClass* cdty = (TypeClass*)cd->type;
    if (!cd->isInterfaceDeclaration() && !cd->isAbstract()) {
        assert(cd->ir.irStruct->init);
        assert(cd->ir.irStruct->constInit);
        assert(cd->ir.irStruct->vtbl);
        assert(cd->ir.irStruct->constVtbl);
    }

    // holds the list of initializers for llvm
    std::vector<LLConstant*> inits;

    ClassDeclaration* cinfo = ClassDeclaration::classinfo;
    DtoForceConstInitDsymbol(cinfo);
    assert(cinfo->ir.irStruct->constInit);

    // def init constant
    LLConstant* defc = cinfo->ir.irStruct->constInit;
    assert(defc);

    LLConstant* c;

    // own vtable
    c = defc->getOperand(0);
    assert(c);
    inits.push_back(c);

    // monitor
    c = defc->getOperand(1);
    inits.push_back(c);

    // byte[] init
    const LLType* byteptrty = getPtrToType(LLType::Int8Ty);
    if (cd->isInterfaceDeclaration() || cd->isAbstract()) {
        c = defc->getOperand(2);
    }
    else {
        c = llvm::ConstantExpr::getBitCast(cd->ir.irStruct->init, byteptrty);
        size_t initsz = getABITypeSize(cd->ir.irStruct->constInit->getType());
        c = DtoConstSlice(DtoConstSize_t(initsz), c);
    }
    inits.push_back(c);

    // class name
    // from dmd
    char *name = cd->ident->toChars();
    size_t namelen = strlen(name);
    if (!(namelen > 9 && memcmp(name, "TypeInfo_", 9) == 0))
    {
        name = cd->toPrettyChars();
        namelen = strlen(name);
    }
    c = DtoConstString(name);
    inits.push_back(c);

    // vtbl array
    if (cd->isInterfaceDeclaration() || cd->isAbstract()) {
        c = defc->getOperand(4);
    }
    else {
        const LLType* byteptrptrty = getPtrToType(byteptrty);
        assert(!cd->ir.irStruct->vtbl->getType()->isAbstract());
        c = llvm::ConstantExpr::getBitCast(cd->ir.irStruct->vtbl, byteptrptrty);
        assert(!cd->ir.irStruct->constVtbl->getType()->isAbstract());
        size_t vtblsz = 0;
        llvm::ConstantArray* constVtblArray = llvm::dyn_cast<llvm::ConstantArray>(cd->ir.irStruct->constVtbl);
        if(constVtblArray) {
            vtblsz = constVtblArray->getType()->getNumElements();
        }
        c = DtoConstSlice(DtoConstSize_t(vtblsz), c);
    }
    inits.push_back(c);

    // interfaces array
    IrStruct* irstruct = cd->ir.irStruct;
    if (cd->isInterfaceDeclaration() || !irstruct->interfaceInfos || cd->isAbstract()) {
        c = defc->getOperand(5);
    }
    else {
        const LLType* t = defc->getOperand(5)->getType()->getContainedType(1);
        c = llvm::ConstantExpr::getBitCast(irstruct->interfaceInfos, t);
        size_t iisz = irstruct->interfaceInfosTy->getNumElements();
        c = DtoConstSlice(DtoConstSize_t(iisz), c);
    }
    inits.push_back(c);

    // base classinfo
    if (cd->baseClass && !cd->isInterfaceDeclaration() && !cd->isAbstract()) {
        DtoDeclareClassInfo(cd->baseClass);
        c = cd->baseClass->ir.irStruct->classInfo;
        assert(c);
        inits.push_back(c);
    }
    else {
        // null
        c = defc->getOperand(6);
        inits.push_back(c);
    }

    // destructor
    if (cd->isInterfaceDeclaration() || cd->isAbstract()) {
        c = defc->getOperand(7);
    }
    else {
        c = build_class_dtor(cd);
    }
    inits.push_back(c);

    // invariant
    if (cd->inv) {
        DtoForceDeclareDsymbol(cd->inv);
        c = cd->inv->ir.irFunc->func;
        c = llvm::ConstantExpr::getBitCast(c, defc->getOperand(8)->getType());
    }
    else {
        c = defc->getOperand(8);
    }
    inits.push_back(c);

    // uint flags
    if (cd->isInterfaceDeclaration() || cd->isAbstract()) {
        c = defc->getOperand(9);
    }
    else {
        unsigned flags = build_classinfo_flags(cd);
        c = DtoConstUint(flags);
    }
    inits.push_back(c);

    // deallocator
    if (cd->aggDelete) {
        DtoForceDeclareDsymbol(cd->aggDelete);
        c = cd->aggDelete->ir.irFunc->func;
        c = llvm::ConstantExpr::getBitCast(c, defc->getOperand(10)->getType());
    }
    else {
        c = defc->getOperand(10);
    }
    inits.push_back(c);

    // offset typeinfo
    if (cd->isInterfaceDeclaration() || cd->isAbstract()) {
        c = defc->getOperand(11);
    }
    else {
        c = build_offti_array(cd, defc->getOperand(11));
    }
    inits.push_back(c);

    // default constructor
    if (cd->defaultCtor && !cd->isInterfaceDeclaration() && !cd->isAbstract()) {
        DtoForceDeclareDsymbol(cd->defaultCtor);
        c = isaConstant(cd->defaultCtor->ir.irFunc->func);
        const LLType* toTy = defc->getOperand(12)->getType();
        c = llvm::ConstantExpr::getBitCast(c, toTy);
    }
    else {
        c = defc->getOperand(12);
    }
    inits.push_back(c);

#if DMDV2

    // xgetMembers
    c = defc->getOperand(13);
    inits.push_back(c);

#else
#endif

    /*size_t n = inits.size();
    for (size_t i=0; i<n; ++i)
    {
        Logger::cout() << "inits[" << i << "]: " << *inits[i] << '\n';
    }*/

    // build the initializer
    const llvm::StructType* st = isaStruct(defc->getType());
    LLConstant* finalinit = llvm::ConstantStruct::get(st, inits);
    //Logger::cout() << "built the classinfo initializer:\n" << *finalinit <<'\n';

    cd->ir.irStruct->constClassInfo = finalinit;
    cd->ir.irStruct->classInfo->setInitializer(finalinit);
}
