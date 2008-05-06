#include <algorithm>

#include "gen/llvm.h"

#include "mtype.h"
#include "aggregate.h"
#include "init.h"
#include "declaration.h"

#include "gen/irstate.h"
#include "gen/tollvm.h"
#include "gen/arrays.h"
#include "gen/logger.h"
#include "gen/structs.h"

#include "ir/irstruct.h"

//////////////////////////////////////////////////////////////////////////////////////////

const llvm::Type* DtoStructType(Type* t)
{
    assert(0);
    std::vector<const llvm::Type*> types;
    return llvm::StructType::get(types);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoStructZeroInit(llvm::Value* v)
{
    assert(gIR);
    uint64_t n = getTypeStoreSize(v->getType()->getContainedType(0));
    //llvm::Type* sarrty = getPtrToType(llvm::ArrayType::get(llvm::Type::Int8Ty, n));
    const llvm::Type* sarrty = getPtrToType(llvm::Type::Int8Ty);

    llvm::Value* sarr = DtoBitCast(v, sarrty);

    llvm::Function* fn = LLVM_DeclareMemSet32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = sarr;
    llargs[1] = llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false);
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    llvm::Value* ret = new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());

    return ret;
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoStructCopy(llvm::Value* dst, llvm::Value* src)
{
    Logger::cout() << "dst = " << *dst << " src = " << *src << '\n';
    assert(dst->getType() == src->getType());
    assert(gIR);

    uint64_t n = getTypeStoreSize(dst->getType()->getContainedType(0));
    //llvm::Type* sarrty = getPtrToType(llvm::ArrayType::get(llvm::Type::Int8Ty, n));
    const llvm::Type* arrty = getPtrToType(llvm::Type::Int8Ty);

    llvm::Value* dstarr = new llvm::BitCastInst(dst,arrty,"tmp",gIR->scopebb());
    llvm::Value* srcarr = new llvm::BitCastInst(src,arrty,"tmp",gIR->scopebb());

    llvm::Function* fn = LLVM_DeclareMemCpy32();
    std::vector<llvm::Value*> llargs;
    llargs.resize(4);
    llargs[0] = dstarr;
    llargs[1] = srcarr;
    llargs[2] = llvm::ConstantInt::get(llvm::Type::Int32Ty, n, false);
    llargs[3] = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0, false);

    return new llvm::CallInst(fn, llargs.begin(), llargs.end(), "", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////
llvm::Constant* DtoConstStructInitializer(StructInitializer* si)
{
    Logger::println("DtoConstStructInitializer: %s", si->toChars());
    LOG_SCOPE;

    TypeStruct* ts = (TypeStruct*)si->ad->type;

    const llvm::StructType* structtype = isaStruct(ts->ir.type->get());
    Logger::cout() << "llvm struct type: " << *structtype << '\n';

    assert(si->value.dim == si->vars.dim);

    std::vector<DUnionIdx> inits;
    for (int i = 0; i < si->value.dim; ++i)
    {
        Initializer* ini = (Initializer*)si->value.data[i];
        assert(ini);
        VarDeclaration* vd = (VarDeclaration*)si->vars.data[i];
        assert(vd);
        llvm::Constant* v = DtoConstInitializer(vd->type, ini);
        inits.push_back(DUnionIdx(vd->ir.irField->index, vd->ir.irField->indexOffset, v));
    }

    DtoConstInitStruct((StructDeclaration*)si->ad);
    return si->ad->ir.irStruct->dunion->getConst(inits);
}

//////////////////////////////////////////////////////////////////////////////////////////

llvm::Value* DtoIndexStruct(llvm::Value* ptr, StructDeclaration* sd, Type* t, unsigned os, std::vector<unsigned>& idxs)
{
    Logger::println("checking for offset %u type %s:", os, t->toChars());
    LOG_SCOPE;

    if (idxs.empty())
        idxs.push_back(0);

    const llvm::Type* llt = getPtrToType(DtoType(t));
    const llvm::Type* st = getPtrToType(DtoType(sd->type));
    if (ptr->getType() != st) {
        assert(sd->ir.irStruct->hasUnions);
        ptr = gIR->ir->CreateBitCast(ptr, st, "tmp");
    }

    for (unsigned i=0; i<sd->fields.dim; ++i) {
        VarDeclaration* vd = (VarDeclaration*)sd->fields.data[i];
        Type* vdtype = DtoDType(vd->type);
        //Logger::println("found %u type %s", vd->offset, vdtype->toChars());
        assert(vd->ir.irField->index >= 0);
        if (os == vd->offset && vdtype == t) {
            idxs.push_back(vd->ir.irField->index);
            ptr = DtoGEP(ptr, idxs, "tmp");
            if (ptr->getType() != llt)
                ptr = gIR->ir->CreateBitCast(ptr, llt, "tmp");
            if (vd->ir.irField->indexOffset)
                ptr = new llvm::GetElementPtrInst(ptr, DtoConstUint(vd->ir.irField->indexOffset), "tmp", gIR->scopebb());
            return ptr;
        }
        else if (vdtype->ty == Tstruct && (vd->offset + vdtype->size()) > os) {
            TypeStruct* ts = (TypeStruct*)vdtype;
            StructDeclaration* ssd = ts->sym;
            idxs.push_back(vd->ir.irField->index);
            if (vd->ir.irField->indexOffset) {
                Logger::println("has union field offset");
                ptr = DtoGEP(ptr, idxs, "tmp");
                if (ptr->getType() != llt)
                    ptr = DtoBitCast(ptr, llt);
                ptr = new llvm::GetElementPtrInst(ptr, DtoConstUint(vd->ir.irField->indexOffset), "tmp", gIR->scopebb());
                std::vector<unsigned> tmp;
                return DtoIndexStruct(ptr, ssd, t, os-vd->offset, tmp);
            }
            else {
                const llvm::Type* sty = getPtrToType(DtoType(vd->type));
                if (ptr->getType() != sty) {
                    ptr = DtoBitCast(ptr, sty);
                    std::vector<unsigned> tmp;
                    return DtoIndexStruct(ptr, ssd, t, os-vd->offset, tmp);
                }
                else {
                    return DtoIndexStruct(ptr, ssd, t, os-vd->offset, idxs);
                }
            }
        }
    }

    size_t llt_sz = getTypeStoreSize(llt->getContainedType(0));
    assert(os % llt_sz == 0);
    ptr = DtoBitCast(ptr, llt);
    return new llvm::GetElementPtrInst(ptr, DtoConstUint(os / llt_sz), "tmp", gIR->scopebb());
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoResolveStruct(StructDeclaration* sd)
{
    if (sd->ir.resolved) return;
    sd->ir.resolved = true;

    Logger::println("DtoResolveStruct(%s): %s", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    if (sd->prot() == PROTprivate && sd->getModule() != gIR->dmodule)
        Logger::println("using a private struct from outside its module");

    TypeStruct* ts = (TypeStruct*)DtoDType(sd->type);

    IrStruct* irstruct = new IrStruct(ts);
    sd->ir.irStruct = irstruct;
    gIR->structs.push_back(irstruct);

    // fields
    Array* arr = &sd->fields;
    for (int k=0; k < arr->dim; k++) {
        VarDeclaration* v = (VarDeclaration*)arr->data[k];
        v->toObjFile();
    }

    bool thisModule = false;
    if (sd->getModule() == gIR->dmodule)
        thisModule = true;

    // methods
    arr = sd->members;
    for (int k=0; k < arr->dim; k++) {
        Dsymbol* s = (Dsymbol*)arr->data[k];
        if (FuncDeclaration* fd = s->isFuncDeclaration()) {
            if (thisModule || (fd->prot() != PROTprivate)) {
                fd->toObjFile();
            }
        }
        else if (s->isAttribDeclaration()) {
            s->toObjFile();
        }
        else {
            Logger::println("Ignoring dsymbol '%s' in this->members of kind '%s'", s->toPrettyChars(), s->kind());
        }
    }

    /*for (int k=0; k < sd->members->dim; k++) {
        Dsymbol* dsym = (Dsymbol*)(sd->members->data[k]);
        dsym->toObjFile();
    }*/

    Logger::println("doing struct fields");

    const llvm::StructType* structtype = 0;
    std::vector<const llvm::Type*> fieldtypes;

    if (irstruct->offsets.empty())
    {
        Logger::println("has no fields");
        fieldtypes.push_back(llvm::Type::Int8Ty);
        structtype = llvm::StructType::get(fieldtypes);
    }
    else
    {
        Logger::println("has fields");
        unsigned prevsize = (unsigned)-1;
        unsigned lastoffset = (unsigned)-1;
        const llvm::Type* fieldtype = NULL;
        VarDeclaration* fieldinit = NULL;
        size_t fieldpad = 0;
        int idx = 0;
        for (IrStruct::OffsetMap::iterator i=irstruct->offsets.begin(); i!=irstruct->offsets.end(); ++i) {
            // first iteration
            if (lastoffset == (unsigned)-1) {
                lastoffset = i->first;
                assert(lastoffset == 0);
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
                sd->ir.irStruct->hasUnions = true;
                i->second.var->ir.irField->index = idx;
            }
            // intersecting offset?
            else if (i->first < (lastoffset + prevsize)) {
                size_t s = getABITypeSize(i->second.type);
                assert((i->first + s) <= (lastoffset + prevsize)); // this holds because all types are aligned to their size
                sd->ir.irStruct->hasUnions = true;
                i->second.var->ir.irField->index = idx;
                i->second.var->ir.irField->indexOffset = (i->first - lastoffset) / s;
            }
            // fresh offset
            else {
                // commit the field
                fieldtypes.push_back(fieldtype);
                irstruct->defaultFields.push_back(fieldinit);
                if (fieldpad) {
                    fieldtypes.push_back(llvm::ArrayType::get(llvm::Type::Int8Ty, fieldpad));
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
            fieldtypes.push_back(llvm::ArrayType::get(llvm::Type::Int8Ty, fieldpad));
            irstruct->defaultFields.push_back(NULL);
        }

        Logger::println("creating struct type");
        structtype = llvm::StructType::get(fieldtypes);
    }

    // refine abstract types for stuff like: struct S{S* next;}
    if (irstruct->recty != 0)
    {
        llvm::PATypeHolder& pa = irstruct->recty;
        llvm::cast<llvm::OpaqueType>(pa.get())->refineAbstractTypeTo(structtype);
        structtype = isaStruct(pa.get());
    }

    assert(ts->ir.type == 0);
    ts->ir.type = new llvm::PATypeHolder(structtype);

    if (sd->parent->isModule()) {
        gIR->module->addTypeName(sd->mangle(),structtype);
    }

    gIR->structs.pop_back();

    gIR->declareList.push_back(sd);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDeclareStruct(StructDeclaration* sd)
{
    if (sd->ir.declared) return;
    sd->ir.declared = true;

    Logger::println("DtoDeclareStruct(%s): %s", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    TypeStruct* ts = (TypeStruct*)DtoDType(sd->type);

    std::string initname("_D");
    initname.append(sd->mangle());
    initname.append("6__initZ");

    llvm::GlobalValue::LinkageTypes _linkage = DtoExternalLinkage(sd);
    llvm::GlobalVariable* initvar = new llvm::GlobalVariable(ts->ir.type->get(), true, _linkage, NULL, initname, gIR->module);
    sd->ir.irStruct->init = initvar;

    gIR->constInitList.push_back(sd);
    if (DtoIsTemplateInstance(sd) || sd->getModule() == gIR->dmodule)
        gIR->defineList.push_back(sd);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoConstInitStruct(StructDeclaration* sd)
{
    if (sd->ir.initialized) return;
    sd->ir.initialized = true;

    Logger::println("DtoConstInitStruct(%s): %s", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    IrStruct* irstruct = sd->ir.irStruct;
    gIR->structs.push_back(irstruct);

    // make sure each offset knows its default initializer
    for (IrStruct::OffsetMap::iterator i=irstruct->offsets.begin(); i!=irstruct->offsets.end(); ++i)
    {
        IrStruct::Offset* so = &i->second;
        llvm::Constant* finit = DtoConstFieldInitializer(so->var->type, so->var->init);
        so->init = finit;
        so->var->ir.irField->constInit = finit;
    }

    const llvm::StructType* structtype = isaStruct(sd->type->ir.type->get());

    // go through the field inits and build the default initializer
    std::vector<llvm::Constant*> fieldinits_ll;
    size_t nfi = irstruct->defaultFields.size();
    for (size_t i=0; i<nfi; ++i) {
        llvm::Constant* c;
        if (irstruct->defaultFields[i] != NULL) {
            c = irstruct->defaultFields[i]->ir.irField->constInit;
            assert(c);
        }
        else {
            const llvm::ArrayType* arrty = isaArray(structtype->getElementType(i));
            std::vector<llvm::Constant*> vals(arrty->getNumElements(), llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false));
            c = llvm::ConstantArray::get(arrty, vals);
        }
        fieldinits_ll.push_back(c);
    }

    // generate the union mapper
    sd->ir.irStruct->dunion = new DUnion; // uses gIR->topstruct()

    // always generate the constant initalizer
    if (!sd->zeroInit) {
        Logger::println("Not zero initialized");
        //assert(tk == gIR->gIR->topstruct()().size());
        #ifndef LLVMD_NO_LOGGER
        Logger::cout() << "struct type: " << *structtype << '\n';
        for (size_t k=0; k<fieldinits_ll.size(); ++k) {
            Logger::cout() << "Type:" << '\n';
            Logger::cout() << *fieldinits_ll[k]->getType() << '\n';
            Logger::cout() << "Value:" << '\n';
            Logger::cout() << *fieldinits_ll[k] << '\n';
        }
        Logger::cout() << "Initializer printed" << '\n';
        #endif
        sd->ir.irStruct->constInit = llvm::ConstantStruct::get(structtype,fieldinits_ll);
    }
    else {
        Logger::println("Zero initialized");
        sd->ir.irStruct->constInit = llvm::ConstantAggregateZero::get(structtype);
    }

    gIR->structs.pop_back();

    // emit typeinfo
    if (sd->getModule() == gIR->dmodule && sd->llvmInternal != LLVMnotypeinfo)
        sd->type->getTypeInfo(NULL);
}

//////////////////////////////////////////////////////////////////////////////////////////

void DtoDefineStruct(StructDeclaration* sd)
{
    if (sd->ir.defined) return;
    sd->ir.defined = true;

    Logger::println("DtoDefineStruct(%s): %s", sd->toChars(), sd->loc.toChars());
    LOG_SCOPE;

    assert(sd->type->ty == Tstruct);
    TypeStruct* ts = (TypeStruct*)sd->type;
    sd->ir.irStruct->init->setInitializer(sd->ir.irStruct->constInit);

    sd->ir.DModule = gIR->dmodule;
}

//////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////   D UNION HELPER CLASS   ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

DUnion::DUnion()
{
    DUnionField* f = NULL;
    IrStruct* topstruct = gIR->topstruct();
    bool unions = false;
    for (IrStruct::OffsetMap::iterator i=topstruct->offsets.begin(); i!=topstruct->offsets.end(); ++i)
    {
        unsigned o = i->first;
        IrStruct::Offset* so = &i->second;
        const llvm::Type* ft = so->init->getType();
        size_t sz = getABITypeSize(ft);
        if (f == NULL) { // new field
            fields.push_back(DUnionField());
            f = &fields.back();
            f->size = sz;
            f->offset = o;
            f->init = so->init;
            f->initsize = sz; 
            f->types.push_back(ft);
        }
        else if (o == f->offset) { // same offset
            if (sz > f->size)
                f->size = sz;
            f->types.push_back(ft);
            unions = true;
        }
        else if (o < f->offset+f->size) {
            assert((o+sz) <= (f->offset+f->size));
            unions = true;
        }
        else {
            fields.push_back(DUnionField());
            f = &fields.back();
            f->size = sz;
            f->offset = o;
            f->init = so->init;
            f->initsize = sz;
            f->types.push_back(ft);
        }
    }

    /*{
        LOG_SCOPE;
        Logger::println("******** DUnion BEGIN");
        size_t n = fields.size();
        for (size_t i=0; i<n; ++i) {
            Logger::cout()<<"field #"<<i<<" offset: "<<fields[i].offset<<" size: "<<fields[i].size<<'('<<fields[i].initsize<<")\n";
            LOG_SCOPE;
            size_t nt = fields[i].types.size();
            for (size_t j=0; j<nt; ++j) {
                Logger::cout()<<*fields[i].types[j]<<'\n';
            }
        }
        Logger::println("******** DUnion END");
    }*/
}

static void push_nulls(size_t nbytes, std::vector<llvm::Constant*>& out)
{
    assert(nbytes > 0);
    std::vector<llvm::Constant*> i(nbytes, llvm::ConstantInt::get(llvm::Type::Int8Ty, 0, false));
    out.push_back(llvm::ConstantArray::get(llvm::ArrayType::get(llvm::Type::Int8Ty, nbytes), i));
}

llvm::Constant* DUnion::getConst(std::vector<DUnionIdx>& in)
{
    std::sort(in.begin(), in.end());
    std::vector<llvm::Constant*> out;

    size_t nin = in.size();
    size_t nfields = fields.size();

    size_t fi = 0;
    size_t last = 0;
    size_t ii = 0;
    size_t os = 0;

    for(;;)
    {
        if (fi == nfields) break;

        bool nextSame = (ii+1 < nin) && (in[ii+1].idx == fi);

        if (ii < nin && fi == in[ii].idx)
        {
            size_t s = getABITypeSize(in[ii].c->getType());
            if (in[ii].idx == last)
            {
                size_t nos = in[ii].idxos * s;
                if (nos && nos-os) {
                    assert(nos >= os);
                    push_nulls(nos-os, out);
                }
                os = nos + s;
            }
            else
            {
                os = s;
            }
            out.push_back(in[ii].c);
            ii++;
            if (!nextSame)
            {
                if (os < fields[fi].size)
                    push_nulls(fields[fi].size - os, out);
                os = 0;
                last = fi++;
            }
            continue;
        }

        // default initialize if necessary
        if (ii == nin || fi < in[ii].idx)
        {
            DUnionField& f = fields[fi];
            out.push_back(f.init);
            if (f.initsize < f.size)
                push_nulls(f.size - f.initsize, out);
            last = fi++;
            os = 0;
            continue;
        }
    }

    std::vector<const llvm::Type*> tys;
    size_t nout = out.size();
    for (size_t i=0; i<nout; ++i)
        tys.push_back(out[i]->getType());

    const llvm::StructType* st = llvm::StructType::get(tys);
    return llvm::ConstantStruct::get(st, out);
}





















