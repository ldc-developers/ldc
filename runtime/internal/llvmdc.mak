# Makefile to build the LLVMDC compiler runtime D library for Linux
# Designed to work with GNU make
# Targets:
#	make
#		Same as make all
#	make lib
#		Build the compiler runtime library
#   make doc
#       Generate documentation
#	make clean
#		Delete unneeded files created by build process

LIB_TARGET=libtango-base-llvmdc.a
LIB_MASK=libtango-base-llvmdc*.a

LIB_TARGET_C=libtango-base-c-llvmdc.a
LIB_MASK_C=libtango-base-c-llvmdc*.a

CP=cp -f
RM=rm -f
MD=mkdir -p

#CFLAGS=-O3 $(ADD_CFLAGS)
CFLAGS=$(ADD_CFLAGS)

#DFLAGS=-release -O3 -inline -w $(ADD_DFLAGS)
DFLAGS=-w $(ADD_DFLAGS)

#TFLAGS=-O3 -inline -w $(ADD_DFLAGS)
TFLAGS=-w $(ADD_DFLAGS)

DOCFLAGS=-version=DDoc

CC=gcc
LC=llvm-ar rsv
CLC=ar rsv
DC=llvmdc
LLC=llvm-as

LIB_DEST=..

.SUFFIXES: .s .S .c .cpp .d .ll .html .o .bc

.s.o:
	$(CC) -c $(CFLAGS) $< -o$@

.S.o:
	$(CC) -c $(CFLAGS) $< -o$@

.c.o:
	$(CC) -c $(CFLAGS) $< -o$@

.cpp.o:
	g++ -c $(CFLAGS) $< -o$@

.d.bc:
	$(DC) -c $(DFLAGS) $< -of$@

.d.html:
	$(DC) -c -o- $(DOCFLAGS) -Df$*.html llvmdc.ddoc $<

targets : lib doc
all     : lib doc
lib     : llvmdc.lib llvmdc.clib
doc     : llvmdc.doc

######################################################
OBJ_C= \
    monitor.o \
    critical.o

OBJ_BASE= \
    aaA.bc \
    aApply.bc \
    aApplyR.bc \
    adi.bc \
    arrayInit.bc \
    cast.bc \
    dmain2.bc \
    eh.bc \
    genobj.bc \
    lifetime.bc \
    memory.bc \
    qsort2.bc \
    switch.bc \
    invariant.bc \

OBJ_UTIL= \
    util/console.bc \
    util/ctype.bc \
    util/string.bc \
    util/utf.bc

OBJ_LLVMDC= \
    llvmdc/bitmanip.bc \
    llvmdc/vararg.bc

OBJ_TI= \
    typeinfo/ti_AC.bc \
    typeinfo/ti_Acdouble.bc \
    typeinfo/ti_Acfloat.bc \
    typeinfo/ti_Acreal.bc \
    typeinfo/ti_Adouble.bc \
    typeinfo/ti_Afloat.bc \
    typeinfo/ti_Ag.bc \
    typeinfo/ti_Aint.bc \
    typeinfo/ti_Along.bc \
    typeinfo/ti_Areal.bc \
    typeinfo/ti_Ashort.bc \
    typeinfo/ti_byte.bc \
    typeinfo/ti_C.bc \
    typeinfo/ti_cdouble.bc \
    typeinfo/ti_cfloat.bc \
    typeinfo/ti_char.bc \
    typeinfo/ti_creal.bc \
    typeinfo/ti_dchar.bc \
    typeinfo/ti_delegate.bc \
    typeinfo/ti_double.bc \
    typeinfo/ti_float.bc \
    typeinfo/ti_idouble.bc \
    typeinfo/ti_ifloat.bc \
    typeinfo/ti_int.bc \
    typeinfo/ti_ireal.bc \
    typeinfo/ti_long.bc \
    typeinfo/ti_ptr.bc \
    typeinfo/ti_real.bc \
    typeinfo/ti_short.bc \
    typeinfo/ti_ubyte.bc \
    typeinfo/ti_uint.bc \
    typeinfo/ti_ulong.bc \
    typeinfo/ti_ushort.bc \
    typeinfo/ti_void.bc \
    typeinfo/ti_wchar.bc

ALL_OBJS= \
    $(OBJ_BASE) \
    $(OBJ_UTIL) \
    $(OBJ_TI) \
    $(OBJ_LLVMDC)

######################################################

ALL_DOCS=

######################################################

llvmdc.lib : $(LIB_TARGET)

$(LIB_TARGET) : $(ALL_OBJS)
	$(RM) $@
	$(LC) $@ $(ALL_OBJS)

llvmdc.clib : $(LIB_TARGET_C)

$(LIB_TARGET_C) : $(OBJ_C)
	$(RM) $@
	$(CLC) $@ $(OBJ_C)

llvmdc.doc : $(ALL_DOCS)
	echo No documentation available.

######################################################

clean :
	find . -name "*.di" | xargs $(RM)
	$(RM) $(ALL_OBJS)
	$(RM) $(OBJ_C)
	$(RM) $(ALL_DOCS)
	$(RM) $(LIB_MASK)
	$(RM) $(LIB_MASK_C)

install :
	$(MD) $(LIB_DEST)
	$(CP) $(LIB_MASK) $(LIB_DEST)/.
	$(CP) $(LIB_MASK_C) $(LIB_DEST)/.
