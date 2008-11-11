# Makefile to build the composite D runtime library for Linux
# Designed to work with GNU make
# Targets:
#	make
#		Same as make all
#	make lib
#		Build the runtime library
#   make doc
#       Generate documentation
#	make clean
#		Delete unneeded files created by build process

LIB_TARGET=libdruntime-ldc.a
DUP_TARGET=libdruntime.a
LIB_MASK=libdruntime*.a

DIR_CC=common
DIR_RT=compiler/ldc
DIR_GC=gc/basic

CP=cp -f
RM=rm -f
MD=mkdir -p

CC=gcc
LC=$(AR) -qsv
DC=ldc2

LIB_DEST=../lib

ADD_CFLAGS=
ADD_DFLAGS=

targets : lib doc
all     : lib doc

######################################################

ALL_OBJS=

######################################################

ALL_DOCS=

######################################################

lib : $(ALL_OBJS)
	make -C $(DIR_CC) -fldc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_RT) -fldc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_GC) -fldc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	$(RM) $(LIB_TARGET)
	$(LC) $(LIB_TARGET) `find $(DIR_CC) -name "*.o" | xargs echo`
	$(LC) $(LIB_TARGET) `find $(DIR_RT) -name "*.o" | xargs echo`
	$(LC) $(LIB_TARGET) `find $(DIR_GC) -name "*.o" | xargs echo`
	$(RM) $(DUP_TARGET)
	$(CP) $(LIB_TARGET) $(DUP_TARGET)

doc : $(ALL_DOCS)
	make -C $(DIR_CC) -fldc.mak doc DC=$(DC)
	make -C $(DIR_RT) -fldc.mak doc DC=$(DC)
	make -C $(DIR_GC) -fldc.mak doc DC=$(DC)

######################################################

clean :
	find . -name "*.di" | xargs $(RM)
	$(RM) $(ALL_OBJS)
	$(RM) $(ALL_DOCS)
	make -C $(DIR_CC) -fldc.mak clean
	make -C $(DIR_RT) -fldc.mak clean
	make -C $(DIR_GC) -fldc.mak clean
	$(RM) $(LIB_MASK)

install :
	make -C $(DIR_CC) -fldc.mak install
	make -C $(DIR_RT) -fldc.mak install
	make -C $(DIR_GC) -fldc.mak install
	$(CP) $(LIB_MASK) $(LIB_DEST)/.
