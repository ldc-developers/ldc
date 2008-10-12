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

LIB_TARGET=libtango-base-ldc.a
LIB_MASK=libtango-base-ldc*.a
LIB_TARGET_C=libtango-base-c-ldc.a
LIB_MASK_C=libtango-base-c-ldc*.a
LIB_NAME_NATIVE=libtango-base-ldc-native
LIB_TARGET_NATIVE=$(LIB_NAME_NATIVE).a

DIR_CC=./common/tango
DIR_RT=../../runtime/internal
DIR_GC=./gc/basic
#DIR_GC=./gc/stub

CP=cp -f
RM=rm -f
MD=mkdir -p

CC=gcc
LC=llvm-ar rsv
CLC=ar rsv
DC=ldc
LLVMLINK=llvm-link
LLC=llc

ADD_CFLAGS=
#ADD_DFLAGS=
ADD_DFLAGS=-g -I`pwd`/common/

targets : lib sharedlib doc
all     : lib sharedlib doc

######################################################

ALL_OBJS=

######################################################

ALL_DOCS=

######################################################

lib : $(ALL_OBJS)
	make -C $(DIR_CC) -fldc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_RT) -fldc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_GC) -fldc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	# could link the three parts into one here, but why should we

sharedlib : $(ALL_OBJS)
	make -C $(DIR_CC) -fldc.mak sharedlib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_RT) -fldc.mak sharedlib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_GC) -fldc.mak sharedlib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	# could link the three parts into one here, but why should we

doc : $(ALL_DOCS)
	make -C $(DIR_CC) -fldc.mak doc
	make -C $(DIR_RT) -fldc.mak doc
	make -C $(DIR_GC) -fldc.mak doc

######################################################

clean :
	find . -name "*.di" | xargs $(RM)
	$(RM) $(ALL_OBJS)
	$(RM) $(ALL_DOCS)
	make -C $(DIR_CC) -fldc.mak clean
	make -C $(DIR_RT) -fldc.mak clean
	make -C $(DIR_GC) -fldc.mak clean
	$(RM) $(LIB_MASK)
	$(RM) $(LIB_MASK_C)
	$(RM) $(LIB_NAME_NATIVE)*

install :
	make -C $(DIR_CC) -fldc.mak install
	make -C $(DIR_RT) -fldc.mak install
	make -C $(DIR_GC) -fldc.mak install
	$(CP) $(LIB_MASK) $(LIB_DEST)/.
	$(CP) $(LIB_MASK_C) $(LIB_DEST)/.
