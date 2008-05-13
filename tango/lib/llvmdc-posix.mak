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

LIB_TARGET=libtango-base-llvmdc.a
LIB_MASK=libtango-base-llvmdc*.a
LIB_TARGET_C=libtango-base-c-llvmdc.a
LIB_MASK_C=libtango-base-c-llvmdc*.a

DIR_CC=./common/tango
DIR_RT=./compiler/llvmdc
#DIR_GC=./gc/basic
DIR_GC=./gc/stub

CP=cp -f
RM=rm -f
MD=mkdir -p

CC=gcc
LC=llvm-ar rsv
CLC=ar rsv
DC=llvmdc

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
	make -C $(DIR_CC) -fllvmdc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	make -C $(DIR_RT) -fllvmdc.mak lib
	make -C $(DIR_GC) -fllvmdc.mak lib DC=$(DC) ADD_DFLAGS="$(ADD_DFLAGS)" ADD_CFLAGS="$(ADD_CFLAGS)"
	find . -name $(LIB_MASK) | xargs $(RM)
	$(LC) $(LIB_TARGET) `find $(DIR_CC) -name "*.bc" | xargs echo`
	$(LC) $(LIB_TARGET) `find $(DIR_RT) -name "*.bc" | xargs echo`
	$(LC) $(LIB_TARGET) `find $(DIR_GC) -name "*.bc" | xargs echo`
	$(CLC) $(LIB_TARGET_C) `find $(DIR_CC) -name "*.o" | xargs echo`
	$(CLC) $(LIB_TARGET_C) `find $(DIR_RT) -name "*.o" | xargs echo`

doc : $(ALL_DOCS)
	make -C $(DIR_CC) -fllvmdc.mak doc
	make -C $(DIR_RT) -fllvmdc.mak doc
	make -C $(DIR_GC) -fllvmdc.mak doc

######################################################

clean :
	find . -name "*.di" | xargs $(RM)
	$(RM) $(ALL_OBJS)
	$(RM) $(ALL_DOCS)
	make -C $(DIR_CC) -fllvmdc.mak clean
	make -C $(DIR_RT) -fllvmdc.mak clean
	make -C $(DIR_GC) -fllvmdc.mak clean
	$(RM) $(LIB_MASK)
	$(RM) $(LIB_MASK_C)

install :
	make -C $(DIR_CC) -fllvmdc.mak install
	make -C $(DIR_RT) -fllvmdc.mak install
	make -C $(DIR_GC) -fllvmdc.mak install
	$(CP) $(LIB_MASK) $(LIB_DEST)/.
	$(CP) $(LIB_MASK_C) $(LIB_DEST)/.
