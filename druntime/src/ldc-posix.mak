# Makefile to build the composite D runtime library for Linux
# Designed to work with GNU make
# Targets:
#	make
#		Same as make all
#	make debug
#		Build the debug version of the library
#   make release
#       Build the release version of the library
#   make doc
#       Generate documentation
#	make clean
#		Delete all files created by build process

# Essentials

LIBDIR:=../lib
DOCDIR:=../doc
LIBBASENAME=libdruntime.a

DIR_CC=common
DIR_RT=compiler/ldc
DIR_GC=gc/stub

# Symbols

DMD="ldc2 -vv"
CLC=ar rsv

# Targets

all : debug release doc unittest $(LIBDIR)/$(LIBBASENAME)

# unittest :
# 	$(MAKE) -fdmd-posix.mak lib MAKE_LIB="unittest"
# 	dmd -unittest unittest ../import/core/stdc/stdarg \
# 		-defaultlib="$(DUP_TARGET)" -debuglib="$(DUP_TARGET)"
# 	$(RM) stdarg.o
# 	./unittest

debug release unittest :
	@$(MAKE) DMD=$(DMD) -C $(DIR_CC) -fposix.mak $@
	@$(MAKE) DMD=$(DMD) -C $(DIR_RT) -fposix.mak $@
	@$(MAKE) DMD=$(DMD) -C $(DIR_GC) -fposix.mak $@
ifeq ($(DMD),ldc2 -vv)
	@$(CLC) $(LIBDIR)/$@/$(LIBBASENAME) \
		$(LIBDIR)/$@/libdruntime-core.a \
		$(LIBDIR)/$@/libdruntime-rt-ldc.a \
		$(LIBDIR)/$@/libdruntime-gc-stub.a
else
	@$(DMD) -lib -of$(LIBDIR)/$@/$(LIBBASENAME) \
		$(LIBDIR)/$@/libdruntime-core.a \
		$(LIBDIR)/$@/libdruntime-rt-dmd.a \
		$(LIBDIR)/$@/libdruntime-gc-basic.a
endif

$(LIBDIR)/$(LIBBASENAME) : $(LIBDIR)/release/$(LIBBASENAME)
	ln -sf $(realpath $<) $@

doc : $(ALL_DOCS)
	$(MAKE) DMD=$(DMD) -C $(DIR_CC) --no-print-directory -fposix.mak doc
#	$(MAKE) DMD=$(DMD) -C $(DIR_RT) --no-print-directory -fposix.mak doc
#	$(MAKE) DMD=$(DMD) -C $(DIR_GC) --no-print-directory -fposix.mak doc

######################################################

clean :
	$(MAKE) DMD=$(DMD) -C $(DIR_CC) -fposix.mak clean
	$(MAKE) DMD=$(DMD) -C $(DIR_RT) -fposix.mak clean
	$(MAKE) DMD=$(DMD) -C $(DIR_GC) -fposix.mak clean
#find . -name "*.di" | xargs $(RM)
	rm -rf $(LIBDIR) $(DOCDIR)

# install :
# 	make -C $(DIR_CC) --no-print-directory -fposix.mak install
# 	make -C $(DIR_RT) --no-print-directory -fposix.mak install
# 	make -C $(DIR_GC) --no-print-directory -fposix.mak install
# 	$(CP) $(LIB_MASK) $(LIB_DEST)/.
# 	$(CP) $(DUP_MASK) $(LIB_DEST)/.

