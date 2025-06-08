// RUN: split-file %s %t
module object;

alias string = immutable(char)[];

template Version(string ident) {
    mixin(`version(` ~ ident ~ `)
        enum Version = true;
    else
        enum Version = false;
`);
}

// RUN: %ldc -d-version=TEST_ORDER -o- -conf=%t/order.conf %s 2>&1
version (TEST_ORDER) {
    static assert(!Version!"File1");
    static assert(Version!"File2");
    static assert(!Version!"File3");
}

// RUN: %ldc -d-version=TEST_APPENDING -o- -conf=%t/appending.conf %s 2>&1
version (TEST_APPENDING) {
    static assert(!Version!"File1_Sec1");
    static assert(!Version!"File1_Sec2");

    static assert(!Version!"File2_Sec1");
    static assert(Version!"File2_Sec2");

    static assert(Version!"File3_Sec1");
    static assert(Version!"File3_Sec2");
}

version (TEST_RPATH) {}
// RUN: not %ldc -d-version=TEST_RPATH -conf=%t/multiple_rpaths.conf -link-defaultlib-shared=true -v %s 2>&1 | FileCheck %s --check-prefix=RPATHS
// RPATHS-NOT: /first_rpath

// RUN: not %ldc -d-version=TEST_RPATH -conf=%t/rpath_clear.conf -link-defaultlib-shared=true -v %s 2>&1 | FileCheck %s --check-prefix=RPATHSCLEAR
// RPATHSCLEAR-NOT: /first_rpath


/+ config dirs used by the tests

//--- order.conf/conf-11
default: {
	switches = [ "-d-version=File1" ]
}
//--- order.conf/conf-102
default: {
	switches = [ "-d-version=File2" ]
}
//--- order.conf/conf-103/conf
default: {
	switches = [ "-d-version=File3" ];
}

//--- appending.conf/01.conf
default: {
	switches ~= [ "-d-version=File1_Sec1" ]
}
".?": {
	switches = [ "-d-version=File1_Sec2" ]
}
//--- appending.conf/02.conf
default: {
	switches ~= [ "-d-version=File2_Sec1" ]
}
".?": {
	switches = [ "-d-version=File2_Sec2" ]
}
//--- appending.conf/03.conf
default: {
	switches ~= [ "-d-version=File3_Sec1" ]
}
".?": {
	switches ~= [ "-d-version=File3_Sec2" ]
}

//--- multiple_rpaths.conf/01.conf
default: {
	switches = [ "-link-defaultlib-shared" ]
	rpath = "/first_rpath";
}
//--- multiple_rpaths.conf/02.conf
default: {
	rpath = "/second_rpath";
}
//--- multiple_rpaths.conf/03.conf
default: {
	// no rpath
}

//--- rpath_clear.conf
default: {
     rpath = "/first_rpath"
}
default: {
     rpath = "" // this should clear the previous value
}

//--- end_of_files
+/
