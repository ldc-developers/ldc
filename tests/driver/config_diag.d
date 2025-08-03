// RUN: split-file %s %t

// RUN: %ldc -o- -conf=%t/invalid_setting.conf %s 2>&1 | FileCheck %s --check-prefix=INV_SET
// INV_SET: Error while reading config file: {{.*}}invalid_setting.conf
// INV_SET-NEXT: Unknown scalar setting named blah-blah

// RUN: %ldc -o- -conf=%t/invalid_append.conf %s 2>&1 | FileCheck %s --check-prefix=APP
// APP: Error while reading config file: {{.*}}invalid_append.conf
// APP-NEXT: line 3: '~=' is not supported with scalar values

module object;

/+ config dirs used by the tests

//--- invalid_setting.conf
default:
{
	blah-blah = "12";
};

//--- invalid_append.conf
default:
{
	rpath ~= "/path";
}
//--- end_of_files
+/
