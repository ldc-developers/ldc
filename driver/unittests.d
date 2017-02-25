module driver.unittests;

version(unittest):

import driver.config;
import driver.configfile;

import core.stdc.stdio;
import core.stdc.string;


void writeToFile(const(char)* filepath, const(char)* text)
{
    FILE *fp = fopen(filepath, "w");
    assert(fp, "Cannot open test file for writing: "~filepath[0 .. strlen(filepath)]);

    fputs(text, fp);
    fclose(fp);
}


// testing driver.configfile.replace
unittest
{
    enum pattern = "pattern";
    enum test1 = "find the pattern in a sentence";
    enum test2 = "find the pattern";
    enum test3 = "pattern in a sentence";
    enum test4 = "a pattern, yet other patterns";

    assert(replace(test1, pattern, "word") == "find the word in a sentence");
    assert(replace(test2, pattern, "word") == "find the word");
    assert(replace(test3, pattern, "word") == "word in a sentence");
    assert(replace(test4, pattern, "word") == "a word, yet other words");
}


unittest
{
    enum confstr =
`// This configuration file uses libconfig.
// See http://www.hyperrealm.com/libconfig/ for syntax details.
// The default group is required
default:
{
    // 'switches' holds array of string that are appends to the command line
    // arguments before they are parsed.
    switches = [
        "-I/opt/dev/ldc/runtime/druntime/src",
        "-I/opt/dev/ldc/runtime/profile-rt/d",
        "-I/opt/dev/ldc/runtime/phobos",
        "-L-L/opt/dev/build/ldc/llvm-3.9.1-Debug/lib",
        "-defaultlib=phobos2-ldc,druntime-ldc",
        "-debuglib=phobos2-ldc-debug,druntime-ldc-debug"
    ];
    test_cat = "concatenated" " multiline"
                " strings";
};
`;

    enum filename = "ldc_config_test.conf";

    writeToFile(filename, confstr);
    scope(exit) remove(filename);

    auto settings = parseConfigFile(filename);

    assert(settings.length == 1);
    assert(settings[0].name == "default");
    assert(settings[0].type == Setting.Type.group);
    auto grp = cast(GroupSetting)settings[0];
    assert(grp.children.length == 2);

    assert(grp.children[0].name == "switches");
    assert(grp.children[0].type == Setting.Type.array);
    auto arr = cast(ArraySetting)grp.children[0];
    assert(arr.vals.length == 6);
    assert(arr.vals[0] == "-I/opt/dev/ldc/runtime/druntime/src");
    assert(arr.vals[1] == "-I/opt/dev/ldc/runtime/profile-rt/d");
    assert(arr.vals[2] == "-I/opt/dev/ldc/runtime/phobos");
    assert(arr.vals[3] == "-L-L/opt/dev/build/ldc/llvm-3.9.1-Debug/lib");
    assert(arr.vals[4] == "-defaultlib=phobos2-ldc,druntime-ldc");
    assert(arr.vals[5] == "-debuglib=phobos2-ldc-debug,druntime-ldc-debug");

    assert(grp.children[1].name == "test_cat");
    assert(grp.children[1].type == Setting.Type.scalar);
    auto scalar = cast(ScalarSetting)grp.children[1];
    assert(scalar.val == "concatenated multiline strings");
}
