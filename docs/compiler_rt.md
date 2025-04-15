# Configuring LDC to use compiler-rt libraries

LDC provides some functionalities like PGO, sanitizers and fuzzing, which require the presence of some libraries that are part of the compiler-rt LLVM sub-project.
This document aims to describe how to tell LDC the locations of these libraries, based on your installation method.
It is meant mostly for people using LDC through a Linux package manager or manually building it, which includes package maintainers.

## Using the prebuilt packages or [dlang.org/install.sh](https://dlang.org/install.sh)

The tarballs at https://github.com/ldc-developers/ldc/releases come with the compiler-rt libraries and you don't need to do any configuration.
The dlang.org install.sh script also uses these tarballs so the same things applies to it.

## Using a Linux distribution's package manager

Given a survey of the available options at the time of writing,
most distributions don't add a dependency on compiler-rt nor do they configure LDC to find them when installed manually,
therefore, it is needed to do both of these steps manually.

### Installing compiler-rt

The name of the actual package varies quite a lot across distributions but you can usually find it be searching for `compiler-rt` or `libclang`.
Below you will find a list of popular Linux distributions and the name of the `compiler-rt` package:

- Debian/Ubuntu: `libclang-rt-dev`
- Fedora: `compiler-rt`
- Arch: `compiler-rt`
- Gentoo: `compiler-rt-sanitizers`

### Adding the compiler-rt path to ldc2.conf

After you've installed the packages chances are that the libraries still won't be found because the path differs from what LDC expects.
To solve this you need to first determine the path of the libraries.
Again, this path depends on distribution but it is easy to find if you use your package manager to list the contents of the `compiler-rt` package installed in the previous step.

Usually the path has the form: `<some_directory>/lib/clang/<major_version>/lib/linux/`.
Where  `<major_version>` is a number like 18, 17, etc.
`linux` may be, instead, a target triple like `x86_64-redhat-linux-gnu`.
If the directory you found contains a bunch of `libclang_rt.*` files then you've found the right path, if not, try again.

For simplicity, below you can find the paths of some Linux distributions.
Remember to adapt them to the appropriate version of the package you have installed.

- Debian/Ubuntu: `/usr/lib/llvm-<major_version>/lib/clang/<major_version>/lib/linux/`
- Fedora: `/usr/lib/clang/<major_version>/lib/x86_64-redhat-linux-gnu/`
- Arch: `/usr/lib/clang/<major_version>/lib/linux/`
- Gentoo: `/usr/lib/clang/<major_version>/lib/linux/`

You now need to edit the ldc2 configuration file.
Since you're installing ldc2 through your package manager the config file is probably in `/etc/ldc2.conf`.
If it's not there it should be in `../etc/ldc2.conf` relative to the directory of the ldc2 executable.

Use your favorite text editor to edit the file and jump to the `lib-dirs = ` snippet.

On recent versions you should see:
```
    lib-dirs = [
        "/usr/lib",
        "/usr/lib/clang/17/lib/linux", // compiler-rt directory
    ];
```
You should edit the line with the `compiler-rt directory` comment and put the path you determined above in there.

On older versions you wouldn't have the second entry so you would only see:
```
    lib-dirs = [
        "/usr/lib",
    ];
```

In that case just add another line with the path surrounded  by `"`.

In the end, you should have:
```
    lib-dirs = [
        "/usr/lib",
		"YOUR_PATH_HERE", // compiler-rt directory
    ];
```
With or without the `compiler-rt` comment.

## Building from source

When building from source you can pass a number of options to cmake related to compiler-rt libraries.

### `COMPILER_RT_BASE_DIR`

This option tells cmake a path that will be transformed, assuming standard layout, in the final compiler-rt directory.
If you are using a distribution tarball for llvm then the default value for this option, `<LLVM_LIB_DIR>/clang`, should be the correct one.

If on Linux and using the llvm provided through your package manager set this option to something like `/usr/lib/clang`.
How to find the correct path is described above, in the Linux section.

### `COMPILER_RT_LIBDIR_OS`

This option only applies to non-Mac Unixes and refers to the final directory name in the full path to the compiler-rt libraries.

If the path is `[...]/clang/<version>/lib/linux` then this value should be `linux`.
If it is `[...]/clang/<version>/lib/x86_64-redhat-linux-gnu` then this value should be `x86_64-redhat-linux-gnu`.

### `LDC_INSTALL_LLVM_RUNTIME_LIBS_ARCH`

The option only applied to non-Mac Unixes and refers to the architecture component in the filename of the libraries.
It only makes sense specifying this if `LDC_INSTALL_LLVM_RUNTIME_LIBS` is set to `ON`, see below.

If the filename of the libraries is `libclang_rt.asan-x86_64.a` then this value should be `x86_64`.
If the filename of the libraries is `libclang_rt.asan.a` then this value should be `""` (empty).

### `LDC_INSTALL_LLVM_RUNTIME_LIBS`

This option tells cmake to copy the library files from the compiler-rt directory specified above to the library directory of ldc2.
If you enable this you don't need to configure anything else, the libraries will be found when you run ldc2.

### `COMPILER_RT_LIBDIR_CONFIG`

The final path that is stored in the configuration file in regards to compiler-rt is `${COMPILER_RT_BASE_DIR}/${LLVM_MAJOR_VERSION}/lib/${COMPILER_RT_LIBDIR_OS}`.
If this setting doesn't match your layout, as a last resort, you can specify your custom path with `-DCOMPILER_RT_LIBDIR_CONFIG` and that unaltered path will be stored in the config.

## Checking that everything works

Try to compile the empty program below with `ldc2 -fprofile-generate -vv sample.d`
```d
void main () {}
```

And check the end of the output:
```
[...]
*** Linking executable ***
Searching profile runtime: /usr/lib/libldc_rt.profile.a
Searching profile runtime: /usr/lib/llvm-17/lib/clang/17/lib/linux/libldc_rt.profile.a
Searching profile runtime: /usr/lib/libldc_rt.profile.a
Searching profile runtime: /usr/lib/llvm-17/lib/libldc_rt.profile.a
Searching profile runtime: /usr/lib/libclang_rt.profile-x86_64.a
Searching profile runtime: /usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.profile-x86_64.a
Found, linking with /usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.profile-x86_64.a
Linking with:
[...]
```
The import line is `Found, linking with ...`.
If you see this it means that you configured LDC correctly and the library was found.

If you're missing that line, like in:
```
*** Linking executable ***
Searching profile runtime: /usr/lib/libldc_rt.profile.a
Searching profile runtime: /usr/lib/libldc_rt.profile.a
Searching profile runtime: /usr/lib/llvm-17/lib/libldc_rt.profile.a
Searching profile runtime: /usr/lib/libclang_rt.profile-x86_64.a
Searching profile runtime: /usr/lib/libclang_rt.profile-x86_64.a
Searching profile runtime: /usr/lib/llvm-17/lib/libclang_rt.profile-x86_64.a
Searching profile runtime: /usr/lib/clang/17.0.6/lib/linux/libclang_rt.profile-x86_64.a
Searching profile runtime: /usr/lib/clang/17.0.6/lib/linux/libclang_rt.profile-x86_64.a
Searching profile runtime: /usr/lib/llvm-17/lib/clang/17.0.6/lib/linux/libclang_rt.profile-x86_64.a
Linking with:
```
You should recheck the paths and make the necessary adjustments in the config file until `ldc2` can find it.
