# Macroni
Macroni is an MLIR dialect that adds macro expansions to VAST's tower of IRS.

## Table of Contents
- [Macroni](#macroni)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Setting up](#setting-up)
  - [Running Macroni](#running-macroni)
  - [Testing](#testing)
  - [License](#license)

<!-- See https://github.com/trailofbits/vast for template -->

## Requirements

The [latest release](https://github.com/lifting-bits/cxx-common/releases/) of
Trail of Bits' cxx-common repository. Choose the release that has `pasta` in the
name and matches your platform. This provides `llvm`, `mlir`, and a version of
`clang` with patches to track macro expansion information; Macroni requires all
these dependencies.

We also recommend installing `ccache` to improve build times.

## Setting up

Clone Macroni and download its required submodules:

```bash
git clone https://github.com/trailofbits/macroni.git
git submodule init
git submodule update
```

Set the following environment variables:

| Variable name          | Definition                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `CMAKE_INSTALL_PREFIX` | The path to where you would like to install Macroni to.                            |
| `LLVM_EXTERNAL_LIT`    | The path to your install of `llvm-lit`. This enables automated testing of Macroni. |
| `CMAKE_TOOLCHAIN_FILE` | The path to your `cxx-common` version's toolchain file (`vcpkg.cmake`).            |
| `CMAKE_C_COMPILER`     | The path to your `cxx-common` version's C compiler.                                |
| `CMAKE_CXX_COMPILER`   | The path to your `cxx-common` version's C++ compiler.                              |

**Tip**: If you are developing in Visual Studio or Visual Studio Code, use the
provided CMakePresets.json file to make your development easier. Just edit the
file by assigning values to these variables in the `cacheVariables` object. Note
that the `CMAKE_TOOLCHAIN_FILE` should be set separately by assigning a value to
the `toolchainFile` field.

Configure Macroni:

```bash
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_INSTALL_PREFIX=... \
      -DLLVM_EXTERNAL_LIT=... \
      -DLLVM_ENABLE_RTTI=ON \
      -DMACRONI_ENABLE_TESTING=ON \
      -DVCPKG_TARGET_TRIPLET=x64-linux \
      -DVCPKG_HOST_TRIPLET=x64-linux-rel \
      -DCMAKE_C_COMPILER=... \
      -DCMAKE_CXX_COMPILER=... \
      -DCMAKE_TOOLCHAIN_FILE=...\
      -Smacroni \
      -B${CMAKE_INSTALL_PREFIX} \
      -G "Ninja Multi-Config"
```


Build the `macronify` binary for translating C code to a mix of `VAST` and
`Macroni` MLIR code:
```bash
cmake --build ${MACRONI_BUILD_DIR} --config Debug --target macronify -j 8 --
```

This will build the `macronify` binary and place it in the directory specified
by the `-B` option in the configure command.

## Running Macroni
Once the `macronify` binary has been built, you can run it on a C source file.
Assuming `MACRONIFY` and `INPUT` are environment variables set to the paths to
the `macronify` binary and a C source file, respectively, this command would
look like:

```bash
${MACRONIFY} -xc ${INPUT}
```

To run Macroni's conversions, pass the `--convert` option:

```bash
${MACRONIFY} -xc ${INPUT} --convert
```

This will convert certain macros and constructs found in the Linux kernel into
operations of Macroni's `Kernel` dialect.

## Testing

Macroni uses [llvm-lit](https://llvm.org/docs/CommandGuide/lit.html) to automate
testing. To set up Macroni's test suite, configure Macroni as normal, with the
additional option `-DMACRONI_ENABLE_TESTING=ON`. Then run the following command
to run Macroni's test suite:

```bash
cmake --build ${MACRONI_BUILD_DIR} --config Debug --target check-macroni -j 8 --
```

## License
Macroni is licensed according to the Apache 2.0 license. Macroni links against
and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with
[LLVM](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT)
exceptions.
