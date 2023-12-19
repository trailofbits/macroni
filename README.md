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

## Requirements
- [Clang 17 and LLVM 17](https://apt.llvm.org/). If on Debian/Ubuntu:
  ```
  wget https://apt.llvm.org/llvm.sh
  chmod +x llvm.sh
  sudo ./llvm.sh 17
  sudo apt install libclang-17-dev libmlir-17-dev mlir-17-tools
  ```
- [`gap`](https://github.com/lifting-bits/gap)
- [`PASTA`](https://github.com/trailofbits/pasta/)
- [`VAST`](https://github.com/trailofbits/vast)
  - Macroni currently uses [this
    fork](https://github.com/trailofbits/vast/tree/mx_codegen) of VAST, so we
    recommend you install this one to run Macroni.

We also recommend installing [`ccache`](https://ccache.dev/) to improve build times.

## Setting up
Clone Macroni:

```bash
git clone https://github.com/trailofbits/macroni.git
```

We offer a `CMakePresets.json` file to ease configuration and building. This
file relies on a number of environment variables (e.g,
`$env{MACRONI_Clang_DIR}`), so to use this presets file you must define these
environment variables first. Once you have done that you can configure Macroni
with:

```bash
cmake --preset macroni-ninja-multiconfig
```

Build and install Macroni:
```bash
cmake --build --preset macroni-ninja-multiconfig -t install
```

## Running Macroni
Once the `macronify` binary has been built, you can run it on a C source file.
Assuming `macronify` has been installed and its location added to your path:

```bash
macronify -xc some_file.c
```

## Testing

TODO: Update this section to reflect latest changes to Macroni

## License
Macroni is licensed according to the Apache 2.0 license. Macroni links against
and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with
[LLVM](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT)
exceptions.
