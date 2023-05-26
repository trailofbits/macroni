#
# Copyright (c) 2023, Trail of Bits, Inc.
# All rights reserved.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.
#

# TODO(bpp): Check if # TODO(bpp): Add MACRONI_ENABLE_{VAST,PASTA}_SUBMODULE
# options are enabled and only set these definitions if so

set(PASTA_BOOTSTRAP_MACROS OFF)
set(PASTA_BOOTSTRAP_TYPES OFF)
set(PASTA_ENABLE_TESTING OFF)
set(PASTA_ENABLE_INSTALL OFF)
add_subdirectory(pasta)


add_subdirectory(vast)