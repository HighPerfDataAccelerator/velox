# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_path(UCX_INCLUDE_DIR NAMES ucp/api/ucp.h)
find_library(UCX_UCP_LIBRARY NAMES ucp libucp)
find_library(UCX_UCS_LIBRARY NAMES ucs libucs)
find_library(UCX_UCT_LIBRARY NAMES uct libuct)
find_library(UCX_UCM_LIBRARY NAMES ucm libucm)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ucx REQUIRED_VARS UCX_INCLUDE_DIR UCX_UCP_LIBRARY UCX_UCS_LIBRARY)

if(ucx_FOUND)
  foreach(component UCP UCS UCT UCM)
    string(TOLOWER "${component}" component_lower)
    if(UCX_${component}_LIBRARY AND NOT TARGET ucx::${component_lower})
      add_library(ucx::${component_lower} UNKNOWN IMPORTED)
      set_target_properties(
        ucx::${component_lower}
        PROPERTIES
          IMPORTED_LOCATION "${UCX_${component}_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES "${UCX_INCLUDE_DIR}"
      )
    endif()
  endforeach()
endif()
