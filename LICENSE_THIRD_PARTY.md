Copyrights and Licenses for Third-Party Software in the MDL SDK
===============================================================

The MDL SDK software contains code written by third parties.  Such software will
have its own individual license file in the directory in which it appears.
This file will describe the copyrights, license, and restrictions which apply
to that code.

The following pieces of software have additional or alternate copyrights,
licenses, and/or restrictions:

* LLVM:   [src/mdl/jit/llvm/dist/LICENSE.TXT](src/mdl/jit/llvm/dist/LICENSE.TXT)
* libzip: [src/base/lib/libzip/](src/base/lib/libzip/)
* zlib:   [src/base/lib/zlib/README](src/base/lib/zlib/README)
* robin_hood:   [src/base/lib/robin_hood/LICENSE](src/base/lib/robin_hood/LICENSE)
* Coco/R: [src/mdl/compiler/coco/LICENSE](src/mdl/compiler/coco/LICENSE) (1)
* tinyxml2:
  [src/base/lib/tinyxml2/tinyxml2.h](src/base/lib/tinyxml2/tinyxml2.h)
  [src/base/lib/tinyxml2/tinyxml2.cpp](src/base/lib/tinyxml2/tinyxml2.cpp)
  [examples/thirdparty/tinyxml2/include/tinyxml2.h](examples/thirdparty/tinyxml2/include/tinyxml2.h)
  [examples/thirdparty/tinyxml2/src/tinyxml2.cpp](examples/thirdparty/tinyxml2/src/tinyxml2.cpp)
* Dear ImGui with stb: [examples/thirdparty/imgui/LICENSE.txt](examples/thirdparty/imgui/LICENSE.txt) (2)
* D3DX12 Helper Library: [examples/thirdparty/d3dx12/license.txt](examples/thirdparty/d3dx12/license.txt) (2)
* fx-gltf: [examples/thirdparty/gltf/fx/license.txt](examples/thirdparty/gltf/fx/license.txt) (2)
* nlohmann/json: [examples/thirdparty/gltf/nlohmann/license.txt](examples/thirdparty/gltf/nlohmann/license.txt) (2)
* Google Noto Fonts: [examples/thirdparty/content/fonts/LICENSE.txt](examples/thirdparty/content/fonts/LICENSE.txt) (2)

The documentation uses the following fonts, CSS, and JavaScript frameworks with
additional or alternate copyrights, licenses, and/or restrictions:

* jQuery v3.6.4: [doc/js/jquery/jquery-3.6.4.min.js](doc/js/jquery/jquery-3.6.4.min.js), see https://jquery.org/license
* jQuery v3.7.1: [doc/base_module/ext/jquery/jquery-3.7.1.min.js](doc/base_module/ext/jquery/jquery-3.7.1.min.js),
  [doc/core_definitions/ext/jquery/jquery-3.7.1.min.js](doc/core_definitions/ext/jquery/jquery-3.7.1.min.js), see https://jquery.org/license
* jQuery UI v1.14.1: [doc/base_module/ext/jquery/jquery-ui-1.14.1.custom/LICENSE.txt](doc/base_module/ext/jquery/jquery-ui-1.14.1.custom/LICENSE.txt),
  [doc/core_definitions/ext/jquery/jquery-ui-1.14.1.custom/LICENSE.txt](doc/core_definitions/ext/jquery/jquery-ui-1.14.1.custom/LICENSE.txt)
* Linux Libertine font: [doc/css/linux-libertine/SIL Open Font License.txt](doc/css/linux-libertine/SIL Open Font License.txt),
  [doc/base_module/ext/fonts/linux-libertine/SIL Open Font License.txt](doc/base_module/ext/fonts/linux-libertine/SIL Open Font License.txt),
  [doc/core_definitions/ext/fonts/linux-libertine/SIL Open Font License.txt](doc/core_definitions/ext/fonts/linux-libertine/SIL Open Font License.txt)
* Noto Sans font: [doc/css/noto-sans/OFL.txt](doc/css/noto-sans/OFL.txt),
  [doc/base_module/ext/fonts/noto-sans/OFL.txt](doc/base_module/ext/fonts/noto-sans/OFL.txt),
  [doc/core_definitions/ext/fonts/noto-sans/OFL.txt](doc/core_definitions/ext/fonts/noto-sans/OFL.txt)
* Roboto Mono font: [doc/css/roboto-mono/LICENSE.txt](doc/css/roboto-mono/LICENSE.txt),
  [doc/base_module/ext/fonts/roboto-mono/LICENSE.txt](doc/base_module/ext/fonts/roboto-mono/LICENSE.txt),
  [doc/core_definitions/ext/fonts/roboto-mono/LICENSE.txt](doc/core_definitions/ext/fonts/roboto-mono/LICENSE.txt)

**Notes**

(1) Coco/R is only used for code generation in the build process and is not
    integrated by itself in the MDL SDK binaries.

(2) Dear ImGui, stb, D3DX12, fx-gltf, nlohmann/json, and the Google Noto Fonts are only used in the examples
    and are not integrated by themselves in the MDL SDK binaries.
