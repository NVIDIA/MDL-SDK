MDL SDK Copyright and License
=============================

Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Images and Resource Files License
---------------------------------

The following images are licensed under the [Creative Commons Attribution-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nd/4.0/) ![Creative Commons License](https://i.creativecommons.org/l/by-nd/4.0/80x15.png):

* [doc/images/mdl_material_examples.jpg](doc/images/mdl_material_examples.jpg)
* [doc/images/mdl_local_coordinates.jpg](doc/images/mdl_local_coordinates.jpg)
* [doc/images/mdl_icon.png](doc/images/mdl_icon.png)

Images (.png, .jpg and .hdr file extensions), lightprofile 
files (.ies file extension), and BSDF measurement files (.mbsdf file extension) in the 
[examples/mdl/nvidia/](examples/mdl/nvidia/) directory, are collectively called MDL Example 
Resources and are licensed under the
[CC0 1.0 Universal Public Domain Dedication](http://creativecommons.org/publicdomain/zero/1.0/).
![CC0 License](https://licensebuttons.net/p/zero/1.0/80x15.png)
To the extent possible under law, NVIDIA Corporation has waived all copyright 
and related or neighboring rights to the MDL Example Resources. 


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

The documentation uses the following web fonts, CSS, and JavaScript frameworks with 
additional or alternate copyrights, licenses, and/or restrictions:

* jQuery v3.2.1: [doc/js/jquery/jquery-3.2.1.min.js](doc/js/jquery/jquery-3.2.1.min.js), see https://jquery.org/license
* jQuery UI v1.12.1: [doc/core_definitions/ext/jquery/jquery-ui-1.12.1/LICENSE.txt](doc/core_definitions/ext/jquery/jquery-ui-1.12.1/LICENSE.txt),
  [doc/base_module/ext/jquery/jquery-ui-1.12.1/LICENSE.txt](doc/base_module/ext/jquery/jquery-ui-1.12.1/LICENSE.txt)
* normalize.css v1.1.3: [doc/core_definitions/kt_css/normalize.css](doc/core_definitions/kt_css/normalize.css),
  [doc/base_module/kt_css/normalize.css](doc/base_module/kt_css/normalize.css)
* Libre Baskerville: [doc/css/librebaskerville/LICENSE-LibreBaskerville.txt](doc/css/librebaskerville/LICENSE-LibreBaskerville.txt)
* Source Sans Pro: [doc/css/sourcesanspro/LICENSE-SourceSansPro.txt](doc/css/sourcesanspro/LICENSE-SourceSansPro.txt)
* Source Code Pro: [doc/css/sourcecodepro/LICENSE-SourceCodePro.txt](doc/css/sourcecodepro/LICENSE-SourceCodePro.txt)

**Notes**

(1) Coco/R is only used for code generation in the build process and is not
    integrated by itself in the MDL SDK binaries.

(2) Dear ImGui, stb, D3DX12, fx-gltf, nlohmann/json, and the Google Noto Fonts are only used in the examples 
    and are not integrated by themselves in the MDL SDK binaries.
