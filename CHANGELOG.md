Change Log
==========

MDL SDK 2018.1 (307800.1800): 09 Aug 2018
-----------------------------------------------

- Initial open source release
- ABI compatible with the MDL SDK 2018.1 (307800.1800) binary release (see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))
- The following features are only available in the binary release and excluded in the source code release:
  - MDL distiller
  - Texture baking (see *examples/mdl_sdk/execution_cuda* for example code for texture baking)
  - GLSL compiler back end
- Added: MDL Core API, a lower-level compiler API in the MDL SDK (see *src/prod/lib/mdl_core* and *doc/mdl_coreapi*)
- Added: examples for the MDL Core API (see *examples/mdl_core* and *doc/mdl_coreapi*)
