Change Log
==========

MDL SDK 2019.1 (317500.1752): 16 May 2019
-----------------------------------------------

ABI compatible with the MDL SDK 2019.1 (317500.1752) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL 1.5 Language Specification

    - A new cast operator has been added to support assignments between
      structurally equivalent user defined structure types and value
      equivalent enumeration types to support workflows with the new
      MDLE format. Beginning with MDL 1.5, `cast` is a reserved word.
    - A new field named `hair` of type `hair_bsdf` has been added to the
      material type, which represents the shading model applicable for hair
      primitives. Beginning with MDL 1.5, `hair_bsdf` is a reserved word.
    - A new elemental distribution function `df::chiang_hair_bsdf` has been
      added as a hair shading model.
    - A new distribution function modifier `df::measured_factor` has been
      added to support microfacet coloring based on the angle between the
      half-vector and the shading normal in addition to the angle between
      the half-vector and the incoming ray direction.
    - Annotations have been added to annotation declarations.
    - A new standard annotation `origin()` has been added, which is used in
      the MDLE file format to reference the original declarations of
      refactored elements.
    - The new Appendix D -- MDLE File Format defines a new container format
      for a self contained MDL material or function including all of its
      dependencies and resources.
    - The new Appendix E -- Internationalization defines the use of XLIFF
      files for the localization of MDL string annotations.

- General

    - A new function `IMdle_api::get_hash()` has been added.
    - A new function `IMdl_compiler::get_module_db_name()` has been added.
    - The MDLE file format version has been bumped to `1.0`.
    - MDLE files now use the new `anno::origin` annotation rather than a custom one.
    - A new interface `mi::neuraylib::IValue_string_localized` has been added.
    - A new function `IType_factory::is_compatible()` has been added to check if one MDL
      type can be cast to another.
    - A new function `IExpression_factory::create_cast()` has been added.
    - A new configuration interface `IMdl_configuration` has been added, which can be used
      to control the behavior of the SDK regarding the automatic insertion of casts when
      assigning compatible but different types to arguments of MDL instances.
    - The `IMdl_discovery_api` has been extended to also support discovery of resources and
      XLIFF files.

- MDL Compiler and Backends

    - A new backend `mi::neuraylib::IMdl_compiler::MB_HLSL` for `HLSL` code generation
      has been added. Please refer to the `dxr` example for an illustrative path tracer
      using it.
    - The CUDA/OptiX backend expects some new functions in the user provided renderer runtime
      to allow using resources unknown at compile-time via argument blocks with class compilation:
        - `bool tex_texture_isvalid(Texture_handler_base const *self, tct_uint texture_idx)`
        - `void tex_resolution_3d(int result[3], Texture_handler_base const *self, tct_uint texture_idx)`
        - `bool df_light_profile_isvalid(Texture_handler_base const *self, tct_uint resource_idx)`
        - `tct_float df_light_profile_power(Texture_handler_base const *self, tct_uint resource_idx)`
        - `tct_float df_light_profile_maximum(Texture_handler_base const *self, tct_uint resource_idx)`
        - `bool df_bsdf_measurement_isvalid(Texture_handler_base const *self, tct_uint resource_idx)`
      The `tex_resolution_3d()` function fills the width, height and depth for the given
      texture index into the respective result entry.
      The other functions are implementations for the corresponding MDL functions.
      See `examples/mdl_sdk/shared/texture_support_cuda.h` for an example implementation.
    - The compiler support for the ternary operator on material and material sub types has been
      improved. Several materials that caused compile errors before are now compiled flawless.
    - The MDL compiler now correctly issues an error when the called object is not a function,
      detecting (wrong) code like f()().
    - The compiler generated now correct MDL code when exporting functions containing a
      dangling if construct.
    - The compiler now (correctly) forbids the use of resource types as parameter types
      in annotations.
    - The PTX backend does not use global counters to generate temporary identifiers anymore,
      this greatly improves PTX cache hits.
    - The following MDL 1.5 features are now supported by the MDL compiler:
        - `hair_bsdf()` type
        - `df::chiang_hair_bsdf()`
        - `anno::origin` annotation
        - support for annotations on annotation declarations
    - The backends were upgraded to use the LLVM 7 library. This means, that the LLVM-IR
      backend now produces LLVM 7 IR code.
    - Allow generating code for `bsdf()` if "compile_constants" options is "on" (default).

- MDL SDK examples
    - A new Direct3D 12 example `dxr` has been added which illustrates how to use
      the `HLSL` back-end in an RTX-based real-time path tracer.
    - Support for thin-walled materials has been added to `example_df_cuda`.

**Fixed Bugs**

- General

    - Failing MDLE creation when an argument of the MDLE prototype is connected to a function
      which returns a user-defined type has been fixed.
    - A bug leading to different output (and therefore different hashes) when exporting the
      same MDLE more than once has been fixed.
    - A failure (error code -8) when creating presets from functions with user-defined
      return types has been fixed.
    - A failure (error code -8) when creating function presets from MDL modules
      with versions < 1.3 has been fixed.
    - When exporting presets from MDLE definitions or MDL definitions containing calls to MDLE
      definitions in their arguments, the MDLE code is now inlined into the new module, rather
      than resulting in invalid MDL.
    - The missing `origin` annotation on the main definition of an MDLE file has been added.
    - Issues resolving MDLE files on UNC file paths have been fixed.
    - Missing imports for user-defined function return types which caused MDLE creation
      to fail, have been added.
    - The conversion of array constructors to MDL AST expressions has been fixed.
    - The use of implicit conversion functions inside `enable-if` expressions is no longer
      forbidden.

- MDL Compiler and Backends
    - Unnecessarily slow code generation for very big output has been fixed.
    - Wrong code generation for `df::bsdf_measurement_isvalid()` has been fixed.
    - Creating DAG call nodes for ternary operators returning non-builtin types has been fixed.
    - For the native and PTX backends, wrong order of array elements returned by `math::modf()`
      has been fixed.
    - Native code execution on Mac has been fixed.
    - Struct member access for MDLE types containing a dot in their file path has been fixed.


MDL SDK 2019 (314800.830): 20 Feb 2019
-----------------------------------------------

ABI compatible with the MDL SDK 2019 (314800.830) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - This release contains a preview version of the new MDL encapsulated (MDLE) file format. Please note that MDLE files generated with this SDK version will be invalidated with the next update.
    - A new API component `mi::neuraylib::IMdle_api` has been added, which can be used to create MDLE files.
    - The function `mi::neuraylib::IMdl_compiler::load_module()` can now load MDLE files. Please note that the option `bool "experimental"` has to be set in the `mi::neuraylib::IMdl_execution_context` in order to enable support for `MDL 1.5`, which is needed for MDLE files.
    - The functions
        - `mi::neuraylib::IMaterial_instance::is_default()` and
        - `mi::neuraylib::IFunction_call::is_default()` have been added.
    - The standalone tool `mdlm` has been extended with MDLE specific commands.
    - The class `mi::neuraylib::IBaker` has been extended to support baking of constants of type `mi::IBoolean`.

- MDL SDK examples
    - A new example `example_mdle` has been added to illustrate the use of MDLE files.
    - The example `example_df_cuda` has been adapted to allow loading and rendering of MDLE files.
    - A new example `example_generate_mdl_identifier` that illustrates how to generate a valid mdl identifier, e.g., a module name, has been added.
    - The example `mdl_browser` has been extended to display MDL keywords in the info tooltip as well as above the description in list view mode.

**Fixed Bugs**

- General
    - A bug when translating light profile and bsdf measurement constructors from the MDL SDK API representation to the MDL Core representation has been fixed.

- MDL Compiler and Backends
    - The hash calculation for struct field access DAG calls for the PTX code cache has been fixed.
    - The handling of array parameters in class compilation has been fixed.
    - A crash when trying to fold an invalid array constructor has been fixed.
    - Missing parentheses when printing operators with the same precedence as MDL has been fixed (`"a/(b*c)"` was printed as `"a/b*c"`).
    - A potential crash when generating code for distribution functions has been fixed.
    - Wrong error messages `"varying call from uniform function"` have been fixed, which were generated by the MDL compiler under rare circumstances for struct declarations.
    - Wrong error messages `"function preset's return type must be 'uniform T' not 'T'"` have been fixed, which were generated by the MDL compiler for function variants if the original function always returns a uniform result but its return type was not declared as uniform T.
    - A discrepancy between code execution on CPU and GPU for constant folding of
    `tt sqrt(c) (c < 0)` has been fixed. Now `NaN` is computed for both.


MDL SDK 2018.1.2 (312200.1281): 11 Dec 2018
-----------------------------------------------

ABI compatible with the MDL SDK 2018.1.2 (312200.1281) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL 1.5 Language Specification
    - A first pre-release draft of the NVIDIA Material Definition Language 1.5: Appendix E - Internationalization has been added to the documentation set.

- General
    - Support for the internationalization of MDL string annotations has been added. See the MDL 1.5 Language Specification for details.
    - A new API component `mi::neuraylib::IMdl_i18n_configuration` has been added, which can be used to query and change MDL internationalization settings.
    - A new standalone tool to create XLIFF files has been added. See `i18n`.
    - Calling `mi::neuraylib::ITransaction::remove()` on an MDL module will cause the module and all its definitions and other dependencies to be removed from the database as soon as it is no longer referenced by another module, material instance or function call. The actual removal is triggered by calling `mi::neuraylib::ITransaction::commit()`.
    - A new API component `mi::neuraylib::Mdl_compatibility_api` has been added which allows to test archives and modules for compatibility.
    - A new standalone tool to manage MDL archives has been added. See `mdlm`.
    - A new API class `mi::neuraylib::IMdl_execution_context` intended to pass options to and receive messages from the MDL compiler has been added.
    - A new API class `mi::neuraylib::IMessage` intended to propagate MDL compiler and SDK messages has been added.
    - A new API function `mi::neuraylib::IMdl_factory::create_execution_context` has been added.
    - The signatures of the API functions
        - `mi::neuraylib::IMaterial_instance::create_compiled_material()`
        - `mi::neuraylib::IMdl_compiler::load_module()`
        - `mi::neuraylib::IMdl_compiler::load_module_from_string()`
        - `mi::neuraylib::IMdl_compiler::export_module()`
        - `mi::neuraylib::IMdl_compiler::export_module_to_string()`
        - `mi::neuraylib::IMdl_backend::translate_environment()`
        - `mi::neuraylib::IMdl_backend::translate_material_expression()`
        - `mi::neuraylib::IMdl_backend::translate_material_df()`
        - `mi::neuraylib::IMdl_backend::translate_material()`
        - `mi::neuraylib::IMdl_backend::create_link_unit()`
        - `mi::neuraylib::IMdl_backend::translate_link_unit()`
        - `mi::neuraylib::ILink_unit::add_environment()`
        - `mi::neuraylib::ILink_unit::add_material_expression()`
        - `mi::neuraylib::ILink_unit::add_material_df()`
        - `mi::neuraylib::ILink_unit::add_material()`
      
      have been changed to use the new class `mi::neuraylib::IMdl_execution_context`.
      The old versions have been deprecated and prefixed with `deprecated_`. They can
      be restored to their original names by setting the preprocessor define
      `MI_NEURAYLIB_DEPRECATED_9_1`.
    - The API functions
        - `mi::neuraylib::IMdl_backend::translate_material_expression_uniform_state()`
        - `mi::neuraylib::IMdl_backend::translate_material_expressions()`
      
      have been deprecated and prefixed with `deprecated_`. They can be restored to
      their original names by setting the preprocessor define `MI_NEURAYLIB_DEPRECATED_9_1`.
    - The utility classes
        - `mi::neuraylib::Definition_wrapper` and
        - `mi::neuraylib::Argument_editor`
      
      have been extended to provide member access functions.

- MDL Compiler and Backends
    - Support for automatic derivatives for 2D texture lookups has been added to the PTX,
  Native x86 and LLVM IR backends. This feature can be enabled via the new backend option "texture_runtime_with_derivs". Please refer to the "Example for Texture Filtering with Automatic Derivatives" documentation for more details.
    - Measured EDFs and BSDFs can now be translated to PTX, Native x86 and LLVM IR. Note that the texture runtime needs to be extended with utility functions that enable runtime access to the data.
    - Spot EDFs can now be translated to PTX, Native x86 and LLVM IR.
    - The `nvidia::df` module has been removed.

- MDL SDK examples
    - Support for automatic derivatives has been added to the `example_execution_native`,`example_execution_cuda` and `example_df_cuda` examples, which can be enabled via a command line option.
    - The `example_execution_native` example has been extended to allow to specify materials on the command line. It is now also possible to enable the user-defined texture runtime via a command line switch.
    - The CUDA example texture runtime has been extended with support for measured EDF and BSDF data.
    - The MDL Browser is now available as a QT QML Module which can also be integrated in non-qt based applications.
    - Initial support for class compiled parameters of type `Texture`, `Light_profile`, and `Bsdf_measurement` has been added to `example_df_cuda`. So far, it is only possible to switch between all loaded resources, new resources cannot be added.

**Fixed Bugs**

- General
    - An error when exporting presets where MDL definitions used in the arguments require a different version than the prototype definition has been fixed.

- MDL Compiler and Backends
    - A missing check for validity of refracted directions has been added to the generated code for the evaluation of microfacet BSDFs.
    - Incorrect code generation for `math::length()` with the atomic types `float` and `double` has been fixed.
    - The computation of the minimum correction pattern in the MDL compiler has been fixed.
    - The compilation of || and && inside DAG IR has been fixed.
    - Pre and post increment/decrement operators when inlined into DAG-IR have been fixed.
    - Previously missing mixed vector/atomic versions of `math::min()` and `math::max()`
  have been added.
    - The handling of (wrong) function references inside array constructor and init constructor has been fixed, producing better MDL compiler error messages.
    - The hash computation of lambda functions with parameters has been fixed.
    - If an absolute file url is given for a module to be resolved AND this module exists in the module cache, the module cache is used to determine its file name. This can speed up file resolution and allows the creation of presets even if the original module is not in the module path anymore.
    - A memory leak in the JIT backend has been fixed.
    - The generated names of passed expressions for code generation have been fixed.

**Known Restrictions**

- When generating code for distribution functions, the parameter `global_distribution` on spot and measured EDF's is currently ignored and assumed to be false.


MDL SDK 2018.1.1 (307800.2890): 15 Sep 2018
-----------------------------------------------

ABI compatible with the MDL SDK 2018.1.1 binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    -  A new API function `mi::neuraylib::ILink_unit::add_material` has been added to
       translate multiple distribution functions and expressions of a material at once.
    -  A new API function `mi::neuraylib::IMdl_backend::translate_material` has been added to
       translate multiple distribution functions and expressions of a material at once.
    -  A new function `mi::neuraylib::ICompiled_material::get_connected_function_db_name`
       has been added.
    -  A new function `IImage_api::create_mipmaps` has been added.
    -  A new parameter has been added to the functions `mi::neuraylib::ITarget_code::execute*`
       to allow passing in user-defined texture access functions.

- MDL Compiler and Backends
    - Diffuse EDFs can now be translated to PTX, native x86 and LLVM IR.
    - Support for passing custom texture access functions has been added to the Native backend.
      The built-in texture handler can be disabled via the new backend option
      `"use_builtin_resource_handler"`.

- MDL SDK examples
    - The `example_df_cuda` example now features simple path tracing inside the sphere to
      enable rendering of transmitting BSDFs.
    - To allow loading of multiple materials within a module, a wildcard suffix "*" is now
      supported in the material name command line parameter of the `example_df_cuda` example.
    - The `example_df_cuda` has been updated to illustrate the use of the new function
      `mi::neuraylib::ILink_unit::add_material`.
    - The `example_execution_native` has been extended to illustrate the use of user-defined
      texture access functions.
    - The `example_mdl_browser` can now be built on Mac OS.

**Fixed Bugs**

- General
    - The handling of archives containing a single module has been fixed in the
      `mi::neuraylib::IMdl_discovery_api`.
    - The handling of relative search paths has been fixed in the
      `mi::neuraylib::IMdl_discovery_api`.

- MDL Compiler and Backends
    - Various fixes have been applied to the code generated for BSDF's:
        - The computation of the `evaluate()` function for glossy refraction has been fixed
          (`df::simple_glossy_bsdf, df::microfacet*`).
        - The `sample()` functions for layering and mixing now properly compute the full PDF
          including the non-selected components.
        - The implementation of `df::color_clamped_mix()` has been fixed
          (the PDF was incorrect and BSDFs potentially got skipped).
        - All mixers now properly clamp weights to 0..1.
        - Total internal reflection is now discarded for glossy BSDF
          (`df::simple_glossy_bsdf`, `df::microfacet*`) with mode `df::scatter_transmit`,
          as defined in the MDL specification.
    - Incorrect code generation for `math::normalize()` with the atomic types `float` and
      `double` has been fixed.
    - The generation of function names for array index functions for modules in packages
      has been fixed.
    - In rare cases, compilation of a df* function could result in undeclared parameter names
      (missing _param_X error), which has been fixed.
    - The compilation of MDL presets of re-exported materials has been fixed.
    - In rare cases, the original name of a preset was not computed, which has been fixed.


MDL SDK 2018.1 (307800.1800): 09 Aug 2018
-----------------------------------------------

- Initial open source release
- ABI compatible with the MDL SDK 2018.1 (307800.1800) binary release (see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))
- The following features are only available in the binary release and excluded in the
  source code release:
    - MDL distiller
    - Texture baking (see *examples/mdl_sdk/execution_cuda* for example code for texture baking)
    - GLSL compiler back end
- Added: MDL Core API, a lower-level compiler API in the MDL SDK (see *src/prod/lib/mdl_core* and *doc/mdl_coreapi*)
- Added: examples for the MDL Core API (see *examples/mdl_core* and *doc/mdl_coreapi*)
