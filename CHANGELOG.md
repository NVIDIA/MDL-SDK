Change Log
==========
MDL SDK 2020.1 (334300.2228): 11 Aug 2020
-----------------------------------------------

ABI compatible with the MDL SDK 2020.1 (334300.2228) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL 1.6 Language Specification

    - Hyperlinks have been added to the MDL Specification PDF document.

- General

    - Enabled support for MDL modules whose names contains parentheses, brackets, or commas.
    - The interface `IMdl_entity_resolver` has been redesigned. Support for resolving resources has
      been added.
    - The new interface `IMdl_module_transformer` allows to apply certain transformations on MDL
      modules.
    - Various API methods have been added in order to reduce the error-prone parsing of MDL-related
      names: To retrieve DB names from MDL names use `get_db_module_name()` and
      `get_db_definition_name()` on `IMdl_factory`. To retrieve parts of the MDL name from the
      corresponding DB element use `get_mdl_package_component_count()`,
      `get_mdl_package_component_name()`, and `get_mdl_simple_name()` on `IModule`;
      `get_mdl_module_name()`, `get_mdl_simple_name()` on `IMaterial_definition`; and
      `get_mdl_module_name()`, `get_mdl_simple_name()`, and `get_mdl_parameter_type_name()` on
      `IFunction_definition` and `IAnnotation_definition`.
    - Added a new overload of `IModule::get_function_overloads()` that accepts a simple name
      and an array of parameter type names instead of two strings. This avoids the ambiguity when
      parsing parentheses and commas. The old overload is deprecated and still available if
      `MI_NEURAYLIB_DEPRECATED_11_1` is defined.
    - Improved recursive MDL module reloading: changed the traversal order from pre-order to
      post-order traversal, avoid flagging a module as changed if it did not change at all.
    - Improved `Definition_wrapper`: the creation of functions calls for template-like MDL
      functions requires now an actual argument list since the dummy defaults for such functions
      easily lead to function calls with the wrong types in the signature.
    - Added more options to control the generation of compiled materials in class compilation mode:
      Folding of enum and bool parameters, folding of individual parameters, folding of cutout
      opacity, and folding of transparent layers.
    - Added methods to retrieve the MDL version of modules, and the MDL version when a particular
      function or material definition was added to (and, if applicable, removed from) the MDL
      specification.
    - Added methods to retrieve the MDL system and user paths.
    - The legacy behavior of `df::simple_glossy_bsdf` can now be controlled via the interface
      `IMdl_configuration`.
    - The return type of `IFunction_definition::get_body()` has been changed from 
      `const IExpression_direct_call*` to `const IExpression*`.

- MDL Compiler and Backends

    - Added support for target code serialization in the HLSL and PTX backends. See the new
      methods `get_backend_kind()`, `supports_serialization()`, `serialize()`, and
      `get_argument_layout_count()` on `ITarget_code`, and
      `IMdl_backend::deserialize_target_code()`. The new context option
      `"serialize_class_instance_data"` for `ITarget_code::serialize()` controls whether
      per-instance data or only per-class data is serialized.
    - Allow total internal reflection for glossy BSDFs with mode `df::scatter_transmit` (libbsdf).
    - When derivatives are enabled, `state::position()` is now derivable. Thus, the `"position"`
      field of `Shading_state_material_with_derivs` is now a derivative type.
    - Added `"meters_per_scene_unit"` field to `Shading_state_material`. It is used, when folding
      of `state::meters_per_scene_unit()` and `state::scene_units_per_meter()` has been disabled
      via the new `IMdl_execution_context` `"fold_meters_per_scene_unit"` option.
    - Added derivative support for matrices.
    - Added derivative support for scene data functions. Requires new texture runtime functions
      `scene_data_lookup_deriv_float`, `scene_data_lookup_deriv_float2`,
      `scene_data_lookup_deriv_float3`, `scene_data_lookup_deriv_float4`, and
      `scene_data_lookup_deriv_color` (see `texture_support_cuda.h` in the MDL SDK examples
      for the prototypes).
    - Added `mi::neuraylib::ICompiled_material::depends_on_uniform_scene_data()` analyzing
      whether any `scene::data_lookup_uniform_*()` functions are called by a material instance.
    - Implemented per function render state usage in `ITarget_code`.
    - Avoid reporting deprecated warnings, if current entity is already deprecated.

- MDL SDK examples

    - Examples Shared
        - Added utility headers for strings, enums, I/O, OS, and MDL specific tasks to be used in
          the examples. Updated examples to make use of the new utility headers.
        - Added GUI classes to illustrate MDL parameter editing and to unify user interfaces in
          examples in the future.

    - Example DXR
        - Added a new more structured user interface with various new features including the
          loading of scenes and environments from the menu, the replacement of materials,
          compilation and parameter folding options, and parameter editing in instance compilation
          mode.
        - Integrated the MDL browser (if built) for the replacement of a selected material.
        - Added shader caching to improve loading times (has to be enabled with option
          `--enable_shader_cache`).

    - GLTF Support
        - Added `KHR_materials_clearcoat` support, also in Example DXR.

    - MDL plugin for Arnold
        - Added a new example to illustrate the integration of MDL into an existing advanced CPU
          renderer.

    - Example Code Generation
        - Added a new example to illustrate HLSL and PTX code generation.

    - Example OptiX 7
        - Added a new example to illustrate the use MDL code as OptiX callable programs in a
          closest hit shader, and alternatively, how to link the MDL code directly into a
          per-material closest hit shader for better runtime performance.

    - Example Native
        - Added missing scene data functions of custom texture runtime.

**Fixed Bugs**

- General

    - Fixed documentation of `Bsdf_evaluate_data` structs: eval function results are output-only,
      not input/output.
    - Fixed compilation of materials using the array length operator.
    - Fixed crash on CentOS 7.1 when importing non-trivial MDL modules.
    - Fixed incorrect behavior during function call creation when implicit casts were enabled.

- MDL Compiler and Backends

    - Fixed file resolution during re-export of MDLE modules.
    - Fixed missing clearing of context messages when creating a link unit.
    - Fixed detection of absolute file names on Windows for MDLEs on a network share.
    - Fixed support for the read-only segment and resources inside function bodies when compiling
      for the native target.
    - Fixed rare crash/memory corruption than could occur on MDLE creation.
    - Fixed possible crash when inlining a function containing a `for (i = ...)` loop statement.
    - Fixed potential crash in the auto importer when imports of the current module are erroneous.
    - Fixed handling of suppressed warnings if notes are attached to them, previously these were
      attached to other messages.
    - Fixed possible crash in generating MDLE when array types are involved.
    - Fixed printing of initializers containing sequence expressions, it is `T v = (a,b);`, not `T
      v = a, b;`.
    - Improved AST optimizer:
        - Write optimized `if` conditions back.
        - Write optimized sub-expressions of binary expressions back.
        - Handle `constant && x`, `constant || x`, `x && constant`, `x || constant`.
    - Fixed folding of calls to `state::meters_per_scene_unit()` and
      `state::scene_units_per_meter()` in non-inlined functions.
    - Fixed wrong code generation for int to float conversions with derivatives.
    - Fixed a bug in the generated HLSL code that caused wrong calculations because loads were
      erroneously placed after calls modifying memory.
    - Fixed checking of valid MDL identifiers (names starting with `"do"` were treated as keywords,
      but not `"do"` itself).
    - Fixed overload resolution for MDL operators.
    - Fixed crash in MDL runtime when using nonexistent image files with MDL.
    - Fixed invalid translation of `int` to `float` conversion with derivatives enabled.
    - Fixed broken `math::sincos()` on vectors.
    - Fixed failing MDLE creation due to several missing or non-exported entities (constants,
      annotations).
    - Fixed failing MDLE creation if the main module was < MDL 1.6, but imported an MDL 1.6 module.
    - Fixed failing MDLE creation if non-absolute imports of `::base` were used.
    - Fixed rare crashes occurring when the array constructor is used in annotations.
    - Fixed lost enumeration of BSDF data textures used by the libbsdf multiscatter.

- MDL SDK examples

    - Examples Shared
        - Fixed failing CUDA checks when minimizing application.

MDL SDK 2020.0.2 (327300.6313): 28 May 2020
-------------------------------------------

ABI compatible with the MDL SDK 2020.0.2 (327300.6313) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL Compiler and Backends

    - Reduced the minimum roughness threshold for microfacet BSDFs from 1e-3 to 1e-7 to
      make them usable for mirrors and clear glass, which is inefficient but could be required by
      ubershaders.
    - Added `"ro_data_segment"` field to `Shading_state_environment` (`"ro_data_segment_offset"` for
      HLSL).
    - Use `"direction"` for the field name of `Shading_state_environment` (HLSL only).
    - Made `state::position()` derivable.

**Fixed Bugs**

- MDL Compiler and Backends

    - Fixed some rare cases were resources inside MDL functions got lost.
    - Fixed crash in MDL code generators due to MDL core compiler missing some error messages when
      a (wrong) member selection has the same name like an enum constant.
    - Fixed rare NaN in microfacet sampling.
    - Fixed error value of `ITarget_code::get_body_*()` functions.
    - Fixed return value of `ITarget_code::create_argument_block()` when required resource callback
      is missing.
    - Fixed read-only data segment data not being set for native lambdas.
    - Fixed resource enumeration when compiling multiple expressions in a link unit with
      `add_material()`: ensure that resources in material bodies are enumerated first.


MDL SDK 2020.0.1 (327300.3640): 30 Mar 2020
-------------------------------------------

ABI compatible with the MDL SDK 2020.0.1 (327300.3640) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General

    - The standalone tools mdlm and i18n have been extended to support Unicode package/module
      names.
    - On Windows, we recommend Visual Studio 2017 or 2019. Visual Studio 2015 is still the minimum
      requirement.

- MDL SDK examples

    - `example_dxr`:
        - Enhanced dependency tracking when reloading materials to also update indirectly affected
          materials.
        - Only the compiled material hash is considered to detect reusable generated code which
          allows to reuse existing materials for structurally equal instances.
        - Skip format conversion for multi-scatter lookup data.
        - Added support for larger glTF scenes up to 2GB.
        - Improved cleanup when loading erroneous scenes.

**Fixed Bugs**

- General

    - Fixed handling of resources inside function bodies. Previously, these resources were not
      found under some conditions, causing black textures for instance.
    - Fixed too strict error checks for creation of function calls of the array index operator, the
      ternary operator, and the cast operator.
    - Fixed creation of variants without specifying any annotations where the annotations of the
      prototype were erroneously copied to the variants.
    - Fixed loading of string-based modules with absolute file paths for resources.
    - Fixed documentation of generated code interfaces: The results of eval functions are
      output-only, not in/out.

- MDL Compiler and Backends

    - Fixed a subtle bug in one of the code caches, which caused ignored argument changes under
      some complex conditions. Typically, boolean parameters were vulnerable, but could happen to
      parameters of any type.
    - Fixed MDL archive tool failures with Unicode package names. The MDL version of such archives
      is now automatically set to MDL 1.6 as lowest necessary version.
    - A bug in the resource handling was fixed that previously caused resources to be resolved and
      loaded more that once, possibly leading to failures if search paths had been changed in
      between.
    - Fixed the MDL core compiler's analysis pass. Some analysis info was computed but not
      annotated, causing JIT failures on functions that consists of a single expression body only.


MDL SDK 2020 (327300.2022): 22 Feb 2020
-----------------------------------------------

ABI compatible with the MDL SDK 2020 (327300.2022) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General

    - A new function `mi::neuraylib::IMdl_compiler::get_df_data_texture()` has been added.
    - A new function `mi::neuraylib::ITarget_code::get_texture_df_data_kind()` has been
      added.
    - A new enum `mi::neuraylib::Df_data_kind` has been added.
    - A new flag on `mi::neuraylib::IMdl_configuration` instructs the MDL compiler to keep the
      names of let expressions and expose them as temporaries on MDL material and function definitions.
      This brings the structure of the material/function definition presented in the API closer to the
      one in the .mdl file.
    - The FreeImage plugin is now based on FreeImage 3.18.0.

- MDL Compiler and Backends

    - Support for the chiang `hair_bsdf` has been added to the code generator for distribution
      functions.

    - Changes to the internal representation of builtin MDL operators.
      MDL supports a variety of operators, potentially featuring an endless number of instances:
      - array index `operator[]`
      - array length symbol
      - ternary operator `?:`

      Previously, 'local' definitions were created for every used instance of these operators in an MDL
      module:
      - array index operator on type T in module M: `M::T@(T,int)`
      - array length symbol on type T in module M: `M::T.len(T)`
      - ternary operator on type T in module M: `M::operator?(bool,T,T)`

      This representation had several drawbacks:
      - there might be one definition for the same operator in every module
      - if the operator was not used inside the source of a module, it was not created
      
      Especially the second point lead to several problems in the editing application. Hence, starting
      with the 2020.0.0 release, the internal representation was changed and operators are now
      represented by 'global' template-like definitions:
      - array index operator: `operator[](<0>[],int)`
      - array length operator: `operator_len(<0>[])`
      - ternary operator: `operator?(bool,<0>,<0>)`

      In addition, the name of the cast operator was changed from `operator_cast()` to
      `operator_cast(<0>)`.

      Drawback: When inspecting the types of the operators definition, 'int' is returned for the
      template types, but this might be changed in the future by expanding the type system.

    - Support for HLSL scene data renderer runtime functions has been added. See the 
     `scene_data_*` functions in `mdl_renderer_runtime.hlsl` MDL SDK DXR example for an
      example implementation.

- MDL SDK examples

    - `example_dxr`:
        - The texture loading pipeline has been simplified and support for UDIM textures has been
          added.
        - Support for scene data introduced in MDL 1.6 (prim vars) has been added.
    - `example_df_cuda`:
        - Support for evaluation of hair-bsdfs on an analytical cylinder has been added.


**Fixed Bugs**

- MDL Compiler and Backends

    - Support for multiple multiscatter textures in one target code object has been fixed.

    - Support for multiscatter textures with disabled `resolve_resources` backend option has
      been fixed.

    - Multiple HLSL code generation problems leading to non-compilable code have been fixed.


MDL SDK 2019.2 (325000.1814): 18 Dec 2019
-----------------------------------------------

ABI compatible with the MDL SDK 2019.2 (325000.1814) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL 1.6 Language Specification

    - The file path resolution algorithm has been changed to treat weak relative
      paths the same as strict relative paths if the referring MDL
      module has an MDL version of 1.6 or higher. Furthermore,
      the error checks have been simplified to only protect relative paths from
      referring to files in other search paths.
    - The import of standard library modules has been changed in all examples
      to use absolute path imports.
    - An additional way of defining functions has been added using an expression
      instead of a procedural function body.
    - Let-expression can also be applied to functions defined using an expression.
    - The limitation has been removed that package names and module names can
      only be identifiers.
    - The new using `alias` declaration has been added to enable the use of Unicode
      names for module names and package names.
    - The description has been clarified that standard module names shadow only
      modules of the same fully qualified name while modules in subpackages can have
      a standard module name as their unqualified name.
    - The new `scene` standard library module has been added with
      `data_isvalid`, `data_lookup_ltype`, and
      `data_lookup_uniform_ltype` functions.
    - The new `multiscatter_tint` parameter has been added to all glossy BSDF
      models to enable energy loss compensation at higher roughness values.
    - The new `df::sheen_bsdf` bidirectional scattering distribution function has
      been added.
    - The new `df::tint` modifier overload has been added for the hair bidirectional
      scattering distribution function.
    - The new `df::tint` modifier overload has been added for the separate tinting
      of the reflective and transmissive light paths of a base BSDF.

- General

    - The new API functions
         - `mi::neuraylib::IModule::reload()`
         - `mi::neuraylib::IModule::reload_from_string()`
         - `mi::neuraylib::IModule::is_valid()`
         - `mi::neuraylib::IMaterial_definition::is_valid()`
         - `mi::neuraylib::IFunction_definition::is_valid()`
         - `mi::neuraylib::IMaterial_instance::is_valid()`
         - `mi::neuraylib::IMaterial_instance::repair()`
         - `mi::neuraylib::IFunction_call::is_valid()`
         - `mi::neuraylib::IFunction_call::repair()`
         - `mi::neuraylib::ICompiled_material::is_valid()`
      have been added to support reloading of MDL modules.
    - The requirements on MDL module names have been relaxed according to the MDL 1.6
      Specification to allow loading of modules with Unicode names.
    - The new API functions
        - `mi::neuraylib::ITarget_code::get_callable_function_df_handle_count()` and
        - `mi::neuraylib::ITarget_code::get_callable_function_df_handle()`
      have been added.
    - The new API function `mi::neuraylib::ITarget_code::get_texture_df_data()`
      has been added.
    - `mi::neuraylib::IMdl_compiler::load_module()` can now be called from multiple
      threads to allow loading modules in parallel. To support custom thread blocking
      the new interfaces
        - `mi::neuraylib::IMdl_loading_wait_handle ` and
        - ` mi::neuraylib::IMdl_loading_wait_handle_factory`
      have been added.
    - The new API functions
        - `mi::neuraylib::IMaterial_definition::get_body()`
        - `mi::neuraylib::IMaterial_definition::get_temporary_count()`
        - `mi::neuraylib::IMaterial_definition::get_temporary()`
        - `mi::neuraylib::IFunction_definition::get_body()`
        - `mi::neuraylib::IFunction_definition::get_temporary_count()`
        - `mi::neuraylib::IFunction_definition::get_temporary()`
      have been added.
    - The signature of the function `mi::base::ILogger::message()` has been changed.
    - The API function `mi::neuraylib::ITransaction::edit()` has been adapted to disallow
      editing of database elements of type `mi::neuraylib::IMaterial_definition` and
      `mi::neuraylib::IFunction_definition`.
    - Support for multiple occurrence of the same annotation has been added to
      `mi::neuraylib::Annotation_wrapper`.
    - Support for deprecated features guarded by `MI_NEURAYLIB_DEPRECATED_8_1` and
      `MI_NEURAYLIB_DEPRECATED_9_1` has been removed.

- MDL Compiler and Backends

    - Support for MDL 1.6 has been added to the MDL core compiler.
    - Limited support for MDL 1.6 features has been added to the backends, in
      particular, the `scene` module is supported, but currently no code is generated
      for interrogating the renderer, hence always the default value is returned.
    - Support for MDL 1.6 has been added to the generated code for distribution
      functions, that is `df::tint(reflection_tint, transmission_tint)`,
      `df::sheen_bsdf()`, the `multiscatter_tint` parameter of all BSDFs exposing this
      in MDL 1.6 (note that this requires that all four dimensions of
      `Bsdf_sample_data::xi` are set).
    - For evaluating parts of distribution functions that are named by handles,
      `Bsdf_evaluate_data` and `Bsdf_auxiliary_data` are adapted to select individual
      handles.
    - A new backend option `df_handle_slot_mode` to select how evaluate and auxiliary
      data is passed between the generated code and the render has been added.
    - The `bsdf` field of `Bsdf_evaluate_data` is split into `bsdf_diffuse` and
      `bsdf_glossy`.
    - The `Bsdf_sample_data` structure now requires a 4th uniform random number and
      returns the handle of the sampled distribution part.
    - Inlining of functions containing constant declarations into the DAG has been
      implemented.
    - Support for light-path-expressions in generated code for distribution functions
      via handles has been added.
    - Support for retrieving albedo and normal in generated code for distribution
      functions via generated auxiliary functions has been added.
    - The entity resolver has been sped up for built-in modules in some cases where it
      is clear that the module can only be read from the MDL root.
    - The memory size of the DAG representation has been slightly reduced by
      internalizing all DAG signatures.
    - The DAG representation now uses unsafe math operations, especially `x * 0 = 0`
      for floating point values.

- MDL SDK examples

    - Example `df_cuda` has been adapted to illustrate how to evaluate parts of the
      distribution functions named by handles via light path expressions (LPEs).
    - All examples have been adapted to support processing of Unicode command line
      arguments.

**Fixed Bugs**

- General

    - An issue in the light profile parser has been fixed: For IESNA LM-63-2002 files
     the `ballast-lamp` value incorrectly acted as multiplier for the intensity.

- MDL Compiler and Backends

    - Several issues in the generated code for distribution functions have been fixed:
        - Bugs in the computation of the pdf and eval functions of
          `df::ward_geisler_moroder_bsdf` have been fixed.
        - Incorrect pdf computation (for `sample`, `eval`, and `pdf` functions) in
          `df::ward_geisler_moroder_bsdf` and `df::backscattering_glossy_reflection_bsdf`
          have been fixed.
        - Fixed a missing re-scale of pseudorandom numbers for v-cavities based masking,
          leading to biased results for `df::scatter_reflect_transmit`.
        - Only use refraction-based half vector for Fresnel-layering, not for all curve
          layering operations.
        - Add simple inside/outside material support based on IOR comparison to determine
          which IOR to override in Fresnel layering. This fixes incorrect rendering when
          BSDFs of type `df::scatter_reflect` and `df::scatter_transmit` are layered using
          `df::fresnel_layer`, in particular missing total internal reflection.
    - The implementation of `math::isnan()` and `math::isfinite()` has been fixed for
      vector types.
    - Printing of quiet NaNs for HLSL has been fixed.
    - A crash in the MDL core compiler that could occur if exported types contain errors
      in their default initializers has been fixed.
    - Wrong function names generated from `debug::assert()` calls when placed after a
      while loop have been fixed.
    - The name of the `anno::deprecated()` parameter has been fixed, it is `description`,
      not `message`.
    - The export of MDL modules containing relative imports has been fixed, access to
      the imported entities is now generated correctly.


MDL SDK 2019.1.4 (317500.5028): 27 Aug 2019
-----------------------------------------------

ABI compatible with the MDL SDK 2019.1.4 (317500.5028) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General

    - A new function `mi::neuraylib::IValue_texture::get_gamma()` has been added.
    - The following new functions have been added to the target code generation:
        - `mi::neuraylib::ITarget_code::execute_bsdf_auxiliary()`
        - `mi::neuraylib::ITarget_code::execute_edf_auxiliary()`
    - A new function `mi::neuraylib::ICompiled_material::get_surface_opacity()`
      has been added.

- MDL Compiler and Backends

    - Code generation for auxiliary methods has been added on distribution
      functions for potential use in AI-denoising.
    - The spectral color constructor `color(float[<N>],float[N])`,
      `math::emission_color()`, and `math::blackboby()` are now supported in the
      JIT backend.
    - More optimizations regarding elemental constructors in the DAG
      representation have been implemented.
    - Map XOR operators on Boolean values to NOT-EQUAL in the HLSL backend to be
      compatible to the SLANG compiler.

- MDL SDK examples

    - The example programs `example_dxr` and `example_df_cuda` have been extended
      to illustrate the use of auxiliary functions.
    - A modified version of `example_dxr` has been added to illustrate the usage
      of MDL in a multi-threaded context.
    - Camera controls have been improved and new options have been added to
      the example program `example_dxr`.

**Fixed Bugs**

- General

    - The export of MDLE files from in-memory MDL modules has been fixed.

- MDL Compiler and Backends

    - Temporary exponential creation of DAG nodes when using derivatives has
      been fixed.
    - Code generation of parameters reused multiple times in a derivative context
      has been fixed.
    - Relative imports including "." and ".." have been fixed.
    - Duplicate global variables in generated HLSL code have been fixed.
    - Invalid code generation for HLSL for special materials has been fixed.
    - Indeterministic rare compilation errors regarding unknown functions have
      been fixed.
    - Indeterministic rare hangs during compilation with multiple threads have
      been fixed.
    - Under rare condition the code cache could return HLSL code instead of PTX
      and vice versa. This has been fixed.
    - The code cache that was not working under several conditions has been fixed.
    - The handling of the `?:` operator on arrays inside the DAG representation
      has been fixed such that it computes the right name now.
    - The handling of unresolved resource paths in the target code has been fixed.
      Previously all resources were mapped to index 1.
    - A crash in the code generator when handling uniform matrix expressions with
      automatic derivatives enabled has been fixed.
    - A crash in the HLSL code generator for non-default optimization levels has
      been fixed.
    - A crash when adding distribution functions to a link unit after adding non-
      distribution functions has been fixed.


MDL SDK 2019.1.1 (317500.2554): 08 Jun 2019
-----------------------------------------------

ABI compatible with the MDL SDK 2019.1.1 (317500.2554) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General

    - A new API class `mi::neuraylib::IAnnotation_definition` has been added.
    - The following new functions have been added:
        - `mi::neuraylib::IAnnotation::get_definition()`
        - `mi::neuraylib::IModule::get_annotation_definition_count()`
        - `mi::neuraylib::IModule::get_annotation_definition(Size)`
        - `mi::neuraylib::IModule::get_annotation_definition(const char*)`
    - Added the Boolean option `fold_ternary_on_df` to fold ternary operators on
      distribution function types in material bodies even in class compilation mode:
      Some uber-materials extensively use the ternary operator to switch BSDFs. This
      causes a lot of generated code and may not be supported by some renderers.
      Enabling this option will compile all arguments that control the condition of
      such a ternary operator in. Changing any of these arguments will then require
      a recompilation.

- MDL SDK examples

    - Support for the glTF extension `KHR_materials_pbrSpecularGlossiness` has been
      added to `example_dxr`.

**Fixed Bugs**

- General

    - Wrong copying of gamma values during cloning of texture values has been fixed.
      This especially happened for default parameters during material instantiation.
    - Fixed problem with dangling user defined types when a module is removed;
      previously removing a module left all its user defined types alive, preventing
      creating new (different) ones when the deleted module was reloaded.
    - Fixed failure when generating MDLE files that reference `intensity_mode` values.

- MDL Compiler and Backends

    - Fixed wrong optimization of empty do-while loops in the core compiler.
    - Fixed imports starting with `.` or `..`, which caused wrong package names before.
    - Fixed printing of float and double constants when MDL code was generated;  not
      enough digits were used previously, causing lost precision when this code was
      compiled again.
    - Fixed a problem where sometimes several uninitialized variables were generated and
      copied around in generated HLSL code.
    - Fixed generation of useless copies, like `t = t;` in generated HLSL code.
    - Generated better HLSL code for vector constructor calls, like
     `floatX(a.x, x.y, ...) => a.xy...`
    - JIT compilation of functions involving `sin` and `cos` with the same argument on
      Mac OS has been fixed.
    - The implementation of `df::color_weighted_layer` has been fixed.

- MDL SDK examples

    - `findGLEW` has been fixed in the build script to work with CMake 3.15.


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
    `sqrt(c)` for  `c < 0` has been fixed. Now `NaN` is computed for both.


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
