Change Log
==========
MDL SDK 2023.1.0 (373000.1077): 14 Dec 2023
-----------------------------------------------


ABI compatible with the MDL SDK 2023.1.0 (373000.1077) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Known Issues and Restrictions**

- MDLE export can fail in this release with broken .mlde files that contain malformed MDL code.

**Added and Changed Features**

- General
    - Added `rotate_around_x/y/z` functions to `::nvidia::support_definitions`.
    - The MDL SDK comes now with unit tests. Building of the unit tests is
      controlled via the `cmake` option `MDL_ENABLE_UNIT_TESTS`. Unit
      tests can be executed from the command line via CTest, `make test`,
      or the Visual Studio solution. Some unit tests require the `idiff`
      tool from `OpenImageIO`.
    - The baker uses now multiple threads.
    - The deprecated `FreeImage` plugin as been removed.
    - The recommended `vcpkg` version has been updated. It is now also
      recommended to install `GLEW` and `GLFW` via `vcpkg`. The
      feature flag `tools` for `OpenImageIO` is now required by some
      unit tests.
    - The database used by the MDL SDK supports now multiple parallel
      transactions. See the documentation of `ITransaction` for visibility
      and conflict resolution. This change includes adding support for
      `ITransaction::abort()`. As a consequence, if an `ITransaction` is
      released without committing or aborting, it is now automatically aborted
      with a warning instead of being committed with an error message.
    - The new API component `ILogging_configuration` has been added to the
      MDL SDK. The API methods `IMdl_configuration::set_logger()` and
      `get_logger()` have been deprecated. Use
      `ILogging_configuration::set_receiving_logger()` and
      `ILogging_configuration::get_receiving/forwarding_logger()` instead. The
      old methods are still available if `MI_NEURAYLIB_DEPRECATED_14_1` is
      defined. Use `ILogging_configuration::set_log_prefix(0)` to prevent
      the forwarding logger from automatically adding the severity to the log
      message.
    - Additional performance improvements, in particular with a focus on
      creation of compiled materials.
    - The new methods `IType_factory::get_mdl_type_name()` and
      `create_from_mdl_type_name()` allow to serialize and deserialize types via
      their type names.
    - The new method `IType_factory::get_mdl_module_name()` returns the
      name of the MDL module that defines a given type. This is primarily
      useful for enum and struct types.
    - The method `IType_array::get_deferred_size()` returns now the
      "simple" symbol name, e.g., "N" instead of the fully qualified one.
    - The method `IExpression_factory::create_direct_call()` allows now to
      create calls to unexported functions. Note that such calls can only be
      used in the same module as the called function.
    - When loading textures via an MDL module, failures do no longer cause a
      dummy instance of `ITexture` to be created and
      `IValue_texture::get_value()` returns now a NULL pointer.
    - Python Bindings:
        - Added more missing interface functions to the bindings.
        - Removed all generated `"declare_interface"`-types.
        - Added post build step to strip unused types and functions and
          marked constructors invalid.
        - Fixed `"error"`-out-parameters by providing a ReturnCode type that can be passed
          by reference.
        - Added `UUID` comparison for interface types.
        - Updated the binding of enums which are now Python enums in the appropriate scope.
        - Extended unit tests written in Python.
        - Added a coverage report option for the unit tests written in Python.
        - Added missing and fixed existing `get/set_value` functions for various `IData` class bindings.
        - Mapping now `mi::Size` to `Sint64` to handle `-1` returns correctly.
        - Removed the `IAttribute_set` function from scene elements.
      
- MDL Compiler and Backends
    - Removed unused `exception_state` parameter from generated functions for non-native backends
      to improve performance. Needs update in renderers calling these functions.
    - Let generated functions for material expressions of base types and vector types return
      their values directly instead via a result buffer by setting the new backend option
      `"lambda_return_mode"` to `"value"`. Only supported by the PTX and the LLVM-IR backend.
    - Generated auxiliary functions now separate albedo into diffuse and glossy, similar to the evaluate functions.
    - Generated auxiliary functions now also report roughness.  

- MDL Distiller and Baker
    - Debugging features for mdltlc: `debug_name` and `debug_print` statements
      in MDLTL rules files.
    - Added mdltl unit tests.

- MDL SDK examples
    - Replaced compiler define `NO_DIRECT_CALL` by command line parameter
      `--use-direct-call` to make it easier to try both variants.
    - Adapted MDL SDK df_cuda and OptiX 7 examples to use the new value return mode.
      The MDL Core df_cuda example still uses the old default mode (`"sret"`).
    - AxF Example:
        - Updated to Pantora 1.9.0.
        - Added search path command line options.
    - Example DXR:
        - Changed hardware sampler mode to clamp and implemented repeat by software
          to fix edge cases when cropping.
        - Added a MaterialX to MDL version number parameter in preparation for
          upcoming MaterialX releases.
        - Added options to render multiple auxiliary outputs to file in one run.

**Fixed Bugs**

- General
    - base.mdl:
        - Improved tangent space handling for bump maps. Noise-based bump mapping is now
          oriented correctly for object and world space coordinate sources.
        - Additionally, coordinate transforms change the orientation consistently now.
          This adds one field to `base::texture_coordinate_info`.
    - Added a work-around to handle user defined constants inside the code that is added using
      `add_function()` to an existing MDL module in the `MDL_module_builder` interface.
      This can only handle cases where the user defined type is either defined locally
      or imported using an absolute path.
    - Python Binding: Fixed proxy parameter handling of `"IMdl_distiller_api::create_baker"`
      and `"ILight_profile::reset_*"` functions.

- MDL Compiler and Backends
    - Fixed auto-import of enum conversion operators.
    - Fixed a case where the `auto-importer` was not able to import conversion operators
      (`enum-to-int`), which caused wrong prefixed constructors when exporting MDL,
      e.g. `base::int(value)`.
    - Adapted data layout for PTX to match data layout used by CUDA compiler
      to avoid problems with misaligned data types.
    - Fixed wrong function indices set for init functions in single-init mode,
      when `ILink_unit::add_material()` is called more than once.
    - Fixed invalid CUDA prototypes returned by
      `mi::neuraylib::ITarget_code::get_callable_function_prototype()`.
    - Fixed `texremapu` in `base.mdl` for GLSL resulting in undefined behaviour
      for negative texture coordinates.
    - Improved handling of invalid MDL code in the MDL compiler.
    
- MDL Distiller and Baker
    - Fixed printing of `NaN`, `+inf` and `-inf` constants in the `mdl_distiller_cli`
      command line utility. They are now printed as `(0.0/0.0)`, `(1.0/0.0)` and `(-1.0/0.0)`
      respectively, same as in the MDL compiler and SL backends.
    - mdltlc: Fixed code generation for rules with node names.
    - mdltlc: Fixed code generation for creation of conditional expressions.
    
- MDL SDK examples
    - Example DXR: Fixed resource creation warnings reported with the
      `"--gpu-debug"` option on Windows 11.    

MDL SDK 2023.0.6 (367100.5773): 03 Nov 2023
-----------------------------------------------

ABI compatible with the MDL SDK 2023.0.6 (367100.5773) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - Various performance improvements, in particular with a focus on creation of
      compiled materials.
    - Python Bindings:
        - Added `get` and `set` value functions to the bindings of types in `mi::data` and
          added corresponding tests.
        - Removed the `IAttribute_set` interface from the bindings of the `IScene_element` types.

- MDL Compiler and Backends
    - Optimized high-level (GLSL/HLSL) code generator to reduce code size.
    - Added new backend option "`hlsl_remap_functions`": This allows to remap MDL functions
      (including state functions) to user implemented  Native HLSL implementations.
    
- MDL Distiller and Baker
    - Renamed `mdl_distiller` command line tool to `mdl_distiller_cli` to more
      clearly separate it from the distiller plugin of the same name.
      
**Fixed Bugs**

- General
    - Fixed `IFactory::compare()` for `IString` and `IRef` on Linux on ARM.
    - Python Bindings:
        - Fixed the binding for the `ITile::get_pixel()` and 
          `ITile::set_pixel()` functions.
        - Mapped `mi::Size` to `signed integer` in python to allow for
          comparing against `-1`.
        - Deprecated the tuple return of functions that have an `float*` out parameter in C++.
          Now an `ReturnCode` object is passed in and out as Python parameter.
        - Removed unused classes and functions from the bindings.
    
- MDL Distiller and Baker
    - mdltlc: Fixed matching on nested attribute expressions.
    - Fixed missing `enum` to `int` conversion operator an auto-imports,
      which caused compilation errors in rare cases.
    - Fixed context information when compiling entities in the DAG-backend
      (fixes only some asserts in debug mode).
    - Fixed HLSL/GLSL code generation for access to single element compound types,
      like arrays of length `1` or structs with only one field.

MDL SDK 2023.0.4 (367100.4957): 05 Oct 2023
-----------------------------------------------

ABI compatible with the MDL SDK 2023.0.4 (367100.4957) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - Python Bindings:
        - Added binding for the (built-in) entity resolver and added unit tests.
        - Accessing functions of invalid interfaces do not crash anymore but
          instead throw python exceptions.
        - Added more unit tests.
        - Extend the wrapper around MDL type (`Type`):
            - Give access to vectors and arrays size and element type.
            - Give access to matrices size.
        - Convert low level `IValues` to python friendly data types:
            - Extended to give access to file path for textures, light profiles
              and BSDF measurements.

- MDL SDK examples
    - Example DXR:
        - Reduced the HLSL glue code and added dynamic function selection
          for generated shader functions.
        - Added support for backface scattering and emission.
        - Handle `meters_per_scene_unit` at runtime and expose the parameter to the UI.
        - MaterialX resource resolution now uses the MDL entity resolver as fallback
          to handle tiled resources.
        - Handle the collapse flag of the `in_group` annotations.
        - Improved measured BSDF runtime implementation numerically.
        - Camera pose fitting now sets near and far plane distances.
        - Added support for `KHR_materials_iridescence` glTF extension.
    - Example Traversal:
        - Removed the preprocessor directive that disabled the distiller option.

**Fixed Bugs**

- General
    - Catch memory allocation failures in the OpenImageIO plugin while exporting images.
    - Python Bindings: Fixed the mdl_distiller plugin path in the scripts for
      running the examples.
    - Argument expressions that are created by the API are no longer optimized by the MDL
      compiler, but stay "unmodified" until arguments in class compilation mode are created.
      This makes the generated arguments more "deterministic" for users.
    - Fixed export of uv-tile textures. Only first tile was exported.
    - Fixed export of animated textures when frame number differs from frame ID.
        
- MDL Compiler and Backends
    - Fixed translation of vector access with non-constant index in some cases for HLSL/GLSL.
    - Fixed bit-operations on integer fields in structs containing derivable values.
    - HLSL/GLSL: The compiler uses now name mangling on struct types instead of
      the very simple old connection with '_'.
    - Fixed bug that caused crashes when several MDL modules import each other
      within a special order.
    - Material expressions which path prefix is "`geometry.displacement`" are now created in
      the displacement context.
    - Fixed code generation for re-exported MDL entities.
    - Fixed parsing of resource sets inside container files (MDLE).
    - Fixed ownership of types in created distribution functions which could lead to crashes
      under certain complex conditions.
    - Fixed crashes due to "missing functions" which are requested from wrong modules,
      for instance `state::cos()`.
    - Fixed printing of package name components which are MDL keywords and require
      quoting as Unicode identifiers.

- MDL SDK examples
    - Example DXR:
        - Fixed the UI parameter mapping for struct parameters.
        - Reviewed the UV coordinate handling and improved its documentation.
        - Added missing out of bounds check when reading `NV_materials_mdl` nodes.
        - Treat alpha channels in glTF textures as linear in cases where the RGB data
          is in sRGB color space.

MDL SDK 2023.0.2 (367100.3997): 01 Aug 2023
-----------------------------------------------

ABI compatible with the MDL SDK 2023.0.2 (367100.3997) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - Improved installation instructions (mention `VCPKG_PLATFORM_TOOLSET` for
      Windows, add 7.5 as the currently highest supported OptiX version).

**Fixed Bugs**

- General
    - Fixed export of `uvtile` textures referenced in MDL modules.
    - Improved installation instructions for installation of thirdparty dependencies via
      `apt-get`.
    - Fixed wrong reference counts when cloning instances of `IPointer` and
      `IConst_pointer` via `IFactory::clone()` without using the option
      `DEEP_ASSIGNMENT_OR_CLONE`.

- MDL Compiler and Backends
    - All backends: Fixed code generation for material expressions with path prefix
      "geometry.normal".
    - Any unsupported code in GLSL/HLSL code generation will now issue an "Internal JIT backend"
      error. The `<ERROR>` marker in the generated source will remain.
    - Fixed GLSL alias type definitions inside the GLSL backend
      (no more "missing conversion constructor" errors).
    - Module transformer: Fixed missing constant declarations when used in some annotations
      in non-root modules that are inlined.
    - Module transformer: Do not loose `anno::native()` annotations when inlining a module.
    - Fixed MDL compiler options containing double typed values
      (causing wrong `limits::DOUBLE_M[IN|AX]`).
    - Fixed the necessary precision to print floats and doubles so it can read back
      without losses.
    - Fixed missing elemental constructor for locally defined structs in the DAG- representation.
    - Fixed compiler crashes for: 
        - Invalid call expressions.
        - Invalid array types with a let expression as the array size.
    - Compiler now reports proper errors when a function is returned from a function.
    - Fixed PTX code generation that causes a GPU crash when the load/store instructions
      are generated with a wrong address space.
    - Fixed HLSL/GLSL backends generating float constants with not enough precision.
      In particular, the `::limits::MIN_FLOAT` constant was generated wrong,
      causing rounding to zero later.

- MDL Distiller and Baker
    - Fixed bug in command line `mdl_distiller` MDL printer for `::tex` module imports
      when textures are used in material parameter defaults.
    - Fixed mixer normalization in mdltlc for patterns with attributes.

MDL SDK 2023.0.0 (367100.2992): 03 Jun 2023
-----------------------------------------------

ABI compatible with the MDL SDK 2023.0.0 (367100.2992) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL 1.8 Language Specification
    - Updated version to 1.8.2.
    - Removed meaningless references to a preprocessor.
    - Reduced the requirement on thin-walled materials that they *should*
      have equal transmission from both sides.
    - Added part to supported selector string components for OpenEXR.
    - Clarified that MDL comments can be nested.
    - Extend list of separators in MDL and clarify their purpose.
    - Clarified that *unicode identifiers* cannot be empty.
    - Added the operator function call to the precedence table of all operators.
    - Added the cast operator to the precedence table of all operators.
    - Moved `operator` from the set of words reserved for future use
      to the reserved words in use.
    - Added section for *unicode identifiers*.
    - Clarified that the use of variables in let-expressions is read-only.
    - Clarified that binary logical operators may or may not use
      short-circuit evaluation.
    - Added definition of lvalues and rvalues.
    - Removed redundant subsections that `light_profile` and
      `bsdf_measurement` have no members.
    - Added requirement to check for array out-of-bounds access and
      to return default-constructed values in this case.
    - Clarified that function parameters are lvalues in their procedural
      implementation bodies.
    - Added that function parameters can only be used as read-only rvalues
      in functions defined by an expression.
    - Added that function parameters can only be used as read-only rvalues
      in let-expression.
    - Changed the grammar for the let-expression from *unary_expression*
      to *assignment_expressions* for the expression affected by
      the variables. This ensures the right scope precedence, e.g., the
      let-expression ends at a sequence operator.
    - Removed using alias declarations. They are replaced by Unicode
      identifiers directly supported as package and module names.
    - Added operator function call syntax for all applicable operators.
    - Clarified that material parameters can only be used as read-only rvalues
      in material definitions.
    - Clarified that material parameters can only be used as read-only rvalues
      in let-expression.
    - Clarified that let-bound variables cannot be used in their own initializer expressions.
    - Fixed the accidentally dropped grammar production rule for the
      import statements in MDL files.
    - Removed the support for annotations on packages with the `package.pkg` file.
    - Added Unicode identifiers for package and module names.
    - Changed productions for *import_path*, *qualified_import* to allow
      Unicode identifiers in package and module names.
    - Added a `collapsed` parameter to the `anno::in_group`
      standard annotation to control the UI presentation of groups.
    - Added the standard annotation `node_output_port_default` and the
      respective enumeration type `node_port_mode` to control the
      default initialization of output ports for external shade graph
      nodes when they are created for MDL functions.
    - Added the standard annotation `native` to hint that a function or material
      might have a native implementation in the integration.
    - Added the standard annotation `noinline` to hint that a function or material
      should not be inlined by the compiler.
    - Added that texture spaces and their respective tangents return in
      volume shading their respective values at the surface entry point
      of the current light-transport path.
    - Added that `state::texture_space` and the related tangent
      state functions return the zero vector in case the index is out of
      the range of available texture spaces.
    - Clarified the exact interpolation curve for `math::smoothstep` and that
      it is undefined for a zero or reversed interpolation range.
    - Added `float4x4` as supported data type for scene data lookup.
    - Removed the requirement on the scene data lookup functions to
      provide literal strings as arguments for the scene data names. They
      are now uniform strings and can be connected to uniform string
      parameters of a material of function definition.
    - Added a diffuse reflection BSDF modeling spherical Lambertian scatterers.
    - Added an approximative Mie VDF model for fog and cloud-like media.
    - Added a reference to the approximation used for the Mie scattering model of `fog_vdf`.
    - Clarified that the `particle_size` parameter of the new
      `fog_vdf` approximative Mie VDF model refers to the particle diameter.
    - Added `custom_curve_layer` and `directional_factor` as
      eligible base BSDF for the `thin_film` BSDF modifier.
    - Clarified that the `measured_factor` modifier BSDF allows
      reflectivities above one and it is in the responsibility of a material
      author to ensure energy conservation.
    - Added optional trailing commas to the productions of
      *parameter_list*, *argument_list*,
      *enum_type_declaration* and *annotation_block*.
    - Added a new Appendix G on external bindings and a section for the recommended
      binding of MDL functions to nodes with multiple output ports.

- General
    - `base.mdl`: Added an implementation of `base::abbe_number_ior`.
    - Improved precision of conversion of spectra to color, in particular for spectra
      with narrow peaks.
    - Refactored `base::coordinate_projection()` to reduce size of generated GPU code.
    - Added `mi::neuraylib::IMdl_backend::execute_init()` method to allow calling init functions
      in single-init mode for the native backend.
    - i18n translation:
        - Added support for the new boolean parameter to the `in_group annotation` (collapsed).
        - Added support for legacy "`in_group$1.7`" annotation.
        - i18n.exe tool:
            - Issue warning when annotation is not supported.
            - Added support for the new added boolean parameter in the `in_group annotation`
              (collapsed).
            - Added support for legacy "`in_group$1.7`" annotation.
    - The parameters of the MDL array constructor have been changed from "0",
      "1", etc. to "value0", "value1", etc. to match the MDL specification. This
      also affects paths to fields in the compiled material. Such paths are e.g.
      used by `ICompiled_material::lookup_sub_expression()`,
      `IMdl_distiller_api::create_baker()`, various methods on
      `IMdl_backend` and the `Target_function_description` struct.
      Hardcoded paths need to be adapted. A frequently occurring pattern is
      `".components.<number>."` which needs to be changed to
      `".components.value<numner>."`.
    - Removed unused enumerator value `DS_SEQUENCE` from
      `mi::neuraylib::IFunction_definition::Semantics`.
    - Changed BMP import of alpha channels with OpenImageIO. While the correct
      behavior is debatable, it restores the historical behavior of FreeImage.
    - A `selector` parameter has been added to the methods
      `IImage::set_from_canvas()`, and
      `IImage_api::create_canvas_from_buffer()`, and
      `IImage_api::create_canvas_from_reader()`. The old signatures are
      deprecated. They are still available if `MI_NEURAYLIB_DEPRECATED_14_0` is defined.
    - The signatures of `create_texture()`, `create_light_profile()`,
      and `create_bsdf_measurement()` on `IMdl_factory` have been
      changed: These methods take now a `IMdl_execution_context` pointer
      as last argument instead of the optional `mi::Sint32` pointer. The old
      signatures are deprecated. They are still available if
      `MI_NEURAYLIB_DEPRECATED_14_0` is defined.
    - The method `IImage::create_mipmaps()` has been renamed to
      `IImage::create_mipmap()`. The old name is deprecated. It is still
      available if `MI_NEURAYLIB_DEPRECATED_14_0` is defined.
    - It is now considered an error to apply the "A" selector to pixel types
      without alpha channel (instead of assuming a fake alpha channel with values
      1.0f or 255).
    - The interface `IImage_plugin` has been changed to allow selector
      support and better plugin selection (See the API reference documentation for details):
        - The method `supports_selectors()` has been added.
        - The method `open_for_reading()` has an additional parameter for the
          selector string.
        - The `test()` method takes now a reader instead of a buffer and the
          file size.
    - The OpenImageIO plugin supports now also non-RGBA selectors. Right now this
      works only for OpenEXR textures, including OpenEXR multipart images (use
      the part name as selector or first selector component).
    - Full Unicode support (including code points outside of the basic
      multilingual planes) for MDL source code and filenames.
    - Enabled distiller support for the open source build and in various examples.
    - The minimum CMake version has been increased to 3.21 (Windows only).
    - The OpenImageIO plugin replaces the previously used FreeImage plugin. See
      INSTALL.md for details about the installation of OpenImageIO. All examples
      have been converted to use the OpenImageIO plugin instead of the FreeImage plugin.
    - The API reference documentation is now built as part of the build process if
      doxygen is available. This can be disabled via the option
      `MDL_BUILD_DOCUMENTATION`. GraphViz/dot (optional) is used to generate nicer
      inheritance diagrams.
    - Deprecated `ILink_unit::add_environment()`. Use `ILink_unit::add_function()`
      with `ILink_unit::FEC_ENVIRONMENT` as second parameter instead. The old method is still
      available if `MI_NEURAYLIB_DEPRECATED_14_0` is defined.
    - Improved documentation for the various subclasses of `IExpression`,
      in particular `IExpression_temporary` and `Expression_parameter`.
    - The module builder supports now upgrades to MDL version 1.8, i.e., it expands
      namespace aliases to the corresponding Unicode identifiers, and finally
      removes the namespace aliases themselves.
    - Added array support to the argument editor. New overloads of
      `Argument_editor::set_value()` and `get_value()` allow now to set or get entire
      arrays in one call.
    - The OpenImageIO plugin for TIFF has been changed such that unassociated alpha
      mode is used for export. This avoids rounding errors and information loss
      for small alpha values.
    - The MDL exporter now uses TIFF instead of EXR when exporting textures with an
      alpha channel. This avoids rounding errors and information loss for small alpha values.
    - Various API methods related to resource loading have been fixed to properly
      report all error conditions. This required changing the existing error codes
      for those methods. The affected API methods are:
        - `mi::neuraylib::IImage::reset_file()`
        - `mi::neuraylib::IImage::reset_reader(IReader*,...)`
        - `mi::neuraylib::IImage::reset_reader(IArray*,...)`
        - `mi::neuraylib::ILightprofile::reset_file()`
        - `mi::neuraylib::ILightprofile::reset_reader()`
        - `mi::neuraylib::IBsdf_measurement::reset_file()`
        - `mi::neuraylib::IBsdf_measurement::reset_reader()`
        - `mi::neuraylib::IVolume_data::reset_file()`
        - `mi::neuraylib::IVolume_data::reset_reader()`
        - `mi::neuraylib::IMdl_factory::create_texture()`
        - `mi::neuraylib::IMdl_factory::create_light_profile()`
        - `mi::neuraylib::IMdl_factory::create_bsdf_measurement()`
        - `mi::neuraylib::IImport_api::import_bsdf_data()`
    - Updated jquery to version 3.6.4 and/or applied workarounds to fix a potential
      XSS vulnerability.
    - Replaced `ITarget_code::get_body_texture_count()` by
      `ITarget_code::get_texture_is_body_resource(mi::Size)` to query the information for the
      individual resources; same for light profiles and BSDF measurements.
    - Allow to disable all example dependencies if the SDK and core examples are disabled.
    - The finding python CMake script now falls back to the system installation if no path is
      specified and emits a warning in that case.
    - Changed the Clang version and existence checks so that 12.0.1 is no longer
      a hard requirement.
    - Disable LLVM CMake warning about `host=x64`.
    - `IFunction_definition::get_mangled_name()` was added. For definitions representing
       real MDL functions this returns the Itanium ABI inspired mangled name used by backends
       when MDL code is compiled to target code.
    - `IUnit::add_function()` was added, `IUnit::add_environment()` is now deprecated.
    - `IUnit::add_function()` for function definitions added by 3ds.
    - `nvidia::core_definitions`
        - Texture transformations changed from uniform to varying.
        - Added triplanar texturing and blending of normals.

- MDL Compiler and Backends
    - libbsdf: Added support for `df::dusty_diffuse_reflection_bsdf`.
    - Optimized ternary operators on `df::thin_film()` where the base BSDF is identical
      to the BSDF of the other case of the ternary.
    - Added "enable_pdf" backend option to allow disabling code generation of PDF methods
      for distribution functions.
    - MDL Core language support:
        - MDL 1.8 support is now complete and always enabled, no need for `"mdl_next"` option.
          Especially the following features were added:
            - `scene::lookup_data_float4x4()`.
            - `anno::native()`, `anno::node_output_port_default()`.
            - No lvalues inside let expressions and single expression functions are allowed.
            - Removed alias declarations.
        - `"mdl_next"` option is currently a no-op.
        - Improved some error messages.
    - Implemented normalization of `df::thin_film()` on the DAG.
    - libbsdf: Avoid PDF code duplication caused by nested layerer DFs.
    - Handle `\u`, `\U`, `\x`, and `\X` escapes more like C++
        - `\u` takes exactly 4 digits, it's an error if less are given.
        - `\U` takes exactly 8 digits, it's an error if less are given.
        - `\x` or `\X` without a digit is an error.
        - `\x` or `\X` with an overflow is a warning.
        - Illegal Unicode codepoints generated using `\u` or `\U` are now an error.
    - Wrong UTF-8 encodings in MDL files are now an error.

- MDL SDK examples
    - i18n, mdlx and the Python examples load now also the DDS plugin.
    - Enabled fast-math for all CUDA examples and avoid calculations on doubles.
    - Examples Shared
        - Added file IO utility to get the basename and file extension of a given path.
        - Added file IO utility to allow recursive folder creations.
        - Added file IO utility to normalize paths by removing `"/../"`.
    - Example Code Generation
        - Added options to adapt normal and microfacet roughness.
        - Added option to compile in single-init mode.
    - Example DF CUDA
        - Switched to another random number generator to provide numbers in the range `[0, 1)`
          as expected by the generated code.
    - Example DF Native
        - Use texture results to improve performance.
        - Use single-init mode for code generation to avoid costly reevaluations of expressions.
    - Example DXR
        - Added partial support for `EXT_lights_ies` glTF extension.
        - Added support for MDL modules, BSDF measurements, and light profiles embedded in gltf
          and glb files.
        - Use single-init mode for code generation to avoid costly reevaluations of expressions.
        - Added support for the Slang shader compiler and added options to control HLSL
          compilation.
        - Updated DXC compiler dependency to July 2022 and made it non-optional to be able to use
          a new compiler interface.
        - Fixed a minor HLSL issue when compiling without `"WITH_ENUM_SUPPORT"`.
        - Allow to use `"*"` as placeholder for the first material inside a given MDL module
          on CLI or within glTF.
        - MaterialX support improvements:
            - Allow to specify a MaterialX node to be rendered directly on CLI or within a glTF
              file.
            - Application side handling of resource paths defined in MaterialX.
            - Added CLI option `-g` to write the generated MDL module to a given path.
            - Added CLI options to match the MaterialX test renderings: constant background color,
              uv transformations, camera setup, and environment rotation.
        - Switched MaterialX SDK to 1.38.7.
        - MaterialX resource resolution is now handled by the example code to better reflect
          MDL search paths.
    - Example OptiX 7
        - Default to one sample per launch to improve experience on low-end GPUs.
        - Use real transforms instead of identity matrices.
        - Improved documentation about the required dependencies.

**Fixed Bugs**

- General
    - Mark BSDF and EDF data structures as 16-byte aligned to avoid a crash due to misaligned
      stores in generated CPU code.
    - Fixed IES parser for files with IES:LM-63-2019 standard.
    - The GLSL/HLSL backends now generate "Internal backend error" messages if some code could
      not be translated into the target language.
    - Added support for some previously unsupported constructs in GLSL/HLSL.

- MDL Compiler and Backends
    - MDL Core: Due to an inconsistence in the MDL Specification the MDL compiler
      was parsing erroneously some types of expressions after the keyword
      `'in'` in the `'let'` expressions, leading to a compiler error and
      resulting in that only unary type of expressions could be parsed correctly.
    - libbsdf:
        - Fixed the handling of total internal reflection for microfacet BSDFs evalation
          in `df::scatter_transmit mode`.
        - Correctly handle `df::thin_film` with a thickness of zero, independent of the
          IOR value.
    - Fixed rare crash in HLSL code generator regarding translation of load instructions.
    - Fixed missing module cache usage for non-inlined functions for link units.
    - Fixed special `df::thin_film()` handling for PTX backend.
    - Fixed crash for zero sized arrays in argument block layout builder.
    - Fixed inlining of functions with deferred-size arrays as parameters.
    - Fixed several bugs in the MDL compiler when compiling syntactically invalid MDL code.
    - Fixed bug in the attribute handling of the Distiller.
    - Fixed endless loop in the MDL compiler that could happen if a preset is defined with some
      literals under care conditions.
    - Fixed potential crash that could occur when an if and its else branch contain
      identical expressions.
    - Fixed garbage issued in the "called object is not a function" error message.
    - Fixed a crash when a single expression body function references an array length identifier.
    - Fixed a crash in code generation when a MDL `cast<>` operator was used as an argument
      for an select instruction.
    - Fixed non-working `cast<T>(cast<S>(x)) ==> cast<T>(x)` optimization.
    - Fixed missing support for {\tt rvalues} of constants in the JIT Backend.
    - Fixed optimizer does not replace constants by literal values in loop initializers.
    - Fixed DAG BE generating wrong signatures when struct insert operations are necessary.
    - Fixed handling of global scope operator, `::x` may now distinct from `x`.
    - Printing now `\U` escape codes with 8 digits instead of 6 when exporting MDL.
    - Fixed a race-condition in the JIT for the native backend leading to some error messages
      in cases with many CPU cores.
    - Fixed opt_level 0 code generation for the PTX backend.
    - Fixed alignment of vector, matrix, array, color and struct types in argument blocks
      for GLSL/HLSL.
    - Improved error message when default constructing a struct which has unresolved field types.
    - Fixed auto-generated name for single init function if none was provided.
      Previously duplicate symbol names were generated causing linker errors when using multiple
      materials with the PTX backend.
    - libbsdf:
        - Fixed PDF of `df::backscattering_glossy_reflection_bsdf()`.
        - Fixed wrong multiscatter texture for `df::ward_geisler_moroder_bsdf()`.
        - Fixed importance sampling of `df::ward_geisler_moroder_bsdf` if `multiscatter` is on.
        - Fixed computation of diffuse part of pdf for all BSDFs with `multiscatter_tint`.
        - Fixed behavior of `df::thin_film` film with thickness `0.0`.
        - Improved color correctness of `df::thin_film`.
    - Fixed computation of pdf in measured BSDF in native runtime.
    - Fixed handling of loops with multiple exits for GLSL/HLSL.

- MDL SDK examples
    - Fixed compilation error with Vulkan SDK 1.3.239.0 and newer.
    - Fixed pdf computation in measured BSDF for HLSL and CUDA example runtimes.
    - Examples Shared
        - Fixed out-of-bounds errors in renderer runtime.
    - Example DF Native
        - Removed memory allocations in render function to improve performance.
    - Example DXR
        - Fixed a minor HLSL issue when compiling without `"WITH_ENUM_SUPPORT"`.
        - Fixed a crash when reloading a module after changing a MDL function signature.
        - Fixed a crash when a function that is not a material was selected for rendering.
        - Fixed a crash when having arrays of structs as material parameter.
    - Example OptiX 7
        - Fixed wrong origin used for shadow ray.
        - Fixed missing normalization and transformation of normals and tangents.

MDL SDK 2022.1.7 (363600.4887): 18 Apr 2023
-----------------------------------------------

ABI compatible with the MDL SDK 2022.1.7 (363600.4887) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Fixed Bugs**

- MDL Compiler and Backends
    - Fixed rare crash in HLSL code generator regarding translation of load instructions.
    - Fixed crash for zero sized arrays in argument block layout builder.
    - Several bug fixed in the MDL compiler when compiling syntactically invalid MDL code.

MDL SDK 2022.1.6 (363600.3938): 20 Mar 2023
-----------------------------------------------

ABI compatible with the MDL SDK 2022.1.6 (363600.3938) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - Improved performance of texture import via the OpenImageIO plugin, in particular when only
      the metadata is needed.

- MDL Compiler and Backends
    - Improved output messages for errors in annotations.

**Fixed Bugs**

- General
    - Fixed usability and layout problems in API reference documentation.
    - Added missing images in the documentation for `nvidia::core_definitions`.
    - Fixed layout of API reference documentation for types using the `__align__(x)` macro.
    - Fixed `ITargetCode` serialization for cases when there are multiple materials in the
      link unit.

- MDL Compiler and Backends
    - Fixed potential crashes in the code generator when a module name contains a '.'.
    - Fixed crash in the MDL compiler caused by invalid constant declarations.
    - Fixed compiling error in LLVM for VS2017.
    - Fixed some cases where invalid MDL source code can lead to compiler crashes.

MDL SDK 2022.1.4 (363600.2768): 14 Feb 2023
-----------------------------------------------

ABI compatible with the MDL SDK 2022.1.4 (363600.2768) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - The priority of the FreeImage and OpenImageIO plugin has been decreased by one level, such
      that it is possible again to override the default plugin with a custom image plugin.

**Fixed Bugs**

- General
    - Fix loading of uv-tilesets/animated textures with filenames containing meta-characters of
      regular expressions.

- MDL Compiler and Backends
    - Fixed a crash related to parameters declared with `auto` type.
    - Fixed wrong compilation and/or possible crashes if a function with an assignment of an atomic
      type to a vector type was inlined into a material body.
    - Fixed generated code for MDL math functions that are directly mapped to HLSL/GLSL intrinsics
      when several overloads of the same intrinsic are used.
    - Fixed uninitialized variables in generated HLSL code in some cases when the layer and the
      base BSDF of a layering distribution function are the same.
    - Fixed support for `modf()` and `sincos()` for HLSL and GLSL.
    - Fixed a crash in the MDL compiler if a variant of a struct constructor is created.
    - Fixed calculation of return type for variants of auto declared functions.
    - Allow the creation of variants of the material type itself.
    - Fixed wrong optimization of the body of function variants.
    - Fixed crash when processing invalid annotations.
    - Fixed handling of struct fields with incomplete type.
    - Fixed crash when the same namespace is declared again.
    - Fixed negative results in `df::fresnel_factor` in numerical corner cases (libbsdf).

MDL SDK 2022.1.1 (363600.1657): 14 Dec 2022
-----------------------------------------------

ABI compatible with the MDL SDK 2022.1.1 (363600.1657) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - Fixed a memory leak when copying canvases (regression since MDL SDK 2022).
    - Restored compatibility of `base::perlin_noise_bump_texture()`,
      `base::worley_noise_bump_texture()`, and `base::flow_noise_bump_texture()` with previous
      versions (differences were noticeable, in particular with strong bump factors).

MDL SDK 2022.1 (3636300.1420): 05 Dec 2022
------------------------------------------

ABI compatible with the MDL SDK 2022.1 (3636300.1420) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - This release contains a preview of a new image plugin based on OpenImageIO.
      It is intended to replace the FreeImage plugin in the future. As of now,
      the new image plugin is disabled by default and not yet supported.
    - Improved Windows implementations of `mi::base::Lock` and `MI::THREAD::Lock` to use
      `SRWLOCK instead` of `CRITICAL_SECTION` primitives.
    - In order to start the examples, it is no longer necessary to explicitly set the
      `LD_LIBRARY_PATH` (on Linux) or `DYLD_LIBRARY_PATH` (on MacOS X).
    - Allowed varying values on parameters of `base::rotation_translation_scale()`.
    - Added the context option `"export_resources_with_module_prefix"` (defaults to true).
      If set to false, the module prefix is avoided if resources are exported as part of an
      exported MDL module.
      While the correct behavior is debatable, it restores the historical behavior of FreeImage.
    - Python Bindings
        - Added high-level Python binding module pymdl.py.
        - Generated .sh and .bat scripts to run Python examples without manually setting
          PATH and PYTHONPATH.

- MDL Compiler and Backends
    - Added support for Unicode identifiers to the MDL compiler (for MDL version >= 1.8).
    - Added implementations for `state::geometry_tangent_u()` and
      `state_geometry_tangent_v()` functions in JIT generated code.
      Before they were returning `0`.

- MDL SDK examples
    - Example Code Generation
        - Added command line option `--warn-spectrum-conv`. It warns if a spectrum constructor
          is converted into RGB.
    - Example DXR
        - Switched MaterialX SDK to 1.38.5.
        - Added support for textures embedded in gltf and glb files.
        - Added support for a material selector to choose one of multiple materials
          defined in an .mtlx file.
        - Enabled the encoded names option.
        - Added support for BSDF measurement resources.
        - Added support for light profile resources.
    - MDL Browser
        - Enabled the encoded names option.
    - Example Python Bindings
        - Added an example to illustrate the use of pymdl.py for inspection.

**Fixed Bugs**

- General
    - Fixed conversion of spectral to color for the case that the minimum wavelength
      is greater than 380 nanometers.
    - Fixed artifacts in `base::perlin_noise_bump_texture()`, `base::worley_noise_bump_texture()`
      and `base::flow_noise_bump_texture()` when using small cell sizes.
    - Fixed `IFunction_call::reset_argument()` for cases where a default parameter referenced
      the directly preceding parameter.
    - Fixed `IFunction_call::set_argument()` to reject expressions that contain (forbidden)
      nested parameter references. Similarly for `IFunction_definion::create_call()`.
    - Changed the compression level for export of PNG files back to 5
      for better speed/space tradeoff.
    - Fixed `IType_factory::is_compatible()}` for enums (the sets of enumeration values need
      to be equal, not just one a subset of the other).
    - MDL plugin for Arnold
        - Added checks on shutdown to prevent crashes when the plugin is not initialized properly.

- MDL Compiler and Backends
    - Fixed crash inside the MDL core compiler when an enum value is used as the right hand side
      of a select expression.
    - Fixed crash in the MDL core compiler when the qualified name of an import declaration
      is ill formed.
    - Changed the behavior of state functions when used with an invalid texture space.
      Now they return always zero.

- MDL SDK examples
    - Example DXR
        - Fixed two memory leaks and added diagnostics to check for leaks in the future.
        - Fixed a rare crash due the usage of a command list from different threads.
        - Fixed handling of MaterialX absolute paths containing a ':' character.

    - Example Python Bindings
        - Fixed the debug build of the MDL Python Bindings in VS when linking against a
          release built of the Python interpreter.

MDL SDK 2022.0.1 (359000.3383): 21 Sept 2022
-----------------------------------------------

ABI compatible with the MDL SDK 2022.0.1 (359000.3383) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - The freeimage and dds plugins can now individually be excluded from the build.
    - Improved Windows building with alternative CMake generators.
    - Python bindings
        - Updated the SWIG file in order to support pymdlsdk.
          `IFunction_definition.get_mdl_version()` in Python.
        - Handle Mdl_version as output parameter. `(since, removed) = mdl_func.get_mdl_version()`.

- MDL Compiler and Backends
    - Added compiler context option "mdl_next": Setting this option will enable preliminary
      features from the upcoming MDL 1.8 Version, especially:
        - Full utf8 identifiers in MDL.
        - Lifted restriction on scene access functions: The name of a scene data can be now any
          expression of type uniform string.

- MDL SDK examples
    - MDL Example df_vulkan
        - Added dummy implementations for unsupported runtime functions.

**Fixed Bugs**

- MDL Compiler and Backends
    - Fixed error message regarding wrong array type issued two times.
    - Speed up material compilation: Instantiation of a material instance is now faster due to
      less database queries and lesser reference counted operations.
    - Fixed module inliner handling of relative resource paths.
    - Fixed libbsdf handling of Fresnel factors.

- MDL SDK examples
    - MDL plugin for Arnold
        - Fixed edf init call (in rare cases it was accessing undefined data).
    - MDL Example df_vulkan
        - Removed alpha channel before exporting images.

MDL SDK 2022 (359000.2512): 06 Aug 2022
-----------------------------------------------

ABI compatible with the MDL SDK 2022.0 (359000.2512) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General
    - Added contribution instructions and a Contributor License Agreement (CLA)
      form to the open source release.
    - Increased the required Python version to 3.8 as this is required by LLVM.
    - Added support for building the MDL SDK and the Examples on MacOS on ARM.
    - Added methods on `IImage_api` to clone canvases and tiles,
      to convert the pixel type and for gamma correction on tiles.
    - Disabling the "materials are functions" feature is now deprecated. To that end,
      the interface `IMaterial_definition` has been deprecated and most of the
      interface `IMaterial_instance` has been deprecated. Similarly, for the
      corresponding enumerators of Element_type and
      `IMdl_evaluator_api::is_material_parameter_enabled()`. See the documentation
      linked from `IMaterial_instance` for details. The interfaces with the full set of
      methods and the enumerators are still available if `MI_NEURAYLIB_DEPRECATED_13_0`
      is defined.
    - Added `IMdl_factory::is_valid_mdl_identifier()`. Using this method avoids the
      need to hardcode the list of MDL keywords yourself. The example "Generate MDL
      identifier" makes now use of this method.
    - Both overloads of `IModule::get_function_overloads()` now also accept the simple
      name to identify the material or function definition.
    - Disabling encoded names is now deprecated. (There are no related API changes,
      disabling just emits a corresponding warning.)
    - Added Python bindings wrapper for `IMdl_evaluator_api` to support additional
      MDL annotations.

- MDL Compiler and Backends
    - In this version, all backends depend on LLVM now. The LLVM version was lifted from 8.0 to
      12.0.1.
      This change is only visible, if LLVM-IR is chosen as the output format.
    - This version uses GLSL as a new target language. This new backend supports all capabilities
      of the already
      existing native, PTX, and HLSL backends, especially:
        - It can compile functions.
        - It can compile materials (by using the libbsdf).
    - The native runtime now blends between frames for animated textures.

- MDL SDK examples
    - AxF to MDL
        - New example that demonstrates how to convert X-Rite AxF file format to MDL.
    - Example DXR
        - Added support for the NV_materials_mdl glTF vendor extension.
        - Added error logging for DRED page fault data.
    - Example Execution GLSL VK
        - New example that demonstrates how to execute material subexpressions generated
          with the GLSL backend in Vulkan.
    - Example DF Vulkan
        - New example that demonstrates how to integrate MDL into a Vulkan based renderer
          with the GLSL backend.
    - Added a Python example that shows how to create an MDL expression graph and
      how to write the created material to disk.
    - Update MaterialX dependency to version 1.38.4.

**Fixed Bugs**

- General
    - Fixed handling of MDL cast operators in the SDK.
    - Fixed typos and descriptions in support_definitions.mdl.
    - Fixed issue with Python bindings and deprecated `IMaterial_definition` interface.
    - Fixed handling of the uniform modifier in the MDL module builder that caused correct
      graph constructions to fail.

- MDL Compiler and Backends
    - Fixed missing struct constructors of re-exported structure types.
    - Fixed bug in material hash calculation.

- MDL SDK examples
    - Example DXR
        - Fixed crash when loading a new scene caused by fences being signaled too early
          due to a race condition.


MDL SDK 2021.1.4 (349500.10153): 24 May 2022
-----------------------------------------------

ABI compatible with the MDL SDK 2021.1.4 (349500.10153) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL Compiler and Backends
    - libbsdf: Implemented clarified `df::thin_film` specification - it now propagates the thin
      film coating parameters to its base and, in case the base is `df::fresnel_layer` or
      `df::fresnel_factor`, the correct coating effect on the Fresnel term is computed there.

- MDL SDK examples

    - Example DF CUDA
        - Added support for cutouts and backface bsdf/edf.

    - Example DF Native
        - Added support for derivatives.
        - Added support shadow rays inside cutouts.

**Fixed Bugs**

- General
    - Fixed mipmapping of 2D texture access in the native texture runtime
      (used with native code generation).

- MDL Compiler and Backends
    - libbsdf: Fixed numerical corner case in `df::measured_factor` causing broken auxiliary
      buffer on the native backend.
    - Removed `double` precision computations in libbsdf implementation, causing `double` type used in
      HLSL/native/PTX.

- MDL SDK examples

    - Example DF Native
        - Fixed wrong auxiliary outputs.

MDL SDK 2021.1.2 (349500.8766): 05 Apr 2022
-----------------------------------------------

ABI compatible with the MDL SDK 2021.1.2 (349500.8766) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Fixed Bugs**

- General

    - Remove wrong error message about failures to construct MDL file paths when using the module
      builder.

- MDL Compiler and Backends

    - Remove invalid optimization in DAG hashing.

MDL SDK 2021.1.1 (349500.8264): 18 Mar 2022
-----------------------------------------------

ABI compatible with the MDL SDK 2021.1.1 (349500.8264) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Known Issues**

- The version text in the MI_NEURAYLIB_PRODUCT_VERSION_STRING macro
  in the version.h include file contains the previous version number
  of "2021.1" instead of "2021.1.1".

**Added and Changed Features**

- Image File Format Plugins

    - Support .rgb extension for textures in the SGI file format.

**Fixed Bugs**

- General

    - Fixed filename extension mismatch when exporting textures referenced from MDL modules.
      Under certain circumstances, the texture was copied, but got a different filename extension,
      causing problems importing the MDL module again.
    - Fixed creation of function calls of the cast operator if the target type has frequency
      qualifiers. Similarly, fixed creation of function calls of the ternary operator if the
      argument types have frequency qualifiers.
    - Fixed handling of memory allocation failures in `IImage_api::create_canvas()/create_tile()`
      methods.
    - Also encode the simple name of function definitions. For almost all functions this does not
      make any change since the simple name is usually an identifier, except for a couple of
      operators from the builtins module.

- MDL Compiler and Backends

    - libbsdf: Fixed incorrect child normal orientation usage in `df::(color_)fresnel_layer`,
      `df::(color_)custom_curve_layer` and `df::(color_)measured_curve_layer` for backside hits.
    - HLSL backend: Fixed code generation for scene data access functions inside automatically
      derived expressions.

MDL SDK 2021.1 (349500.7063): 26 Jan 2022
-----------------------------------------

ABI compatible with the MDL SDK 2021.1 (349500.7063) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL 1.7 Language Specification

    - Removed the draft status of this document.

    - Clarified that constants can be declared locally as well, which is
      working since MDL 1.0.

    - Clarified that the `thin_film` modifier applies only to
      directly connected BSDFs that have a Fresnel term.

    - Clarified that the `state::normal()` orientation is
      facing outward from the volume, with thin-walled materials
      treated as enclosing an infinitesimally thin volume, such that
      both sides are facing outward and the normal always points to the
      observer.

- General

    - The module `::core_definitions` requires now MDL 1.6 and contains a new material
      `surface_falloff`.
    - Added support for texture selectors. The selector can be queried with methods on `IImage`,
      `ITexture`, `IVolume_data`, `IValue_texture`, and `ITarget_code`. For non-volume data
      textures, the supported selectors are restricted to `"R"`, `"G"`, `"B"`, and `"A"` for now.
    - Renamed `IVolume_data::get_channel_name()` to `IVolume_data::get_selector()` for consistency
      with `IImage` and `ITexture`. The old method is still available if
      `MI_NEURAYLIB_DEPRECATED_12_1` is defined.
    - The signature of `IMdl_factory::create_texture()` has been extended to specify the selector.
      The old signature is deprecated and still available if `MI_NEURAYLIB_DEPRECATED_12_1` is
      defined.
    - Added `IImage_api::create_canvas_from_reader()` to enable canvas creation directly from a
      reader (in addition to `create_canvas_from_buffer()`).
    - Added the methods `get_pixel_type_for_channel()` and `extract_channel()` to `IImage_api`,
      which are useful for extracting RGBA channels from existing textures.
    - Added a warning to catch some wrong implementations of
      `IMdl_resolved_module::get_module_name()`.
    - Improved resource enumeration on modules. The new method `IModule::get_resource()` returns
      an `IValue_resource` with all details, including gamma mode and selector. The old methods
      returning the individual bits are deprecated and still available if
      `MI_NEURAYLIB_DEPRECATED_12_1` is defined.
    - Added overloads of `IValue_factory::create()` that accept a range annotation as argument, and
      a type and an entire annotation block. This makes it simpler to create values that observe
      range annotations. Modified `Definition_wrapper` to make use of the latter if a parameter has
      no default.
    - Added `IFunction_call::reset_argument()` which sets an argument back to the parameter default
      (if present), or an value observing given range annotations, or a default-constructed value.
    - Extended `IValue_factory::compare()` and `IExpression_factory::compare()` to support floating
      point comparisons with an optional epsilon range.
    - The materials-are-functions feature is now enabled by default. See the documentation
      referenced from `IMdl_configuration::set_materials_are_functions()`. All examples have been
      updated to avoid usage of `IMaterial_definition` completely, and `IMaterial_instance` as much
      as possible.
    - Added an `user_data` option to the `IMdl_execution_context`, which allows the user to pass
      its own interface pointer around, in particular to the methods of `IMdl_entity_resolver`.
    - Improved performance of editing of instances of `IMdl_function_call`, in particular for
      instances with a large set of arguments. Depending on the exact circumstances, this can cut
      the time for a full edit cycle (create transaction, create argument editor, change argument,
      destroy argument editor, commit transaction) in half. An additional speedup can be obtained
      by making use of the additional optional argument of `Argument_editor`.
    - Extended `IExpression_factory::compare()` to support deep comparisons of call expressions.
      Useful to figure out whether an exporter needs to export an argument, or can rely on the
      corresponding default on the definition.
    - Export to EXR takes now the quality parameter into account: a value of 50 or less selects
     `half` as channel type.
    - Added `create_reader()` and `create_writer()` to `IMdl_impexp_api`.
    - Changed the behavior of `IImage_api::adjust_gamma()` to include the alpha channel. This also
      affects export operations (if `force_default_gamma` is set) and the MDL texture runtime if
      derivatives are enabled.
    - Added an implementation variant based on `std::atomic_uint32_t` to `Atom32`. This results
      in a large speedup on ARM, and on a rather small speedup on Windows.
    - Added support for animated textures:
        - The signature of various methods on `IImage` and `ITexture` has been changed. The frame
          index has been added as first parameter and the order of uvtile index and mipmap level
          has been flipped. The default arguments have been removed. The old signatures are still
          available if `MI_NEURAYLIB_DEPRECATED_12_1` is defined. Methods to query the mapping
          between frame index and frame number have been added.
        - The method `uvtile_marker_to_string()` on `IMdl_impexp_api` and `IExport_api` has been
          renamed to `frame_uvtile_marker_to_string()`. It is still available under the old name if
          `MI_NEURAYLIB_DEPRECATED_12_1` is defined. The method `uvtile_string_to_marker()` on both
          interfaces has been deprecated without replacement. The last component of
          `IImage::get_original_filename()` is an alternative (if available), or construct a custom
          string from scratch.
        - The interface `IMdl_resolved_resource` has been split such that it represents an array of
          instances of the new interface `IMdl_resolved_resource_element`, where each element
          corresponds to a texture frame (only one for non-animated textures and other resources).
        - The frame parameter has been added to various method of the runtime interfaces
          `Texture_handler_vtable` and `Texture_handler_deriv_vtable`. A new member `m_tex_frame`
          to support the intrinsics `tex::first_frame()` and `tex::last_frame()` has been added.
        - The examples "DF Native", "DXR", and "Execution Native" have been extended accordingly.
    - Added an overload of `IMdl_factory::clone()` that allows cloning of an execution context.
      Useful for local changes to the context options.
    - Disabled WEBP export in the FreeImage plugin due to memory access violations.
    - Changed `mi::math::gamma_correction()` to include the alpha channel, as it is done already in
      other places.

- MDL Compiler and Backends

    - Added support for libbsdf normal adaption for CUDA and native runtime. The `--an` option for
      the `df_cuda` and `df_native` examples demonstrates the feature (the dummy implementation
      does not change the normal, though).
    - The libbsdf implementations of the functions `df::diffuse_reflection_bsdf` and
      `df::diffuse_transmission_bsdf` now compensate for energy loss caused by differences of
      shading and geometric normal.
    - Added frame sequences resolution using frame markers in resources to the MDL core compiler.
    - Added an error message to the JIT backend that is issued if the user specifies
      a state module that does not contain all necessary functions.
    - Removed a redundant call to the entity resolver in the MDL core compiler when
      importing modules.

- MDL SDK examples

    - Examples Shared
        - All examples load now the `dds` plugin by default.
    - Example DF Native
        - Added support for custom texture runtime.
    - Example DXR
        - Added a new dependency to the DirectX Shader Compiler, which is optional for Windows SDK
          10.0.20348.0 and later.
        - Write a memory dump file in case an application crash occurs.
        - Added support for the glTF extension `"KHR_materials_emissive_strength"` to
          the example renderer and to the MDL support module.
    - Example Python Bindings
        - Added many more interface to the Python bindings, e.g., `IBaker`, `ICanvas`,
          `ICompiled_material`, `IImage_api`, `IMdl_Module_builder`,
          `IPlugin_configuration`, and `ITile`, and extended the examples.

**Fixed Bugs**

- General

    - Fixed a bug in mdlm and i18n if the environment variable `HOME` is not set.
    - Fixed a bug in i18n which caused command MDL search paths other than `"SYSTEM"` and `"USER"`
      to get ignored.
    - Fixed `IFactory::clone()` when dealing with untyped arrays.
    - Fixed `IMdl_backend::deserialize_target_code()` such that the internal DF texture are no
      longer missing under certain circumstances.
    - Fixed handling of weak imports for MDL < 1.6 if an external entity resolver is set. The
      semantic of weak imports is now handled by the core compiler itself, an external entity
      resolver sees now only absolute or strictly relative module names.
    - Fixed `ICompiled_material::get_hash()` to take the material slots for `surface.emission.mode`
      and `backface.emission.mode` into account. Added enumerators `SLOT_SURFACE_EMISSION_MODE` and
      `SLOT_BACKFACE_EMISSION_MODE` to `mi::neuraylib::Material_slot`. Also added `SLOT_FIRST` and
      `SLOT_LAST` to support easier enumeration over all material slots.
    - Fixed a crash with default constructed BSDF measurements during MDL export/MDLE creation.
    - Fixed gamma value of pink 1x1 default textures.
    - Fixed a race condition for accessing global core JIT backend options from
      different threads, which could have caused overwritten options or a crash,
      by using the thread local core thread context options instead.

- MDL Compiler and Backends

    - Improved numerical stability in `base::coordinate_projection()`.
    - Improved performance of `math::blackbody()` implementation.
    - Fixed incorrect normal flip for strongly bumped normal input (libbsdf).
    - libbsdf: Fixed incorrect flipping of the shading normal for strongly bumped normals. Note
      that libbsdf requires that state input shading and geometric agree on sideness (it has been
      forgiving with respect to that due to this bug).
    - libbsdf: Fixed a numerical issue for `df::fresnel_factor()` (for ior == 0).
    - libbsdf: Fixed implementation of albedo for `df::tint(reflect, transmit)`.
    - Fixed handling of resource sets if used inside MDLE archives.
    - Fixed a crash inside the MDL core compiler if a material preset
      with too many arguments is compiled.

- MDL SDK examples

    - Example OptiX 7
        - Fixed normal orientation (as libbsdf needs side consistency between geometric and shading
          normal).


MDL SDK 2021.0.4 (344800.9767): 26 Nov 2021
-------------------------------------------

ABI compatible with the MDL SDK 2021.0.4 (344800.9767) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Fixed Bugs**

- General

    - Fixed a rare case of incorrect handling of user-defined type names for
      structs and enums when encoded names were enabled.

- MDL Compiler and Backends

    - Fixed non-deterministic behavior with `sincos` calls.


MDL SDK 2021.0.3 (344800.8726): 02 Nov 2021
-------------------------------------------

ABI compatible with the MDL SDK 2021.0.3 (344800.8726) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Fixed Bugs**

- MDL Compiler and Backends

    - Apply thin film only if thickness > 0.0 (libbsdf).


MDL SDK 2021.0.2 (344800.7839): 07 Oct 2021
-------------------------------------------

ABI compatible with the MDL SDK 2021.0.2 (344800.7839) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL SDK examples

    - Example DXR
        - Added support for various glTF extensions to the example renderer and the MDL support
          module. The new supported extensions are: `KHR_texture_transform`,
          `KHR_materials_transmission`, `KHR_materials_sheen`, `KHR_materials_specular`,
          `KHR_materials_ior`, and `KHR_materials_volume`.
        - Added support for `state::animation_time()` when the animation mode is enabled in the
          renderer settings.
        - Added support for volume attenuation to the renderer.
        - Update MaterialX dependency to version 1.38.1.

**Fixed Bugs**

- MDL Compiler and Backends

    - Fixed incorrect BSDF evaluation for `df::sheen_bsdf` with a transmitting `"multiscatter"`
      BSDF parameter.
    - Fixed `df::thin_film` implementation for the case material IOR < thin film IOR
      (libbsdf).
    - Fixed an internal compiler crash.
    - Creation of compiled materials fails now (instead of crashing) if some internal error
      condition is detected.

- MDL SDK examples

    - Example DXR
        - Fixed the UV coordinate transformations to match the glTF specification. Now the
          V-coordinate is simply flipped by applying v = 1.0f - v to improve UDIM support.
        - Metallic-roughness texture lookups are fixed and will now be handled strict to the glTF
          specification.


MDL SDK 2021.0.1 (344800.4174): 17 Aug 2021
-------------------------------------------

ABI compatible with the MDL SDK 2021.0.1 (344800.4174) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General

    - Reduced memory usage for DDS textures with subformat BC7.
    - Added a new execution context option "warning" to silence compiler warnings or to promote
      them to errors. See `IMdl_execution_context` for details.
    - Disabled warnings for deprecated MDL materials and functions by default.

- MDL Compiler and Backends

    - Reduced compilation times.
    - Updated `nvidia::core_definitions` with new functionality.

- MDL SDK examples

    - Example Code Generation
        - Added an option to set number of texture result slots.

**Fixed Bugs**

- MDL Compiler and Backends

    - Fixed handling of `nvidia::baking` annotations.
    - Fixed lambda results handling in single-init mode for non-HLSL.
    - Fixed uncomputed cosine in `sheen_bsdf`'s multiscatter (was broken for a
      transmitting multiscatter component, libbsdf).
    - Improved numerical stability in `base::coordinate_projection`.
    - Fixed missing `color_weighted_layer()` inside the transmission analysis.
    - Fixed thin film factor implementation (libbsdf).


MDL SDK 2021 (344800.2052): 01 Jun 2021
---------------------------------------

ABI compatible with the MDL SDK 2021 (344800.2052) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL 1.7 Language Specification

    - OpenVDB has been added as supported 3D texture format.
    - Minimum required versions have been added for OpenEXR, Ptex, and OpenVDB to conform to the
      VFX Reference Platform CY2021.
    - Supported texture selector string values have been added for each supported texture format.
    - Texture sequences with a sequence marker have been added to the definition of texture file
      paths.
    - The `auto` placeholder type specifier has been added for variable declarations and function
      return types.
    - The `float` type is required to use a 32-bit representation following the IEEE 754 single
      precision standard. Floating point operations for `float` and `double` may deviate but shall
      not interrupt nor terminate processing.
    - The `int` type is required to be a 32-bit signed integer in two's complement form with
      wrap-around operations and without exceptions.
    - The restrictions on the texture types, light profile data type, and measured BSDF data type
      have been removed. They can now be used for local variable types, return types, field types
      in user defined structures, and element type of arrays.
    - A selector string parameter has been added to the `texture_2d` and `texture_3d` constructors
      to support texture formats with multiple data sets and channels in a file. The `anno::usage`
      standard annotation can be used on texture parameters to pre-select selector values in an
      integration.
    - The operators `=`, `==`, and `!=` have been added for the texture types, light profile data
      type, and measured BSDF data type.
    - Emission has been added to the volumetric material properties.
    - The return type of a material definition can have an annotation.
    - The description of various standard annotations, like `in_group` and `ui_order`, mention
      their wider applicability to more elements in MDL.
    - The `usage` standard annotation on materials is recommended to be used on the return type and
      not the material definition to match the recommendation for function declarations.
    - The hyperbolic trigonometric functions `cosh`, `sinh`, and `tanh` have been added to the
      standard math library.
    - The offset functions `width_offset`, `height_offset`, and `depth_offset` have been added to
      give full access to OpenVDB bounding box information.
    - The functions `first_frame` and `last_frame` have been added to give access to the texture
      animation bounds.
    - The transform function `grid_to_object_space` has been added to give access to OpenVDB
      bounding box information in MDL object space.
    - A `frame` parameter has been added to the `width`, `height`, `depth`, `width_offset`,
     `height_offset`, `depth_offset`, and `grid_to_object_space` texture functions to select frames
     in texture sequences.
    - A `frame` parameter has been added to the `texture_2d` and `texture_3d` variants of the
     `lookup_`*ltype* and `texel_`*ltype* family of texture function overloads to select
     frames in texture sequences.
    - The uniform modifier has been removed from the `tint` parameter of the EDF `tint` modifier.
    - The VDF `tint` modifier has been added.
    - An overloaded variant of the `directional_factor` modifier has been added for EDFs.
    - The `sheen_bsdf` has been changed to a modifier by adding a BSDF `multiscatter` parameter.
    - The uniform modifier has been removed from the `weight` field of the `edf_component` and
      `color_edf_component` structures and the upper limit has been removed for the weight.
    - The `color_vdf_component` structure and VDF overloads for the `color_normalized_mix` and
      `color_clamped_mix` mixing distribution functions have been added.
    - The mix distribution functions `unbounded_mix` and `color_unbounded_mix` have been added for
      BSDF, EDF, and VDF.
    - An Appendix F has been added defining MDL search path conventions.

- General

    - The MDL SDK now also supports the ARM platform on Linux (`aarch64-linux-gnu`).
    - This release changes the naming convention used for the DB elements of modules, material
      definitions, and function definitions. This is necessary to avoid problems that exist with
      the old naming scheme when names contain certain meta-characters. See
      `IMdl_configuration::set_encoded_names_enabled()` for details. This feature is enabled by
      default.
    - Added the new interface `IMdl_module_builder` which allows incremental building of new MDL
      modules, as well as editing of existing MDL modules. This interface allows the definition of
      new struct and enum types, and of new functions and materials (including variants). It also
      supports references to previous entities from the same module, explicit control of frequency
      modifiers for parameters, and parameters without defaults. An entity can also be removed from
      a module (if that entity is unreferenced). The new interface can be obtained from
      `IMdl_factory::create_module_builder()`. See also the new `create_module` example. The method
      `IMdl_factory::create_variants()` is deprecated and still available if
      `MI_NEURAYLIB_DEPRECATED_12_0` is defined.
    - The API can be configured to treat materials as if they are simply functions with the return
      type `material`. This means interfaces like `IFunction_definition` and `IFunction_call` can
      also be used for materials. See `IMdl_configuration::set_materials_are_functions()` for
      details. This feature is disabled by default. It will be enabled by default in a future
      release.
    - The new method `IMdl_factory::uniform_analysis()` allows to check the `uniform` property of
      an expression graph.
    - Improved performance for loading of MDL modules, in particular parallel loading of modules,
      and reloading of modules.
    - Added `force_default_gamma` parameter to `IMdl_impexp_api::export_canvas()` and
      `IExport_api::export_canvas()` to perform an automatic gamma adjustment based on the pixel
      type chosen for export.
    - The implementation of `IFunction_definition::get_thumbnail()` and
      `IMaterial_definition::get_thumbnail()` has been changed to compute the value lazily on
      demand.
    - The system locale used in methods of `IMdl_i18n_configuration` is restricted to two-letter
      strings to follow ISO 639-1.
    - Reduced lock scope during MDL module loading. This avoids callbacks to the entity resolver
      and related interfaces while holding this lock.
    - Added `get_mdl_parameter_type_name()`, `get_return_type()` and `get_semantic()` on
      `IMaterial_definition` for consistency with function definition. Likewise, added
      `get_return_type()` on `IMaterial_instance`.
    - Added `IMdl_impexp_api::get_mdl_module_name()` to obtain the MDL module name for a module
      identified by its file name.
    - Added `IType_factory::clone()` for type lists.
    - Made log messages from plugins available.
    - Updated recommended version numbers and use C++ 17 to meet the requirements for the VFX
      reference platform 2021.
    - A Python binding for the MDL SDK has been added. This binding consists of a shared library
      generated from the API headers using SWIG, and some additional helper functions for ease of
      use. In addition, there is a stub Python module to make that shared library accessible from
      Python code. See "Language Bindings" in the API reference for more information. Examples to
      demonstrate working with the Python binding are included as well.
    - In the API, the array constructor now uses --as all other function definitions-- named
      arguments instead of positional arguments. The previously irrelevant parameter names are now
      enforced to be "0", "1", and so on.
    - Added support for the "target model" compilation mode.
    - Added a context option `"remove_dead_parameters"` to control the removal of dead
      parameter in instances of `ICompiled_materials`. Dead parameters only survive if this
      options is set (enabled by default). Setting it to `false` produces a compatible argument
      block layout independent of the target material mode.
    - Reduced the verbosity of the `cmake` configuration output.

- MDL Compiler and Backends

    - Added support for MDL 1.7.
    - The MDL compiler warns now if a literal value would loose precision in a implicit (or
      explicit) conversion.
    - Improved half vector computation for custom-curve/measured-curve layering: assume refraction
      for non-thin-walled materials to loose less energy for non-physical glass materials
      constructed from separate BRDF and BTDF.
    - Protect custom-curve evaluations against cosines > 1 to avoid numerical corner cases.
    - Fixed textures with different gamma modes being reported as one texture in `ITarget_code`,
      when resource resolving was disabled.
    - Restricted several annotations to real functions (i.e. not materials):
      `intrinsic()`, `throws()`, `const_expr()`, `noinline()`.
    - Slightly improved generated HLSL code: the HLSL optimizer can now fold constructions like
      `vector3(a.x, a.y, a.z)` into `a`.
    - Added support for backend option `use_renderer_adapt_normal` to HLSL backend.
    - Improved speed of the DAG compiler computing material hashes.
    - Added some mathematical identities for math functions in the DAG optimizer.
    - Allowed annotation `ui_order()` on functions and materials.
    - `IMdl_backend::translate_environment()` accepts now a function that returns a
      `base::texture_return` layout-compatible type.
    - MDL 1.7 support in libbsdf:
      - Unbounded EDF mix.
      - (Color-)unbounded mix for BSDFs.
      - `df::sheen_bsdf`'s multiscatter parameter.
      - `df::directional_factor` for EDFs.
    - Avoid duplicate calls to common code for ternary BSDF operators and for distribution function
      modifiers to reduce the code size for HLSL after compilation by the DirectXShaderCompiler
      (libbsdf).

- Image File Format Plugins

    - The FreeImage plugin is now based on FreeImage trunk r1859 (fixes (at least) CVE-2019-12211
      and CVE-2019-12212).
    - Huge speedup for loading of progressive JPEGs if only the metadata is needed.
    - Added error message with details if a DDS texture can not be loaded.
    - The `dds` plugin has been enhanced to support more DDS subformats, in particular
      BC4/5/6/7-compressed textures.

- MDL SDK examples

    - Dropped support for Kepler GPUs.
    - Example Create Module
        - New example to demonstrate the new interface `IMdl_module_builder`.
    - Example DF Native
        - New example to demonstrate the integration of a MDL into a renderer by using the native
          (CPU) backend.
    - Example Traversal
        - Added MDL search path handling.

**Fixed Bugs**

- General

    - Fixed `IFunction_definition::create_function_call()` for the special case of the array
      constructor: This function uses positional arguments. Ensure that user-supplied argument
      names are ignored, and instead "0", "1", and so on, are actually used.
    - Fixed the methods `ILink_unit::add_material_path()`  and
      `ILink_unit::add_material_df()` if the path involves a template-like function.
    - Fixed checks for cycles in call graphs in
      `IMaterial_instance::create_compiled_material()`.
    - Fixed the warnings about removed support for deprecation macros in
      `include/mi/neuraylib/version.h` to support MS Visual Studio.
    - Removed bogus error message if comparison of MDLEs returns inequality.
    - Fixed handling of resources in reloaded modules.
    - Fixed module builder such that MDL file paths are used for resources, not plain OS file
      names.
    - Fixed wrong handling of encoded MDL names for user-defined type names and names with suffix
      indicating older MDL versions.
    - Improved documentation and examples to demonstrate how to set the gamma mode, in particular
      when generating MDL source code via the module builder or when creating MDLEs.

- MDL Compiler and Backends

    - All error messages of recursively imported modules are now reported in the list of error
      messages during compilation.
    - The use of an imported user defined structure type with a nested structure type in an
      exported function signature has been fixed.
    - The check for incorrect varying function call attachments to uniform parameters has been
      fixed in the MDL SDK API.
    - The function `Mdl_compiled_material::depends_on_uniform_scenedata` has been fixed.
    - The custom-curve and measured-curve layering has been improved for non-thin-walled
      transmissive materials to reduce energy loss for some cases of modelling glass with these
      components.
    - The import of the `::std` module in modules of lower MDL version has been fixed.
    - Avoid optimizations while adding to link units, making it impossible to select expression
      paths from certain distilled materials.
    - Handle correctly auto-import of types that are used only inside struct types.
    - In some rare cases array constructors of kind `T[](e_1, e_2, ...)` were handled incorrectly
      when exported into an MDLE file. Fixed now.
    - Fixed scope handling for declarations inside `then`/`else` and loop bodies.
    - Disabled tangent approximation in `::base::transform_coordinate()` to fix a performance
      regression.
    - Fixed a potential crash when an array size constant is used in a body of a function that gets
      inlined in the DAG representation.
    - Fixed missing `enable_if` conditions inside material presets.
    - Fixed auxiliary base normals (libbsdf).
    - Fixed handling of first component in unbounded mixers (libbsdf).
    - Reduced energy loss of `df::diffuse_transmission_bsdf` as lower layer of `df::fresnel_layer`,
      `df::custom_curve_layer`, `df::measured_curve_layer` (libbsdf).
    - Fixed incorrect contribution of `df::diffuse_transmission_bsdf` (for reflection directions)
      and `df::diffuse_reflection_bsdf` (for transmission directions) for evaluation with bumped
      normals (libbsdf).
    - Ensure that resources cloned from another module due to a default argument are included into
      this module's resource table.
    - Fixed wrong function names generated for HLSL code (contained sometimes `'.'`).
    - Fixed wrong return code when setting options for LLVM-based backends.
    - Use non-refracted transmission directions for non-Fresnel curve layering (libbsdf). This
      restores the behavior from 2020.1.2.
    - Fixed a crash when importing a function with texture-typed default arguments.
    - Fixed a crash when `df::tint(color,edf)` was used in some context.
    - Fixed computation of derivation info for builtin functions that do not have a declaration.
    - Fixed code generation not restoring import tables of modules.
    - Fixed calculation of the lambda call result index in PTX backend.
    - Fixed wrong `static` storage modifier for functions in HLSL.
    - Fixed generated function names for HLSL, no more `"func0"` etc.
    - Fixed rare code generation failure accessing the first member of a nested compound type
      (HLSL).
    - Fixed the pdf computation of all microfacet BSDFs in mode `df::scatter_reflect_transmit` to
      contain the selection probability (libbsdf).
    - Fixed a crash if a reserved keyword is used as a type name.
    - Fixed inlining of functions when argument expressions referenced parameters of the caller.
    - Improved error reporting on broken MDL archives and MDLEs.
    - Check upon archive creation that user-provided manifest keys form a valid identifier.
    - Fixed creation of annotations for function variants (sometimes a wrong module name was used.
    - Fixed potential crash with re-exported `enable_if()` annotations.

- MDL SDK examples

    - Fixed compilation with VS 2019 and CUDA 11.


MDL SDK 2020.1.2 (334300.5582): 12 Nov 2020
-----------------------------------------------

ABI compatible with the MDL SDK 2020.1.2 (334300.5582) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- MDL Compiler and Backends

    - A new HLSL backend option `use_renderer_adapt_microfacet_roughness` has been added, which
      allows a renderer to adapt the roughness values provided to microfacet BSDFs right before
      using them. The prototype of the function the renderer has to provide is `float2
      mdl_adapt_microfacet_roughness(Shading_state_material state, float2 roughness_uv)`.
    - A new execution context option `ignore_noinline` has been added, which allows to ignore
      `anno::noinline()` annotations, enabling inlining when creating a compiled material.
      Previously this happened later when generating code for distribution functions. But
      optimizing at this time could lead to a changed DAG which may not contain the nodes requested
      by the user anymore.

**Fixed Bugs**

- General

    - Fixed a crash in the `i18n` tool when accessing module annotations.

- MDL Compiler and Backends

    - Fixed wrong optimization for ternary operators selecting different vector elements in HLSL
      always returning the true expression.
    - Fixed wrong PTX version used for sm_86.
    - In single-init mode, don't let a requested `geometry.normal` expression calculate the normal
      again.
    - Fixed analysis of derivative variants of functions not being recognized as depending on
      `state::normal()`.
    - Reduced number of texture result slots used in generated init functions.
    - Do not generate HLSL code containing `min16int` to ensure compatibility to Slang.
    - Fixed translation of conversion of an 8-bit to a 32-bit integer for HLSL.


MDL SDK 2020.1.1 (334300.4226): 29 Sep 2020
-------------------------------------------

ABI compatible with the MDL SDK 2020.1.1 (334300.4226) binary release
(see [https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk))

**Added and Changed Features**

- General

    - Thumbnail paths are now resolved when they are requested. Before, the resolving was done
      during the module loading.
    - A new backend option `eval_dag_ternary_strictly` has been added, which enables strict
      evaluation of ternary operators (?:) on the DAG to reduce code size. By default it is enabled.
    - Improved generated code of compiled materials and lambda functions to benefit from CSE across
      arguments of the root node.

- MDL Compiler and Backends

    - Added single-init mode for a set of functions added to a link unit, allowing all these
      functions to reuse values calculated in the init function and stored in the texture results
      field of the state struct. To enable this mode, the first path in the target function
      description list given to `ILink_unit::add_material()` must be `"init"`. (Note: the init
      function will not be marked as `ITarget_code::DK_BSDF` anymore.)

- MDL SDK examples

    - Examples Shared
        - Enabled CMake option for linking MSVC dynamic runtime (/MD) instead of static (/MT) by
          default.

    - Example Code Generation
        - Updated MaterialX support to incorporate latest changes from the MaterialX github
          repository (branch 1.3.8).

    - Example DXR
        - Added `-e` option to specify which expressions to compile.

**Fixed Bugs**

- MDL Compiler and Backends

    - Fixed `IFunction_call::get_arguments()` for the array constructor, such that it always
      uses `"0"`, `"1"`, and so on as argument names.
    - Fixed failing MDLE export if the `tex::gamma_mode` type is only referenced by an
      annotation.
    - Fixed storing matrices in texture results taking up all the space without much benefit.
    - Fixed failure to add functions to link units when the path involves template-like
      functions.


MDL SDK 2020.1 (334300.2228): 11 Aug 2020
-----------------------------------------

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
    - If an absolute file URL is given for a module to be resolved AND this module exists in the module cache, the module cache is used to determine its file name. This can speed up file resolution and allows the creation of presets even if the original module is not in the module path anymore.
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
