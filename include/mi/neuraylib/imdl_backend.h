/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/
/// \file
/// \brief Interfaces related to MDL compiler backends.

#ifndef MI_NEURAYLIB_IMDL_BACKEND_H
#define MI_NEURAYLIB_IMDL_BACKEND_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/ivalue.h>
#include <mi/neuraylib/target_code_types.h>

namespace mi {

namespace neuraylib {

class IBuffer;
class ICompiled_material;
class IFunction_call;
class ILink_unit;
class IMdl_execution_context;
class ITarget_code;
class ITarget_argument_block;
class ITransaction;

struct Target_function_description;

/** \addtogroup mi_neuray_mdl_compiler
@{
*/

/// MDL backends allow to transform compiled material instances or function calls into target code.
class IMdl_backend : public
    mi::base::Interface_declare<0x9ecdd747,0x20b8,0x4a8a,0xb1,0xe2,0x62,0xb2,0x62,0x30,0xd3,0x67>
{
public:
    /// Sets a backend option.
    ///
    /// The following options are supported by all backends:
    /// - \c "compile_constants": If true, compile simple constants into functions returning
    ///                           constants. If false, do not compile simple constants but return
    ///                           error -4. Possible values: \c "on", \c "off". Default: \c "on".
    /// - \c "fast_math": Enables/disables unsafe floating point optimization. Possible values:
    ///   \c "on", \c "off". Default: \c "on".
    /// - \c "opt_level": Set the optimization level. Possible values:
    ///   * \c "0": no optimization
    ///   * \c "1": no inlining, no expensive optimizations
    ///   * \c "2": full optimizations, including inlining (default)
    /// - \c "num_texture_spaces": Set the number of supported texture spaces.
    ///   Default: \c "32".
    /// - \c "enable_auxiliary": Enable code generation for auxiliary methods on distribution
    ///                          functions. For BSDFs, these compute albedo approximations and
    ///                          normals. For EDFs, the functions exist only as placeholder for
    ///                          future use. Possible values: \c "on", \c "off". Default: \c "off".
    /// - \c "df_handle_slot_mode": When using light path expressions, individual parts of the
    ///                             distribution functions can be selected using "handles".
    ///                             The contribution of each of those parts has to be evaluated
    ///                             during rendering. This option controls how many parts are
    ///                             evaluated with each call into the generated "evaluate" and
    ///                             "auxiliary" functions and how the data is passed.
    ///                             Possible values: \c "none", \c "fixed_1", \c "fixed_2",
    ///                             \c "fixed_4", \c "fixed_8", and \c "pointer", while \c "pointer"
    ///                             is not available for all backends. Default: \c "none".
    /// The following options are supported by the NATIVE backend only:
    /// - \c "use_builtin_resource_handler": Enables/disables the built-in texture runtime.
    ///   Possible values: \c "on", \c "off". Default: \c "on".
    ///
    /// The following options are supported by the PTX, LLVM-IR, native and HLSL backend:
    ///
    /// - \c "inline_aggressively": Enables/disables aggressive inlining. Possible values:
    ///   \c "on", \c "off". Default: \c "off".
    /// - \c "eval_dag_ternary_strictly": Enables/disables strict evaluation of ternary operators
    ///   on the DAG. Possible values:
    ///   \c "on", \c "off". Default: \c "on".
    /// - \c "enable_exceptions": Enables/disables support for exceptions through runtime function
    ///   calls. For PTX, this options is always treated as disabled. Possible values:
    ///   \c "on", \c "off". Default: \c "on".
    /// - \c "enable_ro_segment": Enables/disables the creation of the read-only data segment
    ///   calls. Possible values:
    ///   \c "on", \c "off". Default: \c "off".
    /// - \c "num_texture_results": Set the size of the text_results array in the MDL SDK
    ///   state in number of float4 elements. The array has to be provided by the renderer and
    ///   must be provided per thread (for example as an array on the stack) and will be filled
    ///   in the init function created for a material and used by the sample, evaluate and pdf
    ///   functions, if the size is non-zero.
    ///   Default: \c "0".
    /// - \c "texture_runtime_with_derivs": Enables/disables derivative support for texture lookup
    ///   functions. If enabled, the user-provided texture runtime has to provide functions with
    ///   derivative parameters for the texture coordinates.
    ///   Possible values:
    ///   \c "on", \c "off". Default: \c "off".
    /// - \c "scene_data_names": Comma-separated list of names for which scene data may be
    ///   available in the renderer.
    ///   For names not in the list, \c scene::data_isvalid will always return false and
    ///   the \c scene::data_lookup_* functions will always return the provided default value.
    ///   Use \c "*" to specify that scene data for any name may be available.
    ///   Default: \c ""
    /// - \c "visible_names": Comma-separated list of names functions which will be
    ///   visible in the generated code (empty string means no special restriction).
    ///   Can especially be used in combination with \c "llvm_renderer_module" binary option to
    ///   limit the number of functions for which target code will be generated.
    ///   Default: \c ""
    ///
    /// The following options are supported by the LLVM-IR backend only:
    /// - \c "enable_simd": Enables/disables the use of SIMD instructions. Possible values:
    ///   \c "on", \c "off". Default: \c "on".
    /// - \c "write_bitcode": Enables/disables the creation of the LLVM bitcode instead of LLVM IR.
    ///   Possible values:
    ///   \c "on", \c "off". Default: \c "off".
    ///
    /// The following options are supported by the PTX backend only:
    /// - \c "sm_version": Specifies the SM target version. Possible values:
    ///   \c "20", \c "30", \c "35", \c "37", \c "50", \c "52", \c "60", \c "61", \c "62".
    ///   \c "70", \c "75", \c "80", and \c "86". Default: \c "20".
    /// - \c "enable_ro_segment": Enables/disables the creation of the read-only data segment
    ///   calls. Possible values:
    ///   \c "on", \c "off". Default: \c "off".
    /// - \c "link_libdevice": Enables/disables linking of libdevice before PTX is generated.
    ///   Possible values: \c "on", \c "off". Default: \c "on".
    /// - \c "output_format": Selects the output format of the backend.
    ///   Possible values:
    ///   \c "PTX", \c "LLVM-IR", \c "LLVM-BC". Default: \c "PTX".
    /// - \c "tex_lookup_call_mode": Selects how tex_lookup calls will be generated.
    ///   See \subpage mi_neuray_ptx_texture_lookup_call_modes for more details.
    ///   Possible values:
    ///   * \c "vtable": generate calls through a vtable call (default)
    ///   * \c "direct_call": generate direct function calls
    ///   * \c "optix_cp": generate calls through OptiX bindless callable programs
    ///
    /// The following options are supported by the HLSL backend only:
    /// - \c "hlsl_use_resource_data": If enabled, an extra user define resource data struct is
    ///   passed to all resource callbacks.
    ///   Possible values:
    ///   \c "on", \c "off". Default: \c "off".
    /// - \c "df_handle_slot_mode": The option \c "pointer" is not available (see above).
    /// - \c "use_renderer_adapt_microfacet_roughness": If enabled, the generated code expects
    ///   the renderer to provide a function with the prototype
    ///   \c "float2 mdl_adapt_microfacet_roughness(Shading_state_material state, float2 roughness_uv)"
    ///   which can adapt the roughness of microfacet BSDFs. For sheen_bsdf, the same roughness will
    ///   be provided in both dimensions and only the \c x component of the result will be used.
    ///   Possible values:
    ///   \c "on", \c "off". Default: \c "off".
    ///
    /// \param name       The name of the option.
    /// \param value      The value of the option.
    /// \return
    ///                   -  0: Success.
    ///                   - -1: Unknown option.
    ///                   - -2: Unsupported value.
    virtual Sint32 set_option( const char* name, const char* value) = 0;

    /// Sets a binary backend option.
    ///
    /// The following options are supported by the LLVM backends:
    /// - \c "llvm_state_module": Sets a user-defined implementation of the state module.
    /// - \c "llvm_renderer_module": Sets a user-defined LLVM renderer module which will be linked
    ///   and optimized together with the generated code.
    ///
    /// \param name       The name of the option.
    /// \param data       The data for the option. If \c NULL is passed, the option is cleared.
    /// \param size       The size of the data.
    /// \return
    ///                   -  0: Success.
    ///                   - -1: Unknown option.
    ///                   - -2: Unsupported value.
    virtual Sint32 set_option_binary(
        const char* name,
        const char* data,
        Size size) = 0;

    /// Returns the representation of a device library for this backend if one exists.
    ///
    /// \param[out] size  The size of the library.
    /// \return           The device library or \c NULL if no library exists for this backend.
    virtual const Uint8* get_device_library( Size &size) const = 0;

    /// Transforms an MDL environment function call into target code.
    ///
    /// The prototype of the resulting function will roughly look like this:
    ///
    /// \code
    ///     void FNAME(
    ///         void                                            *result,
    ///         mi::neuraylib::Shading_state_environment const  *state,
    ///         void const                                      *res_data,
    ///         void const                                      *exception_state);
    /// \endcode
    ///
    /// The \c result buffer must be big enough for the expression result.
    ///
    /// \param transaction                 The transaction to be used.
    /// \param call                        The MDL function call for the environment.
    /// \param fname                       The name of the generated function. If \c NULL is passed,
    ///                                    \c "lambda" will be used.
    /// \param[inout] context              A pointer to an
    ///                                    #mi::neuraylib::IMdl_execution_context which can be used
    ///                                    to pass compilation options to the MDL compiler. The
    ///                                    following options are supported by this operation:
    ///                                    - Float32 "meters_per_scene_unit": The conversion
    ///                                      ratio between meters and scene units for this
    ///                                      material. Default: 1.0f.
    ///                                    - Float32 "wavelength_min": The smallest
    ///                                      supported wavelength. Default: 380.0f.
    ///                                    - Float32 "wavelength_max": The largest supported
    ///                                      wavelength. Default: 780.0f.
    ///                                    .
    ///                                    During material translation, messages like errors and
    ///                                    warnings will be passed to the context for
    ///                                    later evaluation by the caller. Can be \c NULL.
    ///                                    Possible error conditions:
    ///                                    - Invalid parameters (\c NULL pointer).
    ///                                    - Invalid expression.
    ///                                    - The backend failed to generate target code for the
    ///                                      function.
    /// \return                            The generated target code, or \c NULL in case of failure.
    virtual const ITarget_code* translate_environment(
        ITransaction* transaction,
        const IFunction_call* call,
        const char* fname,
        IMdl_execution_context* context) = 0;

    /// Transforms an expression that is part of an MDL material instance to target code.
    ///
    /// The prototype of the resulting function will roughly look like this:
    ///
    /// \code
    ///     void FNAME(
    ///         void                                         *result,
    ///         mi::neuraylib::Shading_state_material const  *state,
    ///         void const                                   *res_data,
    ///         void const                                   *exception_state,
    ///         char const                                   *arg_block_data);
    /// \endcode
    ///
    /// The \c result buffer must be big enough for the expression result.
    ///
    /// \param transaction     The transaction to be used.
    /// \param material        The compiled MDL material.
    /// \param path            The path from the material root to the expression that should be
    ///                        translated, e.g., \c "geometry.displacement".
    /// \param fname           The name of the generated function. If \c NULL is passed, \c "lambda"
    ///                        will be used.
    /// \param[inout] context  A pointer to an
    ///                        #mi::neuraylib::IMdl_execution_context which can be used
    ///                        to pass compilation options to the MDL compiler. Currently, no
    ///                        options are supported by this operation.
    ///                        During material translation, messages like errors and
    ///                        warnings will be passed to the context for
    ///                        later evaluation by the caller. Can be \c NULL.
    ///                        Possible error conditions:
    ///                        - Invalid parameters (\c NULL pointer).
    ///                        - Invalid path (non-existing).
    ///                        - The backend failed to generate target code for the expression.
    ///                        - The requested expression is a constant.
    ///                        - Neither BSDFs, EDFs, VDFs, nor resource type expressions can
    ///                          be handled.
    ///                        - The backend does not support compiled MDL materials obtained
    ///                              from class compilation mode.
    /// \return                The generated target code, or \c NULL in case of failure.
    virtual const ITarget_code* translate_material_expression(
        ITransaction* transaction,
        const ICompiled_material* material,
        const char* path,
        const char* fname,
        IMdl_execution_context* context) = 0;

    /// Transforms an MDL distribution function to target code.
    /// Note that currently this is only supported for BSDFs.
    /// For a BSDF it results in four functions, suffixed with \c "_init", \c "_sample",
    /// \c "_evaluate" and \c "_pdf".
    ///
    /// \param transaction    The transaction to be used.
    /// \param material       The compiled MDL material.
    /// \param path           The path from the material root to the expression that
    ///                       should be translated, e.g., \c "surface.scattering".
    /// \param base_fname     The base name of the generated functions.
    ///                       If \c NULL is passed, \c "lambda" will be used.
    /// \param[inout] context A pointer to an
    ///                       #mi::neuraylib::IMdl_execution_context which can be used
    ///                       to pass compilation options to the MDL compiler. The
    ///                       following options are supported by this operation:
    ///                       - bool "include_geometry_normal". If true, the \c
    ///                       "geometry.normal" field will be applied to the MDL state prior
    ///                       to evaluation of the given DF (default: true).
    ///                       .
    ///                       During material translation, messages like errors and
    ///                       warnings will be passed to the context for
    ///                       later evaluation by the caller. Can be \c NULL.
    ///                       Possible error conditions:
    ///                       -  Invalid parameters (\c NULL pointer).
    ///                       -  Invalid path (non-existing).
    ///                       -  The backend failed to generate target code for the material.
    ///                       -  The requested expression is a constant.
    ///                       -  Only distribution functions are allowed.
    ///                       -  The backend does not support compiled MDL materials obtained
    ///                          from class compilation mode.
    ///                       -  The backend does not implement this function, yet.
    ///                       -  EDFs are not supported.
    ///                       -  VDFs are not supported.
    ///                       -  The requested BSDF is not supported, yet.
    /// \return               The generated target code, or \c NULL in case of failure.
    virtual const ITarget_code* translate_material_df(
        ITransaction* transaction,
        const ICompiled_material* material,
        const char* path,
        const char* base_fname,
        IMdl_execution_context* context) = 0;

    /// Transforms (multiple) distribution functions and expressions of a material to target code.
    /// For each distribution function this results in four functions, suffixed with \c "_init",
    /// \c "_sample", \c "_evaluate", and \c "_pdf". Functions can be selected by providing a list
    /// of \c Target_function_descriptions. Each of them needs to define the \c path, the root
    /// of the expression that should be translated. After calling this function, each element of
    /// the list will contain information for later usage in the application,
    /// e.g., the \c argument_block_index and the \c function_index.
    ///
    /// \param transaction              The transaction to be used.
    /// \param material                 The compiled MDL material.
    /// \param function_descriptions    The list of descriptions of function to translate.
    /// \param description_count        The size of the list of descriptions.
    /// \param[inout] context           A pointer to an
    ///                                 #mi::neuraylib::IMdl_execution_context which can be used
    ///                                 to pass compilation options to the MDL compiler. The
    ///                                 following options are supported for this operation:
    ///                                 - bool "include_geometry_normal". If true, the \c
    ///                                   "geometry.normal" field will be applied to the MDL
    ///                                   state prior to evaluation of the given DF (default true).
    ///                                 .
    ///                                 During material compilation messages like errors and
    ///                                 warnings will be passed to the context for
    ///                                 later evaluation by the caller. Can be \c NULL.
    /// \return              The generated target code, or \c NULL in case of failure.
    ///                      In the latter case, the return code in the failing description is
    ///                      set to -1 and the context, if provided, contains an error message.
    virtual const ITarget_code* translate_material(
        ITransaction* transaction,
        const ICompiled_material* material,
        Target_function_description* function_descriptions,
        Size description_count,
        IMdl_execution_context* context) = 0;

    /// Creates a new link unit.
    ///
    /// \param transaction     The transaction to be used.
    /// \param[inout] context  A pointer to an
    ///                        #mi::neuraylib::IMdl_execution_context which can be used
    ///                        to pass compilation options to the MDL compiler.
    ///                        The following options are supported for this operation:
    ///                         - "internal_space"
    ///                         - "fold_meters_per_scene_unit"
    ///                         - "meters_per_scene_unit"
    ///                         - "wavelength_min"
    ///                         - "wavelength_max"
    ///                        During material translation, messages like errors and
    ///                        warnings will be passed to the context for
    ///                        later evaluation by the caller.
    ///                        Can be \c NULL.
    ///                        Possible error conditions:
    ///                        - The JIT backend is not available.
    /// \return                The generated link unit, or \c NULL in case of failure.
    virtual ILink_unit* create_link_unit(
        ITransaction* transaction,
        IMdl_execution_context* context) = 0;

    /// Transforms a link unit to target code.
    ///
    /// \param lu             The link unit to translate.
    /// \param[inout] context A pointer to an
    ///                       #mi::neuraylib::IMdl_execution_context which can be used
    ///                       to pass compilation options to the MDL compiler.
    ///                       During material translation, messages like errors and
    ///                       warnings will be passed to the context for
    ///                       later evaluation by the caller.
    ///                       There are currently no options
    ///                       supported by this operation. Can be \c NULL.
    ///                       Possible error conditions:
    ///                       - Invalid link unit.
    ///                       - The JIT backend failed to compile the unit.
    /// \return               The generated link unit, or \c NULL in case of failure.
    virtual const ITarget_code* translate_link_unit(
        const ILink_unit* lu, IMdl_execution_context* context) = 0;

    /// Restores an instance of \c ITarget_code from a buffer.
    /// Deserialization can fail for outdated input date, which is not an error. Check the context
    /// messages for details.
    ///
    /// \param buffer         The buffer containing the serialized target code to restore.
    /// \param[inout] context A pointer to an #mi::neuraylib::IMdl_execution_context which can be
    ///                       used to pass serialization options. Currently there are no options
    ///                       supported for this operation.
    ///                       During the serialization messages like errors and warnings will be
    ///                       passed to the context for later evaluation by the caller.
    ///                       Can be \c NULL.
    ///                       Possible error conditions:
    ///                         - Serialization is not supported for this kind of back-end.
    ///                         - Corrupt input data, invalid header.
    ///                       Expected failure conditions that raise an info message:
    ///                         - Protocol version mismatch, deserialization invalid.
    ///                         - MDL SDK version mismatch, deserialization invalid.
    /// \return               The restored object in case of success or \c NULL otherwise.
    virtual const ITarget_code* deserialize_target_code(
        const IBuffer* buffer,
        IMdl_execution_context* context) const = 0;

    /// Restores an instance of \c ITarget_code from a buffer.
    /// Deserialization can fail for outdated input date, which is not an error. Check the context
    /// messages for details.
    ///
    /// \param buffer_data    The buffer containing the serialized target code to restore.
    /// \param buffer_size    The size of \c buffer_data.
    /// \param[inout] context A pointer to an #mi::neuraylib::IMdl_execution_context which can be
    ///                       used to pass serialization options. Currently there are no options
    ///                       supported for this operation.
    ///                       During the serialization messages like errors and warnings will be
    ///                       passed to the context for later evaluation by the caller.
    ///                       Can be \c NULL.
    ///                       Possible error conditions:
    ///                         - Serialization is not supported for this kind of back-end.
    ///                         - Corrupt input data, invalid header.
    ///                       Expected failure conditions that raise an info message:
    ///                         - Protocol version mismatch, deserialization invalid.
    ///                         - MDL SDK version mismatch, deserialization invalid.
    /// \return               The restored object in case of success or \c NULL otherwise.
    virtual const ITarget_code* deserialize_target_code(
        const Uint8* buffer_data,
        Size buffer_size,
        IMdl_execution_context* context) const = 0;
};

/// A callback interface to allow the user to handle resources when creating new
/// #mi::neuraylib::ITarget_argument_block objects for class-compiled materials when the
/// arguments contain textures not known during compilation.
class ITarget_resource_callback : public
    mi::base::Interface_declare<0xe7559a88,0x9a9a,0x41d8,0xa1,0x9c,0x4a,0x52,0x4e,0x4b,0x7b,0x66>
{
public:
    /// Returns a resource index for the given resource value usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// The index 0 is always an invalid resource reference.
    /// For #mi::neuraylib::IValue_texture values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_texture().
    /// For mi::mdl::IValue_light_profile values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_light_profile().
    /// For mi::mdl::IValue_bsdf_measurement values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_bsdf_measurement().
    ///
    /// You can use #mi::neuraylib::ITarget_code::get_known_resource_index() to handle resources
    /// which were known during compilation of the target code object.
    ///
    /// See \ref mi_neuray_ptx_texture_lookup_call_modes for more details about texture handlers
    /// for the PTX backend.
    ///
    /// \param resource  the resource value
    ///
    /// \return  a resource index or 0 if no resource index can be returned
    virtual Uint32 get_resource_index(IValue_resource const *resource) = 0;

    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    virtual Uint32 get_string_index(IValue_string const *s) = 0;
};

/// Represents an argument block of a class-compiled material compiled for a specific target.
///
/// The layout of the data is given by the corresponding #mi::neuraylib::ITarget_value_layout
/// object.
///
/// See \ref mi_neuray_compilation_modes for more details.
class ITarget_argument_block : public
    mi::base::Interface_declare<0xf2a5db20,0x85ab,0x4c41,0x8c,0x5f,0x49,0xc8,0x29,0x4a,0x73,0x65>
{
public:
    /// Returns the target argument block data.
    virtual const char* get_data() const = 0;

    /// Returns the target argument block data.
    virtual char* get_data() = 0;

    /// Returns the size of the target argument block data.
    virtual Size get_size() const = 0;

    /// Clones the argument block (to make it writeable).
    virtual ITarget_argument_block *clone() const = 0;
};

/// Structure representing the state during traversal of the nested layout.
struct Target_value_layout_state {
    Target_value_layout_state(mi::Uint32 state_offs = 0, mi::Uint32 data_offs = 0)
        : m_state_offs(state_offs)
        , m_data_offs(data_offs)
    {}

    /// The offset inside the layout state structure.
    mi::Uint32 m_state_offs;

    /// The offset which needs to be added to the element data offset.
    mi::Uint32 m_data_offs;
};

/// Represents the layout of an #mi::neuraylib::ITarget_argument_block with support for nested
/// elements.
///
/// The structure of the layout corresponds to the structure of the arguments of the
/// compiled material not of the original material.
/// Especially note, that the i'th argument of a compiled material does not in general correspond
/// to the i'th argument of the original material.
///
/// See \ref mi_neuray_compilation_modes for more details.
class ITarget_value_layout : public
    mi::base::Interface_declare<0x1003351f,0x0c31,0x4a9d,0xb9,0x99,0x90,0xb5,0xe4,0xb4,0x71,0xe3>
{
public:
    /// Returns the size of the target argument block.
    virtual Size get_size() const = 0;

    /// Get the number of arguments / elements at the given layout state.
    ///
    /// \param state  The layout state representing the current nesting within the
    ///               argument value block. The default value is used for the top-level.
    virtual Size get_num_elements(
        Target_value_layout_state state = Target_value_layout_state()) const = 0;

    /// Get the offset, the size and the kind of the argument / element inside the argument
    /// block at the given layout state.
    ///
    /// \param[out]  kind      Receives the kind of the argument.
    /// \param[out]  arg_size  Receives the size of the argument.
    /// \param       state     The layout state representing the current nesting within the
    ///                        argument value block. The default value is used for the top-level.
    ///
    /// \return  the offset of the requested argument / element or \c "~mi::Size(0)" if the state
    ///          is invalid.
    virtual Size get_layout(
        IValue::Kind &kind,
        Size &arg_size,
        Target_value_layout_state state = Target_value_layout_state()) const = 0;

    /// Get the layout state for the i'th argument / element inside the argument value block
    /// at the given layout state.
    ///
    /// \param i      The index of the argument / element.
    /// \param state  The layout state representing the current nesting within the argument
    ///               value block. The default value is used for the top-level.
    ///
    /// \return  the layout state for the nested element or a state with \c "~mi::Uint32(0)" as
    ///          m_state_offs if the element is atomic.
    virtual Target_value_layout_state get_nested_state(
        Size i,
        Target_value_layout_state state = Target_value_layout_state()) const = 0;

    /// Set the value inside the given block at the given layout state.
    ///
    /// \param[inout] block           The argument value block buffer to be modified.
    /// \param[in] value              The value to be set. It has to match the expected kind.
    /// \param[in] resource_callback  Callback for retrieving resource indices for resource values.
    /// \param[in] state              The layout state representing the current nesting within the
    ///                               argument value block. The default value is used for the
    ///                               top-level.
    ///
    /// \return
    ///                      -  0: Success.
    ///                      - -1: Invalid parameters, block or value is a \c NULL pointer.
    ///                      - -2: Invalid state provided.
    ///                      - -3: Value kind does not match expected kind.
    ///                      - -4: Size of compound value does not match expected size.
    ///                      - -5: Unsupported value type.
    virtual Sint32 set_value(
        char *block,
        IValue const *value,
        ITarget_resource_callback *resource_callback,
        Target_value_layout_state state = Target_value_layout_state()) const = 0;
};

/// Represents target code of an MDL backend.
class ITarget_code : public
    mi::base::Interface_declare<0xefca46ae,0xd530,0x4b97,0x9d,0xab,0x3a,0xdb,0x0c,0x58,0xc3,0xac>
{
public:
    /// The potential state usage properties.
    enum State_usage_property {
        SU_POSITION              = 0x0001u,        ///< uses state::position()
        SU_NORMAL                = 0x0002u,        ///< uses state::normal()
        SU_GEOMETRY_NORMAL       = 0x0004u,        ///< uses state::geometry_normal()
        SU_MOTION                = 0x0008u,        ///< uses state::motion()
        SU_TEXTURE_COORDINATE    = 0x0010u,        ///< uses state::texture_coordinate()
        SU_TEXTURE_TANGENTS      = 0x0020u,        ///< uses state::texture_tangent_*()
        SU_TANGENT_SPACE         = 0x0040u,        ///< uses state::tangent_space()
        SU_GEOMETRY_TANGENTS     = 0x0080u,        ///< uses state::geometry_tangent_*()
        SU_DIRECTION             = 0x0100u,        ///< uses state::direction()
        SU_ANIMATION_TIME        = 0x0200u,        ///< uses state::animation_time()
        SU_ROUNDED_CORNER_NORMAL = 0x0400u,        ///< uses state::rounded_corner_normal()

        SU_ALL_VARYING_MASK      = 0x07FFu,        ///< set of varying states

        SU_TRANSFORMS            = 0x0800u,        ///< uses uniform state::transform*()
        SU_OBJECT_ID             = 0x1000u,        ///< uses uniform state::object_id()

        SU_ALL_UNIFORM_MASK      = 0x1800u,        ///< set of uniform states

        SU_FORCE_32_BIT = 0xFFFFFFFFu //   Undocumented, for alignment only
    }; // can be or'ed

    typedef Uint32 State_usage;

    enum Texture_shape {
        Texture_shape_invalid      = 0, ///< Invalid texture.
        Texture_shape_2d           = 1, ///< Two-dimensional texture.
        Texture_shape_3d           = 2, ///< Three-dimensional texture.
        Texture_shape_cube         = 3, ///< Cube map texture.
        Texture_shape_ptex         = 4, ///< PTEX texture.
        Texture_shape_bsdf_data    = 5, ///< Three-dimensional texture representing a BSDF data
                                        ///  table.
        Texture_shape_FORCE_32_BIT = 0xFFFFFFFFu //   Undocumented, for alignment only
    };

    /// Language to use for the callable function prototype.
    enum Prototype_language {
        SL_CUDA,
        SL_PTX,
        SL_HLSL,
        SL_GLSL,             // \if MDL_SOURCE_RELEASE Reserved\else GLSL\endif.
        SL_NUM_LANGUAGES
    };

    /// Possible kinds of distribution functions.
    enum Distribution_kind
    {
        DK_NONE,
        DK_BSDF,
        DK_HAIR_BSDF,
        DK_EDF,
        DK_INVALID
    };

    /// Possible kinds of callable functions.
    enum Function_kind {
        FK_INVALID,
        FK_LAMBDA,
        FK_SWITCH_LAMBDA,
        FK_ENVIRONMENT,
        FK_CONST,
        FK_DF_INIT,
        FK_DF_SAMPLE,
        FK_DF_EVALUATE,
        FK_DF_PDF,
        FK_DF_AUXILIARY
    };

    /// Possible texture gamma modes.
    enum Gamma_mode {
        GM_GAMMA_DEFAULT,
        GM_GAMMA_LINEAR,
        GM_GAMMA_SRGB,
        GM_GAMMA_UNKNOWN
    };

    /// Returns the kind of backend this information belongs to.
    virtual IMdl_backend_api::Mdl_backend_kind get_backend_kind() const = 0;

    /// Returns the represented target code in ASCII representation.
    virtual const char* get_code() const = 0;

    /// Returns the length of the represented target code.
    virtual Size get_code_size() const = 0;

    /// Returns the number of callable functions in the target code.
    virtual Size get_callable_function_count() const = 0;

    /// Returns the name of a callable function in the target code.
    ///
    /// The name of a callable function is specified via the \c fname parameter of
    /// #mi::neuraylib::IMdl_backend::translate_environment() and
    /// #mi::neuraylib::IMdl_backend::translate_material_expression().
    ///
    /// \param index      The index of the callable function.
    /// \return           The name of the \p index -th callable function, or \c NULL if \p index
    ///                   is out of bounds.
    virtual const char* get_callable_function( Size index) const = 0;

    /// \name Texures
    //@{
    
    /// Returns the number of texture resources used by the target code.
    virtual Size get_texture_count() const = 0;

    /// Returns the number of texture resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid texture).
    ///
    /// \return           The body texture count or \c ~0ull, if the value is invalid due to
    ///                   more than one call to a link unit add function.
    virtual Size get_body_texture_count() const = 0;

    /// Returns the name of a texture resource used by the target code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The name of the DB element associated the texture resource of the given
    ///                   index, or \c NULL if \p index is out of range or the texture does not
    ///                   exist in the database.
    virtual const char* get_texture( Size index) const = 0;

    /// Returns the MDL URL of a texture resource used by the target code if no database
    /// element is associated to the resource.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The MDL URL of the texture resource of the given
    ///                   index, or \c NULL if \p index is out of range or the texture
    ///                   exists in the database.
    virtual const char* get_texture_url( Size index) const = 0;

    /// Returns the owner module name of a relative texture url.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The owner module name of the texture resource of the given
    ///                   index, or \c NULL if \p index is out of range or the owner
    ///                   module is not provided.
    virtual const char* get_texture_owner_module( Size index) const = 0;

    /// Returns the gamma mode of a texture resource used by the target code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The gamma of the texture resource of the given
    ///                   index, or \c GM_GAMMA_UNKNOWN if \p index is out of range.
    virtual Gamma_mode get_texture_gamma( Size index) const = 0;

    /// Returns the texture shape of a given texture resource used by the target code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The shape of the texture resource of the given
    ///                   index, or \c Texture_shape_invalid if \p index is out of range.
    virtual Texture_shape get_texture_shape( Size index) const = 0;

    /// Returns the distribution function data kind of a given texture resource used by the target
    /// code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The distribution function data kind of the texture resource of the given
    ///                   index, or \c DFK_INVALID if \p index is out of range.
    virtual Df_data_kind get_texture_df_data_kind( Size index) const = 0;

    /// Returns the distribution function data this texture refers to.
    ///
    /// \note Calling this function is only meaningful in case #get_texture_shape() returns
    /// #mi::neuraylib::ITarget_code::Texture_shape_bsdf_data.
    ///
    /// \param index      The index of the texture resource.
    /// \param [out] rx   The resolution of the texture in x.
    /// \param [out] ry   The resolution of the texture in y.
    /// \param [out] rz   The resolution of the texture in z.
    /// \return           A pointer to the texture data, if the texture is a distribution function
    ///                   data texture, \c NULL otherwise.
    virtual const Float32* get_texture_df_data(
        Size index,
        Size &rx,
        Size &ry,
        Size &rz) const = 0;

    //@}
    /// \name Light profiles
    //@{

    /// Returns the number of light profile resources used by the target code.
    virtual Size get_light_profile_count() const = 0;

    /// Returns the number of light profile resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid light profile).
    ///
    /// \return           The body light profile count or \c ~0ull, if the value is invalid due to
    ///                   more than one call to a link unit add function.
    virtual Size get_body_light_profile_count() const = 0;

    /// Returns the name of a light profile resource used by the target code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The name of the DB element associated the light profile resource of the
    ///                   given index, or \c NULL if \p index is out of range.
    virtual const char* get_light_profile(Size index) const = 0;

    /// Returns the MDL URL of a light profile resource used by the target code if no database
    /// element is associated to the resource.
    ///
    /// \param index      The index of the light profile resource.
    /// \return           The MDL URL of the light profile resource of the given
    ///                   index, or \c NULL if \p index is out of range or the light profile
    ///                   exists in the database.
    virtual const char* get_light_profile_url( Size index) const = 0;

    /// Returns the owner module name of a relative light profile URL.
    ///
    /// \param index      The index of the light profile resource.
    /// \return           The owner module name of the light profile resource of the given
    ///                   index, or \c NULL if \p index is out of range or the owner
    ///                   module is not provided.
    virtual const char* get_light_profile_owner_module( Size index) const = 0;

    //@}
    /// \name BSDF measurements
    //@{

    /// Returns the number of bsdf measurement resources used by the target code.
    virtual Size get_bsdf_measurement_count() const = 0;

    /// Returns the number of BSDF measurement resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid BSDF measurement).
    ///
    /// \return           The body BSDF measurement count or \c ~0ull, if the value is invalid due
    ///                   to more than one call to a link unit add function.
    virtual Size get_body_bsdf_measurement_count() const = 0;

    /// Returns the name of a bsdf measurement resource used by the target code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The name of the DB element associated the bsdf measurement resource of
    ///                   the given index, or \c NULL if \p index is out of range.
    virtual const char* get_bsdf_measurement(Size index) const = 0;

    /// Returns the MDL URL of a BSDF measurement resource used by the target code if no database
    /// element is associated to the resource.
    ///
    /// \param index      The index of the BSDF measurement resource.
    /// \return           The MDL URL of the BSDF measurement resource of the given
    ///                   index, or \c NULL if \p index is out of range or the BSDF measurement
    ///                   exists in the database.
    virtual const char* get_bsdf_measurement_url( Size index) const = 0;

    /// Returns the owner module name of a relative BSDF measurement URL.
    ///
    /// \param index      The index of the BSDF measurement resource.
    /// \return           The owner module name of the BSDF measurement resource of the given
    ///                   index, or \c NULL if \p index is out of range or the owner
    ///                   module is not provided.
    virtual const char* get_bsdf_measurement_owner_module( Size index) const = 0;

    //@}

    /// Returns the number of constant data initializers.
    virtual Size get_ro_data_segment_count() const = 0;

    /// Returns the name of the constant data segment at the given index.
    ///
    /// \param index   The index of the data segment.
    /// \return        The name of the constant data segment or \c NULL if the index is out of
    ///                bounds.
    virtual const char* get_ro_data_segment_name( Size index) const = 0;

    /// Returns the size of the constant data segment at the given index.
    ///
    /// \param index   The index of the data segment.
    /// \return        The size of the constant data segment or 0 if the index is out of bounds.
    virtual Size get_ro_data_segment_size( Size index) const = 0;

    /// Returns the data of the constant data segment at the given index.
    ///
    /// \param index   The index of the data segment.
    /// \return        The data of the constant data segment or \c NULL if the index is out of
    ///                bounds.
    virtual const char* get_ro_data_segment_data( Size index) const = 0;

    /// Returns the number of code segments of the target code.
    virtual Size get_code_segment_count() const = 0;

    /// Returns the represented target code segment in ASCII representation.
    ///
    /// \param index   The index of the code segment.
    /// \return        The code segment or \c NULL if the index is out of bounds.
    virtual const char* get_code_segment( Size index) const = 0;

    /// Returns the length of the represented target code segment.
    ///
    /// \param index   The index of the code segment.
    /// \return        The size of the code segment or \c 0 if the index is out of bounds.
    virtual Size get_code_segment_size( Size index) const = 0;

    /// Returns the description of the target code segment.
    ///
    /// \param index   The index of the code segment.
    /// \return        The code segment description or \c NULL if the index is out of bounds.
    virtual const char* get_code_segment_description( Size index) const = 0;

    /// Returns the potential render state usage of the target code.
    ///
    /// If the corresponding property bit is not set, it is guaranteed that the
    /// code does not use the associated render state property.
    virtual State_usage get_render_state_usage() const = 0;

    /// Returns the number of target argument blocks.
    virtual Size get_argument_block_count() const = 0;

    /// Get a target argument block if available.
    ///
    /// \param index   The index of the target argument block.
    ///
    /// \return  the captured argument block or \c NULL if no arguments were captured or the
    ///          index was invalid.
    virtual const ITarget_argument_block *get_argument_block(Size index) const = 0;

    /// Create a new target argument block of the class-compiled material for this target code.
    ///
    /// \param index              The index of the base target argument block of this target code.
    /// \param material           The class-compiled MDL material which has to fit to this
    ///                           \c ITarget_code, i.e. the hash of the compiled material must be
    ///                           identical to the one used to generate this \c ITarget_code.
    /// \param resource_callback  Callback for retrieving resource indices for resource values.
    ///
    /// \return  the generated target argument block or \c NULL if no arguments were captured
    ///          or the index was invalid.
    virtual ITarget_argument_block *create_argument_block(
        Size index,
        const ICompiled_material *material,
        ITarget_resource_callback *resource_callback) const = 0;

    /// Returns the number of target argument block layouts.
    virtual Size get_argument_layout_count() const = 0;

    /// Get a captured arguments block layout if available.
    ///
    /// \param index   The index of the target argument block.
    ///
    /// \return  the layout or \c NULL if no arguments were captured or the index was invalid.
    virtual const ITarget_value_layout *get_argument_block_layout(Size index) const = 0;

    /// Returns the number of string constants used by the target code.
    virtual Size get_string_constant_count() const = 0;

    /// Returns the string constant used by the target code.
    ///
    /// \param index    The index of the string constant.
    /// \return         The string constant that is represented by the given index, or \c NULL
    ///                 if \p index is out of range.
    virtual const char* get_string_constant(Size index) const = 0;

    /// Returns the resource index for use in an \c ITarget_argument_block of resources already
    /// known when this \c ITarget_code object was generated.
    ///
    /// \param transaction  Transaction to retrieve resource names from tags.
    /// \param resource     The resource value.
    virtual Uint32 get_known_resource_index(
        ITransaction* transaction,
        IValue_resource const *resource) const = 0;

    /// Returns the prototype of a callable function in the target code.
    ///
    /// \param index   The index of the callable function.
    /// \param lang    The language to use for the prototype.
    ///
    /// \return The prototype or NULL if \p index is out of bounds or \p lang cannot be used
    ///         for this target code.
    virtual const char* get_callable_function_prototype( Size index, Prototype_language lang)
        const = 0;

    /// Returns the distribution kind of a callable function in the target code.
    ///
    /// \param index   The index of the callable function.
    ///
    /// \return The distribution kind of the callable function
    ///         or \c DK_INVALID if \p index was invalid.
    virtual Distribution_kind get_callable_function_distribution_kind( Size index) const = 0;

    /// Returns the function kind of a callable function in the target code.
    ///
    /// \param index   The index of the callable function.
    ///
    /// \return The kind of the callable function or \c FK_INVALID if \p index was invalid.
    virtual Function_kind get_callable_function_kind( Size index) const = 0;

    /// Get the index of the target argument block to use with a callable function.
    /// \note All DF_* functions of one material DF use the same target argument block.
    ///
    /// \param index   The index of the callable function.
    ///
    /// \return The index of the target argument block for this function or ~0 if not used.
    virtual Size get_callable_function_argument_block_index( Size index) const = 0;

    /// Run this code on the native CPU.
    ///
    /// \param[in]  index       The index of the callable function.
    /// \param[in]  state       The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[out] result      The result will be written to.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given index does not
    ///          refer to an environment function.
    ///
    /// \note This allows to execute any compiled function on the CPU.
    virtual Sint32 execute_environment(
        Size index,
        const Shading_state_environment& state,
        Texture_handler_base* tex_handler,
        Spectrum_struct* result) const = 0;

    /// Run this code on the native CPU with the given captured arguments block.
    ///
    /// \param[in]  index       The index of the callable function.
    /// \param[in]  state       The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]  cap_args    The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object will be used, if any.
    /// \param[out] result      The result will be written to.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given index does not refer to
    ///          a material expression
    ///
    /// \note This allows to execute any compiled function on the CPU. The result must be
    ///       big enough to take the functions result.
    virtual Sint32 execute(
        Size index,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args,
        void* result) const = 0;

    /// Run the BSDF init function for this code on the native CPU.
    ///
    /// This function updates the normal field of the shading state with the result of
    /// \c "geometry.normal" and, if the \c "num_texture_results" backend option has been set to
    /// non-zero, fills the text_results fields of the state.
    ///
    /// \param[in]  index       The index of the callable function.
    /// \param[in]  state       The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]  cap_args    The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a BSDF init function
    virtual Sint32 execute_bsdf_init(
        Size index,
        Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;

    /// Run the BSDF sample function for this code on the native CPU.
    ///
    /// \param[in]    index     The index of the callable function.
    /// \param[inout] data      The input and output fields for the BSDF sampling.
    /// \param[in]    state     The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]    cap_args  The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a BSDF sample function
    virtual Sint32 execute_bsdf_sample(
        Size index,
        Bsdf_sample_data *data,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;

    /// Run the BSDF evaluation function for this code on the native CPU.
    ///
    /// \param[in]    index     The index of the callable function.
    /// \param[inout] data      The input and output fields for the BSDF evaluation.
    /// \param[in]    state     The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]    cap_args  The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a BSDF evaluation
    ///         function
    virtual Sint32 execute_bsdf_evaluate(
        Size index,
        Bsdf_evaluate_data_base *data,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;

    /// Run the BSDF PDF calculation function for this code on the native CPU.
    ///
    /// \param[in]    index     The index of the callable function.
    /// \param[inout] data      The input and output fields for the BSDF PDF calculation.
    /// \param[in]    state     The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]    cap_args  The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a BSDF PDF calculation
    ///         function
    virtual Sint32 execute_bsdf_pdf(
        Size index,
        Bsdf_pdf_data *data,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;


    /// Run the BSDF auxiliary calculation function for this code on the native CPU.
    ///
    /// \param[in]    index     The index of the callable function.
    /// \param[inout] data      The input and output fields for the BSDF auxiliary calculation.
    /// \param[in]    state     The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]    cap_args  The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a BSDF PDF calculation
    ///         function
    virtual Sint32 execute_bsdf_auxiliary(
        Size index,
        Bsdf_auxiliary_data_base *data,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;


    /// Run the EDF init function for this code on the native CPU.
    ///
    /// This function updates the normal field of the shading state with the result of
    /// \c "geometry.normal" and, if the \c "num_texture_results" backend option has been set to
    /// non-zero, fills the text_results fields of the state.
    ///
    /// \param[in]  index       The index of the callable function.
    /// \param[in]  state       The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]  cap_args    The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a EDF init function
    virtual Sint32 execute_edf_init(
        Size index,
        Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;

    /// Run the EDF sample function for this code on the native CPU.
    ///
    /// \param[in]    index     The index of the callable function.
    /// \param[inout] data      The input and output fields for the EDF sampling.
    /// \param[in]    state     The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]    cap_args  The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a EDF sample function
    virtual Sint32 execute_edf_sample(
        Size index,
        Edf_sample_data *data,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;

    /// Run the EDF evaluation function for this code on the native CPU.
    ///
    /// \param[in]    index     The index of the callable function.
    /// \param[inout] data      The input and output fields for the EDF evaluation.
    /// \param[in]    state     The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]    cap_args  The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a EDF evaluation
    ///          function
    virtual Sint32 execute_edf_evaluate(
        Size index,
        Edf_evaluate_data_base *data,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;

    /// Run the EDF PDF calculation function for this code on the native CPU.
    ///
    /// \param[in]    index     The index of the callable function.
    /// \param[inout] data      The input and output fields for the EDF PDF calculation.
    /// \param[in]    state     The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]    cap_args  The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a EDF PDF calculation
    ///          function
    virtual Sint32 execute_edf_pdf(
        Size index,
        Edf_pdf_data *data,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;


    /// Run the EDF auxiliary calculation function for this code on the native CPU.
    ///
    /// \param[in]    index     The index of the callable function.
    /// \param[inout] data      The input and output fields for the EDF auxiliary calculation.
    /// \param[in]    state     The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[in]    cap_args  The captured arguments to use for the execution.
    ///                         If \p cap_args is \c NULL, the captured arguments of this
    ///                         \c ITarget_code object for the given callable function will be used,
    ///                         if any.
    ///
    /// \return
    ///    -  0: on success
    ///    - -1: if execution was aborted by runtime error
    ///    - -2: cannot execute: not native code or the given function is not a EDF PDF calculation
    ///          function
    virtual Sint32 execute_edf_auxiliary(
        Size index,
        Edf_auxiliary_data_base *data,
        const Shading_state_material& state,
        Texture_handler_base* tex_handler,
        const ITarget_argument_block *cap_args) const = 0;

    /// Get the number of distribution function handles referenced by a callable function.
    ///
    /// \param func_index   The index of the callable function.
    ///
    /// \return The number of distribution function handles referenced or \c 0, if the callable
    ///         function is not a distribution function.
    virtual Size get_callable_function_df_handle_count( Size func_index) const = 0;

    /// Get the name of a distribution function handle referenced by a callable function.
    ///
    /// \param func_index     The index of the callable function.
    /// \param handle_index   The index of the handle.
    ///
    /// \return The name of the distribution function handle or \c NULL, if the callable
    ///         function is not a distribution function or \p index is invalid.
    virtual const char* get_callable_function_df_handle( Size func_index, Size handle_index)
        const = 0;

    /// Indicates whether the target code can be serialized.
    /// Not all back-ends support serialization.
    virtual bool supports_serialization() const = 0;

    /// Stores the data of this object in a buffer that can written to an external cache.
    /// The object can be restored by a corresponding back-end.
    ///
    /// \param[inout] context A pointer to an #mi::neuraylib::IMdl_execution_context which can be
    ///                       used to pass serialization options. The following options are
    ///                       supported for this operation:
    ///                        - \c bool "serialize_class_instance_data": If true, the argument
    ///                          block, resources, and strings in class instance parameters are
    ///                          serialized as well. Otherwise only body information are stored,
    ///                          which is sufficient to create new argument blocks for a material
    ///                          class. Default: \c true.
    ///                       During the serialization messages like errors and warnings will be
    ///                       passed to the context for later evaluation by the caller.
    ///                       Can be \c NULL.
    ///                       Possible error conditions:
    ///                         - Serialization is not supported for this kind of back-end.
    /// \return               The buffer in case of success and \c NULL otherwise.
    virtual const IBuffer* serialize( IMdl_execution_context* context) const = 0;

    /// Returns the potential render state usage of callable function in the target code.
    ///
    /// If the corresponding property bit is not set, it is guaranteed that the
    /// code does not use the associated render state property.
    ///
    /// \return The potential render state usage of the callable function
    ///         or \c 0 if \p index was invalid.
    virtual State_usage get_callable_function_render_state_usage( Size index) const = 0;
};

/// Represents a link-unit of an MDL backend.
class ILink_unit : public
    mi::base::Interface_declare<0x1df9bbb0,0x5d96,0x475f,0x9a,0xf4,0x07,0xed,0x8c,0x2d,0xfd,0xdb>
{
public:
    /// Add an MDL environment function call as a function to this link unit.
    ///
    /// \param call                       The MDL function call for the environment.
    /// \param fname                      The name of the function that is created.
    /// \param[inout] context             A pointer to an
    ///                                   #mi::neuraylib::IMdl_execution_context which can be used
    ///                                   to pass compilation options to the MDL compiler.
    ///                                   Currently, no options are supported by this operation.
    ///                                   During material compilation messages like errors and
    ///                                   warnings will be passed to the context for
    ///                                   later evaluation by the caller. Can be \c NULL.
    ///                                   Possible error conditions:
    ///                                    - Invalid parameters (\c NULL pointer).
    ///                                    - Invalid expression.
    ///                                    - The backend failed to compile the function.
    /// \return           A return code. The return codes have the following meaning:
    ///                   -  0: Success.
    ///                   - -1: An error occurred. Please check the
    ///                         #mi::neuraylib::IMdl_execution_context for details
    ///                         if it has been provided.
    virtual Sint32 add_environment(
        const IFunction_call    *call,
        const char              *fname,
        IMdl_execution_context  *context = 0) = 0;

    /// Add an expression that is part of an MDL material instance as a function to this
    /// link unit.
    ///
    /// \param inst       The compiled MDL material instance.
    /// \param path       The path from the material root to the expression that should be
    ///                   translated, e.g., \c "geometry.displacement".
    /// \param fname      The name of the function that is created.
    /// \param[inout] context  A pointer to an
    ///                        #mi::neuraylib::IMdl_execution_context which can be used
    ///                        to pass compilation options to the MDL compiler.
    ///                        Currently, no options are supported by this operation.
    ///                        During material compilation messages like errors and
    ///                        warnings will be passed to the context for
    ///                        later evaluation by the caller. Can be \c NULL.
    ///                        Possible error conditions:
    ///                        - The JIT backend is not available.
    ///                        - Invalid field name (non-existing).
    ///                        - invalid function name.
    ///                        - The JIT backend failed to compile the function.
    ///                        - The requested expression is a constant.
    ///                        - Neither BSDFs, EDFs, VDFs, nor resource type expressions can be
    ///                         compiled.
    /// \return           A return code. The return codes have the following meaning:
    ///                   -  0: Success.
    ///                   - -1: An error occurred. Please check the
    ///                         #mi::neuraylib::IMdl_execution_context for details
    ///                         if it has been provided.
    virtual Sint32 add_material_expression(
        const ICompiled_material* inst,
        const char* path,
        const char* fname,
        IMdl_execution_context* context) = 0;

    /// Add an MDL distribution function to this link unit.
    /// Note that currently this is only supported for BSDFs.
    /// For a BSDF it results in four functions, suffixed with \c "_init", \c "_sample",
    /// \c "_evaluate" and \c "_pdf".
    ///
    /// \param material         The compiled MDL material.
    /// \param path             The path from the material root to the expression that
    ///                         should be translated, e.g., \c "surface.scattering".
    /// \param base_fname       The base name of the generated functions.
    ///                         If \c NULL is passed, \c "lambda" will be used.
    /// \param[inout] context   A pointer to an
    ///                         #mi::neuraylib::IMdl_execution_context which can be used
    ///                         to pass compilation options to the MDL compiler. The
    ///                         following options are supported for this operation:
    ///                         - bool "include_geometry_normal". If true, the \c
    ///                           "geometry.normal" field will be applied to the MDL
    ///                           state prior to evaluation of the given DF (default true).
    ///                         .
    ///                         During material compilation messages like errors and
    ///                         warnings will be passed to the context for
    ///                         later evaluation by the caller. Can be \c NULL.
    ///                         Possible error conditions:
    ///                         - Invalid parameters (\c NULL pointer).
    ///                         - Invalid path (non-existing).
    ///                         - The backend failed to generate target code for the material.
    ///                         - The requested expression is a constant.
    ///                         - Only distribution functions are allowed.
    ///                         - The backend does not support compiled MDL materials obtained
    ///                           from class compilation mode.
    ///                         - The backend does not implement this function, yet.
    ///                         - VDFs are not supported.
    ///                         - The requested DF is not supported, yet.
    /// \return           A return code. The return codes have the following meaning:
    ///                   -  0: Success.
    ///                   - -1: An error occurred. Please check the
    ///                         #mi::neuraylib::IMdl_execution_context for details
    ///                         if it has been provided.
    virtual Sint32 add_material_df(
        const ICompiled_material* material,
        const char* path,
        const char* base_fname,
        IMdl_execution_context* context) = 0;

    /// Add (multiple) MDL distribution functions and expressions of a material to this link unit.
    /// Functions can be selected by providing a list of \c Target_function_descriptions.
    /// If the first function in the list uses the path "init", one init function will be generated,
    /// precalculating values which will be used by the other requested functions.
    /// Each other entry in the list needs to define the \c path, the root of the expression that
    /// should be translated.
    /// For each distribution function it results in three or four functions, suffixed with
    /// \c "_init" (if first requested path was not \c "init"), \c "_sample", \c "_evaluate",
    /// and \c "_pdf".
    /// After calling this function, each element of the list will contain information for later
    /// usage in the application, e.g., the \c argument_block_index and the \c function_index.
    ///
    /// \param material                 The compiled MDL material.
    /// \param[inout] function_descriptions The list of descriptions of function to translate.
    /// \param description_count        The size of the list of descriptions.
    /// \param[inout] context           A pointer to an
    ///                                 #mi::neuraylib::IMdl_execution_context which can be used
    ///                                 to pass compilation options to the MDL compiler. The
    ///                                 following options are supported for this operation:
    ///                                 - bool "include_geometry_normal". If true, the \c
    ///                                   "geometry.normal" field will be applied to the MDL
    ///                                   state prior to evaluation of the given DF (default true).
    ///                                 .
    ///                                 During material compilation messages like errors and
    ///                                 warnings will be passed to the context for
    ///                                 later evaluation by the caller. Can be \c NULL.
    /// \return              A return code. The error codes have the following meaning:
    ///                      -  0: Success.
    ///                      - -1: An error occurred while processing the entries in the list.
    ///                            Please check the
    ///                            #mi::neuraylib::IMdl_execution_context for details
    ///                            if it has been provided.
    ///
    /// \note Upon unsuccessful return, function_descriptions.return_code might contain further
    ///       info.
    virtual Sint32 add_material(
        const ICompiled_material*       material,
        Target_function_description*    function_descriptions,
        Size                            description_count,
        IMdl_execution_context*         context) = 0;
};

/// Description of target function
struct Target_function_description
{
    Target_function_description(
        const char* expression_path = 0,
        const char* base_function_name = 0)
        : path(expression_path)
        , base_fname(base_function_name)
        , argument_block_index(~Size(0))
        , function_index(~Size(0))
        , distribution_kind(ITarget_code::DK_INVALID)
        , return_code(~Sint32(0)) // not processed
    {
    }

    /// The path from the material root to the expression that should be translated,
    /// e.g., \c "surface.scattering".
    const char* path;

    /// The base name of the generated functions.
    /// If \c NULL is passed, the function name will be 'lambda' followed by an increasing
    /// counter. Note, that this counter is tracked per link unit. That means, you need to provide
    /// functions names when using multiple link units in order to avoid collisions.
    const char* base_fname;

    /// The index of argument block that belongs to the compiled material the function is
    /// generated from or ~0 if none of the added function required arguments.
    /// It allows to get the layout and a writable pointer to argument data. This is an output
    /// parameter which is available after adding the function to the link unit.
    Size argument_block_index;

    /// The index of the generated function for accessing the callable function information of
    /// the link unit or ~0 if the selected function is an invalid distribution function.
    /// ~0 is not an error case, it just means, that evaluating the function will result in 0.
    /// In case the function is a distribution function, the returned index will be the
    /// index of the \c init function, while \c sample, \c evaluate, and \c pdf will be
    /// accessible by the consecutive indices, i.e., function_index + 1, function_index + 2,
    /// function_index + 3. This is an output parameter which is available after adding the
    /// function to the link unit.
    Size function_index;

    /// Return the distribution kind of this function (or NONE in case expressions). This is
    /// an output parameter which is available after adding the function to the link unit.
    ITarget_code::Distribution_kind distribution_kind;

    /// A return code. For the meaning of the error codes correspond to the codes
    /// \c add_material_expression (code * 10) and \c add_material_df (code * 100).
    ///      0:  Success.
    ///     ~0:  The function has not yet been processed
    ///     -1:  Invalid parameters (\c NULL pointer).
    ///     -2:  Invalid path (non-existing).
    ///     -7:  The backend does not implement this function, yet.
    ///
    ///  codes for expressions, i.e., distribution_kind == DK_NONE
    ///    -10:  The JIT backend is not available.
    ///    -20:  Invalid field name (non-existing).
    ///    -30:  invalid function name.
    ///    -40:  The JIT backend failed to compile the function.
    ///    -50:  The requested expression is a constant.
    ///    -60:  Neither BSDFs, EDFs, VDFs, nor resource type expressions can be compiled.
    ///
    ///  codes for distribution functions, i.e., distribution_kind == DK_BSDF, DK_EDF, ...
    ///   -100:  Invalid parameters (\c NULL pointer).
    ///   -200:  Invalid path (non-existing).
    ///   -300:  The backend failed to generate target code for the material.
    ///   -400:  The requested expression is a constant.
    ///   -500:  Only distribution functions are allowed.
    ///   -600:  The backend does not support compiled MDL materials obtained from
    ///          class compilation mode.
    ///   -700:  The backend does not implement this function, yet.
    ///   -800:  EDFs are not supported. (deprecated, will not occur anymore)
    ///   -900:  VDFs are not supported.
    ///  -1000:  The requested DF is not supported, yet.
    Sint32 return_code;
};

mi_static_assert( sizeof( ITarget_code::State_usage_property) == sizeof( mi::Uint32));

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_BACKEND_H

