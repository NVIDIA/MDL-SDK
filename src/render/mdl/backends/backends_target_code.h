/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief

#ifndef RENDER_MDL_BACKENDS_BACKENDS_TARGET_CODE_H
#define RENDER_MDL_BACKENDS_BACKENDS_TARGET_CODE_H

#include <string>
#include <vector>
#include <map>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

#include <base/data/serial/i_serial_buffer_serializer.h>
#include <base/data/serial/i_serializer.h>

#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>

namespace mi { namespace mdl {
class IGenerated_code_executable;
class IGenerated_code_lambda_function;
} }

namespace mi { namespace neuraylib {
class IBuffer;
} }

namespace MI {

namespace SERIAL { class Buffer_serializer; }
namespace DB { class Transaction; }
namespace NEURAY { class Mdl_llvm_backend; }
namespace MDLRT { class Resource_handler; }

namespace BACKENDS {

class Target_value_layout;

/// Structure containing information about a callable function.
class Callable_function_info : public MI::SERIAL::Serializable
{
public:
    /// Constructor.
    Callable_function_info(
        std::string const &name,
        mi::neuraylib::ITarget_code::Distribution_kind dist_kind,
        mi::neuraylib::ITarget_code::Function_kind kind,
        mi::Size arg_block_index,
        mi::neuraylib::ITarget_code::State_usage state_usage)
    : m_name( name)
    , m_dist_kind( dist_kind)
    , m_kind( kind)
    , m_arg_block_index( arg_block_index)
    , m_state_usage( state_usage)
    {}

    /// Default Constructor used for deserialization.
    Callable_function_info()
        : m_name()
        , m_dist_kind(mi::neuraylib::ITarget_code::DK_INVALID)
        , m_kind(mi::neuraylib::ITarget_code::FK_INVALID)
        , m_arg_block_index(static_cast<mi::Size>(-1))
        , m_state_usage(0)
    {}

    /// The name of the callable function.
    std::string m_name;

    /// The distribution kind of the callable function.
    mi::neuraylib::ITarget_code::Distribution_kind m_dist_kind;

    /// The function kind of the callable function.
    mi::neuraylib::ITarget_code::Function_kind m_kind;

    /// The prototypes for the different languages according to
    /// #mi::neuraylib::ITarget_code::Prototype_language.
    std::vector<std::string> m_prototypes;

    /// The index of the target argument block associated with this function, or ~0 if not used.
    mi::Size m_arg_block_index;

    /// The DF handle name table.
    std::vector<std::string> m_df_handle_name_table;

    /// The state usage of the callable function.
    mi::neuraylib::ITarget_code::State_usage m_state_usage;

    /// Required for serialization.
    MI::SERIAL::Class_id get_class_id() const override
    {
        return MI::SERIAL::class_id_unknown;
    }

    /// Serialize the fields of this structure.
    const MI::SERIAL::Serializable* serialize(
        MI::SERIAL::Serializer* serializer) const override;

    /// Deserialize the fields of this structure.
    MI::SERIAL::Serializable* deserialize(
        MI::SERIAL::Deserializer* deserializer) override;
};

/// Implementation of #mi::neuraylib::ITarget_code.
class Target_code : public mi::base::Interface_implement<mi::neuraylib::ITarget_code>
{
public:

    /// Default Constructor used for deserialization
    /// and as base constructor (C++11 delegated constructor).
    Target_code();

    /// Constructor from executable code.
    ///
    /// \param code             MDL generated executable code
    /// \param transaction      the current transaction
    /// \param string_ids       True if string arguments inside target argument blocks
    ///                         are mapped to identifiers
    /// \param use_derivatives  True if derivative support is enabled for the generated code
    /// \param use_builtin_resource_handler True, if the builtin texture runtime is supposed to be
    ///                         used when running x86 code.
    /// \param be_kind     Kind of back-end that created this target code object.
    Target_code(
        mi::mdl::IGenerated_code_executable* code,
        MI::DB::Transaction* transaction,
        bool string_ids,
        bool use_derivatives,
        bool use_builtin_resource_handler,
        mi::neuraylib::IMdl_backend_api::Mdl_backend_kind be_kind);


    /// Constructor for link mode.
    ///
    /// \param string_ids  True if string arguments inside target argument blocks
    ///                    are mapped to identifiers
    /// \param be_kind     Kind of back-end that created this target code object.
    Target_code(
        bool string_ids,
        mi::neuraylib::IMdl_backend_api::Mdl_backend_kind be_kind);

    /// Finalization method for link mode for executable code.
    void finalize( mi::mdl::IGenerated_code_executable* code,
        MI::DB::Transaction* transaction,
        bool use_derivatives);


    // API methods

    /// Returns the kind of backend this information belongs to.
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind get_backend_kind() const override;

    /// Returns the represented target code in ASCII representation.
    const char* get_code() const override;

    /// Returns the length of the represented target code.
    Size get_code_size() const override;

    /// Returns the number of callable functions in the target code.
    Size get_callable_function_count() const override;

    /// Returns the name of a callable function in the target code.
    ///
    /// The name of a callable function is specified via the \c fname parameter of
    /// #mi::neuraylib::IMdl_backend::translate_environment() and
    /// #mi::neuraylib::IMdl_backend::translate_material_expression().
    ///
    /// \param index      The index of the callable function.
    /// \return           The name of the \p index -th callable function, or \c NULL if \p index
    ///                   is out of bounds.
    const char* get_callable_function(Size index) const override;

    /// Returns the number of texture resources used by the target code.
    Size get_texture_count() const override;

    /// Returns the number of texture resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid texture).
    ///
    /// \return           The body texture count or \c ~0ull, if the value is invalid due to
    ///                   more than one call to a link unit add function.
    Size get_body_texture_count() const override;

    /// Returns the name of a texture resource used by the target code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The name of the DB element associated the texture resource of the given
    ///                   index, or \c NULL if \p index is out of range.
    const char* get_texture(Size index) const override;

    /// Returns the MDL URL of a texture resource used by the target code if no database
    /// element is associated to the resource.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The MDL URL of the texture resource of the given
    ///                   index, or \c NULL if \p index is out of range.
    const char* get_texture_url(Size index) const override;

    /// Returns the owner module name of a relative texture url.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The owner module name of the texture resource of the given
    ///                   index, or \c NULL if \p index is out of range or the owner
    ///                   module is not provided.
    const char* get_texture_owner_module(Size index) const override;

    /// Returns the gamma mode of a texture resource used by the target code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The gamma of the texture resource of the given
    ///                   index, or \c GM_GAMMA_UNKNOWN if \p index is out of range.
    mi::neuraylib::ITarget_code::Gamma_mode get_texture_gamma(Size index) const override;

    /// Returns the texture shape of a given texture resource used by the target code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The shape of the texture resource of the given
    ///                   index, or \c Texture_shape_invalid if \p index is out of range.
    Texture_shape get_texture_shape(Size index) const override;

    /// Returns the distribution function data this texture refers to.
    ///
    /// \param index      The index of the texture resource.
    /// \param [out] rx   The resolution of the texture in x.
    /// \param [out] ry   The resolution of the texture in y.
    /// \param [out] rz   The resolution of the texture in z.
    /// \return           A pointer to the texture data, if the texture is a  distribution function
    ///                   data texture, \c NULL otherwise.
    const mi::Float32* get_texture_df_data(
        mi::Size index,
        mi::Size &rx,
        mi::Size &ry,
        mi::Size &rz) const override;

    /// Returns the distribution function data kind of a given texture resource used by the target
    /// code.
    ///
    /// \param index      The index of the texture resource.
    /// \return           The distribution function data kind of the texture resource of the given
    ///                   index, or \c DFK_INVALID if \p index is out of range.
    mi::neuraylib::Df_data_kind get_texture_df_data_kind(Size index) const override;

    /// Returns the number of constant data initializers.
    Size get_ro_data_segment_count() const override;

    /// Returns the name of the constant data segment at the given index.
    ///
    /// \param index   The index of the data segment.
    /// \return        The name of the constant data segment or \c NULL if the index is out of
    ///                bounds.
    const char* get_ro_data_segment_name(Size index) const override;

    /// Returns the size of the constant data segment at the given index.
    ///
    /// \param index   The index of the data segment.
    /// \return        The size of the constant data segment or 0 if the index is out of bounds.
    Size get_ro_data_segment_size(Size index) const override;

    ///
    /// \param index   The index of the data segment.
    /// \return        The data of the constant data segment or \c NULL if the index is out of
    ///                bounds.
    const char* get_ro_data_segment_data(Size index) const override;

    /// Returns the number of code segments of the target code.
    Size get_code_segment_count() const override;

    /// Returns the represented target code segment in ASCII representation.
    ///
    /// \param index   The index of the code segment.
    /// \return        The code segment or \c NULL if the index is out of bounds.
    const char* get_code_segment(mi::Size index) const override;

    /// Returns the length of the represented target code segment.
    ///
    /// \param index   The index of the code segment.
    /// \return        The size of the code segment or \c 0 if the index is out of bounds.
    mi::Size get_code_segment_size(Size index) const override;

    /// Returns the description of the target code segment.
    ///
    /// \param index   The index of the code segment.
    /// \return        The code segment description or \c NULL if the index is out of bounds.
    const char* get_code_segment_description(Size index) const override;

    /// Returns the potential render state usage of the target code.
    ///
    /// If the corresponding property bit is not set, it is guaranteed that the
    /// code does not use the associated render state property.
    State_usage get_render_state_usage() const override;

    /// Returns the number of target argument blocks / block layouts.
    Size get_argument_block_count() const override;

    /// Get a target argument block if available.
    ///
    /// \param index   The index of the target argument block.
    ///
    /// \returns the captured argument block or \c NULL if no arguments were captured or the
    ///          index was invalid.
    const mi::neuraylib::ITarget_argument_block *get_argument_block(
        Size index) const override;

    /// Create a new target argument block of the class-compiled material for this target code.
    ///
    /// \param index              The index of the base target argument block of this target code.
    /// \param material           The class-compiled MDL material which has to fit to this
    ///                           \c ITarget_code, i.e. the hash of the compiled material must be
    ///                           identical to the one used to generate this \c ITarget_code.
    /// \param resource_callback  Callback for retrieving resource indices for resource values.
    ///
    /// \returns the generated target argument block or \c NULL if no arguments were captured
    ///          or the index was invalid.
    mi::neuraylib::ITarget_argument_block *create_argument_block(
        Size index,
        const mi::neuraylib::ICompiled_material *material,
        mi::neuraylib::ITarget_resource_callback *resource_callback) const override;

    /// Returns the number of target argument blocks / block layouts.
    Size get_argument_layout_count() const override;

    /// Get a captured arguments block layout if available.
    ///
    /// \param index   The index of the target argument block.
    ///
    /// \returns the layout or \c NULL if no arguments were captured or the index was invalid.
    const mi::neuraylib::ITarget_value_layout *get_argument_block_layout(
        Size index) const override;

    /// Returns the number of light profile resources used by the target code.
    Size get_light_profile_count() const override;

    /// Returns the number of light profile resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid light profile).
    ///
    /// \return           The body light profile count or \c ~0ull, if the value is invalid due to
    ///                   more than one call to a link unit add function.
    Size get_body_light_profile_count() const override;

    /// Returns the name of a light profile resource used by the target code.
    const char* get_light_profile(Size index) const override;

    /// Returns the MDL URL of a light profile resource used by the target code.
    ///
    /// \param index      The index of the light profile resource.
    /// \return           The MDL URL of the light profile resource of the given
    ///                   index, or \c NULL if \p index is out of range.
    const char* get_light_profile_url(Size index) const override;

    /// Returns the owner module name of a relative light profile URL.
    ///
    /// \param index      The index of the light profile resource.
    /// \return           The owner module name of the light profile resource of the given
    ///                   index, or \c NULL if \p index is out of range or the owner
    ///                   module is not provided.
    const char* get_light_profile_owner_module(Size index) const override;

    /// Returns the number of BSDF measurement resources used by the target code.
    Size get_bsdf_measurement_count() const override;

    /// Returns the number of BSDF measurement resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid BSDF measurement).
    ///
    /// \return           The body BSDF measurement count or \c ~0ull, if the value is invalid due
    ///                   to more than one call to a link unit add function.
    Size get_body_bsdf_measurement_count() const override;

    /// Returns the name of a BSDF measurement resource used by the target code.
    const char* get_bsdf_measurement(Size index) const override;

    /// Returns the MDL URL of a BSDF measurement resource used by the target code.
    ///
    /// \param index      The index of the BSDF measurement resource.
    /// \return           The MDL URL of the BSDF measurement resource of the given
    ///                   index, or \c NULL if \p index is out of range.
    const char* get_bsdf_measurement_url(Size index) const override;

    /// Returns the owner module name of a relative BSDF measurement URL.
    ///
    /// \param index      The index of the BSDF measurement resource.
    /// \return           The owner module name of the BSDF measurement resource of the given
    ///                   index, or \c NULL if \p index is out of range or the owner
    ///                   module is not provided.
    const char* get_bsdf_measurement_owner_module(Size index) const override;

    /// Returns the number of string constants used by the target code.
    Size get_string_constant_count() const override;

    /// Returns the string constant used by the target code.
    ///
    /// \param index    The index of the string constant.
    /// \return         The string constant that is represented by the given index, or \c NULL
    ///                 if \p index is out of range.
    const char* get_string_constant(Size index) const override;

    /// Returns the resource index for use in an \c ITarget_argument_block of resources already
    /// known when this \c ITarget_code object was generated.
    ///
    /// \param transaction  Transaction to retrieve resource names from tags.
    /// \param resource     The resource value.
    mi::Uint32 get_known_resource_index(
        mi::neuraylib::ITransaction* transaction,
        mi::neuraylib::IValue_resource const *resource) const override;

    /// Returns the prototype of a callable function in the target code.
    ///
    /// \param index   The index of the callable function.
    /// \param lang    The language to use for the prototype.
    ///
    /// \return The prototype or NULL if \p index is out of bounds or \p lang cannot be used
    ///         for this target code.
    const char* get_callable_function_prototype(
        Size index, Prototype_language lang) const override;

    /// Returns the distribution kind of a callable function in the target code.
    ///
    /// \param index   The index of the callable function.
    ///
    /// \return The distribution kind of the callable function
    ///         or \c DK_INVALID if \p index was invalid.
    Distribution_kind get_callable_function_distribution_kind(
        Size index) const override;


    /// Returns the function kind of a callable function in the target code.
    ///
    /// \param index   The index of the callable function.
    ///
    /// \return The kind of the callable function or \c FK_INVALID if \p index was invalid.
    Function_kind get_callable_function_kind(
        Size index) const override;

    /// Get the index of the target argument block to use with a callable function.
    /// \note All DF_* functions of one material DF use the same target argument block.
    ///
    /// \param index   The index of the callable function.
    ///
    /// \return The index of the target argument block for this function or ~0 if not used.
    Size get_callable_function_argument_block_index(Size index) const override;

    /// Get the number of distribution function handles referenced by a callable function.
    ///
    /// \param func_index   The index of the callable function.
    ///
    /// \return The number of distribution function handles referenced or \c 0, if the callable
    ///         function is not a distribution function.
    Size get_callable_function_df_handle_count(Size func_index) const override;

    /// Get the name of a distribution function handle referenced by a callable function.
    ///
    /// \param func_index     The index of the callable function.
    /// \param handle_index   The index of the handle.
    ///
    /// \return The name of the distribution function handle or \c NULL, if the callable
    ///         function is not a distribution function or \p index is invalid.
    const char* get_callable_function_df_handle(Size func_index, Size handle_index)
        const override;

    /// Returns the potential render state usage of callable function in the target code.
    ///
    /// If the corresponding property bit is not set, it is guaranteed that the
    /// code does not use the associated render state property.
    ///
    /// \return The potential render state usage of the callable function
    ///         or \c 0 if \p index was invalid.
    State_usage get_callable_function_render_state_usage(Size index) const override;

    /// Run this code on the native CPU.
    ///
    /// \param[in]  index       The index of the callable function.
    /// \param[in]  state       The core state.
    /// \param[in]  tex_handler Texture handler containing the vtable for the user-defined
    ///                         texture lookup functions. Can be NULL if the built-in resource
    ///                         handler is used.
    /// \param[out] result      The result will be written to.
    ///
    /// \returns
    ///    - 0  on success
    ///    - -1 if execution was aborted by runtime error
    ///    - -2 cannot execute: not native code or the given index does not
    ///         refer to an environment function.
    ///
    /// \note This allows to execute any compiled function on the CPU.
    Sint32 execute_environment(
        Size index,
        const mi::neuraylib::Shading_state_environment& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        mi::Spectrum_struct* result) const override;

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
    /// \returns
    ///    - 0  on success
    ///    - -1 if execution was aborted by runtime error
    ///    - -2 cannot execute: not native code or the given index does not refer to
    ///         a material expression
    ///
    /// \note This allows to execute any compiled function on the CPU. The result must be
    ///       big enough to take the functions result.
    mi::Sint32 execute(
        mi::Size index,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args,
        void* result) const override;

    /// Run the BSDF init function for this code on the native CPU.
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
    /// \returns
    ///    - 0  on success
    ///    - -1 if execution was aborted by runtime error
    ///    - -2 cannot execute: not native code or the given function is not a BSDF init function
    mi::Sint32 execute_bsdf_init(
        mi::Size index,
        mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    /// Run the BSDF sample function for this code on the native CPU.
    ///
    /// \param[in]    index         The index of the callable function.
    /// \param[inout] data          The input and output fields for the BSDF sampling.
    /// \param[in]    state         The core state.
    /// \param[in]    tex_handler   Texture handler containing the vtable for the user-defined
    ///                             texture lookup functions. Can be NULL if the built-in resource
    ///                             handler is used.
    /// \param[in]    cap_args      The captured arguments to use for the execution.
    ///                             If \p cap_args is \c NULL, the captured arguments of this
    ///                             \c ITarget_code object for the given callable function will be
    ///                             used, if any.
    ///
    /// \returns
    ///    - 0  on success
    ///    - -1 if execution was aborted by runtime error
    ///    - -2 cannot execute: not native code or the given function is not a BSDF sample function
    mi::Sint32 execute_bsdf_sample(
        mi::Size index,
        mi::neuraylib::Bsdf_sample_data *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    /// Run the BSDF evaluation function for this code on the native CPU.
    ///
    /// \param[in]    index         The index of the callable function.
    /// \param[inout] data          The input and output fields for the BSDF evaluation.
    /// \param[in]    state         The core state.
    /// \param[in]    tex_handler   Texture handler containing the vtable for the user-defined
    ///                             texture lookup functions. Can be NULL if the built-in resource
    ///                             handler is used.
    /// \param[in]    cap_args      The captured arguments to use for the execution.
    ///                             If \p cap_args is \c NULL, the captured arguments of this
    ///                             \c ITarget_code object for the given callable function will be
    ///                             used, if any.
    ///
    /// \returns
    ///    - 0  on success
    ///    - -1 if execution was aborted by runtime error
    ///    - -2 cannot execute: not native code or the given function is not a BSDF evaluation
    ///         function
    mi::Sint32 execute_bsdf_evaluate(
        mi::Size index,
        mi::neuraylib::Bsdf_evaluate_data_base *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    /// Run the BSDF PDF calculation function for this code on the native CPU.
    ///
    /// \param[in]    index         The index of the callable function.
    /// \param[inout] data          The input and output fields for the BSDF PDF calculation.
    /// \param[in]    state         The core state.
    /// \param[in]    tex_handler   Texture handler containing the vtable for the user-defined
    ///                             texture lookup functions. Can be NULL if the built-in resource
    ///                             handler is used.
    /// \param[in]    cap_args      The captured arguments to use for the execution.
    ///                             If \p cap_args is \c NULL, the captured arguments of this
    ///                             \c ITarget_code object for the given callable function will be
    ///                             used, if any.
    ///
    /// \returns
    ///    - 0  on success
    ///    - -1 if execution was aborted by runtime error
    ///    - -2 cannot execute: not native code or the given function is not a BSDF PDF calculation
    ///         function
    mi::Sint32 execute_bsdf_pdf(
        mi::Size index,
        mi::neuraylib::Bsdf_pdf_data *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    /// Run the BSDF auxiliary calculation function for this code on the native CPU.
    mi::Sint32 execute_bsdf_auxiliary (
        mi::Size index,
        mi::neuraylib::Bsdf_auxiliary_data_base *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;


    /// Run the EDF init function for this code on the native CPU.
    mi::Sint32 execute_edf_init(
        mi::Size index,
        mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    /// Run the EDF sample function for this code on the native CPU.
    mi::Sint32 execute_edf_sample(
        Size index,
        mi::neuraylib::Edf_sample_data *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    /// Run the EDF evaluation function for this code on the native CPU.
    mi::Sint32 execute_edf_evaluate(
        mi::Size index,
        mi::neuraylib::Edf_evaluate_data_base *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    /// Run the EDF PDF calculation function for this code on the native CPU.
    mi::Sint32 execute_edf_pdf(
        mi::Size index,
        mi::neuraylib::Edf_pdf_data *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    /// Run the EDF auxiliary calculation function for this code on the native CPU.
    mi::Sint32 execute_edf_auxiliary(
        mi::Size index,
        mi::neuraylib::Edf_auxiliary_data_base *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const override;

    // non-API methods.

    /// Adds a new callable function to this target code.
    ///
    /// \param name             the name of the function
    /// \param dist_kind        the distribution kind of the function
    /// \param kind             the kind of the function
    /// \param arg_block_index  the argument block index associated with this function or ~0
    ///                         if no argument block is used
    /// \param state_usage      the state usage of the function
    ///
    /// \return  The index of this function.
    size_t add_function(
        const std::string& name,
        Distribution_kind dist_kind,
        Function_kind kind,
        mi::Size arg_block_index,
        State_usage state_usage);

    /// Set a function prototype for a callable function.
    ///
    /// \param index  the index of the callable function
    /// \param lang   the language of the prototype being set
    /// \param proto  the function prototype
    void set_function_prototype( size_t index, Prototype_language lang, const std::string& proto);

    /// Registers a used texture index.
    ///
    /// \param index                 the texture index as used in compiled code
    /// \param name                  the name of the DB element this index refers to.
    /// \param mdl_url               the mdl url.
    /// \param gamma                 texture gamma
    /// \param shape                 the texture shape of the texture
    /// \param sema                  the semantic of the texture, typically \c DS_UNKNOWN.
    void add_texture_index(
        size_t index,
        const std::string& name,
        const std::string& mdl_url,
        float gamma,
        Texture_shape shape,
        mi::mdl::IValue_texture::Bsdf_data_kind df_data_kind);

    /// Registers a used light profile index.
    ///
    /// \param index  the texture index as used in compiled code
    /// \param name   the name of the DB element this index refers to.
    /// \param mdl_url               the mdl url.
    void add_light_profile_index(
        size_t index,
        const std::string& name,
        const std::string& mdl_url);

    /// Registers a used bsdf measurement index.
    ///
    /// \param index  the texture index as used in compiled code
    /// \param name   the name of the DB element this index refers to.
    /// \param mdl_url               the mdl url.
    void add_bsdf_measurement_index(
        size_t index,
        const std::string& name,
        const std::string& mdl_url);

    /// Registers a used string constant index.
    ///
    /// \param index  the string constant index as used in compiled code
    /// \param scons  the string constant this index refers to.
    void add_string_constant_index(size_t index, const std::string& scons);

    /// Set the body resource counts.
    void set_body_resource_counts(
        mi::Size body_texture_counts,
        mi::Size body_light_profile_counts,
        mi::Size body_bsdf_measurement_counts);

    /// Add a new read-only data segment.
    ///
    /// \param name  the name of the segment
    /// \param data  the data of the segment
    /// \param size  the size of the segment in bytes
    void add_ro_segment( const char* name, const unsigned char* data, mi::Size size);

    /// Initializes a target argument block for the class-compiled material for this target code.
    ///
    /// \param index         The index of the target argument block
    /// \param transaction   Transaction to retrieve resource names from tags
    /// \param args          The argument list of the compiled material
    /// \return              The generated target argument block
    void init_argument_block(
        mi::Size index,
        MI::DB::Transaction* transaction,
        const MDL::IValue_list* args);

    /// Returns the resource index for use in an \c ITarget_argument_block of resources already
    /// known when this \c Target_code object was generated.
    ///
    /// \param transaction  Transaction to retrieve resource names from tags.
    /// \param resource     The resource value.
    mi::Uint32 get_known_resource_index(
        MI::DB::Transaction* transaction,
        MI::MDL::IValue_resource const *resource) const;

    /// Add a target argument block layout to this target code.
    ///
    /// \param layout  The layout to add
    ///
    /// \return  The index of the added layout.
    mi::Size add_argument_block_layout(Target_value_layout *layout);

    /// Returns true if string values in the argument block are mapped to IDs.
    bool string_args_mapped_to_ids() const { return m_string_args_mapped_to_ids; }

    /// Get the string identifier for a given string inside the constant table or 0
    /// if the string is not known.
    mi::Uint32 get_string_index(char const *string) const;

    static const mi::Float32* get_df_data_texture(
        mi::mdl::IValue_texture::Bsdf_data_kind kind,
        mi::Size &rx,
        mi::Size &ry,
        mi::Size &rz);

    /// Indicates whether the target code can be serialized.
    bool supports_serialization() const final;

    /// Stores the data of this object into a buffer that can used in an external cache.
    const mi::neuraylib::IBuffer* serialize(
        mi::neuraylib::IMdl_execution_context* context) const final;

    /// Called from the back-end to restore an instance of this class.
    bool deserialize(
        mi::mdl::ICode_generator* code_gen,
        const mi::neuraylib::IBuffer* buffer,
        mi::neuraylib::IMdl_execution_context* context);

private:
    /// Destructor.
    ~Target_code();

private:
    /// If native code was generated, its interface.
    mutable mi::base::Handle<mi::mdl::IGenerated_code_lambda_function> m_native_code;

    /// The kind of backend this information belongs to.
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind m_backend_kind;

    /// The code.
    std::string m_code;

    /// The code segments if any.
    std::vector<std::string> m_code_segments;

    /// The code segments descriptions if any.
    std::vector<std::string> m_code_segment_descriptions;

    /// The list of all callable function infos.
    std::vector<Callable_function_info> m_callable_function_infos;

    /// Helper value type for resource entries.
    struct Resource_info : public MI::SERIAL::Serializable {
        /// Constructor.
        Resource_info(
            std::string const &db_name,
            std::string const &mdl_url,
            std::string const &owner)
        : m_db_name(db_name)
        , m_mdl_url(mdl_url)
        , m_owner_module(owner)
        {
        }

        /// Default Constructor used for deserialization.
        Resource_info()
            : m_db_name()
            , m_mdl_url()
            , m_owner_module()
        {
        }

        /// Get the database name of the resource.
        char const *get_db_name() const
        {
            return m_db_name.c_str();
        }

        /// Get the mdl url of the resource.
        char const *get_mdl_url() const
        {
            return m_mdl_url.c_str();
        }

        /// Get the owner module name of the resource.
        char const *get_owner() const
        {
            return m_owner_module.c_str();
        }

        /// Required for serialization.
        MI::SERIAL::Class_id get_class_id() const final
        {
            return MI::SERIAL::class_id_unknown;
        }

        /// Serialize the fields of this structure.
        const MI::SERIAL::Serializable* serialize(
            MI::SERIAL::Serializer* serializer) const override;

        /// Deserialize the fields of this structure.
        MI::SERIAL::Serializable* deserialize(
            MI::SERIAL::Deserializer* deserializer) override;

    private:
        /// The db name of the resource.
        std::string m_db_name;

        /// The mdl url of the resource.
        std::string m_mdl_url;

        /// The owner module name of the resource.
        std::string m_owner_module;
    };

    /// Helper value type for texture entries.
    struct Texture_info : public Resource_info {
        /// Constructor.
        Texture_info(
            std::string const &db_name,
            std::string const &mdl_url,
            std::string const &owner,
            float gamma,
            Texture_shape shape,
            mi::mdl::IValue_texture::Bsdf_data_kind df_data_kind)
        : Resource_info(db_name, mdl_url, owner)
        , m_gamma(gamma)
        , m_texture_shape(shape)
        , m_df_data_kind(df_data_kind)
        {
        }

        /// Default Constructor used for deserialization.
        Texture_info()
            : Resource_info()
            , m_gamma(0.0f)
            , m_texture_shape(mi::neuraylib::ITarget_code::Texture_shape_invalid)
            , m_df_data_kind(mi::mdl::IValue_texture::BDK_NONE)
        {
        }

        /// Get the texture gamma
        float get_gamma() const { return m_gamma; }

        /// Get the texture shape of the texture.
        Texture_shape get_texture_shape() const { return m_texture_shape; }

        /// Get the semantic of the texture.
        mi::mdl::IValue_texture::Bsdf_data_kind get_df_data_kind() const { return m_df_data_kind; }

        const MI::SERIAL::Serializable* serialize(
            MI::SERIAL::Serializer* serializer) const override;

        MI::SERIAL::Serializable* deserialize(
            MI::SERIAL::Deserializer* deserializer) override;

    private:
        /// Texture gamma.
        float m_gamma;

        /// The shape of the texture.
        mi::neuraylib::ITarget_code::Texture_shape m_texture_shape;

        /// The kind of the texture.
        mi::mdl::IValue_texture::Bsdf_data_kind m_df_data_kind;
    };

    // reduce redundant code be wrapping bsdf, edf, ... calls
    mi::Sint32 execute_df_init_function(
        mi::neuraylib::ITarget_code::Distribution_kind dist_kind,
        mi::Size index,
        mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const;

    // reduce redundant code be wrapping bsdf, edf, ... calls
    mi::Sint32 execute_generic_function(
        mi::neuraylib::ITarget_code::Distribution_kind dist_kind,
        mi::neuraylib::ITarget_code::Function_kind func_kind,
        mi::Size index,
        void *data,
        const mi::neuraylib::Shading_state_material& state,
        mi::neuraylib::Texture_handler_base* tex_handler,
        const mi::neuraylib::ITarget_argument_block *cap_args) const;

    /// The texture resource table.
    std::vector<Texture_info> m_texture_table;

    /// The number of textures coming from the body of expressions
    /// (not only from material arguments). -1 if invalid.
    mi::Size m_body_texture_count;

    /// The light profile resource table.
    std::vector<Resource_info> m_light_profile_table;

    /// The number of light profiles coming from the body of expressions
    /// (not only from material arguments). -1 if invalid.
    mi::Size m_body_light_profile_count;

    /// The bsdf measurement resource table.
    std::vector<Resource_info> m_bsdf_measurement_table;

    /// The number of BSDF measurements coming from the body of expressions
    /// (not only from material arguments). -1 if invalid.
    mi::Size m_body_bsdf_measurement_count;

    /// The string constant table.
    std::vector<std::string> m_string_constant_table;

    /// Helper class for handling segments.
    class Segment : public MI::SERIAL::Serializable {
    public:
        /// Constructor.
        ///
        /// \param name  the name of the segment
        /// \param data  points to the start of the segment blob, will be copied
        /// \param size  the size of the segment blob
        Segment(const char* name, const unsigned char* data, mi::Size size)
            : m_name(name)
            , m_data()
        {
            if (data && size > 0)
            {
                m_data.resize(size);
                memcpy(m_data.data(), data, size);
            }
        }

        /// Default Constructor used for deserialization.
        Segment()
            : m_name("")
            , m_data()
        {
        }

        /// Get the name.
        const char* get_name() const { return m_name.c_str(); }

        /// Get the data.
        const unsigned char* get_data() const { return m_data.data(); }

        /// Get the size.
        mi::Size get_size() const { return m_data.size(); }

        /// Required for serialization.
        MI::SERIAL::Class_id get_class_id() const final
        {
            return MI::SERIAL::class_id_unknown;
        }

        /// Serialize the fields of this structure.
        const MI::SERIAL::Serializable* serialize(
            MI::SERIAL::Serializer* serializer) const final;

        /// Deserialize the fields of this structure.
        MI::SERIAL::Serializable* deserialize(
            MI::SERIAL::Deserializer* deserializer) final;

    private:
        std::string m_name;
        std::vector<unsigned char> m_data;
    };

    /// The list of all segments.
    std::vector<Segment> m_data_segments;

    /// The layouts of the captured arguments blocks.
    std::vector<mi::base::Handle<MI::BACKENDS::Target_value_layout const> > m_cap_arg_layouts;

    /// The captured arguments blocks.
    std::vector<mi::base::Handle<mi::neuraylib::ITarget_argument_block> > m_cap_arg_blocks;

    /// The resource handler if any.
    MDLRT::Resource_handler *m_rh;

    /// The potential render state usage of this code.
    mi::Uint32 m_render_state_usage;

    /// True, if string arguments in the target block are mapped to identifiers.
    bool m_string_args_mapped_to_ids;

    /// True, if the builtin resource handler is supposed to be used when running native code
    bool m_use_builtin_resource_handler;
};

} // namespace BACKENDS

} // namespace MI

#endif // RENDER_MDL_BACKENDS_BACKENDS_TARGET_CODE_H

