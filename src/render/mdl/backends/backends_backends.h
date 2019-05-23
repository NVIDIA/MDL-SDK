/***************************************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef RENDER_MDL_BACKENDS_BACKENDS_BACKENDS_H
#define RENDER_MDL_BACKENDS_BACKENDS_BACKENDS_H

#include <string>
#include <vector>
#include <map>

#include <base/system/main/neuray_cc_conf.h>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/imdl_compiler.h>

namespace mi {
namespace mdl { class IType_struct; class IType; }
namespace neuraylib { class ITarget_code; }
}

namespace MI {

namespace DB { class Transaction; }
namespace MDL {
    class Execution_context;
    class Mdl_compiled_material;
    class Mdl_function_call;
    class IValue;
    class IValue_resource;
    class IValue_string;
}

namespace BACKENDS {

class Link_unit;
class Target_code;

/// LLVM-IR based backends.
class Mdl_llvm_backend
{
public:

    /// Constructor.
    ///
    /// \param kind            The backend kind.
    /// \param compiler        The MDL compiler.
    /// \param jit             The JIT code generator.
    /// \param code_cache      If non-NULL, the code cache.
    /// \param string_ids      If True, string arguments are mapped to string identifiers.
    Mdl_llvm_backend(
        mi::neuraylib::IMdl_compiler::Mdl_backend_kind kind,
        mi::mdl::IMDL* compiler,
        mi::mdl::ICode_generator_jit* jit,
        mi::mdl::ICode_cache *code_cache,
        bool string_ids);

    // API methods

    mi::Sint32 set_option( const char* name, const char* value);

    mi::Sint32 set_option_binary( const char* name, const char* data, mi::Size size);

    const mi::neuraylib::ITarget_code* translate_environment(
        DB::Transaction* transaction,
        const MDL::Mdl_function_call* call,
        const char* fname,
        MDL::Execution_context* context);

    const mi::neuraylib::ITarget_code* translate_material_expression(
        DB::Transaction* transaction,
        const MDL::Mdl_compiled_material* material,
        const char* path,
        const char* fname,
        MDL::Execution_context* context);

    const mi::neuraylib::ITarget_code* translate_material_expressions(
        DB::Transaction* transaction,
        const MDL::Mdl_compiled_material* material,
        const char* const paths[],
        mi::Uint32 path_cnt,
        const char* fname,
        mi::Sint32* errors);

    const mi::neuraylib::ITarget_code* translate_material_expression_uniform_state(
        DB::Transaction* transaction,
        const MDL::Mdl_compiled_material* material,
        const char* path,
        const char* fname,
        const mi::Float32_4_4_struct& world_to_obj,
        const mi::Float32_4_4_struct& obj_to_world,
        mi::Sint32 object_id,
        mi::Sint32* errors);

    const mi::neuraylib::ITarget_code* translate_material_df(
        DB::Transaction* transaction,
        const MDL::Mdl_compiled_material* material,
        const char* path,
        const char* base_fname,
        MDL::Execution_context* context);

    const mi::Uint8* get_device_library( mi::Size &size) const;

    /// Creates a target argument block of the class-compiled material for this backend.
    ///
    /// \param transaction   The transaction to be used.
    /// \param material      The class-compiled MDL material.
    /// \return              The generated target argument block.
    const mi::neuraylib::ITarget_argument_block* create_argument_block(
        DB::Transaction *transaction,
        const MDL::Mdl_compiled_material* material);

    mi::neuraylib::ITarget_code const *translate_link_unit(
        Link_unit const *lu,
        MDL::Execution_context* context);

    /// Get the MDL compiler.
    mi::base::Handle<mi::mdl::IMDL> get_compiler() const { return m_compiler; }

    /// Get the JIT backend.
    mi::base::Handle<mi::mdl::ICode_generator_jit> get_jit_be() const { return m_jit; }

    /// Get the backend kind.
    mi::neuraylib::IMdl_compiler::Mdl_backend_kind get_kind() const { return m_kind; }

    /// If true, the LLVM-IR backend uses SIMD instructions.
    bool get_enable_simd() const { return m_enable_simd; }

    /// If compiling for PTX, get the SM version.
    unsigned get_sm_version() const { return m_sm_version; }

    /// Get the number of supported texture spaces.
    unsigned get_num_texture_spaces() const { return m_num_texture_spaces; }

    /// Get the number of supported texture results.
    unsigned get_num_texture_results() const { return m_num_texture_results; }

    /// If true, compile pure constants into functions.
    bool get_compile_consts() const { return m_compile_consts; }

    /// If true, source code backend emits target language, else LLVM-IR.
    bool get_output_target_lang() const { return m_output_target_lang; }

    /// If true, string arguments are mapped to string identifiers.
    bool get_strings_mapped_to_ids() const { return m_strings_mapped_to_ids; }

    /// If true, derivatives should be calculated.
    bool get_calc_derivatives() const { return m_calc_derivatives; }

private:
    /// The backend kind.
    mi::neuraylib::IMdl_compiler::Mdl_backend_kind m_kind;

    /// If compiling for PTX, the SM version.
    unsigned m_sm_version;

    /// Number of supported texture spaces.
    unsigned m_num_texture_spaces;

    /// The number of supported float4 texture results in the MDL state.
    unsigned m_num_texture_results;

    /// The MDL compiler.
    mi::base::Handle<mi::mdl::IMDL> m_compiler;

    /// The JIT code generator.
    mi::base::Handle<mi::mdl::ICode_generator_jit> m_jit;

    /// The code cache if any.
    mi::base::Handle<mi::mdl::ICode_cache> m_code_cache;

    /// If true, compile pure constants into functions.
    bool m_compile_consts;

    /// If true, SIMD instruction are generated.
    bool m_enable_simd;

    /// If true, source code backends backend emit the target language, else LLVM-IR.
    bool m_output_target_lang;

    /// If true, strings arguments are compiled into string identifiers.
    bool m_strings_mapped_to_ids;

    /// If true, derivatives should be calculated.
    bool m_calc_derivatives;

    /// If true, use the builtin resource handler when running native code
    bool m_use_builtin_resource_handler;
};


/// Implementation of #mi::neuraylib::ITarget_argument_block.
class Target_argument_block : public
    mi::base::Interface_implement<mi::neuraylib::ITarget_argument_block>
{
public:
    /// Constructor allocating but not initializing the target argument block.
    ///
    /// \param arg_block_size  The size of the argument block to allocate.
    Target_argument_block(mi::Size arg_block_size);

    // API methods

    /// Returns the target argument block data.
    const char* get_data() const NEURAY_OVERRIDE;

    /// Returns the target argument block data.
    char* get_data() NEURAY_OVERRIDE;

    /// Returns the size of the target argument block data.
    mi::Size get_size() const NEURAY_OVERRIDE;

    /// Clones the target argument block (to make it writeable).
    ITarget_argument_block *clone() const NEURAY_OVERRIDE;

private:
    /// Destructor.
    ~Target_argument_block();

private:
    /// The size of the argument block data.
    mi::Size m_size;

    /// The target argument block data.
    char *m_data;
};

/// Internal version of the #mi::neuraylib::ITarget_resource_callback callback interface
/// operating on MI::MDL::IValue_resource objects.
class ITarget_resource_callback_internal
{
public:
    /// Returns a resource index for the given resource value usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// The value 0 is always an invalid resource reference.
    /// For #mi::mdl::IValue_texture values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_texture().
    /// For mi::mdl::IValue_light_profile values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_light_profile().
    /// For mi::mdl::IValue_bsdf_measurement values, the first indices correspond to the indices
    /// used with #mi::neuraylib::ITarget_code::get_bsdf_measurement().
    ///
    /// See \ref mi_neuray_ptx_texture_lookup_call_modes for more details about texture handlers
    /// for the PTX backend.
    ///
    /// \param resource  the resource value
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    virtual mi::Uint32 get_resource_index(MI::MDL::IValue_resource const *resource) = 0;

    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    virtual mi::Uint32 get_string_index(MI::MDL::IValue_string const *s) = 0;
};

/// Implementation of #mi::neuraylib::ITarget_value_layout.
/// Wraps an mi::mdl::IGenerated_code_value_layout.
class Target_value_layout : public
    mi::base::Interface_implement<mi::neuraylib::ITarget_value_layout>
{
public:
    /// Constructor.
    ///
    /// \param layout      The argument block layout.
    /// \param string_ids  if True, string argument values are mapped to string identifiers.
    Target_value_layout(
        mi::mdl::IGenerated_code_value_layout const *layout,
        bool string_ids);

    // API methods

    /// Get the size of the target argument block.
    mi::Size get_size() const NEURAY_OVERRIDE;

    /// Get the number of arguments / elements at the given layout state.
    ///
    /// \param state  The layout state representing the current nesting within the
    ///               argument value block. The default value is used for the top-level.
    mi::Size get_num_elements(
        mi::neuraylib::Target_value_layout_state state =
            mi::neuraylib::Target_value_layout_state()) const NEURAY_OVERRIDE;

    /// Get the offset, the size and the kind of the argument / element inside the argument
    /// block at the given layout state.
    ///
    /// \param[out]  kind      Receives the kind of the argument.
    /// \param[out]  arg_size  Receives the size of the argument.
    /// \param       state     The layout state representing the current nesting within the
    ///                        argument value block. The default value is used for the top-level.
    ///
    /// \returns the offset of the requested argument / element or ~0 if the state is invalid.
    mi::Size get_layout(
        mi::neuraylib::IValue::Kind &kind,
        mi::Size &arg_size,
        mi::neuraylib::Target_value_layout_state state =
            mi::neuraylib::Target_value_layout_state()) const NEURAY_OVERRIDE;

    /// Get the layout state for the i'th argument / element inside the argument value block
    /// at the given layout state.
    ///
    /// \param i      The index of the argument / element.
    /// \param state  The layout state representing the current nesting within the argument
    ///               value block. The default value is used for the top-level.
    ///
    /// \returns the layout state for the nested element or ~0 if the element is atomic.
    mi::neuraylib::Target_value_layout_state get_nested_state(
        mi::Size i,
        mi::neuraylib::Target_value_layout_state state =
        mi::neuraylib::Target_value_layout_state()) const NEURAY_OVERRIDE;

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
    mi::Sint32 set_value(
        char *block,
        mi::neuraylib::IValue const *value,
        mi::neuraylib::ITarget_resource_callback *resource_callback,
        mi::neuraylib::Target_value_layout_state state =
            mi::neuraylib::Target_value_layout_state()) const NEURAY_OVERRIDE;

    // Non-API methods

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
    mi::Sint32 set_value(
        char *block,
        MI::MDL::IValue const *value,
        ITarget_resource_callback_internal *resource_callback,
        mi::neuraylib::Target_value_layout_state state =
            mi::neuraylib::Target_value_layout_state()) const;

private:
    /// The MDL argument block.
    mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> m_layout;

    /// If true, string argument values are mapped to string identifiers.
    bool m_strings_mapped_to_ids;
};

} // namespace BACKENDS

} // namespace MI

#endif // RENDER_MDL_BACKENDS_BACKENDS_BACKENDS_H

