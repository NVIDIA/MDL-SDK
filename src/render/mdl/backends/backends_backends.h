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

#ifndef RENDER_MDL_BACKENDS_BACKENDS_BACKENDS_H
#define RENDER_MDL_BACKENDS_BACKENDS_BACKENDS_H

#include <string>
#include <vector>
#include <map>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/imdl_backend.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/neuraylib/itile.h>

#include <io/scene/dbimage/i_dbimage.h>

namespace mi {
namespace mdl { class IType_struct; class IType; }
namespace neuraylib { class ITarget_code; }
}

namespace MI {

namespace DB { class Transaction; }
namespace DBIMAGE { class Image_set; }

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
        mi::neuraylib::IMdl_backend_api::Mdl_backend_kind kind,
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

    mi::mdl::ILink_unit *create_link_unit(
        MDL::Execution_context* context);

    mi::neuraylib::ITarget_code const *translate_link_unit(
        Link_unit const *lu,
        MDL::Execution_context* context);

    /// Update the MDL JIT options from the given parameters.
    void update_jit_options(
        const char *internal_space,
        MDL::Execution_context *context);

    /// Get the MDL compiler.
    mi::base::Handle<mi::mdl::IMDL> get_compiler() const { return m_compiler; }

    /// Get the JIT backend.
    mi::base::Handle<mi::mdl::ICode_generator_jit> get_jit_be() const { return m_jit; }

    /// Get the backend kind.
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind get_kind() const { return m_kind; }

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
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind m_kind;

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
    const char* get_data() const override;

    /// Returns the target argument block data.
    char* get_data() override;

    /// Returns the size of the target argument block data.
    mi::Size get_size() const override;

    /// Clones the target argument block (to make it writeable).
    ITarget_argument_block *clone() const override;

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
    mi::Size get_size() const override;

    /// Get the number of arguments / elements at the given layout state.
    ///
    /// \param state  The layout state representing the current nesting within the
    ///               argument value block. The default value is used for the top-level.
    mi::Size get_num_elements(
        mi::neuraylib::Target_value_layout_state state =
            mi::neuraylib::Target_value_layout_state()) const override;

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
            mi::neuraylib::Target_value_layout_state()) const override;

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
        mi::neuraylib::Target_value_layout_state()) const override;

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
            mi::neuraylib::Target_value_layout_state()) const override;

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

    /// Get the internal IGenerated_code_value_layout
    mi::mdl::IGenerated_code_value_layout const* get_internal_layout() const
    {
        m_layout->retain();
        return m_layout.get();
    }

    /// If true, string argument values are mapped to string identifiers.
    bool strings_mapped_to_ids() const { return m_strings_mapped_to_ids; }

private:
    /// The MDL argument block.
    mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> m_layout;

    /// If true, string argument values are mapped to string identifiers.
    bool m_strings_mapped_to_ids;
};

/// Helper class to store bsdf data textures into the neuray database.
class Df_data_helper
{
public:

    Df_data_helper(
        DB::Transaction *transaction)
        : m_transaction(transaction)
    {
    }

    /// Creates and stores bsdf data textures in the database.
    ///
    /// \return The tag of the texture in the database
    DB::Tag store_df_data(mi::mdl::IValue_texture::Bsdf_data_kind df_data_kind);

    /// Returns the database name for the given df data kind.
    static const char* get_texture_db_name(mi::mdl::IValue_texture::Bsdf_data_kind kind);

private:

    class Df_data_tile : public mi::base::Interface_implement<mi::neuraylib::ITile>
    {

    public:

        /// Constructor
        Df_data_tile(mi::Uint32 rx, mi::Uint32 ry, const float* data)
            : m_resolution_x(rx)
            , m_resolution_y(ry)
            , m_data(data)
        {
        }

        // methods of mi::neuraylib::ITile
        void get_pixel(
            mi::Uint32 x_offset,
            mi::Uint32 y_offset,
            mi::Float32* floats) const final;

        void set_pixel(
            mi::Uint32 x_offset,
            mi::Uint32 y_offset,
            const  mi::Float32* floats) final;

        const char* get_type() const final;

        mi::Uint32 get_resolution_x() const final;

        mi::Uint32 get_resolution_y() const final;

        const void* get_data() const final;

        void* get_data() final;

    private:

        mi::Uint32 m_resolution_x;      ///< resolution in x
        mi::Uint32 m_resolution_y;      ///< resolution in y

        const float* m_data;            ///< data
    };

    class Df_data_canvas : public mi::base::Interface_implement<mi::neuraylib::ICanvas>
    {
    public:

        /// Constructor
        Df_data_canvas(mi::Uint32 rx, mi::Uint32 ry, mi::Uint32 rz, const float *data)
        {
            m_tiles.resize(rz);
            mi::Uint32 offset = rx * ry;
            for (mi::Size i = 0; i < rz; ++i) {
                m_tiles[i] = new Df_data_tile(rx, ry, &data[i*offset]);
            }
        }

        // methods of mi::neuraylib::ICanvas_base

        mi::Uint32 get_resolution_x() const final;

        mi::Uint32 get_resolution_y() const final;

        const char* get_type() const final;

        mi::Uint32 get_layers_size() const final;

        mi::Float32 get_gamma() const final;

        void set_gamma(mi::Float32) final;

        // methods of mi::neuraylib::ICanvas

        mi::Uint32 get_tile_resolution_x() const final;

        mi::Uint32 get_tile_resolution_y() const final;

        mi::Uint32 get_tiles_size_x() const final;

        mi::Uint32 get_tiles_size_y() const final;

        const mi::neuraylib::ITile* get_tile(
            mi::Uint32 pixel_x, mi::Uint32 pixel_y, mi::Uint32 layer = 0) const final;

        mi::neuraylib::ITile* get_tile(
            mi::Uint32 pixel_x, mi::Uint32 pixel_y, mi::Uint32 layer = 0) final;

    private:

        std::vector<mi::base::Handle<Df_data_tile>> m_tiles;
    };

    class Df_image_set : public DBIMAGE::Image_set
    {
    public:

        Df_image_set(mi::neuraylib::ICanvas* canvas)
            : m_canvas(mi::base::make_handle_dup(canvas)) { }

        // methods from DBIMAGE::Image_set
        mi::Size get_length() const final;

        bool is_uvtile() const final;

        bool is_mdl_container() const final;

        void get_uv_mapping( mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const final;

        const char* get_original_filename() const final;

        const char* get_container_filename() const final;

        const char* get_mdl_file_path() const final;

        const char* get_resolved_filename( mi::Size i) const final;

        const char* get_container_membername( mi::Size i) const final;

        mi::neuraylib::IReader* open_reader( mi::Size i) const final;

        mi::neuraylib::ICanvas* get_canvas( mi::Size i) const final;

        const char* get_image_format() const final;

    private:
        mi::base::Handle <mi::neuraylib::ICanvas> m_canvas;
    };

    DB::Tag store_texture(
        mi::Uint32 rx,
        mi::Uint32 ry,
        mi::Uint32 rz,
        const float *data,
        const std::string& tex_name);

private:

    static mi::base::Lock m_lock;
    using Df_data_map = std::map <mi::mdl::IValue_texture::Bsdf_data_kind,std::string>;
    static Df_data_map m_df_data_to_name;

    DB::Transaction *m_transaction;
};

} // namespace BACKENDS

} // namespace MI

#endif // RENDER_MDL_BACKENDS_BACKENDS_BACKENDS_H

