/***************************************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

//#define ADD_EXTRA_TIMERS

#ifdef ADD_EXTRA_TIMERS
#include <chrono>
#endif

#include <cstring>
#include <map>
#include <string>
using namespace std::string_literals;

#include <mi/base/handle.h>
#include <mi/base/types.h>
#include "mi/mdl/mdl_generated_dag.h"
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_types.h>
#include <base/lib/log/i_log_logger.h>
#include <base/data/db/i_db_access.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/scene/mdl_elements/mdl_elements_detail.h> // DETAIL::Type_binder
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/texture/i_texture.h>
#include <render/mdl/runtime/i_mdlrt_resource_handler.h>
#include <render/mdl/backends/backends_target_code.h>

#include <mdl/codegenerators/generator_dag/generator_dag_lambda_function.h>
#include <mdl/codegenerators/generator_dag/generator_dag_tools.h>
#include <mdl/codegenerators/generator_dag/generator_dag_dumper.h>
#include <mdl/jit/generator_jit/generator_jit_libbsdf_data.h>
#include <mdl/compiler/compilercore/compilercore_streams.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include "backends_link_unit.h"
#include "backends_backends.h"

namespace MI {

namespace BACKENDS {

namespace {

template <typename T>
T get_context_option(MDL::Execution_context* context, const char* option)
{
    if (context)
        return context->get_option<T>(option);

    MDL::Execution_context default_context;
    return default_context.get_option<T>(option);
}

/// A name register interface.
class IResource_register {
public:
    /// Register a texture index.
    ///
    /// \param index        the texture index
    /// \param is_resolved  true, if this texture has been resolved and exists in the Neuray DB
    /// \param name         the DB name of the texture at this index, if the texture has been
    ///                     resolved, the unresolved mdl url of the texture otherwise
    /// \param owner_module the owner module name of the texture
    /// \param gamma        the gamma value of the texture
    /// \param selector     the selector of the texture
    /// \param type         the type of the texture
    /// \param df_data_kind the \c DF data kind of the texture
    virtual void register_texture(
        size_t                                     index,
        bool                                       is_resolved,
        char const                                 *name,
        char const                                 *owner_module,
        float                                      gamma,
        char const                                 *selector,
        mi::neuraylib::ITarget_code::Texture_shape type,
        mi::mdl::IValue_texture::Bsdf_data_kind    df_data_kind) = 0;

    /// Updates an already registered resource when it is encountered again.
    ///
    /// \param index        the texture index
    virtual void update_texture(size_t index) = 0;

    /// Return the number of texture resources.
    virtual size_t get_texture_count() const = 0;

    /// Returns the number of texture resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid texture).
    ///
    /// \return           The body texture count or \c ~0ull, if the value is invalid due to
    ///                   multiple translate calls.
    virtual size_t get_body_texture_count() const = 0;

    /// Register a light profile.
    ///
    /// \param index        the light profile index
    /// \param is_resolved  true, if this resource has been resolved and exists in the Neuray DB
    /// \param name         the DB name of this index, if this resource has been resolved,
    ///                     the unresolved mdl url otherwise
    /// \param owner_module the owner module name of the resource
    virtual void register_light_profile(
        size_t                                     index,
        bool                                       is_resolved,
        char const                                 *name,
        char const                                 *owner_module) = 0;

    /// Updates an already registered resource when it is encountered again.
    ///
    /// \param index        the light profile index
    virtual void update_light_profile(size_t index) = 0;

    /// Return the number of light profile resources.
    virtual size_t get_light_profile_count() const = 0;

    /// Returns the number of light profile resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid light profile).
    ///
    /// \return           The body light profile count or \c ~0ull, if the value is invalid due to
    ///                   more than one call to a link unit add function.
    virtual size_t get_body_light_profile_count() const = 0;

    /// Register a BSDF measurement.
    ///
    /// \param index        the BSDF measurement index
    /// \param is_resolved  true, if this resource has been resolved and exists in the Neuray DB
    /// \param name         the DB name of this index, if this resource has been resolved,
    ///                     the unresolved mdl url otherwise
    /// \param owner_module the owner module name of the resource
    virtual void register_bsdf_measurement(
        size_t                                     index,
        bool                                       is_resolved,
        char const                                 *name,
        char const                                 *owner_module) = 0;

    /// Updates an already registered resource when it is encountered again.
    ///
    /// \param index        the BSDF measurement index
    virtual void update_bsdf_measurement(size_t index) = 0;

    /// Return the number of BSDF measurement resources.
    virtual size_t get_bsdf_measurement_count() const = 0;

    /// Returns the number of BSDF measurement resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid BSDF measurement).
    ///
    /// \return           The body BSDF measurement count or \c ~0ull, if the value is invalid due
    ///                   to more than one call to a link unit add function.
    virtual size_t get_body_bsdf_measurement_count() const = 0;
};


namespace // anonymous
{
    // used for a workaround
    bool get_first_tile(DB::Transaction* transaction,
        DB::Tag tag,
        mi::Size& first_frame,
        mi::Sint32& first_uvtile_u,
        mi::Sint32& first_uvtile_v)
    {
        if (!tag || transaction->get_class_id(tag) != TEXTURE::ID_TEXTURE)
            return false;

        DB::Access<TEXTURE::Texture> db_texture(tag, transaction);
        tag = db_texture->get_image();

        if (!tag || transaction->get_class_id(tag) != DBIMAGE::ID_IMAGE)
            return false;

        DB::Access<DBIMAGE::Image> db_image(tag, transaction);
        if (!db_image->is_valid())
            return false;

        first_frame = (int)db_image->get_frame_number(0);

        mi::Sint32 res = db_image->get_uvtile_uv(0, 0, first_uvtile_u, first_uvtile_v);
        if (res != 0)
            return false;

        return true;
    }
}

/// Helper class to enumerate resources in lambda functions.
class Function_enumerator : public mi::mdl::ILambda_resource_enumerator {
public:
    typedef std::map<std::string, size_t>  Resource_index_map;

    /// Constructor.
    ///
    /// \param reg                       an index register interface
    /// \param lambda                    currently processed lambda function
    /// \param db_transaction            the current transaction
    /// \param keep_unresolved_resources if \c true, unresolved resources are kept
    /// \param store_df_data             if \c true, df data tables are stored in the database
    Function_enumerator(
        IResource_register        &reg,
        mi::mdl::ILambda_function *lambda,
        DB::Transaction           *db_transaction,
        bool                      keep_unresolved_resources,
        bool                      store_df_data)
    : m_register(reg)
    , m_lambda(lambda)
    , m_additional_lambda(NULL)
    , m_db_transaction(db_transaction)
    , m_resource_index_map(NULL)
    , m_tex_idx_store(0)
    , m_lp_idx_store(0)
    , m_bm_idx_store(0)
    , m_tex_idx(m_tex_idx_store)
    , m_lp_idx(m_lp_idx_store)
    , m_bm_idx(m_bm_idx_store)
    , m_keep_unresolved_resources(keep_unresolved_resources)
    , m_store_df_data(store_df_data)
    {
    }

    /// Constructor.
    ///
    /// \param reg                       an index register interface
    /// \param lambda                    currently processed lambda function
    /// \param trans                     the current transaction
    /// \param tex_idx                   next texture index.
    /// \param lp_idx                    next light profile index.
    /// \param bm_idx                    next bsdf measurement index.
    /// \param res_map                   a map to track which resources are already known
    /// \param keep_unresolved_resources if \c true, unresolved resources are kept
    /// \param store_df_data             if \c true, df data tables are stored in the database
    Function_enumerator(
        IResource_register        &reg,
        mi::mdl::ILambda_function *lambda,
        DB::Transaction           *db_transaction,
        size_t                    &tex_idx,
        size_t                    &lp_idx,
        size_t                    &bm_idx,
        Resource_index_map        &res_map,
        bool                      keep_unresolved_resources,
        bool                      store_df_data)
    : m_register(reg)
    , m_lambda(lambda)
    , m_additional_lambda(NULL)
    , m_db_transaction(db_transaction)
    , m_resource_index_map(&res_map)
    , m_tex_idx_store(0)
    , m_lp_idx_store(0)
    , m_bm_idx_store(0)
    , m_tex_idx(tex_idx)
    , m_lp_idx(lp_idx)
    , m_bm_idx(bm_idx)
    , m_keep_unresolved_resources(keep_unresolved_resources)
    , m_store_df_data(store_df_data)
    {
    }

    /// Called for a texture resource.
    ///
    /// \param v  the texture resource or an invalid ref
    virtual void texture(mi::mdl::IValue const *v, Texture_usage tex_usage)
    {
        if (m_register.get_texture_count() == 0) {
            // index 0 is always the only invalid texture index
            m_register.register_texture(
                0, false, "", "", 0.0f, "",
                mi::neuraylib::ITarget_code::Texture_shape_invalid,
                mi::mdl::IValue_texture::BDK_NONE);
        }

        if (mi::mdl::IValue_texture const *tex = mi::mdl::as<mi::mdl::IValue_texture>(v)) {
            bool valid = false;
            int width = 0, height = 0, depth = 0;

            int tag_value = tex->get_tag_value();
            if (tag_value == 0) {
                // check, if the tag is mapped
                tag_value = m_lambda->get_resource_tag(tex);
            }

            char const *name = nullptr;
            bool is_resolved = false;

            mi::mdl::IValue_texture::Bsdf_data_kind kind = tex->get_bsdf_data_kind();
            if (kind != mi::mdl::IValue_texture::BDK_NONE) {
                if (m_store_df_data) {
                    Df_data_helper helper(m_db_transaction);
                    tag_value = static_cast<int>(helper.store_df_data(kind).get_uint());
                    is_resolved = tag_value != 0;
                }
                name = Df_data_helper::get_texture_db_name(kind);
            } else {
                is_resolved = tag_value != 0;
            }

            // TODO
            // We are currently fetching the first frame and it's uv tiles as a workaround.
            // The texture attributes and also the uv-tiles for a frame depend
            // on the frame, so using width, height, and depth is wrong.
            mi::Size first_frame_number;
            mi::Sint32 first_uvtile_u;
            mi::Sint32 first_uvtile_v;
            int first_frame, last_frame;
            if (get_first_tile(
                m_db_transaction, DB::Tag(tag_value), first_frame_number, first_uvtile_u, first_uvtile_v))
            {
                MI::MDL::get_texture_attributes(
                    m_db_transaction, DB::Tag(tag_value), first_frame_number,
                    first_uvtile_u, first_uvtile_v, valid, width, height, depth, first_frame, last_frame);
            }
            else {
                valid = false;
            }

            if (valid || m_keep_unresolved_resources) {
                if (!name)
                    name = resource_to_name(tag_value, tex);

                mi::mdl::IValue_texture::gamma_mode gamma_mode = tex->get_gamma_mode();
                mi::mdl::IType_texture::Shape shape = tex->get_type()->get_shape();
                const char* selector = tex->get_selector();

                bool new_entry = true;
                size_t tex_idx;
                if (m_resource_index_map != NULL) {
                    std::string resource_key =
                        std::string(name) + '_' +
                        std::to_string(unsigned(gamma_mode)) + '_' +
                        std::to_string(unsigned(shape)) + "_" +
                        selector;
                    Resource_index_map::const_iterator it(m_resource_index_map->find(resource_key));
                    if (it == m_resource_index_map->end()) {
                        // new entry
                        tex_idx = ++m_tex_idx;
                        new_entry = true;
                        m_resource_index_map->insert(
                            Resource_index_map::value_type(resource_key, tex_idx));
                    } else {
                        // known
                        tex_idx = it->second;
                        new_entry = false;
                    }
                } else {
                    // no map, always new
                    tex_idx = ++m_tex_idx;
                    new_entry = true;
                }

                if (new_entry) {
                    float gamma = gamma_mode == mi::mdl::IValue_texture::gamma_default ? 0.0f :
                        (gamma_mode == mi::mdl::IValue_texture::gamma_linear ? 1.0f : 2.2f);
                    m_register.register_texture(
                        tex_idx,
                        is_resolved,
                        name,
                        /*owner_module=*/"",
                        gamma,
                        selector,
                        get_texture_shape(tex->get_type()),
                        tex->get_bsdf_data_kind());
                } else {
                    // if a later expression uses a resource in the body that has been previously
                    // used in the arguments only
                    m_register.update_texture(tex_idx);
                }

                m_lambda->map_tex_resource(
                    tex->get_kind(),
                    tex->get_string_value(),
                    selector,
                    gamma_mode,
                    tex->get_bsdf_data_kind(),
                    shape,
                    tag_value,
                    tex_idx,
                    /*valid=*/true,
                    width,
                    height,
                    depth);
                if (m_additional_lambda != NULL) {
                    m_additional_lambda->map_tex_resource(
                        tex->get_kind(),
                        tex->get_string_value(),
                        selector,
                        gamma_mode,
                        tex->get_bsdf_data_kind(),
                        shape,
                        tag_value,
                        tex_idx,
                        /*valid=*/true,
                        width,
                        height,
                        depth);
                }

                return;
            }
        }
        // invalid textures are always mapped to zero in the MDL SDK
        m_lambda->map_tex_resource(
            v->get_kind(),
            /*res_url=*/NULL,
            /*res_sel=*/NULL,
            mi::mdl::IValue_texture::gamma_default,
            mi::mdl::IValue_texture::BDK_NONE,
            mi::mdl::IType_texture::TS_2D,
            0,
            0,
            /*valid=*/false,
            0,
            0,
            0);
        if (m_additional_lambda != NULL) {
            m_additional_lambda->map_tex_resource(
                v->get_kind(),
                /*res_url=*/NULL,
                /*res_sel=*/NULL,
                mi::mdl::IValue_texture::gamma_default,
                mi::mdl::IValue_texture::BDK_NONE,
                mi::mdl::IType_texture::TS_2D,
                0,
                0,
                /*valid=*/false,
                0,
                0,
                0);
        }
    }

    /// Called for a light profile resource.
    ///
    /// \param v  the light profile resource or an invalid ref
    virtual void light_profile(mi::mdl::IValue const *v)
    {
        if (m_register.get_light_profile_count() == 0) {
            // index 0 is always the only invalid light profile index
            m_register.register_light_profile(0, /*is_resolved=*/false, "", "");
        }

        if (mi::mdl::IValue_resource const *r = mi::mdl::as<mi::mdl::IValue_resource>(v)) {
            bool valid = false;
            float power = 0.0f, maximum = 0.0f;

            int tag_value = r->get_tag_value();
            if (tag_value == 0) {
                // check, if the tag is mapped
                tag_value = m_lambda->get_resource_tag(r);
            }

            MI::MDL::get_light_profile_attributes(
                m_db_transaction, DB::Tag(tag_value), valid, power, maximum);

            if (valid || m_keep_unresolved_resources) {
                char const *name = resource_to_name(tag_value, r);
                if (m_additional_lambda != NULL) {
                    m_additional_lambda->map_lp_resource(
                        r->get_kind(),
                        r->get_string_value(),
                        tag_value,
                        m_lp_idx,
                        /*valid=*/true,
                        power,
                        maximum);
                }

                bool new_entry = true;
                size_t lp_idx;
                if (m_resource_index_map != NULL) {
                    Resource_index_map::const_iterator it(m_resource_index_map->find(name));
                    if (it == m_resource_index_map->end()) {
                        // new entry
                        lp_idx = ++m_lp_idx;
                        new_entry = true;
                        m_resource_index_map->insert(Resource_index_map::value_type(name, lp_idx));
                    } else {
                        // known
                        lp_idx = it->second;
                        new_entry = false;

                    }
                } else {
                    // no map, always new
                    lp_idx = ++m_lp_idx;
                    new_entry = true;
                }

                if (new_entry) {
                    m_register.register_light_profile(
                        lp_idx, /*is_resolved=*/tag_value != 0, name, "");
                }
                else {
                    m_register.update_light_profile(lp_idx);
                }

                m_lambda->map_lp_resource(
                    r->get_kind(),
                    r->get_string_value(),
                    tag_value,
                    lp_idx,
                    /*valid=*/true,
                    power,
                    maximum);

                return;
            }
        }
        // invalid light profiles are always mapped to zero in the MDL SDK
        m_lambda->map_lp_resource(
            v->get_kind(),
            NULL,
            0,
            0,
            /*valid=*/false,
            0.0f,
            0.0f);
        if (m_additional_lambda != NULL) {
            m_additional_lambda->map_lp_resource(
                v->get_kind(),
                NULL,
                0,
                0,
                /*valid=*/false,
                0.0f,
                0.0f);
        }
    }

    /// Called for a bsdf measurement resource.
    ///
    /// \param v  the bsdf measurement resource or an invalid_ref
    virtual void bsdf_measurement(mi::mdl::IValue const *v)
    {
        if (m_register.get_bsdf_measurement_count() == 0) {
            // index 0 is always the only invalid bsdf measurement index
            m_register.register_bsdf_measurement(0, /*is_resolved=*/false, "", "");
        }

        if (mi::mdl::IValue_resource const *r = mi::mdl::as<mi::mdl::IValue_resource>(v)) {
            bool valid = false;

            int tag_value = r->get_tag_value();
            if (tag_value == 0) {
                // check, if the tag is mapped
                tag_value = m_lambda->get_resource_tag(r);
            }

            MI::MDL::get_bsdf_measurement_attributes(m_db_transaction, DB::Tag(tag_value), valid);

            if (valid || m_keep_unresolved_resources) {
                char const *name = resource_to_name(tag_value, r);
                if (m_additional_lambda != NULL) {
                    m_additional_lambda->map_bm_resource(
                        r->get_kind(), r->get_string_value(), tag_value, m_bm_idx, /*valid=*/true);
                }

                bool new_entry = true;
                size_t bm_idx;
                if (m_resource_index_map != NULL) {
                    Resource_index_map::const_iterator it(m_resource_index_map->find(name));
                    if (it == m_resource_index_map->end()) {
                        // new entry
                        bm_idx = ++m_bm_idx;
                        new_entry = true;
                        m_resource_index_map->insert(Resource_index_map::value_type(name, bm_idx));
                    } else {
                        // known
                        bm_idx = it->second;
                        new_entry = false;
                    }
                } else {
                    // no map, always new
                    bm_idx = ++m_bm_idx;
                    new_entry = true;
                }

                if (new_entry) {
                    m_register.register_bsdf_measurement(
                        bm_idx, /*is_resolved=*/tag_value != 0, name, "");
                } else {
                    m_register.update_bsdf_measurement(bm_idx);
                }

                m_lambda->map_bm_resource(
                    r->get_kind(), r->get_string_value(), tag_value, bm_idx, /*valid=*/true);

                return;
            }
        }
        // invalid bsdf measurements are always mapped to zero in the MDL SDK
        m_lambda->map_bm_resource(v->get_kind(), NULL, 0, 0, /*valid=*/false);
        if (m_additional_lambda != NULL) {
            m_additional_lambda->map_bm_resource(v->get_kind(), NULL, 0, 0, /*valid=*/false);
        }
    }

    /// Set an additional lambda which should receive registered resources.
    void set_additional_lambda(mi::mdl::ILambda_function *additional_lambda) {
        m_additional_lambda = additional_lambda;
    }

private:

    /// Get the DB name of a resource.
    char const *resource_to_name(int tag_value, mi::mdl::IValue_resource const *r)
    {
        DB::Tag tag = DB::Tag(tag_value);
        char const *name = m_db_transaction->tag_to_name(tag);
        if (name != NULL)
            return name;
        if (m_keep_unresolved_resources)
            return r->get_string_value();
        return "";
    }

    /// Get the ITargetcode::Texture_shape from a MDL type.
    static mi::neuraylib::ITarget_code::Texture_shape get_texture_shape(
        mi::mdl::IType_texture const *type)
    {
        switch (type->get_shape()) {
        case mi::mdl::IType_texture::TS_2D:
            return mi::neuraylib::ITarget_code::Texture_shape_2d;
            break;
        case mi::mdl::IType_texture::TS_3D:
            return mi::neuraylib::ITarget_code::Texture_shape_3d;
            break;
        case mi::mdl::IType_texture::TS_CUBE:
            return mi::neuraylib::ITarget_code::Texture_shape_cube;
            break;
        case mi::mdl::IType_texture::TS_PTEX:
            return mi::neuraylib::ITarget_code::Texture_shape_ptex;
            break;
        case mi::mdl::IType_texture::TS_BSDF_DATA:
            return mi::neuraylib::ITarget_code::Texture_shape_bsdf_data;
            break;
        }
        ASSERT(M_BACKENDS, !"Unsupported MDL texture shape");
        return mi::neuraylib::ITarget_code::Texture_shape_invalid;
    }

private:
    /// The index register interface.
    IResource_register &m_register;

    /// The processed lambda function.
    mi::mdl::ILambda_function *m_lambda;

    /// Additional lambda function where textures should be registered.
    mi::mdl::ILambda_function *m_additional_lambda;

    /// The current transaction.
    DB::Transaction *m_db_transaction;

    /// The set of known resources if they are tracked.
    Resource_index_map *m_resource_index_map;

    /// Storage for the current texture index.
    size_t m_tex_idx_store;

    /// Storage for the current light profile index.
    size_t m_lp_idx_store;

    /// Storage for the current bsdf measurement index.
    size_t m_bm_idx_store;

    /// Current texture index.
    size_t &m_tex_idx;

    /// Current light profile index.
    size_t &m_lp_idx;

    /// Current bsdf measurement index.
    size_t &m_bm_idx;

    /// If true, unresolved resources will be retained.
    bool m_keep_unresolved_resources;

    /// If true, DF data textures are stored into the database.
    bool m_store_df_data;
};

/// Helper class for building Lambda functions.
class Lambda_builder {
public:
    /// Constructor.
    Lambda_builder(
        mi::mdl::IMDL   *compiler,
        DB::Transaction *db_transaction,
        bool            compile_consts,
        bool            calc_derivatives)
    : m_compiler(compiler, mi::base::DUP_INTERFACE)
    , m_db_transaction(db_transaction)
    , m_error(0)
    , m_compile_consts(compile_consts)
    , m_calc_derivatives(calc_derivatives)
    {
    }

    /// Get the error code of the last operation.
    mi::Sint32 get_error_code() const { return m_error; }

    /// Get the error code of the last operation.
    const std::string& get_error_string() const { return m_error_string; }

    /// Build a lambda function from a call.
    mi::mdl::ILambda_function *from_call(
        MDL::Mdl_function_call const                        *function_call,
        char const                                          *fname,
        mi::mdl::ILambda_function::Lambda_execution_context execution_ctx)
    {
        if (function_call == NULL) {
            error(-1, "Invalid parameter (NULL pointer)");
            return NULL;
        }

        DB::Tag def_tag = function_call->get_function_definition(m_db_transaction);
        if (!def_tag.is_valid()) {
            error(-2, "Invalid call expression");
            return NULL;
        }
        DB::Access<MDL::Mdl_function_definition> definition(
            def_tag, m_db_transaction);

        if (!definition.is_valid()) {
            error(-2, "Invalid call expression");
            return NULL;
        }

        mi::mdl::IType const *mdl_type = definition->get_core_return_type(m_db_transaction);
        if (mdl_type == NULL) {
            error(-2, "Invalid call expression");
            return NULL;
        }
        mdl_type = mdl_type->skip_type_alias();

        mi::mdl::IDefinition::Semantics sema = definition->get_core_semantic();
        if (sema == mi::mdl::IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR) {
            // need special handling for array constructor because its definition is "broken".
            // However, array constructors are not allowed here
            error(-2, "Unsupported array constructor call expression");
            return NULL;
        }

        bool type_ok = false;
        mi::mdl::IType_struct const *env_call_ret_type = NULL;

        if (execution_ctx == mi::mdl::ILambda_function::LEC_ENVIRONMENT) {
            // check for base::texture_return compatible or color return type if
            // compiling for environment
            if (mi::mdl::IType_struct const *s_type = mi::mdl::as<mi::mdl::IType_struct>(mdl_type)) {
                if (s_type->get_compound_size() == 2) {
                    mi::mdl::IType const *e0_tp = s_type->get_compound_type(0)->skip_type_alias();
                    mi::mdl::IType const *e1_tp = s_type->get_compound_type(1)->skip_type_alias();

                    if (mi::mdl::is<mi::mdl::IType_color>(e0_tp) &&
                        mi::mdl::is<mi::mdl::IType_float>(e1_tp))
                    {
                        type_ok = true;
                        env_call_ret_type = s_type;
                    }
                }
            } else if (mi::mdl::is<mi::mdl::IType_color>(mdl_type)) {
                type_ok = true;
            }

            if (!type_ok) {
                error(-2, "Invalid return type for environment call");
                return NULL;
            }
        }

        mi::base::Handle<mi::mdl::ILambda_function> lambda(
            m_compiler->create_lambda_function(execution_ctx));

        MDL::Mdl_dag_builder builder(
            m_db_transaction, lambda.get(), /*compiled_material=*/NULL);

        mi::mdl::IType_factory *tf = lambda->get_type_factory();
        MDL::DETAIL::Type_binder type_binder(tf);

        mi::Uint32 count = static_cast<mi::Uint32>(function_call->get_parameter_count());

        std::vector<mi::mdl::DAG_call::Call_argument> mdl_arguments(count);
        mi::base::Handle<const MDL::IExpression_list> arguments(
            function_call->get_arguments());
        for (mi::Uint32 i = 0; i < count; ++i) {
            mi::mdl::IType const *parameter_type =
                definition->get_core_parameter_type(m_db_transaction, i);

            mi::base::Handle<MDL::IExpression const> argument(arguments->get_expression(i));
            mdl_arguments[i].arg = builder.int_expr_to_core_dag_node(
                parameter_type, argument.get());
            if (mdl_arguments[i].arg == NULL) {
                error(-2, "Invalid argument");
                return NULL;
            }
            mdl_arguments[i].param_name = function_call->get_parameter_name(i);
            parameter_type = tf->import(parameter_type->skip_type_alias());

            mi::mdl::IType const *argument_type = mdl_arguments[i].arg->get_type();
            mi::Sint32 result = type_binder.check_and_bind_type(parameter_type, argument_type);
            switch (result) {
            case 0:
                // nothing to do
                break;
            case -1:
                error(-2, std::string("Type mismatch for argument \"") +
                    mdl_arguments[i].param_name +
                    "\" of function call \"" +
                    function_call->get_mdl_function_definition() +
                    "\"");
                LOG::mod_log->error(
                    M_BACKENDS, LOG::Mod_log::C_DATABASE, "%s", m_error_string.c_str());
                return NULL;
            case -2:
                error(-2, std::string("Array size mismatch for argument \"") +
                    mdl_arguments[i].param_name +
                    "\" of function call \"" +
                    function_call->get_mdl_function_definition() +
                    "\"");
                LOG::mod_log->error(
                    M_BACKENDS, LOG::Mod_log::C_DATABASE, "%s", m_error_string.c_str());
                return NULL;
            default:
                ASSERT(M_BACKENDS, false);
                error(-2, "Invalid expression");
                return NULL;
            }

            mdl_type = tf->import(mdl_type);
        }

        mi::mdl::DAG_call::Call_argument const *p_arguments = count > 0 ? &mdl_arguments[0] : 0;
        mi::mdl::DAG_node const *body = lambda->create_call(
            MDL::decode_name_with_signature(function_call->get_mdl_function_definition()).c_str(),
            function_call->get_core_semantic(),
            p_arguments,
            count,
            mdl_type);

        // if return type is ::base::texture_return (see above) wrap the
        // DAG node by a select to extract the tint field
        if (env_call_ret_type != NULL) {
            mi::mdl::IType_struct::Field const *field = env_call_ret_type->get_field(0);

            std::string name(env_call_ret_type->get_symbol()->get_name());
            name += '.';
            name += field->get_symbol()->get_name();
            name += '(';
            name += env_call_ret_type->get_symbol()->get_name();
            name += ')';

            mi::mdl::DAG_call::Call_argument args[1];

            args[0].arg        = body;
            args[0].param_name = "s";
            body = lambda->create_call(
                name.c_str(),
                mi::mdl::IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS,
                args,
                1,
                field->get_type());
        }

        lambda->set_body(body);
        if (fname != NULL) {
            lambda->set_name(fname);
        }

        m_error = 0;
        lambda->retain();
        return lambda.get();
    }

    /// Build a lambda function from a material sub-expression.
    mi::mdl::ILambda_function *from_sub_expr(
        MDL::Mdl_compiled_material const *compiled_material,
        char const                       *path,
        char const                       *fname)
    {
        // get the field corresponding to path
        mi::base::Handle<MDL::IExpression const > field(
            compiled_material->lookup_sub_expression(m_db_transaction, path));

        if (!field.is_valid_interface()) {
            error(-2, "Invalid path (non-existing)");
            return NULL;
        }

        // reject constants if not explicitly enabled
        if (!m_compile_consts && field->get_kind() == MDL::IExpression::EK_CONSTANT) {
            error(-4, "The requested expression is a constant");
            return NULL;
        }

        // create a lambda function
        mi::mdl::ILambda_function::Lambda_execution_context lec
            = mi::mdl::ILambda_function::LEC_CORE;
        if (strcmp(path, "geometry.displacement") == 0) {
            // only this is the displacement function
            lec = mi::mdl::ILambda_function::LEC_DISPLACEMENT;
        }
        mi::base::Handle<mi::mdl::ILambda_function> lambda(
            m_compiler->create_lambda_function(lec));

        // reject DFs or resources
        mi::base::Handle<const MDL::IType> field_int_type(field->get_type());
        mi::mdl::IType_factory *tf         = lambda->get_type_factory();
        mi::mdl::IType const   *field_type = MDL::int_type_to_core_type(field_int_type.get(), *tf);
        field_type = field_type->skip_type_alias();
        if (contains_df_type(field_type) || mi::mdl::is<mi::mdl::IType_resource>(field_type)) {
            error(-5, "Neither DFs nor resource type expressions can be handled");
            return NULL;
        }

        // copy the resource map to the lambda
        copy_resource_map(compiled_material, lambda);

        // ... and fill up ...
        MDL::Mdl_dag_builder builder(
            m_db_transaction, lambda.get(), compiled_material);

        // add all material parameters to the lambda function
        for (size_t i = 0, n = compiled_material->get_parameter_count(); i < n; ++i) {
            mi::base::Handle<MI::MDL::IValue const> value(
                compiled_material->get_argument(m_db_transaction, i));
            mi::base::Handle<MI::MDL::IType const> p_type(value->get_type());

            mi::mdl::IType const *tp = MDL::int_type_to_core_type(p_type.get(), *tf);

            size_t idx = lambda->add_parameter(tp, compiled_material->get_parameter_name(i));

            // map the i'th material parameter to this new parameter
            lambda->set_parameter_mapping(i, idx);
        }

        mi::mdl::DAG_node const *body = builder.int_expr_to_core_dag_node(field_type, field.get());
        lambda->set_body(body);
        if (fname != NULL) {
            lambda->set_name(fname);
        }

        m_error = 0;
        lambda->retain();
        return lambda.get();
    }

    /// Build a lambda function from a function definition.
    mi::mdl::ILambda_function *from_function_definition(
        MDL::Mdl_function_definition const                  *fdef,
        char const                                          *fname,
        mi::mdl::ILambda_function::Lambda_execution_context lec)
    {
        if (fdef == NULL) {
            error(-1, "Invalid parameters (NULL pointer)");
            return NULL;
        }

        mi::base::Handle<mi::mdl::ILambda_function> lambda(
            m_compiler->create_lambda_function(lec));

        if (fname != NULL) {
            lambda->set_name(fname);
        } else {
            // if not set, use the mangled name of the function definition
            lambda->set_name(fdef->get_mangled_name(m_db_transaction).c_str());
        }

        size_t parameter_count =  fdef->get_parameter_count();
        std::vector<mi::mdl::DAG_call::Call_argument> mdl_arguments(parameter_count);

        MDL::Mdl_dag_builder builder(
            m_db_transaction, lambda.get(), /*compiled_material=*/NULL);

        mi::mdl::IType_factory *tf = lambda->get_type_factory();
        MDL::DETAIL::Type_binder type_binder(tf);

        for (size_t i = 0; i < parameter_count; ++i) {
            mi::mdl::IType const *type = fdef->get_core_parameter_type(m_db_transaction, i);

            if (mi::mdl::IType_array const *arr_type = mi::mdl::as<mi::mdl::IType_array>(type)) {
                if (!arr_type->is_immediate_sized()) {
                    // compilation of deferred size arrays not supported yet
                    error(-2, "Arguments of deferred size array type unsupported");
                    return NULL;
                }
            }

            type = tf->import(type->skip_type_alias());

            lambda->add_parameter(type, fdef->get_parameter_name(i));
            mdl_arguments[i].param_name = fdef->get_parameter_name(i);
            mdl_arguments[i].arg        = lambda->create_parameter(type, int(i));
        }

        std::string decoded_mdl_name = MDL::decode_name_with_signature(fdef->get_mdl_name());
        mi::mdl::DAG_node const *body = lambda->create_call(
            decoded_mdl_name.c_str(),
            fdef->get_core_semantic(),
            parameter_count > 0 ? &mdl_arguments[0] : NULL,
            parameter_count,
            fdef->get_core_return_type(m_db_transaction));

        lambda->set_body(body);

        m_error = 0;
        lambda->retain();

        return lambda.get();
    }

    /// Build a distribution function from a material df (for example surface.scattering).
    mi::mdl::IDistribution_function *from_material_df(
        const MDL::Mdl_compiled_material* compiled_material,
        const char* path,
        const char* fname,
        bool include_geometry_normal,
        bool allow_double_expr_lambdas)
    {
        // get the field corresponding to path
        mi::base::Handle<const MDL::IExpression> field(
            compiled_material->lookup_sub_expression(m_db_transaction, path));

        if (!field) {
            error(-2, "Invalid path (non-existing)");
            return NULL;
        }

        // reject constants if not explicitly enabled
        if (!m_compile_consts && field->get_kind() == MDL::IExpression::EK_CONSTANT) {
            error(-4, "The requested expression is a constant");
            return NULL;
        }

        // ok, we found the attribute to compile, create a lambda function ...
        mi::base::Handle<mi::mdl::IDistribution_function> dist_func(
            m_compiler->create_distribution_function());

        // but first copy the resource map
        copy_resource_map(compiled_material, dist_func);

        mi::base::Handle<mi::mdl::ILambda_function> root_lambda(dist_func->get_root_lambda());
        if (fname) {
            // set the name of the init function
            std::string init_name(fname);
            init_name += "_init";
            root_lambda->set_name(init_name.c_str());
        }

        // copy the resource map to the lambda
        copy_resource_map(compiled_material, root_lambda);

        // reject non-DFs
        mi::base::Handle<const MDL::IType> field_int_type(field->get_type());
        mi::mdl::IType_factory *tf         = root_lambda->get_type_factory();
        mi::mdl::IType const   *field_type = MDL::int_type_to_core_type(field_int_type.get(), *tf);
        field_type = field_type->skip_type_alias();
        if (!mi::mdl::is<mi::mdl::IType_df>(field_type)) {
            error(-5, "Only distribution functions are allowed");
            return NULL;
        }

        // currently only BSDFs and EDFs are supported
        switch (field_type->get_kind()) {
        case mi::mdl::IType::TK_BSDF:
        case mi::mdl::IType::TK_HAIR_BSDF:
        case mi::mdl::IType::TK_EDF:
            break;

        case mi::mdl::IType::TK_VDF:
            error(-9, "VDFs are not supported");
            return NULL;

        default:
            MDL_ASSERT(false);
            return NULL;
        }

        // ... and fill up ...
        MDL::Mdl_dag_builder builder(
            m_db_transaction, root_lambda.get(), compiled_material);

        // add all material parameters to the lambda function
        for (size_t i = 0, n = compiled_material->get_parameter_count(); i < n; ++i) {
            mi::base::Handle<MI::MDL::IValue const> value(
                compiled_material->get_argument(m_db_transaction, i));
            mi::base::Handle<MI::MDL::IType const>  p_type(value->get_type());

            mi::mdl::IType const *tp = MDL::int_type_to_core_type(p_type.get(), *tf);

            size_t idx = root_lambda->add_parameter(tp, compiled_material->get_parameter_name(i));

            /// map the i'th material parameter to this new parameter
            root_lambda->set_parameter_mapping(i, idx);
        }

        mi::base::Handle<const MI::MDL::IExpression_direct_call> mat_body(
            compiled_material->get_body(m_db_transaction));
        DB::Tag tag = mat_body->get_definition(m_db_transaction);
        if (!tag.is_valid()) {
            error(-2, "Invalid expression");
            return NULL;
        }

        DB::Access<MI::MDL::Mdl_function_definition> definition(tag, m_db_transaction);
        mi::mdl::IType const *mat_type = definition->get_core_return_type(m_db_transaction);

        // disable optimizations to ensure, that the expression paths will stay valid
        mi::mdl::Lambda_function *root_lambda_impl =
            mi::mdl::impl_cast<mi::mdl::Lambda_function>(root_lambda.get());
        bool old_opt = root_lambda_impl->enable_opt(false);

        const mi::mdl::DAG_node *material_constructor =
            builder.int_expr_to_core_dag_node(mat_type, mat_body.get());

        mi::mdl::IDistribution_function::Requested_function req_func(path, fname);

        MDL::Mdl_call_resolver resolver(m_db_transaction);
        mi::mdl::IDistribution_function::Error_code ec = dist_func->initialize(
            material_constructor,
            &req_func,
            1,
            include_geometry_normal,
            m_calc_derivatives,
            allow_double_expr_lambdas,
            &resolver);

        // restore optimizations after expression paths have been processed
        root_lambda_impl->enable_opt(old_opt);

        switch (ec) {
        case mi::mdl::IDistribution_function::EC_NONE:
            break;
        case mi::mdl::IDistribution_function::EC_INVALID_PATH:
            error(-2, "Invalid path");
            return NULL;
        case mi::mdl::IDistribution_function::EC_UNSUPPORTED_BSDF:
        case mi::mdl::IDistribution_function::EC_UNSUPPORTED_EDF:
            error(-10, "unsupported distribution function");
            return NULL;
        case mi::mdl::IDistribution_function::EC_UNSUPPORTED_DISTRIBUTION_TYPE:
        case mi::mdl::IDistribution_function::EC_UNSUPPORTED_EXPRESSION_TYPE:
        case mi::mdl::IDistribution_function::EC_INVALID_PARAMETERS:
            MDL_ASSERT(!"Unexpected error.");
            error(-10, "The requested BSDF is not supported, yet");
            return NULL;
        }

        if (fname && dist_func->get_main_function_count() > 0) {
            mi::base::Handle<mi::mdl::ILambda_function> main_func(dist_func->get_main_function(0));
            main_func->set_name(fname);
        }

        m_error = 0;
        dist_func->retain();
        return dist_func.get();
    }

    /// Build a lambda function from a material sub-expression.
    size_t add_sub_expr(
        mi::mdl::ILambda_function        *ilambda,
        MDL::Mdl_compiled_material const *compiled_material,
        char const                       *path)
    {
        mi::mdl::Lambda_function *lambda = mi::mdl::impl_cast<mi::mdl::Lambda_function>(ilambda);
        mi::mdl::ILambda_function::Lambda_execution_context lec =
            mi::mdl::ILambda_function::LEC_CORE;

        if (strcmp(path, "geometry.displacement") == 0) {
            // only this is the displacement function
            lec = mi::mdl::ILambda_function::LEC_DISPLACEMENT;
        }
        if (lec != lambda->get_execution_context()) {
            // we cannot mix expressions with different contexts so give up
            error(-7, "Mixing displacement and non-displacement expression not possible");
            return 0;
        }

        // get the field corresponding to path
        mi::base::Handle<MDL::IExpression const> field(
            compiled_material->lookup_sub_expression(m_db_transaction, path));

        if (!field.is_valid_interface()) {
            error(-2, "Invalid path");
            return 0;
        }

        // reject constants if not explicitly enabled
        if (!m_compile_consts && field->get_kind() == MDL::IExpression::EK_CONSTANT) {
            error(-4, "The requested expression is a constant");
            return 0;
        }

        // reject DFs or resources
        mi::base::Handle<const MDL::IType> field_int_type(field->get_type());
        mi::mdl::IType_factory *tf         = lambda->get_type_factory();
        mi::mdl::IType const   *field_type = MDL::int_type_to_core_type(field_int_type.get(), *tf);
        field_type = field_type->skip_type_alias();
        if (contains_df_type(field_type) || mi::mdl::is<mi::mdl::IType_resource>(field_type)) {
            error(-5, "distribution functions are not allowed");
            return 0;
        }

        // ... and fill up ...
        MDL::Mdl_dag_builder builder(
            m_db_transaction, lambda, compiled_material);
        mi::mdl::DAG_node const *expr = builder.int_expr_to_core_dag_node(field_type, field.get());

        mi::mdl::DAG_node const *body = lambda->get_body();
        if (body != NULL) {
            lambda->store_root_expr(body);
            lambda->set_body(NULL);
        }

        size_t idx = lambda->store_root_expr(expr);

        m_error = 0;
        return idx;
    }

    /// Enumerates all resources in the arguments of the compiled material.
    ///
    /// \param lambda             The lambda function used for converting the values.
    /// \param compiled_material  The compiled material.
    /// \param enumerator         The enumerator collecting the resources.
    void enumerate_resource_arguments(
        mi::mdl::ILambda_function *lambda,
        MDL::Mdl_compiled_material const *compiled_material,
        Function_enumerator &enumerator)
    {
        mi::mdl::IType_factory *type_factory = lambda->get_type_factory();
        mi::mdl::IValue_factory *value_factory = lambda->get_value_factory();

        for (mi::Size i = 0, n = compiled_material->get_parameter_count(); i < n; ++i) {
            mi::base::Handle<MI::MDL::IValue const> arg_val(
                compiled_material->get_argument(m_db_transaction, i));

            // skip any non-resources
            MI::MDL::IValue::Kind kind = arg_val->get_kind();
            if (kind != MI::MDL::IValue::VK_TEXTURE &&
                kind != MI::MDL::IValue::VK_LIGHT_PROFILE &&
                kind != MI::MDL::IValue::VK_BSDF_MEASUREMENT)
            {
                continue;
            }

            mi::base::Handle<MI::MDL::IType const> p_type(arg_val->get_type());
            mi::mdl::IType const *tp = MDL::int_type_to_core_type(p_type.get(), *type_factory);

            mi::mdl::IValue const *mdl_value = int_value_to_core_value(
                m_db_transaction, value_factory, tp, arg_val.get());
            switch (kind) {
            case MI::MDL::IValue::VK_TEXTURE:
                enumerator.texture(mdl_value, 0);  // texture usage not used by function enumerator
                break;
            case MI::MDL::IValue::VK_LIGHT_PROFILE:
                enumerator.light_profile(mdl_value);
                break;
            case MI::MDL::IValue::VK_BSDF_MEASUREMENT:
                enumerator.bsdf_measurement(mdl_value);
                break;
            default:
                MDL_ASSERT(!"unexpected kind");
                break;
            }
        }
    }

    /// Copy the resource map from the compiled material to a lambda like object.
    ///
    /// \param compiled_material  the compiled material
    /// \param lambda             the destination lambda
    template<typename T>
    static void copy_resource_map(
        MDL::Mdl_compiled_material const *compiled_material,
        T                                 lambda)
    {
        for (size_t i = 0, n = compiled_material->get_resources_count(); i < n; ++i) {
            MI::MDL::Resource_tag_tuple const *e = compiled_material->get_resource_tag_tuple(i);
            lambda->set_resource_tag(
                e->m_kind, e->m_mdl_file_path.c_str(), e->m_selector.c_str(), e->m_tag.get_uint());
        }
    }

private:
    /// Generate an error.
    void error(int code, std::string const &message) {
        m_error_string = message;
        m_error        = code;
    }

    /// Check if the given type contains a *df type.
    ///
    /// \param type  the type to check
    static bool contains_df_type(mi::mdl::IType const *type)
    {
        type = type->skip_type_alias();
        if (mi::mdl::is<mi::mdl::IType_df>(type))
            return true;
        if (mi::mdl::IType_compound const *c_type = mi::mdl::as<mi::mdl::IType_compound>(type)) {
            for (int i = 0, n = c_type->get_compound_size(); i < n; ++i) {
                mi::mdl::IType const *e_tp = c_type->get_compound_type(i);

                if (contains_df_type(e_tp))
                    return true;
            }
        }
        return false;
    }

    /// The MDL compiler.
    mi::base::Handle<mi::mdl::IMDL> m_compiler;

    /// The used transaction.
    DB::Transaction *m_db_transaction;

    /// Reported error code, if any
    mi::Sint32 m_error;

    /// Reported error string, if any
    std::string m_error_string;

    /// True, if constants should be compiled (else return error -4 instead)
    bool m_compile_consts;

    /// If true, derivatives should be calculated.
    bool m_calc_derivatives;
};

} // anonymous

// --------------------- Target code register --------------------

/// A simple name register for target code.
class Target_code_register : public IResource_register {
public:
    struct Res_entry {
        Res_entry(
            size_t                                   index,
            std::string const                        &name,
            std::string const                        &owner_module,
            bool                                     is_resolved,
            bool                                     is_body_resource)
        : m_index(index)
        , m_name(name)
        , m_owner_module(owner_module)
        , m_is_resolved(is_resolved)
        , m_is_body_resource(is_body_resource || name.empty()) // empty resource is body
        {
        }

        size_t       m_index;
        std::string  m_name;
        std::string  m_owner_module;
        bool         m_is_resolved;
        bool         m_is_body_resource;
    };


    struct Texture_entry : public Res_entry {
        Texture_entry(
            size_t                                     index,
            std::string const                          &name,
            std::string const                          &owner_module,
            bool                                       is_resolved,
            bool                                       is_body_resource,
            float                                      gamma,
            std::string const                          &selector,
            mi::neuraylib::ITarget_code::Texture_shape type,
            mi::mdl::IValue_texture::Bsdf_data_kind    df_data_kind)
        : Res_entry(index, name, owner_module, is_resolved, is_body_resource)
        , m_gamma(gamma)
        , m_selector(selector)
        , m_type(type)
        , m_df_data_kind(df_data_kind)
        {
        }

        float                                       m_gamma;
        std::string                                 m_selector;
        mi::neuraylib::ITarget_code::Texture_shape  m_type;
        mi::mdl::IValue_texture::Bsdf_data_kind     m_df_data_kind;
    };

    typedef std::vector<Texture_entry> Texture_resource_table;

    typedef std::vector<Res_entry> Resource_table;

public:
    /// Constructor.
    Target_code_register()
    : m_texture_table()
    , m_body_texture_count(0)
    , m_light_profile_table()
    , m_body_light_profile_count(0)
    , m_bsdf_measurement_table()
    , m_body_bsdf_measurement_count(0)
    , m_in_argument_mode(false)
    {
    }

    /// Destructor.
    virtual ~Target_code_register() = default;

    /// Register a texture index.
    ///
    /// \param index        the texture index
    /// \param is_resolved  true, if this texture has been resolved and exists in the Neuray DB
    /// \param name         the DB name of the texture at this index, if the texture has been
    ///                     resolved, the unresolved mdl url of the texture otherwise
    /// \param owner_module the owner module name of the texture
    /// \param gamma        the gamma value of the texture
    /// \param selector     the selector of the texture
    /// \param type         the type of the texture
    void register_texture(
        size_t                                     index,
        bool                                       is_resolved,
        char const                                 *name,
        char const                                 *owner_module,
        float                                      gamma,
        char const                                 *selector,
        mi::neuraylib::ITarget_code::Texture_shape type,
        mi::mdl::IValue_texture::Bsdf_data_kind    df_data_kind) override
    {
        m_texture_table.push_back(
            Texture_entry(
                index,
                name,
                owner_module,
                is_resolved,
                !m_in_argument_mode,
                gamma,
                selector,
                type,
                df_data_kind));

        // Is a body resource and body resources count has not been marked as invalid?
        if (!m_in_argument_mode && m_body_texture_count != ~0ull)
            ++m_body_texture_count;
    }

    /// Updates an already registered resource when it is encountered again.
    ///
    /// \param index        the texture index
    void update_texture(size_t index) override
    {
        for (auto& it : m_texture_table)
        {
            if (it.m_index != index)
                continue;

            it.m_is_body_resource |= !m_in_argument_mode; // appearance in body sets true
            break;
        }
    }

    /// Return the number of texture resources.
    size_t get_texture_count() const override
    {
        return m_texture_table.size();
    }

    /// Returns the number of texture resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid texture).
    ///
    /// \return           The body texture count or \c ~0ull, if the value is invalid due to
    ///                   more than one call to a link unit add function.
    size_t get_body_texture_count() const override
    {
        return m_body_texture_count;
    }

    /// Register a light profile.
    ///
    /// \param index  the light profile index
    /// \param is_resolved  true, if this resource has been resolved and exists in the Neuray DB
    /// \param name         the DB name of this index, if this resource has been resolved,
    ///                     the unresolved mdl url otherwise
    /// \param owner_module the owner module name of the resource
    void register_light_profile(
        size_t                                     index,
        bool                                       is_resolved,
        char const                                 *name,
        char const                                 *owner_module) override
    {
        m_light_profile_table.push_back(
            Res_entry(index, name, owner_module, is_resolved, !m_in_argument_mode));

        // Is a body resource and body resources count has not been marked as invalid?
        if (!m_in_argument_mode && m_body_light_profile_count != ~0ull)
            ++m_body_light_profile_count;
    }

    /// Updates an already registered resource when it is encountered again.
    ///
    /// \param index        the texture index
    void update_light_profile(size_t index) override
    {
        for (auto& it : m_light_profile_table)
        {
            if (it.m_index != index)
                continue;

            it.m_is_body_resource |= !m_in_argument_mode; // appearance in body sets true
            break;
        }
    }

    /// Return the number of light profile resources.
    size_t get_light_profile_count() const override
    {
        return m_light_profile_table.size();
    }

    /// Returns the number of light profile resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid light profile).
    ///
    /// \return           The body light profile count or \c ~0ull, if the value is invalid due to
    ///                   more than one call to a link unit add function.
    size_t get_body_light_profile_count() const override
    {
        return m_body_light_profile_count;
    }

    /// Register a BSDF measurement.
    ///
    /// \param index        the BSDF measurement index
    /// \param is_resolved  true, if this resource has been resolved and exists in the Neuray DB
    /// \param name         the DB name of this index, if this resource has been resolved,
    ///                     the unresolved mdl url otherwise
    /// \param owner_module the owner module name of the resource
    void register_bsdf_measurement(
        size_t                                     index,
        bool                                       is_resolved,
        char const                                 *name,
        char const                                 *owner_module) override
    {
        m_bsdf_measurement_table.push_back(
            Res_entry(index, name, owner_module, is_resolved, !m_in_argument_mode));

        // Is a body resource and body resources count has not been marked as invalid?
        if (!m_in_argument_mode && m_body_bsdf_measurement_count != ~0ull)
            ++m_body_bsdf_measurement_count;

    }

    /// Updates an already registered resource when it is encountered again.
    ///
    /// \param index        the texture index
    void update_bsdf_measurement(size_t index) override
    {
        for (auto& it : m_bsdf_measurement_table)
        {
            if (it.m_index != index)
                continue;

            it.m_is_body_resource |= !m_in_argument_mode; // appearance in body sets true
            break;
        }
    }

    /// Return the number of BSDF measurement resources.
    size_t get_bsdf_measurement_count() const override
    {
        return m_bsdf_measurement_table.size();
    }

    /// Returns the number of BSDF measurement resources coming from the body of expressions
    /// (not solely from material arguments). These will be necessary regardless of the chosen
    /// material arguments and start at index \c 0 (including the invalid BSDF measurement).
    ///
    /// \return           The body BSDF measurement count or \c ~0ull, if the value is invalid due to
    ///                   more than one call to a link unit add function.
    size_t get_body_bsdf_measurement_count() const override
    {
        return m_body_bsdf_measurement_count;
    }

    /// Retrieve the texture resource table.
    Texture_resource_table const &get_texture_table() const { return m_texture_table; }

    /// Retrieve the light profile resource table.
    Resource_table const &get_light_profile_table() const { return m_light_profile_table; }

    /// Retrieve the texture resource table.
    Resource_table const &get_bsdf_measurement_table() const { return m_bsdf_measurement_table; }

    /// Set whether the next resources will come from arguments.
    void set_in_argument_mode(bool in_argument_mode)
    {
        // going out of argument mode again?
        if (m_in_argument_mode && !in_argument_mode) {
            // if there have already been non-body registered resources,
            // the body counts will become invalid
            if (m_texture_table.size() > m_body_texture_count)
                m_body_texture_count = ~0ull;

            if (m_light_profile_table.size() > m_body_light_profile_count)
                m_body_light_profile_count = ~0ull;

            if (m_bsdf_measurement_table.size() > m_body_bsdf_measurement_count)
                m_body_bsdf_measurement_count= ~0ull;
        }

        m_in_argument_mode = in_argument_mode;
    }

private:
    /// The texture resource table.
    Texture_resource_table m_texture_table;

    /// The number of textures coming from the body of expressions
    /// (not only from material arguments). ~0ull if invalid.
    size_t m_body_texture_count;

    /// The light profile resource table.
    Resource_table m_light_profile_table;

    /// The number of light profiles coming from the body of expressions
    /// (not only from material arguments). ~0ull if invalid.
    size_t m_body_light_profile_count;

    /// The BSDF measurement resource table.
    Resource_table m_bsdf_measurement_table;

    /// The number of BSDF measurements coming from the body of expressions
    /// (not only from material arguments). ~0ull if invalid.
    size_t m_body_bsdf_measurement_count;

    /// True, if all following resources come from material arguments.
    bool m_in_argument_mode;
};

/// Copy Data from the register facility to the target code.
static void fill_resource_tables(Target_code_register const &tc_reg, Target_code *tc)
{
    typedef Target_code_register::Texture_resource_table TRT;

    TRT const &txt_table = tc_reg.get_texture_table();

    for (const auto & entry : txt_table) {
        tc->add_texture_index(
            entry.m_index,
            entry.m_is_resolved ? entry.m_name : "",
            !entry.m_is_resolved ? entry.m_name : "",
            entry.m_gamma,
            entry.m_selector,
            entry.m_type,
            entry.m_df_data_kind,
            entry.m_is_body_resource);
    }

    typedef Target_code_register::Resource_table RT;

    RT const &lp_table = tc_reg.get_light_profile_table();

    for (const auto & entry : lp_table) {
        tc->add_light_profile_index(
            entry.m_index,
            entry.m_is_resolved ? entry.m_name : "",
            !entry.m_is_resolved ? entry.m_name : "",
            entry.m_is_body_resource);
    }

    RT const &bm_table = tc_reg.get_bsdf_measurement_table();

    for (const auto & entry : bm_table) {
        tc->add_bsdf_measurement_index(
            entry.m_index,
            entry.m_is_resolved ? entry.m_name : "",
            !entry.m_is_resolved ? entry.m_name : "",
            entry.m_is_body_resource);
    }

    tc->set_body_resource_counts(
        tc_reg.get_body_texture_count(),
        tc_reg.get_body_light_profile_count(),
        tc_reg.get_body_bsdf_measurement_count());
}

// --------------------- Target argument block class --------------------

Target_argument_block::Target_argument_block(mi::Size arg_block_size)
: m_size(arg_block_size)
, m_data(new char[arg_block_size])
{
    memset(m_data, 0, m_size);
}

Target_argument_block::~Target_argument_block()
{
    delete[] m_data;
    m_data = NULL;
}

char const *Target_argument_block::get_data() const
{
    return m_data;
}

char *Target_argument_block::get_data()
{
    return m_data;
}

mi::Size Target_argument_block::get_size() const
{
    return m_size;
}

mi::neuraylib::ITarget_argument_block *Target_argument_block::clone() const
{
    Target_argument_block *cloned_block = new Target_argument_block(m_size);
    memcpy(cloned_block->get_data(), m_data, m_size);
    return cloned_block;
}

// ---------------------- Target value layout class ---------------------

// Constructor.
Target_value_layout::Target_value_layout(
    mi::mdl::IGenerated_code_value_layout const *layout,
    bool string_ids)
: m_layout(layout, mi::base::DUP_INTERFACE)
, m_strings_mapped_to_ids(string_ids)
{
}

// Get the size of the target argument block.
mi::Size Target_value_layout::get_size() const
{
    if (m_layout == NULL)
        return 0;
    return m_layout->get_size();
}

// Get the number of arguments / elements at the given layout state.
mi::Size Target_value_layout::get_num_elements(
    mi::neuraylib::Target_value_layout_state state) const
{
    if (m_layout == NULL)
        return ~mi::Size(0);
    return m_layout->get_num_elements(
        mi::mdl::IGenerated_code_value_layout::State(state.m_state_offs, state.m_data_offs));
}

// Get the offset, the size and the kind of the argument / element inside the argument
// block at the given layout state.
mi::Size Target_value_layout::get_layout(
    mi::neuraylib::IValue::Kind              &kind,
    mi::Size                                 &arg_size,
    mi::neuraylib::Target_value_layout_state state) const
{
    if (m_layout == NULL) {
        arg_size = 0;
        kind = mi::neuraylib::IValue::VK_INVALID_DF;
        return ~mi::Size(0);
    }

    mi::mdl::IValue::Kind mdl_kind = mi::mdl::IValue::VK_BAD;
    size_t as = arg_size;
    mi::Size offset = m_layout->get_layout(
        mdl_kind,
        as,
        mi::mdl::IGenerated_code_value_layout::State(state.m_state_offs, state.m_data_offs));
    arg_size = as;

    // translate from MDL value kinds to Neuray value kinds
    switch (mdl_kind) {
        #define MAP_KIND(from, to) \
            case mi::mdl::IValue::from: kind = mi::neuraylib::IValue::to; break

        MAP_KIND(VK_BAD,              VK_INVALID_DF);
        MAP_KIND(VK_BOOL,             VK_BOOL);
        MAP_KIND(VK_INT,              VK_INT);
        MAP_KIND(VK_ENUM,             VK_ENUM);
        MAP_KIND(VK_FLOAT,            VK_FLOAT);
        MAP_KIND(VK_DOUBLE,           VK_DOUBLE);
        MAP_KIND(VK_STRING,           VK_STRING);
        MAP_KIND(VK_VECTOR,           VK_VECTOR);
        MAP_KIND(VK_MATRIX,           VK_MATRIX);
        MAP_KIND(VK_ARRAY,            VK_ARRAY);
        MAP_KIND(VK_RGB_COLOR,        VK_COLOR);
        MAP_KIND(VK_STRUCT,           VK_STRUCT);
        MAP_KIND(VK_INVALID_REF,      VK_INVALID_DF);
        MAP_KIND(VK_TEXTURE,          VK_TEXTURE);
        MAP_KIND(VK_LIGHT_PROFILE,    VK_LIGHT_PROFILE);
        MAP_KIND(VK_BSDF_MEASUREMENT, VK_BSDF_MEASUREMENT);

        #undef MAP_KIND
    }

    return offset;
}

// Get the layout state for the i'th argument / element inside the argument value block
// at the given layout state.
mi::neuraylib::Target_value_layout_state Target_value_layout::get_nested_state(
    mi::Size                                 i,
    mi::neuraylib::Target_value_layout_state state) const
{
    if (m_layout == NULL)
        return mi::neuraylib::Target_value_layout_state(~mi::Uint32(0), ~mi::Uint32(0));

    mi::mdl::IGenerated_code_value_layout::State mdl_state =
        m_layout->get_nested_state(
            i,
            mi::mdl::IGenerated_code_value_layout::State(state.m_state_offs, state.m_data_offs));

    return mi::neuraylib::Target_value_layout_state(
        mdl_state.m_state_offs, mdl_state.m_data_offs);
}

// Set the value inside the given block at the given layout state.
mi::Sint32 Target_value_layout::set_value(
    char                                     *block,
    mi::neuraylib::IValue const              *value,
    mi::neuraylib::ITarget_resource_callback *resource_callback,
    mi::neuraylib::Target_value_layout_state state) const
{
    if (block == NULL || value == NULL || resource_callback == NULL)
        return -1;

    mi::neuraylib::IValue::Kind kind;
    mi::Size arg_size;
    mi::Size offs = get_layout(kind, arg_size, state);
    if (value->get_kind() != kind)
        return -3;

    switch (kind) {
        case mi::neuraylib::IValue::VK_BOOL:
            *reinterpret_cast<bool *>(block + offs) =
                static_cast<mi::neuraylib::IValue_bool const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_INT:
            *reinterpret_cast<mi::Sint32 *>(block + offs) =
                static_cast<mi::neuraylib::IValue_int const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_ENUM:
            *reinterpret_cast<mi::Sint32 *>(block + offs) =
                static_cast<mi::neuraylib::IValue_enum const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_FLOAT:
            *reinterpret_cast<mi::Float32 *>(block + offs) =
                static_cast<mi::neuraylib::IValue_float const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_DOUBLE:
            *reinterpret_cast<mi::Float64 *>(block + offs) =
                static_cast<mi::neuraylib::IValue_double const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_STRING:
        {
            if (m_strings_mapped_to_ids)
            {
                mi::Uint32 id =
                    resource_callback != NULL ?
                    resource_callback->get_string_index(
                        static_cast<mi::neuraylib::IValue_string const *>(value)) :
                    0u;
                *reinterpret_cast<mi::Uint32 *>(block + offs) = id;
            }
            else
            {
                // unmapped string are not supported
                *reinterpret_cast<char **>(block + offs) = NULL;
            }
            return 0;
        }

        case mi::neuraylib::IValue::VK_VECTOR:
        case mi::neuraylib::IValue::VK_MATRIX:
        case mi::neuraylib::IValue::VK_ARRAY:
        case mi::neuraylib::IValue::VK_COLOR:
        case mi::neuraylib::IValue::VK_STRUCT:
        {
            mi::neuraylib::IValue_compound const *comp_val =
                static_cast<mi::neuraylib::IValue_compound const *>(value);
            mi::Size num = get_num_elements(state);
            if (comp_val->get_size() != num) return -4;

            // Set all nested values
            for (mi::Size i = 0; i < num; ++i) {
                mi::base::Handle<const mi::neuraylib::IValue> sub_val(comp_val->get_value(i));
                mi::Sint32 err = set_value(
                    block,
                    sub_val.get(),
                    resource_callback,
                    get_nested_state(i, state));
                if (err != 0)
                    return err;
            }
            return 0;
        }

        case mi::neuraylib::IValue::VK_TEXTURE:
        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
        {
            mi::Uint32 index = resource_callback->get_resource_index(
                static_cast<mi::neuraylib::IValue_resource const *>(value));
            *reinterpret_cast<mi::Uint32 *>(block + offs) = index;
            return 0;
        }

        case mi::neuraylib::IValue::VK_INVALID_DF:
        case mi::neuraylib::IValue::VK_FORCE_32_BIT:
        {
            ASSERT(M_BACKENDS, !"unexpected value type");
            return -5;
        }
    }
    ASSERT(M_BACKENDS, !"unsupported value type");
    return -5;
}

// Set the value inside the given block at the given layout state.
mi::Sint32 Target_value_layout::set_value(
    char                                     *block,
    MI::MDL::IValue const                    *value,
    ITarget_resource_callback_internal       *resource_callback,
    mi::neuraylib::Target_value_layout_state state) const
{
    if (block == NULL || value == NULL || resource_callback == NULL)
        return -1;

    mi::neuraylib::IValue::Kind kind;
    mi::Size arg_size;
    mi::Size offs = get_layout(kind, arg_size, state);

    // MI::MDL::IValue::Kind is identical to mi::neuraylib::IValue::Kind so just cast to compare.
    if (mi::neuraylib::IValue::Kind(value->get_kind()) != kind)
        return -3;

    switch (kind) {
        case mi::neuraylib::IValue::VK_BOOL:
            *reinterpret_cast<bool *>(block + offs) =
                static_cast<MI::MDL::IValue_bool const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_INT:
            *reinterpret_cast<mi::Sint32 *>(block + offs) =
                static_cast<MI::MDL::IValue_int const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_ENUM:
            *reinterpret_cast<mi::Sint32 *>(block + offs) =
                static_cast<MI::MDL::IValue_enum const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_FLOAT:
            *reinterpret_cast<mi::Float32 *>(block + offs) =
                static_cast<MI::MDL::IValue_float const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_DOUBLE:
            *reinterpret_cast<mi::Float64 *>(block + offs) =
                static_cast<MI::MDL::IValue_double const *>(value)->get_value();
            return 0;

        case mi::neuraylib::IValue::VK_STRING:
        {
            if (m_strings_mapped_to_ids) {
                mi::Uint32 id =
                    resource_callback != NULL ?
                    resource_callback->get_string_index(
                        static_cast<MI::MDL::IValue_string const *>(value)) :
                    0u;
                *reinterpret_cast<mi::Uint32 *>(block + offs) = id;
            } else {
                // unmapped string are not supported
                *reinterpret_cast<char **>(block + offs) = NULL;
            }
            return 0;
        }

        case mi::neuraylib::IValue::VK_VECTOR:
        case mi::neuraylib::IValue::VK_MATRIX:
        case mi::neuraylib::IValue::VK_ARRAY:
        case mi::neuraylib::IValue::VK_COLOR:
        case mi::neuraylib::IValue::VK_STRUCT:
        {
            MI::MDL::IValue_compound const *comp_val =
                static_cast<MI::MDL::IValue_compound const *>(value);
            mi::Size num = get_num_elements(state);
            if (comp_val->get_size() != num)
                return -4;

            // Set all nested values
            for (mi::Size i = 0; i < num; ++i) {
                mi::base::Handle<MI::MDL::IValue const> sub_val(comp_val->get_value(i));
                mi::Sint32 err = set_value(
                    block,
                    sub_val.get(),
                    resource_callback,
                    get_nested_state(i, state));
                if (err != 0)
                    return err;
            }
            return 0;
        }

        case mi::neuraylib::IValue::VK_TEXTURE:
        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
        {
            mi::Uint32 index = resource_callback->get_resource_index(
                static_cast<MI::MDL::IValue_resource const *>(value));
            *reinterpret_cast<mi::Uint32 *>(block + offs) = index;
            return 0;
        }

        case mi::neuraylib::IValue::VK_INVALID_DF:
        case mi::neuraylib::IValue::VK_FORCE_32_BIT:
        {
            ASSERT(M_BACKENDS, !"unexpected value type");
            return -5;
        }
    }
    ASSERT(M_BACKENDS, !"unsupported value type");
    return -5;
}

// ------------------------- LLVM based link unit -------------------------

static mi::mdl::ICode_generator::Target_language map_target_language(
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind kind)
{
    switch (kind) {
    case mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX:
        return mi::mdl::ICode_generator::TL_PTX;
    case mi::neuraylib::IMdl_backend_api::MB_LLVM_IR:
        return mi::mdl::ICode_generator::TL_LLVM_IR;
    case mi::neuraylib::IMdl_backend_api::MB_GLSL:
        return mi::mdl::ICode_generator::TL_GLSL;
    case mi::neuraylib::IMdl_backend_api::MB_NATIVE:
        return mi::mdl::ICode_generator::TL_NATIVE;
    case mi::neuraylib::IMdl_backend_api::MB_HLSL:
        return mi::mdl::ICode_generator::TL_HLSL;
    default:
        // should not happen
        return mi::mdl::ICode_generator::TL_NATIVE;
    }
}

// Constructor from an LLVM backend.
Link_unit::Link_unit(
    Mdl_llvm_backend       &llvm_be,
    DB::Transaction        *transaction,
    MDL::Execution_context *context)
: m_compiler(llvm_be.get_compiler())
, m_be_kind(llvm_be.get_kind())
, m_unit(llvm_be.create_link_unit(context))
, m_target_code(new Target_code(llvm_be.get_strings_mapped_to_ids(), m_be_kind))
, m_transaction(transaction)
, m_tc_reg(new Target_code_register())
, m_res_index_map()
, m_tex_idx(0)
, m_lp_idx(0)
, m_bm_idx(0)
, m_gen_base_name_suffix_counter(0)
, m_compile_consts(llvm_be.get_compile_consts())
, m_strings_mapped_to_ids(llvm_be.get_strings_mapped_to_ids())
, m_calc_derivatives(llvm_be.get_calc_derivatives())
, m_internal_space(get_context_option<std::string>(context, MDL_CTX_OPTION_INTERNAL_SPACE))
{
}

// Destructor.
Link_unit::~Link_unit()
{
    delete m_tc_reg;
}

/// Conversion helper.
static mi::mdl::ILambda_function::Lambda_execution_context to_lambda_exc(
    Link_unit::Function_execution_context ctx)
{
    switch (ctx) {
    case Link_unit::FEC_ENVIRONMENT:  return mi::mdl::ILambda_function::LEC_ENVIRONMENT;
    case Link_unit::FEC_CORE:         return mi::mdl::ILambda_function::LEC_CORE;
    case Link_unit::FEC_DISPLACEMENT: return mi::mdl::ILambda_function::LEC_DISPLACEMENT;
    }
    MDL_ASSERT(!"Unexpected function execution context");
    return mi::mdl::ILambda_function::LEC_CORE;
}

/// Conversion helper.
static mi::mdl::IGenerated_code_executable::Function_kind  to_function_kind(
    Link_unit::Function_execution_context ctx)
{
    switch (ctx) {
    case Link_unit::FEC_ENVIRONMENT:  return mi::mdl::IGenerated_code_executable::FK_ENVIRONMENT;
    case Link_unit::FEC_CORE:         return mi::mdl::IGenerated_code_executable::FK_LAMBDA;
    case Link_unit::FEC_DISPLACEMENT: return mi::mdl::IGenerated_code_executable::FK_LAMBDA;
    }
    MDL_ASSERT(!"Unexpected function execution context");
    return mi::mdl::IGenerated_code_executable::FK_LAMBDA;
}

// Add an MDL function call as a function to this link unit.
mi::Sint32 Link_unit::add_function(
    MDL::Mdl_function_call const *function_call,
    Function_execution_context   fxc,
    char const                   *fname,
    MDL::Execution_context       *context)
{
    if (function_call == NULL || m_transaction == NULL) {
        MDL::add_error_message(context, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }
    DB::Tag_set tags_seen;
    if (!function_call->is_valid(m_transaction, tags_seen, context)) {
        MDL::add_error_message(context, "Invalid function call.", -1);
        return -1;
    }
    DB::Tag def_tag = function_call->get_function_definition(m_transaction);
    if (!def_tag.is_valid()) {
        MDL::add_error_message(context, "Function does not point to a valid definition.", -1);
        return -1;
    }
    // check internal space configuration
    DB::Access<MDL::Mdl_function_definition> function_definition(def_tag, m_transaction);
    DB::Access<MDL::Mdl_module> module(
        function_definition->get_module(m_transaction), m_transaction);
    mi::base::Handle<mi::mdl::IGenerated_code_dag const> code_dag(module->get_code_dag());

    if (m_internal_space != code_dag->get_internal_space()) {
        MDL::add_error_message(context, "Functions and materials compiled with different "
            "internal_space configurations cannot be mixed.", -1);
        return -1;
    }

    Lambda_builder builder(
        m_compiler.get(),
        m_transaction,
        m_compile_consts,
        m_calc_derivatives);

    mi::base::Handle<mi::mdl::ILambda_function> lambda(
        builder.from_call(function_call, fname, to_lambda_exc(fxc)));
    if (!lambda.is_valid_interface()) {
        MDL::add_error_message(context,
            builder.get_error_string(), builder.get_error_code());
        return -1;
    }

    MDL::Mdl_call_resolver resolver(m_transaction);

    // enumerate resources ...
    bool resolve_resources = get_context_option<bool>(context, MDL_CTX_OPTION_RESOLVE_RESOURCES);
    m_tc_reg->set_in_argument_mode(false);
    Function_enumerator enumerator(
        *m_tc_reg,
        lambda.get(),
        m_transaction,
        m_tex_idx,
        m_lp_idx,
        m_bm_idx,
        m_res_index_map,
        !resolve_resources,
        resolve_resources);
    lambda->enumerate_resources(resolver, enumerator, lambda->get_body());
    if (!resolve_resources)
        lambda->set_has_resource_attributes(false);

    // ... and add it to the compilation unit
    mi::base::Handle<mi::mdl::IGenerated_code_executable> code;

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(m_transaction, mdlc_module->get_module_wait_queue(), {});

    size_t arg_block_index = ~0;
    bool res = m_unit->add(
        lambda.get(),
        &module_cache,
        &resolver,
        to_function_kind(fxc),
        &arg_block_index,
        NULL);
    if (!res) {
        MDL::add_error_message(context,
            "The backend failed to compile the function.", -3);
        return -1;
    }
    ASSERT(M_BACKENDS, arg_block_index == size_t(~0) &&
        "Functions should not have captured arguments");

    return 0;
}

mi::Sint32 Link_unit::add_material_expression(
    MDL::Mdl_compiled_material const *compiled_material,
    char const                       *path,
    char const                       *base_fname,
    MDL::Execution_context           *context)
{
    if (!compiled_material->is_valid(m_transaction, context)) {
        if (context)
            MDL::add_error_message(context,
                "The compiled material is invalid.", -1);
        return -1;
    }
    mi::neuraylib::Target_function_description desc(path, base_fname);
    mi::Sint32 result = add_material(compiled_material, &desc, 1, context);
    // store legacy error code
    if(context)
        context->set_result((desc.return_code != ~0 && desc.return_code < -9)
            ? desc.return_code / 10
            : desc.return_code);
    return result;
}

mi::Sint32 Link_unit::add_material_df(
    MDL::Mdl_compiled_material const *compiled_material,
    char const *path,
    char const *base_fname,
    MDL::Execution_context* context)
{
    if (!compiled_material->is_valid(m_transaction, context)) {
        if (context)
            MDL::add_error_message(context,
                "The compiled material is invalid.", -1);
        return -1;
    }
    mi::neuraylib::Target_function_description desc(path, base_fname);
    mi::Sint32 result = add_material(compiled_material, &desc, 1, context);
    // store legacy error code
    if (context)
        context->set_result((desc.return_code != ~0 && desc.return_code < -99)
            ? desc.return_code / 100
            : desc.return_code);
    return result;
}

mi::Sint32 Link_unit::add_material_single_init(
    MDL::Mdl_compiled_material const             *compiled_material,
    mi::neuraylib::Target_function_description   *function_descriptions,
    mi::Size                                      description_count,
    MDL::Execution_context                       *context)
{
    MDL_ASSERT(
        description_count > 0 &&
        function_descriptions[0].path != NULL &&
        strcmp(function_descriptions[0].path, "init") == 0);

    if (compiled_material == NULL) {
        MDL::add_error_message(context, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }
    if (!compiled_material->is_valid(m_transaction, context)) {
        if (context)
            MDL::add_error_message(context,
                "The compiled material is invalid.", -1);
        return -1;
    }

    mi::base::Handle<MDL::IExpression_factory> ef(MDL::get_expression_factory());
    MDL::Mdl_call_resolver resolver(m_transaction);

    bool resolve_resources =
        get_context_option<bool>(context, MDL_CTX_OPTION_RESOLVE_RESOURCES);
    bool include_geometry_normal =
        get_context_option<bool>(context, MDL_CTX_OPTION_INCLUDE_GEO_NORMAL);

    // check internal space configuration
    if (m_internal_space != compiled_material->get_internal_space()) {
        MDL::add_error_message(context, "Materials compiled with different internal_space "
            "configurations cannot be mixed.", -1);
        return -1;
    }

    // increment once for each add_material invocation
    m_gen_base_name_suffix_counter++;

    // TODO: m_compile_consts?

    // first generate / normalize all function names to be able to take pointers afterwards
    std::vector<std::string> base_fname_list;
    for (mi::Size i = 0; i < description_count; ++i) {
        if (function_descriptions[i].path == NULL) {
            function_descriptions[i].return_code = MDL::add_error_message(
                context,
                "Invalid parameters (NULL pointer) for function at index " + std::to_string(i),
                -1);
            return -1;
        }

        std::string base_fname;

        if (function_descriptions[i].base_fname && function_descriptions[i].base_fname[0]) {
            base_fname = function_descriptions[i].base_fname;
        } else {
            std::stringstream sstr;
            sstr << "lambda_" << m_gen_base_name_suffix_counter
                 << "__" << function_descriptions[i].path;
            base_fname = sstr.str();
        }

        std::replace(base_fname.begin(), base_fname.end(), '.', '_');
        base_fname_list.push_back(base_fname);
    }

    std::vector<mi::mdl::IDistribution_function::Requested_function> func_list;
    for (mi::Size i = 1; i < description_count; ++i) {
        func_list.push_back(
            mi::mdl::IDistribution_function::Requested_function(
                function_descriptions[i].path,
                base_fname_list[i].c_str()));
    }

    mi::base::Handle<mi::mdl::IDistribution_function> dist_func(
        m_compiler->create_distribution_function());

    // copy the resource map
    Lambda_builder::copy_resource_map(compiled_material, dist_func);

    // set init function name
    mi::base::Handle<mi::mdl::ILambda_function> root_lambda(dist_func->get_root_lambda());
    root_lambda->set_name(base_fname_list[0].c_str());

    // copy the resource map to the lambda
    Lambda_builder::copy_resource_map(compiled_material, root_lambda);

    // add all material parameters to the lambda function
    for (size_t i = 0, n = compiled_material->get_parameter_count(); i < n; ++i) {
        mi::base::Handle<MI::MDL::IValue const> value(
            compiled_material->get_argument(m_transaction, i));
        mi::base::Handle<MI::MDL::IType const>  p_type(value->get_type());

        mi::mdl::IType const *tp
            = MDL::int_type_to_core_type(p_type.get(), *root_lambda->get_type_factory());

        size_t idx = root_lambda->add_parameter(tp, compiled_material->get_parameter_name(i));

        // map the i'th material parameter to this new parameter
        root_lambda->set_parameter_mapping(i, idx);
    }

    // get DAG node for material constructor
    mi::base::Handle<const MI::MDL::IExpression_direct_call> mat_body(
        compiled_material->get_body(m_transaction));
    DB::Tag tag = mat_body->get_definition(m_transaction);
    if (!tag.is_valid()) {
        if (context)
            MDL::add_error_message(context, "The material constructor is invalid.", -1);
        return -1;
    }

    DB::Access<MI::MDL::Mdl_function_definition> definition(tag, m_transaction);
    mi::mdl::IType const *mat_type = definition->get_core_return_type(m_transaction);

    // disable optimizations to ensure, that the expression paths will stay valid
    mi::mdl::Lambda_function *root_lambda_impl =
        mi::mdl::impl_cast<mi::mdl::Lambda_function>(root_lambda.get());
    bool old_opt = root_lambda_impl->enable_opt(false);

    MDL::Mdl_dag_builder builder(
        m_transaction, root_lambda.get(), compiled_material);

    mi::mdl::DAG_node const *material_constructor =
        builder.int_expr_to_core_dag_node(mat_type, mat_body.get());

    // initialize distribution function with the list of requested functions,
    // selecting multiply used expressions for expression lambdas and
    // rewriting the graphs to use them
    // Note: We currently don't support storing expression lambdas of type double
    //       in GLSL/HLSL, so disable it.
    mi::mdl::IDistribution_function::Error_code ec = dist_func->initialize(
        material_constructor,
        func_list.data(),
        func_list.size(),
        include_geometry_normal,
        m_calc_derivatives,
        /*allow_double_expr_lambdas=*/!target_is_structured_language(),
        &resolver);

    // restore optimizations after expression paths have been processed
    root_lambda_impl->enable_opt(old_opt);

    if (ec != mi::mdl::IDistribution_function::EC_NONE) {
        if (context)
            MDL::add_error_message(context, "Error initializing material function group.", -1);
        function_descriptions[0].return_code = -1;
        for (size_t i = 1; i < description_count; ++i) {
            switch (func_list[i - 1].error_code) {
            case mi::mdl::IDistribution_function::EC_NONE:
                function_descriptions[i].return_code = 0;
                break;
            case mi::mdl::IDistribution_function::EC_INVALID_PARAMETERS:
                MDL::add_error_message(
                    context,
                    "Invalid parameters for function at index " + std::to_string(i) + '.',
                    -1);
                function_descriptions[i].return_code = -1;
                break;
            case mi::mdl::IDistribution_function::EC_INVALID_PATH:
                MDL::add_error_message(
                    context,
                    "Invalid path (non-existing) for function at index " + std::to_string(i) + '.',
                    -2);
                function_descriptions[i].return_code = -2;
                break;
            case mi::mdl::IDistribution_function::EC_UNSUPPORTED_EXPRESSION_TYPE:
                MDL::add_error_message(
                    context,
                    "Unsupported expression type for function at index " + std::to_string(i) + '.',
                    -1000);
                function_descriptions[i].return_code = -1000;
                break;
            case mi::mdl::IDistribution_function::EC_UNSUPPORTED_DISTRIBUTION_TYPE:
            case mi::mdl::IDistribution_function::EC_UNSUPPORTED_BSDF:
            case mi::mdl::IDistribution_function::EC_UNSUPPORTED_EDF:
                MDL::add_error_message(
                    context,
                    "Unsupported distribution type for function at index " +
                        std::to_string(i) + '.',
                    -1000);
                function_descriptions[i].return_code = -1000;
                break;
            }
        }
        return -1;
    }

    // ... enumerate resources: must be done before we compile ...
    //     all resource information will be collected in root_lambda
    m_tc_reg->set_in_argument_mode(false);
    Function_enumerator enumerator(
        *m_tc_reg, root_lambda.get(), m_transaction, m_tex_idx,
        m_lp_idx, m_bm_idx, m_res_index_map,
        !resolve_resources, resolve_resources);

    for (size_t i = 0, n = dist_func->get_main_function_count(); i < n; ++i) {
        mi::base::Handle<mi::mdl::ILambda_function> main_func(
            dist_func->get_main_function(i));

        root_lambda->enumerate_resources(resolver, enumerator, main_func->get_body());
    }

    if (!resolve_resources)
        root_lambda->set_has_resource_attributes(false);

    // ... enumerate resources of expression lambdas
    for (size_t i = 0, n = dist_func->get_expr_lambda_count(); i < n; ++i) {
        mi::base::Handle<mi::mdl::ILambda_function> lambda(
            dist_func->get_expr_lambda(i));

        // also register the resources in lambda itself,
        // so we see whether it accesses resources
        enumerator.set_additional_lambda(lambda.get());

        lambda->enumerate_resources(resolver, enumerator, lambda->get_body());

        // ... optimize expression lambda
        // (for derivatives, optimization already happened while building derivative info,
        // and doing it again may destroy the analysis result)
        if (!m_calc_derivatives) {
            MDL::Call_evaluator<mi::mdl::ILambda_function> call_evaluator(
                lambda.get(),
                m_transaction,
                resolve_resources);

            lambda->optimize(&resolver, &call_evaluator);
        }
    }

    // ... also enumerate resources from arguments after all body resources are processed ...
    if (compiled_material->get_parameter_count() != 0) {
        Lambda_builder builder(
            m_compiler.get(),
            m_transaction,
            m_compile_consts,
            m_calc_derivatives);

        m_tc_reg->set_in_argument_mode(true);
        builder.enumerate_resource_arguments(root_lambda.get(), compiled_material, enumerator);
    }

    // ... and add to link unit
    size_t arg_block_index = ~0;
    std::vector<size_t> main_func_indices(func_list.size() + 1);  // +1 for init function

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(m_transaction, mdlc_module->get_module_wait_queue(), {});

    if (!m_unit->add(
        dist_func.get(),
        &module_cache,
        &resolver,
        &arg_block_index,
        main_func_indices.data(),
        main_func_indices.size()))
    {
        MDL::convert_and_log_messages(m_unit->access_messages(), context);
        MDL::add_error_message(context,
            "The JIT backend failed to compile the function group.", -300);
        function_descriptions[0].return_code = -300;
        return -1;
    }

    // Fill output field for the init function, which is always added first
    function_descriptions[0].function_index = main_func_indices[0];
    function_descriptions[0].distribution_kind = mi::neuraylib::ITarget_code::DK_NONE;

    // Fill output fields for the other main functions
    for (mi::Size i = 1; i < description_count; ++i) {
        function_descriptions[i].function_index = main_func_indices[i];
        mi::base::Handle<mi::mdl::ILambda_function> main_func(
            dist_func->get_main_function(i - 1));
        mi::mdl::DAG_node const *main_node = main_func->get_body();
        mi::mdl::IType const *main_type = main_node->get_type()->skip_type_alias();
        switch (main_type->get_kind()) {
        case mi::mdl::IType::TK_BSDF:
            function_descriptions[i].distribution_kind = mi::neuraylib::ITarget_code::DK_BSDF;
            break;
        case mi::mdl::IType::TK_HAIR_BSDF:
            function_descriptions[i].distribution_kind = mi::neuraylib::ITarget_code::DK_HAIR_BSDF;
            break;
        case mi::mdl::IType::TK_EDF:
            function_descriptions[i].distribution_kind = mi::neuraylib::ITarget_code::DK_EDF;
            break;
        case mi::mdl::IType::TK_VDF:
            function_descriptions[i].distribution_kind = mi::neuraylib::ITarget_code::DK_INVALID;
            break;
        default:
            function_descriptions[i].distribution_kind = mi::neuraylib::ITarget_code::DK_NONE;
            break;
        }
    }

    // Was a target argument block layout created for this entity?
    if (arg_block_index != size_t(~0)) {
        // Add it to the target code and remember the arguments of the compiled material
        mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> layout(
            m_unit->get_arg_block_layout(arg_block_index));
        mi::Size index = m_target_code->add_argument_block_layout(
            mi::base::make_handle(
            new Target_value_layout(layout.get(), m_strings_mapped_to_ids)).get());
        ASSERT(M_BACKENDS, index == arg_block_index && "Unit and target code should be in sync");

        m_arg_block_comp_material_args.push_back(
            mi::base::make_handle(compiled_material->get_arguments(m_transaction)));
        ASSERT(M_BACKENDS, index == m_arg_block_comp_material_args.size() - 1 &&
               "Unit and arg block material arg list should be in sync");

        (void) index;  // avoid warning about unused variable
    }

    // pass out the block index
    for (size_t i = 0; i < description_count; ++i) {
        function_descriptions[i].argument_block_index = arg_block_index;
        function_descriptions[i].return_code = 0;
    }

    return 0;
}

mi::Sint32 Link_unit::add_material(
    MDL::Mdl_compiled_material const             *compiled_material,
    mi::neuraylib::Target_function_description   *function_descriptions,
    mi::Size                                      description_count,
    MDL::Execution_context                       *context)
{
    // adding a group of functions with a single init function?
    if (description_count > 0 &&
        function_descriptions[0].path != NULL &&
        strcmp(function_descriptions[0].path, "init") == 0)
    {
        return add_material_single_init(
            compiled_material, function_descriptions, description_count, context);
    }

    if (compiled_material == NULL) {
        MDL::add_error_message(context, "Invalid parameters (NULL pointer).", -1);
        return -1;
    }
    if (!compiled_material->is_valid(m_transaction, context)) {
        if (context)
            MDL::add_error_message(context,
                "The compiled material is invalid.", -1);
        return -1;
    }

    // argument block index for the entire material
    // (initialized by the first function that requires material arguments)
    size_t arg_block_index = ~0;

    MDL::Mdl_call_resolver resolver(m_transaction);

    bool resolve_resources =
        get_context_option<bool>(context, MDL_CTX_OPTION_RESOLVE_RESOURCES);
    bool include_geometry_normal =
        get_context_option<bool>(context, MDL_CTX_OPTION_INCLUDE_GEO_NORMAL);

    // check internal space configuration
    if (m_internal_space != compiled_material->get_internal_space()) {
        MDL::add_error_message(context, "Materials compiled with different internal_space "
            "configurations cannot be mixed.", -1);
        return -1;
    }

    // increment once for each add_material invocation
    m_gen_base_name_suffix_counter++;

    // we need to first collect all resources from all expressions to be translated,
    // then remember the number of (body) resources per resource type,
    // and then enumerate the argument resources and translate the expressions
    struct Add_list_item {
        mi::base::Handle<mi::mdl::IDistribution_function> dist_func;
        mi::base::Handle<mi::mdl::ILambda_function> lambda_func;
    };

    std::vector<Add_list_item> add_list_items(description_count);

    Lambda_builder builder(
        m_compiler.get(),
        m_transaction,
        m_compile_consts,
        m_calc_derivatives);

    for (mi::Size i = 0; i < description_count; ++i) {
        if (function_descriptions[i].path == NULL) {
            function_descriptions[i].return_code = MDL::add_error_message(
                context,
                "Invalid parameters (NULL pointer) for function at index " + std::to_string(i),
                -1);
            return -1;
        }

        // get the field corresponding to path
        mi::base::Handle<const MDL::IExpression> field(
            compiled_material->lookup_sub_expression(m_transaction, function_descriptions[i].path));

        if (!field) {
            MDL::add_error_message(context, "Invalid path (non-existing) for function at index "
                + std::to_string(i) + '.', -2);
            function_descriptions[i].return_code = -2;
            return -1;
        }

        // and the type of the field
        mi::base::Handle<const MDL::IType> field_int_type(field->get_type());
        field_int_type = field_int_type->skip_all_type_aliases();

        // use the provided base name or generate one
        std::stringstream sstr;
        if (function_descriptions[i].base_fname && function_descriptions[i].base_fname[0]) {
            sstr << function_descriptions[i].base_fname;
        } else {
            sstr << "lambda_" << m_gen_base_name_suffix_counter
                 << "__" << function_descriptions[i].path;
        }

        std::string function_name = sstr.str();
        std::replace(function_name.begin(), function_name.end(), '.', '_');

        switch (field_int_type->get_kind())
        {
            // DF types that are supported
            case MDL::IType::TK_BSDF:
            case MDL::IType::TK_HAIR_BSDF:
            case MDL::IType::TK_EDF:
            //case MDL::IType::TK_VDF:
            {
                // set infos that are passed back
                switch (field_int_type->get_kind())
                {
                    case MDL::IType::TK_BSDF:
                        function_descriptions[i].distribution_kind =
                            mi::neuraylib::ITarget_code::DK_BSDF;
                        break;

                    case MDL::IType::TK_HAIR_BSDF:
                        function_descriptions[i].distribution_kind =
                            mi::neuraylib::ITarget_code::DK_HAIR_BSDF;
                        break;

                    case MDL::IType::TK_EDF:
                        function_descriptions[i].distribution_kind =
                            mi::neuraylib::ITarget_code::DK_EDF;
                        break;

                    //case MDL::IType::TK_VDF:
                    //    function_descriptions[i].distribution_kind =
                    //        mi::neuraylib::ITarget_code::DK_VDF;
                    //    break;

                    default:
                        MDL::add_error_message(
                            context,
                            "VDFs are not supported for function at index" +
                            std::to_string(i),
                            -900);
                        function_descriptions[i].return_code = -900;
                        function_descriptions[i].distribution_kind =
                            mi::neuraylib::ITarget_code::DK_INVALID;
                        break;
                }

                // convert from an IExpression-based compiled material sub-expression
                // to a DAG_node-based distribution function consisting of
                //  - a main df containing the DF part and array and struct constants and
                //  - a number of expression lambdas containing the non-DF part
                // Note: We currently don't support storing expression lambdas of type double
                //       in GLSL/HLSL, so disable it.
                mi::base::Handle<mi::mdl::IDistribution_function> dist_func(
                    builder.from_material_df(
                        compiled_material,
                        function_descriptions[i].path,
                        function_name.c_str(),
                        include_geometry_normal,
                        /*allow_double_expr_result=*/!target_is_structured_language()));

                if (!dist_func.is_valid_interface())
                {
                    MDL::add_error_message(
                        context, builder.get_error_string(), builder.get_error_code() * 100);
                    function_descriptions[i].return_code = builder.get_error_code() * 100;
                    return -1;
                }

                mi::base::Handle<mi::mdl::ILambda_function> main_func(
                    dist_func->get_main_function(0));
                MDL_ASSERT(main_func);

                // check if the distribution function is the default one, e.g. 'bsdf()'
                // if that's the case we don't need to translate as the evaluation of the function
                // will result in zero, if not explicitly enabled
                const mi::mdl::DAG_node* body = main_func->get_body();
                if (!m_compile_consts && body->get_kind() == mi::mdl::DAG_node::EK_CONSTANT &&
                    mi::mdl::cast<mi::mdl::DAG_constant>(body)->get_value()->get_kind()
                        == mi::mdl::IValue::VK_INVALID_REF)
                {
                    function_descriptions[i].function_index = ~0;
                    break;
                }

                // ... enumerate resources: must be done before we compile ...
                //     all resource information will be collected in root_lambda
                mi::base::Handle<mi::mdl::ILambda_function> root_lambda(
                    dist_func->get_root_lambda());
                m_tc_reg->set_in_argument_mode(false);
                Function_enumerator enumerator(
                    *m_tc_reg, root_lambda.get(), m_transaction, m_tex_idx,
                    m_lp_idx, m_bm_idx, m_res_index_map,
                    !resolve_resources, resolve_resources);
                root_lambda->enumerate_resources(resolver, enumerator, main_func->get_body());
                if (!resolve_resources)
                    root_lambda->set_has_resource_attributes(false);

                size_t expr_lambda_count = dist_func->get_expr_lambda_count();
                for (size_t i = 0; i < expr_lambda_count; ++i)
                {
                    mi::base::Handle<mi::mdl::ILambda_function> lambda(
                        dist_func->get_expr_lambda(i));

                    // also register the resources in lambda itself,
                    // so we see whether it accesses resources
                    enumerator.set_additional_lambda(lambda.get());

                    lambda->enumerate_resources(resolver, enumerator, lambda->get_body());
                }

                // ... optimize all expression lambdas
                // (for derivatives, optimization already happened while building derivative info,
                // and doing it again may destroy the analysis result)
                if (!m_calc_derivatives) {
                    for (size_t i = 0, n = dist_func->get_expr_lambda_count(); i < n; ++i) {
                        mi::base::Handle<mi::mdl::ILambda_function> lambda(
                            dist_func->get_expr_lambda(i));

                        MDL::Call_evaluator<mi::mdl::ILambda_function> call_evaluator(
                            lambda.get(),
                            m_transaction,
                            resolve_resources);

                        lambda->optimize(&resolver, &call_evaluator);
                    }
                }

                add_list_items[i].dist_func = dist_func;
                add_list_items[i].lambda_func = root_lambda;
                break;
            }

            // if not a distribution function, we assume a generic expression
            default:
            {
                // set infos that are passed back
                function_descriptions[i].distribution_kind = mi::neuraylib::ITarget_code::DK_NONE;

                mi::base::Handle<mi::mdl::ILambda_function> lambda(
                    builder.from_sub_expr(
                        compiled_material,
                        function_descriptions[i].path,
                        function_name.c_str()));

                if (!lambda.is_valid_interface())
                {
                    MDL::add_error_message(
                        context, builder.get_error_string(), builder.get_error_code() * 10);
                    function_descriptions[i].return_code = builder.get_error_code() * 10;
                    return -1;
                }

                if (m_calc_derivatives)
                    lambda->initialize_derivative_infos(&resolver);

                // Enumerate resources ...
                m_tc_reg->set_in_argument_mode(false);
                Function_enumerator enumerator(
                    *m_tc_reg, lambda.get(), m_transaction, m_tex_idx,
                    m_lp_idx, m_bm_idx, m_res_index_map,
                    !resolve_resources, resolve_resources);
                lambda->enumerate_resources(resolver, enumerator, lambda->get_body());
                if (!resolve_resources)
                    lambda->set_has_resource_attributes(false);

                add_list_items[i].lambda_func = lambda;
                break;
            }
        }
    }

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(m_transaction, mdlc_module->get_module_wait_queue(), {});

    // now that all expressions are preprocessed and all resources from the bodies are collected,
    // process the arguments and add the expressions to the link unit
    for (mi::Size i = 0; i < description_count; ++i) {
        if (!add_list_items[i].lambda_func)
            continue;

        Function_enumerator enumerator(
            *m_tc_reg, add_list_items[i].lambda_func.get(), m_transaction, m_tex_idx,
            m_lp_idx, m_bm_idx, m_res_index_map,
            !resolve_resources, resolve_resources);

        // ... also enumerate resources from arguments ...
        if (compiled_material->get_parameter_count() != 0) {
            m_tc_reg->set_in_argument_mode(true);
            builder.enumerate_resource_arguments(
                add_list_items[i].lambda_func.get(), compiled_material, enumerator);
        }

        // ... and add it to the compilation unit
        if (add_list_items[i].dist_func) {
            size_t indices[2];
            if (!m_unit->add(
                add_list_items[i].dist_func.get(),
                &module_cache,
                &resolver,
                &arg_block_index,
                indices,
                2))
            {
                MDL::convert_and_log_messages(m_unit->access_messages(), context);
                MDL::add_error_message(context,
                    "The JIT backend failed to compile the function at index "
                    + std::to_string(i) + '.', -300);
                function_descriptions[i].return_code = -300;
                return -1;
            }
            // for distribution functions, let function_index point to the init function
            function_descriptions[i].function_index = indices[0];
        } else {
            size_t index;
            if (!m_unit->add(
                add_list_items[i].lambda_func.get(),
                &module_cache,
                &resolver,
                mi::mdl::IGenerated_code_executable::FK_LAMBDA,
                &arg_block_index,
                &index))
            {
                MDL::convert_and_log_messages(m_unit->access_messages(), context);
                MDL::add_error_message(
                    context, "The JIT backend failed to compile the function at index" +
                    std::to_string(i), -30);
                function_descriptions[i].return_code = -30;
                return -1;
            }
            function_descriptions[i].function_index = index;
        }
    }

    // Was a target argument block layout created for this entity?
    if (arg_block_index != size_t(~0))
    {
        // Add it to the target code and remember the arguments of the compiled material
        mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> layout(
            m_unit->get_arg_block_layout(arg_block_index));
        mi::Size index = m_target_code->add_argument_block_layout(
            mi::base::make_handle(
            new Target_value_layout(layout.get(), m_strings_mapped_to_ids)).get());
        ASSERT(M_BACKENDS, index == arg_block_index && "Unit and target code should be in sync");

        m_arg_block_comp_material_args.push_back(
            mi::base::make_handle(compiled_material->get_arguments(m_transaction)));
        ASSERT(M_BACKENDS, index == m_arg_block_comp_material_args.size() - 1 &&
               "Unit and arg block material arg list should be in sync");

        (void) index;  // avoid warning about unused variable
    }

    // pass out the block index
    for (size_t i = 0; i < description_count; ++i)
    {
        function_descriptions[i].argument_block_index = arg_block_index;
        function_descriptions[i].return_code = 0;
    }

    return 0;
}

// Add an MDL function definition as a function to this link unit.
mi::Sint32 Link_unit::add_function(
    MDL::Mdl_function_definition const *function,
    Function_execution_context         fxc,
    char const                         *name,
    MDL::Execution_context             *context)
{
    Lambda_builder builder(
        m_compiler.get(),
        m_transaction,
        m_compile_consts,
        m_calc_derivatives);

    mi::base::Handle<mi::mdl::ILambda_function> lambda(
        builder.from_function_definition(function, name, to_lambda_exc(fxc)));

    if (!lambda.is_valid_interface()) {
        MDL::add_error_message(
            context, builder.get_error_string(), builder.get_error_code() * 10);
        return builder.get_error_code();
    }

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(m_transaction, mdlc_module->get_module_wait_queue(), {});

    MDL::Mdl_call_resolver resolver(m_transaction);

    // enumerate resources ...
    bool resolve_resources = get_context_option<bool>(context, MDL_CTX_OPTION_RESOLVE_RESOURCES);
    m_tc_reg->set_in_argument_mode(false);
    Function_enumerator enumerator(
        *m_tc_reg,
        lambda.get(),
        m_transaction,
        m_tex_idx,
        m_lp_idx,
        m_bm_idx,
        m_res_index_map,
        !resolve_resources,
        resolve_resources);
    lambda->enumerate_resources(resolver, enumerator, lambda->get_body());
    if (!resolve_resources) {
        lambda->set_has_resource_attributes(false);
    }

    if (m_calc_derivatives) {
        lambda->initialize_derivative_infos(&resolver);
    }

    size_t arg_block_index = ~0;
    size_t index = ~0;

    if (!m_unit->add(
        lambda.get(),
        &module_cache,
        &resolver,
        to_function_kind(fxc),
        &arg_block_index,
        &index))
    {
        MDL::convert_and_log_messages(m_unit->access_messages(), context);
        MDL::add_error_message(
            context, std::string("The JIT backend failed to compile the function ") +
            function->get_mdl_name(), -1);
        return -1;
    }

    if (arg_block_index != size_t(~0)) {
        // Add it to the target code and remember the arguments of the function material
        mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> layout(
            m_unit->get_arg_block_layout(arg_block_index));
        mi::Size index = m_target_code->add_argument_block_layout(
            mi::base::make_handle(
                new Target_value_layout(layout.get(), m_strings_mapped_to_ids)).get());
        ASSERT(M_BACKENDS, index == arg_block_index && "Unit and target code should be in sync");

        //FIXME: should we add to m_arg_block_comp_material_args?
        //m_arg_block_comp_material_args.push_back(
        //    mi::base::make_handle(compiled_material->get_arguments()));
        //ASSERT(M_BACKENDS, index == m_arg_block_comp_material_args.size() - 1 &&
        //    "Unit and arg block material arg list should be in sync");

        (void)index;  // avoid warning about unused variable
    }

    return 0;
}

// Get the number of functions inside this link unit.
mi::Size Link_unit::get_num_functions() const
{
    return m_unit->get_function_count();
}

// Get the name of the i'th function inside this link unit.
char const *Link_unit::get_function_name(mi::Size i) const
{
    return m_unit->get_function_name(i);
}

// Get the distribution kind of the i'th function inside this link unit.
mi::neuraylib::ITarget_code::Distribution_kind Link_unit::get_distribution_kind(mi::Size i) const
{
    return mi::neuraylib::ITarget_code::Distribution_kind(m_unit->get_distribution_kind(i));
}

// Get the function kind of the i'th function inside this link unit.
mi::neuraylib::ITarget_code::Function_kind Link_unit::get_function_kind(mi::Size i) const
{
    return mi::neuraylib::ITarget_code::Function_kind(m_unit->get_function_kind(i));
}

// Get the index of the target argument block layout for the i'th function inside this link
// unit if used.
mi::Size Link_unit::get_function_arg_block_layout_index(mi::Size i) const
{
    return mi::Size(m_unit->get_function_arg_block_layout_index(size_t(i)));
}

// Get the number of target argument block layouts used by this link unit.
mi::Size Link_unit::get_arg_block_layout_count() const
{
    return mi::Size(m_unit->get_arg_block_layout_count());
}

// Get the i'th target argument block layout used by this link unit.
mi::mdl::IGenerated_code_value_layout const *Link_unit::get_arg_block_layout(mi::Size i) const
{
    return m_unit->get_arg_block_layout(size_t(i));
}

// Get the MDL link unit.
mi::mdl::ILink_unit *Link_unit::get_compilation_unit() const
{
    m_unit->retain();
    return m_unit.get();
}


// Get the target code of this link unit.
Target_code *Link_unit::get_target_code() const
{
    m_target_code->retain();
    return m_target_code.get();
}

// ------------------------- LLVM based backend -------------------------

/// Checks whether the given backend supports SIMD instructions.
static bool supports_simd(mi::neuraylib::IMdl_backend_api::Mdl_backend_kind kind)
{
    if (kind == mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX)
        return false;
    if (kind == mi::neuraylib::IMdl_backend_api::MB_GLSL ||
        kind == mi::neuraylib::IMdl_backend_api::MB_HLSL) {
        // FIXME: disabled so far
        return false;
    }
    return true;
}

Mdl_llvm_backend::Mdl_llvm_backend(
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind kind,
    mi::mdl::IMDL                                  *compiler,
    mi::mdl::ICode_generator_jit                   *jit,
    mi::mdl::ICode_cache                           *code_cache,
    bool                                           string_ids)
  : m_kind(kind),
    m_sm_version(20),
    m_num_texture_spaces(32),  // by default the number of texture spaces is 32
    m_num_texture_results(0),
    m_compiler(compiler, mi::base::DUP_INTERFACE),
    m_jit(jit, mi::base::DUP_INTERFACE),
    m_code_cache(code_cache, mi::base::DUP_INTERFACE),
    m_compile_consts(true),
    m_enable_simd(supports_simd(kind)),
    m_output_target_lang(true),
    m_strings_mapped_to_ids(string_ids),
    m_calc_derivatives(false),
    m_use_builtin_resource_handler(true)
{
    mi::mdl::Options &options = m_jit->access_options();

    // by default fast math is on
    options.set_option(MDL_JIT_OPTION_FAST_MATH, "true");

    // by default opt-level is 2
    options.set_option(MDL_JIT_OPTION_OPT_LEVEL, "2");

    // by default the internal space of the renderer is "world"
    options.set_option(MDL_CG_OPTION_INTERNAL_SPACE, "coordinate_world");

    // by default the folding meters_per_scene_unit is enabled
    options.set_option(MDL_CG_OPTION_FOLD_METERS_PER_SCENE_UNIT, "true");

    // by default meters_per_scene_unit is 1
    options.set_option(MDL_CG_OPTION_METERS_PER_SCENE_UNIT, "1");

    // by default wavelength_min is 380
    options.set_option(MDL_CG_OPTION_WAVELENGTH_MIN, "380");

    // by default wavelength_max is 780
    options.set_option(MDL_CG_OPTION_WAVELENGTH_MAX, "780");

    // by default we disable exceptions to comply with MDL 1.8 spec
    options.set_option(MDL_JIT_OPTION_DISABLE_EXCEPTIONS, "true");

    // by default the read-only segment is disabled for all BE except HLSL
    // "for historical reasons"
    options.set_option(MDL_JIT_OPTION_ENABLE_RO_SEGMENT,
        kind == mi::neuraylib::IMdl_backend_api::MB_HLSL ? "true" :"false");

    // by default the maximum size of a constant in the code with enabled read-only segment is 1024
    options.set_option(MDL_JIT_OPTION_MAX_CONST_DATA, "1024");

    // by default we generate LLVM IR
    options.set_option(MDL_JIT_OPTION_WRITE_BITCODE, "false");

    // by default we link libdevice
    options.set_option(MDL_JIT_OPTION_LINK_LIBDEVICE, "true");

    // by default we do NOT use bitangent
    options.set_option(MDL_JIT_OPTION_USE_BITANGENT, "false");

    // by default we do NOT include the uniform state
    options.set_option(MDL_JIT_OPTION_INCLUDE_UNIFORM_STATE, "false");

    // by default we use vtable tex_lookup calls
    options.set_option(MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE, "vtable");

    // by default let the target language decide which return mode to use
    options.set_option(MDL_JIT_OPTION_LAMBDA_RETURN_MODE, "default");

    // do we map strings to identifiers?
    options.set_option(MDL_JIT_OPTION_MAP_STRINGS_TO_IDS, string_ids ? "true" : "false");

    // by default we disable GLSL/HLSL resource data
    options.set_option(MDL_JIT_OPTION_SL_USE_RESOURCE_DATA, "false");

    // by default, no function remap
    options.set_option(MDL_JIT_OPTION_REMAP_FUNCTIONS, "");

    // defaults for HLSL/GLSL are different
    if (kind == mi::neuraylib::IMdl_backend_api::MB_HLSL) {
        options.set_option(MDL_JIT_OPTION_SL_CORE_STATE_API_NAME, "Shading_state_material");
        options.set_option(MDL_JIT_OPTION_SL_ENV_STATE_API_NAME,  "Shading_state_environment");
    } else if (kind == mi::neuraylib::IMdl_backend_api::MB_GLSL) {
        options.set_option(MDL_JIT_OPTION_SL_CORE_STATE_API_NAME, "State");
        options.set_option(MDL_JIT_OPTION_SL_ENV_STATE_API_NAME,  "State_env");
    }

    // by default we don't use a renderer provided microfacet roughness adaption function
    options.set_option(MDL_JIT_OPTION_USE_RENDERER_ADAPT_MICROFACET_ROUGHNESS, "false");

    // by default we expect a texture runtime without derivative support
    options.set_option(MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES, "false");

    // by default we generate no auxiliary methods on distribution functions
    options.set_option(MDL_JIT_OPTION_ENABLE_AUXILIARY, "false");

    // by default we generate pdf functions
    options.set_option(MDL_JIT_OPTION_ENABLE_PDF, "true");
}

/// Currently supported SM versions.
static struct Sm_versions { char const *name; unsigned code; } const known_sms[] = {
    { "20", 20 },
    { "30", 30 },
    { "35", 35 },
    { "37", 37 },
    { "50", 50 },
    { "52", 52 },
    { "60", 60 },
    { "61", 61 },
    { "62", 62 },
    { "70", 70 },
    { "75", 75 },
    { "80", 80 },
    { "86", 86 },
};

static mi::Sint32 set_state_mode_option(
    mi::mdl::Options &options,
    char const *name,
    char const *value)
{
    static char const * const allowed[] = {
        "field", "arg", "func", "zero"
    };

    for (size_t i = 0, n = sizeof(allowed) / sizeof(allowed[0]); i < n; ++i) {
        if (strcmp(allowed[i], value) == 0) {
            options.set_option(name, value);
            return 0;
        }
    }
    return -2;
}

mi::Sint32 Mdl_llvm_backend::set_option(
    char const *name,
    char const *value)
{
    if (name == NULL)
        return -1;
    if (value == NULL)
        return -2;

    // common options
    mi::mdl::Options &jit_options = m_jit->access_options();

    if (strcmp(name, "compile_constants") == 0) {
        if (strcmp(value, "off") == 0) {
            m_compile_consts = false;
        } else if (strcmp(value, "on") == 0) {
            m_compile_consts = true;
        } else {
            return -2;
        }
        return 0;
    }

    if (strcmp(name, "fast_math") == 0) {
        if (strcmp(value, "off") == 0) {
            value = "false";
        } else if (strcmp(value, "on") == 0) {
            value = "true";
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_FAST_MATH, value);
        return 0;
    }
    if (strcmp(name, "opt_level") == 0) {
        if (strcmp(value, "0") == 0 || strcmp(value, "1") == 0 || strcmp(value, "2") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_OPT_LEVEL, value);
            return 0;
        }
        return -2;
    }
    if (strcmp(name, "num_texture_spaces") == 0) {
        unsigned v = 0;
        if (sscanf(value, "%u", &v) != 1) {
            return -2;
        }
        m_num_texture_spaces = v;
        return 0;
    }

    // LLVM specific options
    if (strcmp(name, "jit_warn_spectrum_conversion") == 0) {
        if (strcmp(value, "off") == 0) {
            value = "false";
        } else if (strcmp(value, "on") == 0) {
            value = "true";
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_WARN_SPECTRUM_CONVERSION, value);
        return 0;
    }

    if (strcmp(name, "inline_aggressively") == 0) {
        if (strcmp(value, "off") == 0) {
            value = "false";
        } else if (strcmp(value, "on") == 0) {
            value = "true";
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_INLINE_AGGRESSIVELY, value);
        return 0;
    }
    if (strcmp(name, "eval_dag_ternary_strictly") == 0) {
        if (strcmp(value, "off") == 0) {
            value = "false";
        } else if (strcmp(value, "on") == 0) {
            value = "true";
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_EVAL_DAG_TERNARY_STRICTLY, value);
        return 0;
    }
    if (strcmp(name, "df_handle_slot_mode") == 0) {
        if (strcmp(value, "none") == 0 ||
           (strcmp(value, "fixed_1") == 0) ||
           (strcmp(value, "fixed_2") == 0) ||
           (strcmp(value, "fixed_4") == 0) ||
           (strcmp(value, "fixed_8") == 0) ||
           (strcmp(value, "pointer") == 0 && !target_is_structured_language()))
        {
            jit_options.set_option(MDL_JIT_OPTION_LINK_LIBBSDF_DF_HANDLE_SLOT_MODE, value);
            return 0;
        }
        return -2;
    }
    if (strcmp(name, "libbsdf_flags_in_bsdf_data") == 0)
    {
        if (strcmp(value, "off") == 0) {
            value = "false";
        }
        else if (strcmp(value, "on") == 0) {
            value = "true";
        }
        else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_LIBBSDF_FLAGS_IN_BSDF_DATA, value);
        return 0;
    }
    if (strcmp(name, "enable_auxiliary") == 0)
    {
        if (strcmp(value, "off") == 0) {
            value = "false";
        }
        else if (strcmp(value, "on") == 0) {
            value = "true";
        }
        else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_ENABLE_AUXILIARY, value);
        return 0;
    }
    if (strcmp(name, "enable_pdf") == 0)
    {
        if (strcmp(value, "off") == 0) {
            value = "false";
        }
        else if (strcmp(value, "on") == 0) {
            value = "true";
        }
        else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_ENABLE_PDF, value);
        return 0;
    }
    if (strcmp(name, "enable_exceptions") == 0) {
        // beware, the JIT backend has the inverse option
        if (strcmp(value, "off") == 0) {
            value = "true";
        } else if (strcmp(value, "on") == 0) {
            value = "false";
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_DISABLE_EXCEPTIONS, value);
        return 0;
    }
    if (strcmp(name, "enable_ro_segment") == 0) {
        if (strcmp(value, "off") == 0) {
            value = "false";
        } else if (strcmp(value, "on") == 0) {
            value = "true";
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_ENABLE_RO_SEGMENT, value);
        return 0;
    }
    if (strcmp(name, "max_const_data") == 0) {
        jit_options.set_option(MDL_JIT_OPTION_MAX_CONST_DATA, value);
        return 0;
    }
    if (strcmp(name, "num_texture_results") == 0) {
        unsigned v = 0;
        if (sscanf(value, "%u", &v) != 1) {
            return -2;
        }
        m_num_texture_results = v;
        return 0;
    }

    if (strcmp(name, "texture_runtime_with_derivs") == 0) {
        if (strcmp(value, "off") == 0) {
            value = "false";
            m_calc_derivatives = false;
        } else if (strcmp(value, "on") == 0) {
            value = "true";
            m_calc_derivatives = true;
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES, value);
        return 0;
    }

    if (strcmp(name, "use_renderer_adapt_microfacet_roughness") == 0) {
        // TODO: currently only supported for GLSL/HLSL backend
        if (!target_is_structured_language()) {
            return -1;
        }
        if (strcmp(value, "off") == 0) {
            value = "false";
        } else if (strcmp(value, "on") == 0) {
            value = "true";
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_USE_RENDERER_ADAPT_MICROFACET_ROUGHNESS, value);
        return 0;
    }

    if (strcmp(name, "use_renderer_adapt_normal") == 0) {
        if (strcmp(value, "off") == 0) {
            value = "false";
        } else if (strcmp(value, "on") == 0) {
            value = "true";
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_USE_RENDERER_ADAPT_NORMAL, value);
        return 0;
    }

    if (strcmp(name, "visible_functions") == 0) {
        jit_options.set_option(MDL_JIT_OPTION_VISIBLE_FUNCTIONS, value);
        return 0;
    }

    if (strcmp(name, "lambda_return_mode") == 0) {
        // only supported for CUDA_PTX and LLVM_IR
        if (m_kind != mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX &&
                m_kind != mi::neuraylib::IMdl_backend_api::MB_LLVM_IR)
            return -1;

        if (strcmp(value, "default") == 0) {
        } else if (strcmp(value, "sret") == 0) {
        } else if (strcmp(value, "value") == 0) {
        } else {
            return -2;
        }
        jit_options.set_option(MDL_JIT_OPTION_LAMBDA_RETURN_MODE, value);
        return 0;
    }

    // specific options
    switch (m_kind) {
    case mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX:
        if (strcmp(name, "sm_version") == 0) {
            for (size_t i = 0, n = sizeof(known_sms) / sizeof(known_sms[0]); i < n; ++i) {
                if (strcmp(value, known_sms[i].name) == 0) {
                    m_sm_version = known_sms[i].code;
                    return 0;
                }
            }
            return  -2;
        }
        if (strcmp(name, "link_libdevice") == 0) {
            if (strcmp(value, "off") == 0) {
                value = "false";
            } else if (strcmp(value, "on") == 0) {
                value = "true";
            } else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_OPTION_LINK_LIBDEVICE, value);
            return 0;
        }
        if (strcmp(name, "output_format") == 0) {
            bool enable_bc   = false;
            if (strcmp(value, "PTX") == 0) {
                m_output_target_lang = true;
                enable_bc    = false;
            } else if (strcmp(value, "LLVM-IR") == 0) {
                m_output_target_lang = false;
                enable_bc    = false;
            } else if (strcmp(value, "LLVM-BC") == 0) {
                m_output_target_lang = false;
                enable_bc    = true;
            } else {
                return -2;
            }
            jit_options.set_option(
                MDL_JIT_OPTION_WRITE_BITCODE, enable_bc ? "true" : "false");
            return 0;
        }
        if (strcmp(name, "tex_lookup_call_mode") == 0) {
            if (strcmp(value, "vtable") == 0) {
            } else if (strcmp(value, "direct_call") == 0) {
            } else if (strcmp(value, "optix_cp") == 0) {
            } else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE, value);
            return 0;
        }
        break;

    case mi::neuraylib::IMdl_backend_api::MB_LLVM_IR:
        if (strcmp(name, "enable_simd") == 0) {
            if (strcmp(value, "off") == 0) {
                m_enable_simd = false;
                return 0;
            } else if (strcmp(value, "on") == 0) {
                m_enable_simd = true;
                return 0;
            }
            return -2;
        }
        if (strcmp(name, "write_bitcode") == 0) {
            if (strcmp(value, "off") == 0) {
                value = "false";
            } else if (strcmp(value, "on") == 0) {
                value = "true";
            } else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_OPTION_WRITE_BITCODE, value);
            return 0;
        }
        break;

    case mi::neuraylib::IMdl_backend_api::MB_GLSL:
        if (strcmp(name, "glsl_version") == 0) {
            char const *version = "3.30";
            if (strcmp(value, "100") == 0) {
                version = "1.00";
            } else if (strcmp(value, "150") == 0) {
                version = "1.50";
            } else if (strcmp(value, "300") == 0) {
                version = "3.00";
            } else if (strcmp(value, "310") == 0) {
                version = "3.10";
            } else if (strcmp(value, "330") == 0) {
                version = "3.30";
            } else if (strcmp(value, "400") == 0) {
                version = "4.00";
            } else if (strcmp(value, "410") == 0) {
                version = "4.10";
            } else if (strcmp(value, "420") == 0) {
                version = "4.20";
            } else if (strcmp(value, "430") == 0) {
                version = "4.30";
            } else if (strcmp(value, "440") == 0) {
                version = "4.40";
            } else if (strcmp(value, "450") == 0) {
                version = "4.50";
            } else if (strcmp(value, "460") == 0) {
                version = "4.60";
            } else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_OPTION_GLSL_VERSION, version);
            return 0;
        }
        if (strcmp(name, "glsl_profile") == 0) {
            char const *profile = "core";
            if (strcmp(value, "core") == 0) {
                profile = value;
            } else if (strcmp(value, "es") == 0) {
                profile = value;
            } else if (strcmp(value, "compatibility") == 0) {
                profile = value;
            } else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_OPTION_GLSL_PROFILE, profile);
            return 0;
        }
        if (strcmp(name, "glsl_enabled_extensions") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_GLSL_ENABLED_EXTENSIONS, value);
            return 0;
        }
        if (strcmp(name, "glsl_required_extensions") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_GLSL_REQUIRED_EXTENSIONS, value);
            return 0;
        }
        if (strcmp(name, "glsl_max_const_data") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_GLSL_MAX_CONST_DATA, value);
            return 0;
        }
        if (strcmp(name, "glsl_remap_functions") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_REMAP_FUNCTIONS, value);
            return 0;
        }
        if (strcmp(name, "glsl_state_animation_time_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_ANIMATION_TIME_MODE, value);
        }
        if (strcmp(name, "glsl_state_geometry_normal_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_GEOMETRY_NORMAL_MODE, value);
        }
        if (strcmp(name, "glsl_state_motion_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_MOTION_MODE, value);
        }
        if (strcmp(name, "glsl_state_normal_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_NORMAL_MODE, value);
        }
        if (strcmp(name, "glsl_state_object_id_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_OBJECT_ID_MODE, value);
        }
        if (strcmp(name, "glsl_state_position_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_POSITION_MODE, value);
        }
        if (strcmp(name, "glsl_state_texture_space_max_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_TEXTURE_SPACE_MAX_MODE, value);
        }
        if (strcmp(name, "glsl_state_texture_coordinate_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_TEXTURE_COORDINATE_MODE, value);
        }
        if (strcmp(name, "glsl_state_texture_tangent_u_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_TEXTURE_TANGENT_U_MODE, value);
        }
        if (strcmp(name, "glsl_state_texture_tangent_v_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_TEXTURE_TANGENT_V_MODE, value);
        }
        if (strcmp(name, "glsl_state_geometry_tangent_u_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_GEOMETRY_TANGENT_U_MODE, value);
        }
        if (strcmp(name, "glsl_state_geometry_tangent_v_mode") == 0) {
            return set_state_mode_option(
                jit_options, MDL_JIT_OPTION_SL_STATE_GEOMETRY_TANGENT_V_MODE, value);
        }
        if (strcmp(name, "glsl_place_uniforms_into_ssbo") == 0) {
            if (strcmp(value, "off") == 0) {
                value = "false";
            } else if (strcmp(value, "on") == 0) {
                value = "true";
            } else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_OPTION_GLSL_PLACE_UNIFORMS_INTO_SSBO, value);
            return 0;
        }
        if (strcmp(name, "glsl_uniform_ssbo_name") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_NAME, value);
            return 0;
        }
        if (strcmp(name, "glsl_uniform_ssbo_binding") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_BINDING, value);
            return 0;
        }
        if (strcmp(name, "glsl_uniform_ssbo_set") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_SET, value);
            return 0;
        }

        if (strcmp(name, "glsl_use_resource_data") == 0) {
            if (strcmp(value, "on") == 0) {
                value = "true";
            } else if (strcmp(value, "off") == 0) {
                value = "false";
            } else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_OPTION_SL_USE_RESOURCE_DATA, value);
            return 0;
        }
        break;

    case mi::neuraylib::IMdl_backend_api::MB_NATIVE:
        if (strcmp(name, "use_builtin_resource_handler") == 0) {
            if (strcmp(value, "on") == 0) {
                m_use_builtin_resource_handler = true;
                value = "true";
            }
            else if (strcmp(value, "off") == 0) {
                m_use_builtin_resource_handler = false;
                value = "false";
            }
            else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU, value);
            return 0;
        }
        break;

    case mi::neuraylib::IMdl_backend_api::MB_HLSL:
        if (strcmp(name, "hlsl_use_resource_data") == 0) {
            if (strcmp(value, "on") == 0) {
                value = "true";
            } else if (strcmp(value, "off") == 0) {
                value = "false";
            } else {
                return -2;
            }
            jit_options.set_option(MDL_JIT_OPTION_SL_USE_RESOURCE_DATA, value);
            return 0;
        }
        if (strcmp(name, "hlsl_remap_functions") == 0) {
            jit_options.set_option(MDL_JIT_OPTION_REMAP_FUNCTIONS, value);
            return 0;
        }
        break;

    case mi::neuraylib::IMdl_backend_api::MB_FORCE_32_BIT:
        break;
    }
    return -1;
}

mi::Sint32 Mdl_llvm_backend::set_option_binary(
    char const *name,
    char const *data,
    mi::Size size)
{
    if (strcmp(name, "llvm_state_module") == 0) {
        m_jit->access_options().set_binary_option(
            MDL_JIT_BINOPTION_LLVM_STATE_MODULE, data, size);
        return 0;
    }
    if (strcmp(name, "llvm_renderer_module") == 0) {
        m_jit->access_options().set_binary_option(
            MDL_JIT_BINOPTION_LLVM_RENDERER_MODULE, data, size);
        return 0;
    }
    return -1;
}

void Mdl_llvm_backend::update_jit_context_options(
    mi::mdl::ICode_generator_thread_context &cg_ctx,
    const char                              *internal_space,
    MDL::Execution_context                  *context) const
{
    mi::mdl::Options &cg_opts = cg_ctx.access_options();

    if (internal_space == nullptr) {
        const std::string internal_space_obj =
            get_context_option<std::string>(context, MDL_CTX_OPTION_INTERNAL_SPACE);
        cg_opts.set_option(
            MDL_CG_OPTION_INTERNAL_SPACE, internal_space_obj.c_str());
    } else {
        cg_opts.set_option(MDL_CG_OPTION_INTERNAL_SPACE, internal_space);
    }
    cg_opts.set_option(
        MDL_CG_OPTION_FOLD_METERS_PER_SCENE_UNIT,
        get_context_option<bool>(context, MDL_CTX_OPTION_FOLD_METERS_PER_SCENE_UNIT)
        ? "true" : "false");
    char buf[32];
    snprintf(buf, sizeof(buf), "%f",
        get_context_option<float>(context, MDL_CTX_OPTION_METERS_PER_SCENE_UNIT));
    cg_opts.set_option(MDL_CG_OPTION_METERS_PER_SCENE_UNIT, buf);
    snprintf(buf, sizeof(buf), "%f",
        get_context_option<float>(context, MDL_CTX_OPTION_WAVELENGTH_MIN));
    cg_opts.set_option(MDL_CG_OPTION_WAVELENGTH_MIN, buf);
    snprintf(buf, sizeof(buf), "%f",
        get_context_option<float>(context, MDL_CTX_OPTION_WAVELENGTH_MAX));
    cg_opts.set_option(MDL_CG_OPTION_WAVELENGTH_MAX, buf);
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::translate_environment(
    DB::Transaction              *transaction,
    MDL::Mdl_function_call const *function_call,
    char const                   *fname,
    MDL::Execution_context       *context)
{
    if (transaction == NULL || function_call == NULL) {
        MDL::add_error_message(context, "Invalid parameters (NULL pointer).", -1);
        return NULL;
    }
    DB::Tag_set tags_seen;
    if (!function_call->is_valid(transaction, tags_seen, context)) {
        MDL::add_error_message(context, "Invalid function call.", -1);
        return NULL;
    }
    DB::Tag def_tag = function_call->get_function_definition(transaction);
    if (!def_tag.is_valid()) {
        MDL::add_error_message(context, "Function does not point to a valid definition.", -1);
        return NULL;
    }

    Lambda_builder builder(
        m_compiler.get(),
        transaction,
        m_compile_consts,
        m_calc_derivatives);

    mi::base::Handle<mi::mdl::ILambda_function> lambda(
        builder.from_call(function_call, fname, mi::mdl::ILambda_function::LEC_ENVIRONMENT));
    if (!lambda.is_valid_interface()) {
        MDL::add_error_message(context,
            builder.get_error_string(), builder.get_error_code());
        return NULL;
    }

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(transaction, mdlc_module->get_module_wait_queue(), {});
    MDL::Mdl_call_resolver resolver(transaction);

    // enumerate resources: must be done before we compile
    bool resolve_resources = get_context_option<bool>(context, MDL_CTX_OPTION_RESOLVE_RESOURCES);
    Target_code_register tc_reg;
    Function_enumerator enumerator(tc_reg, lambda.get(), transaction,
        !resolve_resources, resolve_resources);
    lambda->enumerate_resources(resolver, enumerator, lambda->get_body());
    if (!resolve_resources)
        lambda->set_has_resource_attributes(false);

    // now compile
    DB::Access<MDL::Mdl_function_definition> function_definition(def_tag, transaction);
    DB::Access<MDL::Mdl_module> module(
        function_definition->get_module(transaction), transaction);
    mi::base::Handle<mi::mdl::IGenerated_code_dag const> code_dag(module->get_code_dag());
    mi::base::Handle<mi::mdl::ICode_generator_thread_context> cg_ctx(
        m_jit->create_thread_context());

    update_jit_context_options(*cg_ctx.get(), code_dag->get_internal_space(), context);

    mi::base::Handle<mi::mdl::IGenerated_code_executable> code;
    switch (m_kind) {
    case mi::neuraylib::IMdl_backend_api::MB_LLVM_IR:
        code = mi::base::make_handle(
            m_jit->compile_into_llvm_ir(
                lambda.get(),
                &module_cache,
                &resolver,
                cg_ctx.get(),
                m_num_texture_spaces,
                m_num_texture_results,
                m_enable_simd));
        break;
    case mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX:
    case mi::neuraylib::IMdl_backend_api::MB_GLSL:
    case mi::neuraylib::IMdl_backend_api::MB_HLSL:
        code = mi::base::make_handle(
            m_jit->compile_into_source(
                m_code_cache.get(),
                lambda.get(),
                &module_cache,
                &resolver,
                cg_ctx.get(),
                m_num_texture_spaces,
                m_num_texture_results,
                m_sm_version,
                map_target_language(m_kind),
                !m_output_target_lang));
        break;
    case mi::neuraylib::IMdl_backend_api::MB_NATIVE:
        code = mi::base::make_handle(
            m_jit->compile_into_environment(
                lambda.get(),
                &module_cache,
                &resolver,
                cg_ctx.get()));
        break;
    default:
        break;
    }

    if (!code.is_valid_interface()) {
        MDL::add_error_message(context,
            "The backend failed to generate target code for the function.", -3);
        return NULL;
    }

    MDL::convert_and_log_messages(code->access_messages(), context);

    if (!code->is_valid()) {
        MDL::add_error_message(context,
            "The backend failed to generate target code for the function.", -3);
        return NULL;
    }

    Target_code *tc = new Target_code(
        code.get(),
        transaction,
        m_strings_mapped_to_ids,
        m_calc_derivatives,
        m_use_builtin_resource_handler,
        m_kind);

    // Enter the resource-table here
    fill_resource_tables(tc_reg, tc);

    // copy data segments
    for (size_t i = 0, n = code->get_data_segment_count(); i < n; ++i) {
        mi::mdl::IGenerated_code_executable::Segment const *desc = code->get_data_segment(i);

        if (desc) {
            tc->add_ro_segment(
                desc->name,
                desc->data,
                desc->size);
        }
    }

    // copy the string constant table
    for (size_t i = 0, n = code->get_string_constant_count(); i < n; ++i) {
        tc->add_string_constant_index(i, code->get_string_constant(i));
    }

    return tc;
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::translate_material_expression(
    DB::Transaction                  *transaction,
    MDL::Mdl_compiled_material const *compiled_material,
    char const                       *path,
    char const                       *fname,
    MDL::Execution_context           *context)
{
    if (!transaction || !compiled_material || !path) {
        MDL::add_error_message(context, "Invalid parameters (NULL pointer).", -1);
        return NULL;
    }
    if (!compiled_material->is_valid(transaction, context)) {
        MDL::add_error_message(context, "Compiled material is invalid.", -1);
        return NULL;
    }

    Lambda_builder builder(
        m_compiler.get(),
        transaction,
        m_compile_consts,
        m_calc_derivatives);

    mi::base::Handle<mi::mdl::ILambda_function> lambda(
        builder.from_sub_expr(compiled_material, path, fname));
    if (!lambda.is_valid_interface()) {
        MDL::add_error_message(context,
            builder.get_error_string(), builder.get_error_code());
        return NULL;
    }

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(transaction, mdlc_module->get_module_wait_queue(), {});
    MDL::Mdl_call_resolver resolver(transaction);

    if (m_calc_derivatives)
        lambda->initialize_derivative_infos(&resolver);

    // ... enumerate resources: must be done before we compile ...
    bool resove_resources = get_context_option<bool>(context, MDL_CTX_OPTION_RESOLVE_RESOURCES);
    Target_code_register tc_reg;
    Function_enumerator enumerator(tc_reg, lambda.get(), transaction,
        !resove_resources, resove_resources);
    lambda->enumerate_resources(resolver, enumerator, lambda->get_body());
    if (!resove_resources)
        lambda->set_has_resource_attributes(false);

    // ... also enumerate resources from arguments ...
    if (compiled_material->get_parameter_count() != 0) {
        tc_reg.set_in_argument_mode(true);
        builder.enumerate_resource_arguments(lambda.get(), compiled_material, enumerator);
    }

    mi::base::Handle<mi::mdl::ICode_generator_thread_context> cg_ctx(
        m_jit->create_thread_context());

    // ... and compile
    update_jit_context_options(*cg_ctx.get(), compiled_material->get_internal_space(), context);

    mi::base::Handle<mi::mdl::IGenerated_code_executable> code;
    switch (m_kind) {
    case mi::neuraylib::IMdl_backend_api::MB_LLVM_IR:
        code = mi::base::make_handle(
            m_jit->compile_into_llvm_ir(
                lambda.get(),
                &module_cache,
                &resolver,
                cg_ctx.get(),
                m_num_texture_spaces,
                m_num_texture_results,
                m_enable_simd));
        break;
    case mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX:
    case mi::neuraylib::IMdl_backend_api::MB_GLSL:
    case mi::neuraylib::IMdl_backend_api::MB_HLSL:
        code = mi::base::make_handle(
            m_jit->compile_into_source(
                m_code_cache.get(),
                lambda.get(),
                &module_cache,
                &resolver,
                cg_ctx.get(),
                m_num_texture_spaces,
                m_num_texture_results,
                m_sm_version,
                map_target_language(m_kind),
                !m_output_target_lang));
        break;
    case mi::neuraylib::IMdl_backend_api::MB_NATIVE:
        code = mi::base::make_handle(
            m_jit->compile_into_generic_function(
                lambda.get(),
                &module_cache,
                &resolver,
                cg_ctx.get(),
                m_num_texture_spaces,
                m_num_texture_results,
                /*transformer=*/NULL));
        break;
    default:
        break;
    }

    if (!code.is_valid_interface()) {
        MDL::add_error_message(context,
            "The backend failed to generate target code for the function.", -3);
        return NULL;
    }

    MDL::convert_and_log_messages(code->access_messages(), context);

    if (!code->is_valid()) {
        MDL::add_error_message(context,
            "The backend failed to generate target code for the function.", -3);
        return NULL;
    }

    Target_code *tc = new Target_code(
        code.get(),
        transaction,
        m_strings_mapped_to_ids,
        m_calc_derivatives,
        m_use_builtin_resource_handler,
        m_kind);

    // Enter the resource-table here
    fill_resource_tables(tc_reg, tc);

    if (compiled_material->get_parameter_count() != 0) {
        mi::base::Handle<const MI::MDL::IValue_list> args(
            compiled_material->get_arguments(transaction));
        tc->init_argument_block(0, transaction, args.get());
    }

    // copy data segments
    for (size_t i = 0, n = code->get_data_segment_count(); i < n; ++i) {
        mi::mdl::IGenerated_code_executable::Segment const *desc = code->get_data_segment(i);

        if (desc) {
            tc->add_ro_segment(
                desc->name,
                desc->data,
                desc->size);
        }
    }

    // copy the string constant table
    for (size_t i = 0, n = code->get_string_constant_count(); i < n; ++i) {
        tc->add_string_constant_index(i, code->get_string_constant(i));
    }

    return tc;
}

const mi::neuraylib::ITarget_code* Mdl_llvm_backend::translate_material_df(
    DB::Transaction* transaction,
    const MDL::Mdl_compiled_material* compiled_material,
    const char* path,
    const char* base_fname,
    MDL::Execution_context* context)
{
    if (!compiled_material->is_valid(transaction, context)) {
        MDL::add_error_message(context, "Compiled material is invalid.", -1);
        return NULL;
    }

    Lambda_builder lambda_builder(
        m_compiler.get(),
        transaction,
        m_compile_consts,
        m_calc_derivatives);

    // convert from an IExpression-based compiled material sub-expression
    // to a DAG_node-based distribution function consisting of
    //  - a main df containing the DF part and array and struct constants and
    //  - a number of expression lambdas containing the non-DF part
    // Note: We currently don't support storing expression lambdas of type double
    //       in GLSL/HLSL, so disable it.
    mi::base::Handle<mi::mdl::IDistribution_function> dist_func(
        lambda_builder.from_material_df(
            compiled_material,
            path,
            base_fname,
            get_context_option<bool>(context, MDL_CTX_OPTION_INCLUDE_GEO_NORMAL),
            /*allow_double_expr_lambda=*/!target_is_structured_language()));
    if (!dist_func.is_valid_interface()) {
       MDL::add_error_message(
           context, lambda_builder.get_error_string(), lambda_builder.get_error_code());
        return NULL;
    }

    mi::base::Handle<mi::mdl::ILambda_function> root_lambda(dist_func->get_root_lambda());

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(transaction, mdlc_module->get_module_wait_queue(), {});
    MDL::Mdl_call_resolver resolver(transaction);

    // ... enumerate resources: must be done before we compile ...
    //     all resource information will be collected in root_lambda
    bool resolve_resources = get_context_option<bool>(context, MDL_CTX_OPTION_RESOLVE_RESOURCES);
    Target_code_register tc_reg;
    Function_enumerator enumerator(tc_reg, root_lambda.get(), transaction,
        !resolve_resources, resolve_resources);
    for (size_t i = 0, n = dist_func->get_main_function_count(); i < n; ++i) {
        mi::base::Handle<mi::mdl::ILambda_function> main_func(
            dist_func->get_main_function(i));
        root_lambda->enumerate_resources(resolver, enumerator, main_func->get_body());
    }
    if (!resolve_resources)
        root_lambda->set_has_resource_attributes(false);

    size_t expr_lambda_count = dist_func->get_expr_lambda_count();
    for (size_t i = 0; i < expr_lambda_count; ++i) {
        mi::base::Handle<mi::mdl::ILambda_function> lambda(dist_func->get_expr_lambda(i));

        // also register the resources in lambda itself, so we see whether it accesses resources
        enumerator.set_additional_lambda(lambda.get());

        lambda->enumerate_resources(resolver, enumerator, lambda->get_body());
    }

    // ... also enumerate resources from arguments ...
    if (compiled_material->get_parameter_count() != 0) {
        tc_reg.set_in_argument_mode(true);
        lambda_builder.enumerate_resource_arguments(
            root_lambda.get(), compiled_material, enumerator);
    }

    // ... and compile
    mi::base::Handle<mi::mdl::ICode_generator_thread_context> cg_ctx(
        m_jit->create_thread_context());

    update_jit_context_options(*cg_ctx.get(), compiled_material->get_internal_space(), context);

    mi::base::Handle<mi::mdl::IGenerated_code_executable> code;
    switch (m_kind) {
    case mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX:
    case mi::neuraylib::IMdl_backend_api::MB_GLSL:
    case mi::neuraylib::IMdl_backend_api::MB_HLSL:
        code = mi::base::make_handle(
            m_jit->compile_distribution_function_gpu(
                dist_func.get(),
                &module_cache,
                &resolver,
                cg_ctx.get(),
                m_num_texture_spaces,
                m_num_texture_results,
                m_sm_version,
                map_target_language(m_kind),
                !m_output_target_lang));
        break;
    case mi::neuraylib::IMdl_backend_api::MB_NATIVE:
        code = mi::base::make_handle(
            m_jit->compile_distribution_function_cpu(
                dist_func.get(),
                &module_cache,
                &resolver,
                cg_ctx.get(),
                m_num_texture_spaces,
                m_num_texture_results));
        break;
    default:
        break;
    }

    if (!code.is_valid_interface()) {
        MDL::add_error_message(
            context, "The backend failed to generate target code for the material.", -3);
        return NULL;
    }

    MDL::convert_and_log_messages(code->access_messages(), context);

    if (!code->is_valid()) {
        MDL::add_error_message(
            context, "The backend failed to generate target code for the material.", -3);
        return NULL;
    }

    Target_code *tc = new Target_code(
        code.get(),
        transaction,
        m_strings_mapped_to_ids,
        m_calc_derivatives,
        m_use_builtin_resource_handler,
        m_kind);

    // Enter the resource-table here
    fill_resource_tables(tc_reg, tc);

    if (compiled_material->get_parameter_count() != 0) {
        mi::base::Handle<const MI::MDL::IValue_list> args(
            compiled_material->get_arguments(transaction));
        tc->init_argument_block(0, transaction, args.get());
    }

    // copy data segments
    for (size_t i = 0, n = code->get_data_segment_count(); i < n; ++i) {
        mi::mdl::IGenerated_code_executable::Segment const *desc = code->get_data_segment(i);

        if (desc) {
            tc->add_ro_segment(
                desc->name,
                desc->data,
                desc->size);
        }
    }

    // copy the string constant table
    for (size_t i = 0, n = code->get_string_constant_count(); i < n; ++i) {
        tc->add_string_constant_index(i, code->get_string_constant(i));
    }

    return tc;
}

mi::Uint8 const *Mdl_llvm_backend::get_device_library(
    mi::Size &size) const
{
    if (m_kind == mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX) {
        size_t              s  = 0;
        unsigned char const *r = m_jit->get_libdevice_for_gpu(s);

        size = mi::Size(s);
        return r;
    }
    size = 0;
    return NULL;
}

mi::mdl::ILink_unit *Mdl_llvm_backend::create_link_unit(
    MDL::Execution_context* context)
{
    mi::base::Handle<mi::mdl::ICode_generator_thread_context> cg_ctx(
        m_jit->create_thread_context());

    update_jit_context_options(*cg_ctx.get(), NULL, context);

    if (m_jit.is_valid_interface()) {
        return m_jit->create_link_unit(
            cg_ctx.get(),
            map_target_language(get_kind()),
            get_enable_simd(),
            get_sm_version(),
            get_num_texture_spaces(),
            get_num_texture_results());
    }
    return NULL;
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::translate_link_unit(
    Link_unit const *lu,
    MDL::Execution_context* context)
{
    mi::base::Handle<mi::mdl::ICode_generator_thread_context> cg_ctx(
        m_jit->create_thread_context());

    update_jit_context_options(*cg_ctx.get(), lu->get_internal_space(), context);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module(/*deferred=*/false);
    MDL::Module_cache module_cache(lu->get_transaction(), mdlc_module->get_module_wait_queue(), {});

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

    mi::base::Handle<mi::mdl::IGenerated_code_executable> code(m_jit->compile_unit(
        cg_ctx.get(),
        &module_cache,
        mi::base::make_handle(lu->get_compilation_unit()).get(),
        !m_output_target_lang));

    if (!code.is_valid_interface()) {
        MDL::add_error_message(context,
            "The JIT backend failed to compile the unit.", -2);
        return NULL;
    }

    MDL::convert_and_log_messages(code->access_messages(), context);

    if (!code->is_valid()) {
        MDL::add_error_message(context,
            "The JIT backend failed to compile the unit.", -2);
        return NULL;
    }

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

    mi::base::Handle<Target_code> tc(lu->get_target_code());
    tc->finalize(code.get(), lu->get_transaction(), m_calc_derivatives);

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
#endif

    // Enter the resource-table here
    fill_resource_tables(*lu->get_tc_reg(), tc.get());

    // Copy the string constant table: This must be done before the target argument
    // block is created, because it might contain string values (mapped to IDs).
    for (size_t i = 0, n_args = code->get_string_constant_count(); i < n_args; ++i) {
        tc->add_string_constant_index(i, code->get_string_constant(i));
    }

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
#endif

    // Create all target argument blocks, now that all resources are known
    {
        std::vector<mi::base::Handle<MDL::IValue_list const> > const &args =
            lu->get_arg_block_comp_material_args();
        DB::Transaction *trans = lu->get_transaction();
        for (size_t i = 0, n_args = args.size(); i < n_args; ++i) {
            tc->init_argument_block(
                i,
                trans,
                args[i].get());
        }
    }

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
#endif

    // copy data segments
    for (size_t i = 0, n = code->get_data_segment_count(); i < n; ++i) {
        mi::mdl::IGenerated_code_executable::Segment const *desc = code->get_data_segment(i);

        if (desc) {
            tc->add_ro_segment(
                desc->name,
                desc->data,
                desc->size);
        }
    }

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();

    std::chrono::duration<double> et = t6 - t1;
    printf("TLU |||| Total time                 : %f seconds.\n", et.count());

    et = t2 - t1;
    printf("TLU | Compile unit                  : %f seconds.\n", et.count());

    et = t3 - t2;
    printf("TLU | Finalize target code          : %f seconds.\n", et.count());

    et = t4 - t3;
    printf("TLU | Fill res/string tables        : %f seconds.\n", et.count());

    et = t5 - t4;
    printf("TLU | Init argument block           : %f seconds.\n", et.count());

    et = t6 - t5;
    printf("TLU | Add ro data segment           : %f seconds.\n", et.count());
#endif

    tc->retain();
    return tc.get();
}

mi::base::Lock Df_data_helper::m_lock;

Df_data_helper::Df_data_map Df_data_helper::m_df_data_to_name =
{
    { mi::mdl::IValue_texture::Bsdf_data_kind::BDK_BECKMANN_VC_MULTISCATTER,            "bsdf_data_beckmann_vc_ms_texture" },
    { mi::mdl::IValue_texture::Bsdf_data_kind::BDK_BECKMANN_SMITH_MULTISCATTER,         "bsdf_data_beckmann_smith_ms_texture" },
    { mi::mdl::IValue_texture::Bsdf_data_kind::BDK_GGX_VC_MULTISCATTER,                 "bsdf_data_ggx_vcavities_ms_texture" },
    { mi::mdl::IValue_texture::Bsdf_data_kind::BDK_GGX_SMITH_MULTISCATTER,              "bsdf_data_ggx_smith_ms_texture" },
    { mi::mdl::IValue_texture::Bsdf_data_kind::BDK_WARD_GEISLER_MORODER_MULTISCATTER,   "bsdf_data_geisler_moroder_ms_texture" },
    { mi::mdl::IValue_texture::Bsdf_data_kind::BDK_SIMPLE_GLOSSY_MULTISCATTER,          "bsdf_data_simple_glossy_ms_texture" },
    { mi::mdl::IValue_texture::Bsdf_data_kind::BDK_BACKSCATTERING_GLOSSY_MULTISCATTER,  "bsdf_data_bs_glossy_ms_texture" },
    { mi::mdl::IValue_texture::Bsdf_data_kind::BDK_SHEEN_MULTISCATTER,                  "bsdf_data_sheen_ms_texture" }
};

DB::Tag Df_data_helper::store_df_data(mi::mdl::IValue_texture::Bsdf_data_kind df_data_kind)
{
    mi::base::Lock::Block block(&m_lock);

    size_t rx=0, ry=0, rz=0;
    const auto& entry = m_df_data_to_name.find(df_data_kind);
    ASSERT(M_BACKENDS, entry != m_df_data_to_name.end());

    std::string &bsdf_tex_name = entry->second;

    DB::Tag tag = m_transaction->name_to_tag(bsdf_tex_name.c_str());
    if (!tag.is_valid()) {

        mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data_resolution(df_data_kind, rx, ry, rz);
        size_t s;
        unsigned char const *data = mi::mdl::libbsdf_data::get_libbsdf_multiscatter_data(df_data_kind, s);
        tag = store_texture(rx, ry, rz, reinterpret_cast<const float*>(data), bsdf_tex_name);
    }
    return tag;
}

DB::Tag Df_data_helper::store_texture(
    mi::Uint32 rx,
    mi::Uint32 ry,
    mi::Uint32 rz,
    const float *data,
    const std::string& tex_name)
{
    std::vector<mi::base::Handle<mi::neuraylib::ITile>> tiles(rz);
    mi::Uint32 offset = rx * ry;
    for (mi::Size i = 0; i < rz; ++i)
        tiles[i] = new Df_data_tile(rx, ry, &data[i*offset]);

    SYSTEM::Access_module<IMAGE::Image_module> image_module(false);
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(image_module->create_canvas(tiles, 1.0f));

    std::vector<mi::base::Handle<mi::neuraylib::ICanvas>> canvases(1);
    canvases[0] = canvas;
    mi::base::Handle<IMAGE::IMipmap> mipmap(image_module->create_mipmap(canvases));

    DBIMAGE::Image *img = new DBIMAGE::Image();
    img->set_mipmap(m_transaction, mipmap.get(), /*selector*/ nullptr, mi::base::Uuid{0,0,0,0});
    std::string img_name = tex_name + "_img";
    DB::Tag img_tag = m_transaction->store_for_reference_counting(img, img_name.c_str());

    TEXTURE::Texture *tex = new TEXTURE::Texture();
    tex->set_gamma(1.0f);
    tex->set_image(img_tag);
    DB::Tag tex_tag = m_transaction->store(tex, tex_name.c_str());

    return tex_tag;
}

char const* Df_data_helper::get_texture_db_name(mi::mdl::IValue_texture::Bsdf_data_kind kind)
{
    auto const &entry = m_df_data_to_name.find(kind);
    if (entry == m_df_data_to_name.end())
        return nullptr;
    return entry->second.c_str();
}

void Df_data_helper::Df_data_tile::get_pixel(
    mi::Uint32 x_offset,
    mi::Uint32 y_offset,
    mi::Float32* floats) const
{
    mi::Size p = ((y_offset * (mi::Size)m_resolution_x) + x_offset);
    floats[0] = floats[1] = floats[2] = m_data[p];
    floats[3] = 1.0f;
}

void Df_data_helper::Df_data_tile::set_pixel(
    mi::Uint32 x_offset,
    mi::Uint32 y_offset,
    const  mi::Float32* floats)
{
    // pixel data cannot be changed.
    ASSERT( M_BACKENDS, false);
    return;
}

const char* Df_data_helper::Df_data_tile::get_type() const
{
    return "Float32";
}

mi::Uint32 Df_data_helper::Df_data_tile::get_resolution_x() const
{
    return m_resolution_x;
}

mi::Uint32 Df_data_helper::Df_data_tile::get_resolution_y() const
{
    return m_resolution_y;
}

const void* Df_data_helper::Df_data_tile::get_data() const
{
    return m_data;
}

void* Df_data_helper::Df_data_tile::get_data()
{
    // pixel data cannot be changed.
    ASSERT( M_BACKENDS, false);
    return nullptr;
}

} // namespace BACKENDS

} // namespace MI

