/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

// examples/mdl_core/shared/example_shared_backends.h
//
// Code shared by some examples

#ifndef EXAMPLE_SHARED_BACKENDS_H
#define EXAMPLE_SHARED_BACKENDS_H

#include "example_shared.h"

#include <OpenImageIO/imageio.h>

/// Representation of textures, holding meta and image data.
class Texture_data
{
public:
    /// Constructor for an invalid texture.
    Texture_data()
        : m_path()
        , m_gamma_mode(mi::mdl::IValue_texture::gamma_default)
        , m_shape(mi::mdl::IType_texture::TS_2D)
        , m_bsdf_data(nullptr)
        , m_width(0)
        , m_height(0)
        , m_depth(0)
    {
    }

    /// Constructor from an MDL IValue_texture.
    Texture_data(
        mi::mdl::IValue_texture const* tex,
        mi::mdl::ICode_generator_jit* jit_be,
        mi::mdl::IEntity_resolver* resolver)
        : m_path(tex->get_string_value())
        , m_gamma_mode(tex->get_gamma_mode())
        , m_shape(tex->get_type()->get_shape())
        , m_bsdf_data(nullptr)
        , m_width(0)
        , m_height(0)
        , m_depth(0)
    {
        if (tex->get_bsdf_data_kind() != mi::mdl::IValue_texture::BDK_NONE) {
            load_bsdf_data(tex, jit_be);
            return;
        }

        // Our OpenImageIO plugin supports only 2D textures
        if (m_shape != mi::mdl::IType_texture::TS_2D)
            return;

        load_image(resolver);
    }

    /// Constructor from a file path.
    Texture_data(
        char const* path,
        mi::mdl::IEntity_resolver* resolver)
        : m_path(path)
        , m_gamma_mode(mi::mdl::IValue_texture::gamma_default)
        , m_shape(mi::mdl::IType_texture::TS_2D)
        , m_bsdf_data(nullptr)
        , m_width(0)
        , m_height(0)
        , m_depth(0)
    {
        load_image(resolver);
    }

    /// Get texture url
    const char* get_url() const { return m_path.c_str(); }

    /// Returns true, if the texture was loaded successfully.
    bool is_valid() const { return m_image || m_bsdf_data != nullptr; }

    /// Get the texture width.
    mi::Uint32 get_width() const { return m_width; }

    /// Get the texture height.
    mi::Uint32 get_height() const { return m_height; }

    /// Get the texture depth. Currently only depth 1 is supported.
    mi::Uint32 get_depth() const { return m_depth; }

    /// Get the pixel format or nullptr.
    const char* get_pixel_type() const {
        return m_pixel_type.empty() ? nullptr : m_pixel_type.c_str();
    }

    /// Get the gamma mode.
    mi::mdl::IValue_texture::gamma_mode get_gamma_mode() const { return m_gamma_mode; }

    /// Get the texture shape.
    mi::mdl::IType_texture::Shape get_shape() const { return m_shape; }

    /// Get the texture image data.
    std::shared_ptr<OIIO::ImageInput> get_image() const { return m_image; }

    /// Get the bsdf texture data if present.
    unsigned char const* get_bsdf_data() const { return m_bsdf_data; }

private:
    /// Load the image and get the meta data.
    ///
    /// \param resolver  the entity resolver allowing to find resources in the configured
    ///                  MDL search path and supporting MDL archives
    void load_image(mi::mdl::IEntity_resolver* resolver)
    {
        mi::base::Handle<mi::mdl::IMDL_resource_set> resource_set(
            resolver->resolve_resource_file_name(
                m_path.c_str(),
                /*owner_file_path=*/ nullptr,
                /*owner_name=*/ nullptr,
                /*pos=*/ nullptr,
                /*ctx*/ nullptr));
        if (!resource_set)
            return;

        if (resource_set->get_udim_mode() != mi::mdl::NO_UDIM || resource_set->get_count() != 1)
            return;

        mi::base::Handle<mi::mdl::IMDL_resource_element const> elem(resource_set->get_element(0));
        if (!elem || elem->get_count() != 1)
            return;

        const char* filename = elem->get_filename(0);
        m_image = OIIO::ImageInput::open(filename);
        if (!m_image)
            return;

        const OIIO::ImageSpec& spec = m_image->spec();
        m_width = spec.width;
        m_height = spec.height;
        m_depth = 1;
    }

    /// Load the bsdf data.
    void load_bsdf_data(mi::mdl::IValue_texture const* tex, mi::mdl::ICode_generator_jit* jit_be)
    {
        size_t res_u = 0, res_v = 0, res_w = 0;
        size_t data_size = 0;
        const char* pixel_type;

        mi::mdl::IValue_texture::Bsdf_data_kind bdk = tex->get_bsdf_data_kind();
        switch (bdk)
        {
        case mi::mdl::IValue_texture::BDK_MICROFLAKE_SHEEN_GENERAL:
            if (!jit_be->get_libbsdf_general_data_resolution(bdk, res_u, res_v, res_w, pixel_type))
                return;
            m_bsdf_data = jit_be->get_libbsdf_general_data(bdk, data_size);
            break;

        default:  // all other bsdf data tables are used for multi-scattering
            if (!jit_be->get_libbsdf_multiscatter_data_resolution(
                bdk, res_u, res_v, res_w, pixel_type))
                return;
            m_bsdf_data = jit_be->get_libbsdf_multiscatter_data(bdk, data_size);
            break;
        }

        if (m_bsdf_data == nullptr)
            return;

        m_width = res_u;
        m_height = res_v;
        m_depth = res_w;
        m_pixel_type = pixel_type;
    }

private:
    std::string m_path;
    mi::mdl::IValue_texture::gamma_mode m_gamma_mode;
    mi::mdl::IType_texture::Shape m_shape;
    std::shared_ptr<OIIO::ImageInput> m_image;
    unsigned char const* m_bsdf_data;
    mi::Uint32 m_width;
    mi::Uint32 m_height;
    mi::Uint32 m_depth;
    std::string m_pixel_type;
};


/// Helper class to handle the string table of a target code, allowing it to grow
/// when new user strings are registered.
class String_constant_table
{
    typedef std::map<std::string, unsigned> String_map;
public:
    /// Constructor.
    String_constant_table()
        : m_string_constants_map()
        , m_strings()
        , m_max_len(0u)
    {
    }

    /// Get the ID for a given string, return 0 if the string does not exist in the table.
    unsigned lookup_id_for_string(const char* name) const {
        String_map::const_iterator it(m_string_constants_map.find(name));
        if (it != m_string_constants_map.end())
            return it->second;

        return 0u;
    }

    /// Get the ID for a given string, register it if it does not exist, yet.
    unsigned get_id_for_string(const char* name) {
        String_map::const_iterator it(m_string_constants_map.find(name));
        if (it != m_string_constants_map.end())
            return it->second;

        // the user adds a string that is NOT in the code and we have not seen so far, add it
        // and assign a new id
        unsigned n_id = unsigned(m_string_constants_map.size() + 1);

        m_string_constants_map[name] = n_id;
        m_strings.reserve((n_id + 63) & ~63);
        m_strings.push_back(name);

        size_t l = strlen(name);
        if (l > m_max_len)
            m_max_len = l;
        return n_id;
    }

    /// Get the length of the longest string in the string constant table.
    size_t get_max_length() const { return m_max_len; }

    /// Get number of strings
    size_t get_number_of_strings() const {
        return m_strings.size();
    }

    /// Get the string for a given ID, or nullptr if this ID does not exist.
    const char* get_string(unsigned id) {
        if (id == 0 || id - 1 >= m_strings.size())
            return nullptr;
        return m_strings[id - 1].c_str();
    }

    /// Get all string constants used inside a target code and their maximum length.
    void get_all_strings(
        const mi::mdl::IGenerated_code_executable* target_code)
    {
        m_max_len = 0;
        // ignore the 0, it is the "Not-a-known-string" entry
        m_strings.reserve(target_code->get_string_constant_count());
        for (size_t i = 1, n = target_code->get_string_constant_count(); i < n; ++i) {
            const char* s = target_code->get_string_constant(i);
            size_t l = strlen(s);
            if (l > m_max_len)
                m_max_len = l;
            m_string_constants_map[s] = (unsigned)i;
            m_strings.push_back(s);
        }
    }

private:
    String_constant_table(String_constant_table const&) = delete;
    String_constant_table& operator=(String_constant_table const&) = delete;

private:
    String_map               m_string_constants_map;
    std::vector<std::string> m_strings;
    size_t                   m_max_len;
};


/// Class responsible for managing resources.
class Resource_collection : public mi::mdl::ILambda_resource_enumerator,
    public mi::mdl::IGenerated_code_value_callback
{
public:
    /// Constructor.
    ///
    /// \param mdl            the MDL compiler
    /// \param jit_be         the JIT backend
    /// \param call_resolver  the call name resolver
    Resource_collection(
        mi::mdl::IMDL* mdl,
        mi::mdl::ICode_generator_jit* jit_be,
        mi::mdl::ICall_name_resolver& call_resolver)
        : m_jit_be(jit_be, mi::base::DUP_INTERFACE)
        , m_entity_resolver(mdl->create_entity_resolver(nullptr))
        , m_call_resolver(call_resolver)
        , m_cur_lambda()
        , m_textures()
        , m_texture_map()
        , m_string_constant_table()
    {
    }

    /// Destructor.
    ~Resource_collection()
    {
        for (auto tex : m_textures)
            delete tex;
    }

    /// Called for an enumerated texture resource.
    /// Registers the texture in this collection and in the current lambda, if set.
    ///
    /// \param t          the texture resource or an invalid_ref
    /// \param tex_usage  the potential usage of the texture
    virtual void texture(mi::mdl::IValue const* t, Texture_usage tex_usage) override
    {
        texture_impl(t);
    }

    /// Called for an enumerated light profile resource.
    ///
    /// \param t  the light profile resource or an invalid_ref
    virtual void light_profile(mi::mdl::IValue const* t) override
    {
        auto lp = mi::mdl::as<mi::mdl::IValue_light_profile>(t);

        // not supported in this example
        std::cerr << "warning: Light profiles are not supported by the MDL Core examples.\n"
            "         However, the loaded material references the light profile:\n"
            << "         " << lp->get_string_value() << "\n";
        (void)t;
    }

    /// Called for an enumerated bsdf measurement resource.
    ///
    /// \param t  the bsdf measurement resource or an invalid_ref
    virtual void bsdf_measurement(mi::mdl::IValue const* t) override
    {
        auto bm = mi::mdl::as<mi::mdl::IValue_bsdf_measurement>(t);

        // not supported in this example
        std::cerr << "warning: Measured BSDFs are not supported by the MDL Core examples.\n"
            "         However, the loaded material references the BSDF measurement:\n"
            << "         " << bm->get_string_value() << "\n";
        (void)t;
    }

    /// Sets the current lambda used for registering the resources.
    void set_current_lambda(mi::mdl::ILambda_function* lambda)
    {
        m_cur_lambda = mi::base::make_handle_dup(lambda);
    }

    /// Collect and map all resources used by the given lambda function.
    void collect(mi::mdl::ILambda_function* lambda)
    {
        mi::base::Handle<mi::mdl::ILambda_function> old_lambda = m_cur_lambda;
        m_cur_lambda = mi::base::make_handle_dup(lambda);

        lambda->enumerate_resources(m_call_resolver, *this, lambda->get_body());

        m_cur_lambda = old_lambda;
    }

    /// Collect and map all resources used by the given distribution function.
    void collect(mi::mdl::IDistribution_function* dist_func)
    {
        mi::base::Handle<mi::mdl::ILambda_function> root_lambda(dist_func->get_root_lambda());
        mi::base::Handle<mi::mdl::ILambda_function> old_lambda = m_cur_lambda;

        // all resources will be registered in the main lambda
        m_cur_lambda = mi::base::make_handle_dup(root_lambda.get());

        for (size_t i = 0, n = dist_func->get_main_function_count(); i < n; ++i) {
            mi::base::Handle<mi::mdl::ILambda_function> main_func(dist_func->get_main_function(i));
            root_lambda->enumerate_resources(m_call_resolver, *this, main_func->get_body());
        }

        for (size_t i = 0, n = dist_func->get_expr_lambda_count(); i < n; ++i) {
            mi::base::Handle<mi::mdl::ILambda_function> expr_lambda(dist_func->get_expr_lambda(i));
            expr_lambda->enumerate_resources(m_call_resolver, *this, expr_lambda->get_body());
        }

        m_cur_lambda = old_lambda;
    }

    /// Returns the textures list.
    std::vector<Texture_data*> const& get_textures() const {
        return m_textures;
    }

    /// Returns the resource index for the given resource value usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// \param resource  the resource value
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    virtual mi::Uint32 get_resource_index(mi::mdl::IValue_resource const* resource) override
    {
        switch (resource->get_kind()) {
        case mi::mdl::IValue::VK_TEXTURE:
            return texture_impl(resource);
        default:
            break;
        }
        return 0;
    }

    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    virtual mi::Uint32 get_string_index(mi::mdl::IValue_string const* s) override
    {
        return m_string_constant_table.lookup_id_for_string(s->get_value());
    }

    /// Collect all string from the given executable code.
    void collect_strings(mi::base::Handle<const mi::mdl::IGenerated_code_executable> code)
    {
        m_string_constant_table.get_all_strings(code.get());
    }

    /// Get the string table, so it can grow.
    String_constant_table& get_string_constant_table() const {
        return m_string_constant_table;
    }

private:
    /// Called for an enumerated texture resource.
    /// Registers the texture in this collection and in the current lambda, if set.
    ///
    /// \param t  the texture resource or an invalid_ref
    ///
    /// \returns the resource index or 0 if no resource index can be returned
    mi::Uint32 texture_impl(mi::mdl::IValue const* t)
    {
        mi::mdl::IValue_texture const* tex_val = mi::mdl::as<mi::mdl::IValue_texture>(t);
        if (tex_val == nullptr)
            return 0;

        // add the invalid texture, if this is the first texture
        if (m_textures.empty()) {
            m_textures.push_back(new Texture_data());
        }

        Texture_data* tex;
        size_t index;

        std::string cache_key = std::string(tex_val->get_string_value());
        cache_key += "_" + std::to_string(tex_val->get_gamma_mode());
        cache_key += "_" + std::to_string(tex_val->get_bsdf_data_kind());
        auto it = m_texture_map.find(cache_key);
        if (it == m_texture_map.end()) {
            tex = new Texture_data(tex_val, m_jit_be.get(), m_entity_resolver.get());
            m_textures.push_back(tex);
            index = m_textures.size() - 1;
            m_texture_map[cache_key] = unsigned(index);
        }
        else {
            index = size_t(it->second);
            tex = m_textures[index];
        }

        // is there a lambda function to register the texture?
        if (m_cur_lambda) {
            if (tex->is_valid()) {
                m_cur_lambda->map_tex_resource(
                    tex_val->get_kind(),
                    tex_val->get_string_value(),
                    tex_val->get_selector(),
                    tex_val->get_gamma_mode(),
                    tex_val->get_bsdf_data_kind(),
                    tex_val->get_type()->get_shape(),
                    tex_val->get_tag_value(),
                    index,
                    /*valid=*/ true,
                    tex->get_width(),
                    tex->get_height(),
                    tex->get_depth());
            }
            else {
                // invalid texture are always mapped to zero
                m_cur_lambda->map_tex_resource(
                    tex_val->get_kind(),
                    tex_val->get_string_value(),
                    tex_val->get_selector(),
                    tex_val->get_gamma_mode(),
                    tex_val->get_bsdf_data_kind(),
                    tex_val->get_type()->get_shape(),
                    tex_val->get_tag_value(),
                    0,
                    /*valid=*/ false,
                    0,
                    0,
                    0);
            }
        }

        return mi::Uint32(index);
    }

    Resource_collection(Resource_collection const&) = delete;
    Resource_collection& operator=(Resource_collection const&) = delete;

private:
    /// The MDL JIT backend to retrieve BSDF texture data.
    mi::base::Handle<mi::mdl::ICode_generator_jit> m_jit_be;

    /// The MDL entity resolver for accessing resources.
    mi::base::Handle<mi::mdl::IEntity_resolver> m_entity_resolver;

    /// The MDL call name resolver.
    mi::mdl::ICall_name_resolver& m_call_resolver;

    /// The current lambda for which resources should be registered.
    mi::base::Handle<mi::mdl::ILambda_function> m_cur_lambda;

    /// List of loaded textures.
    std::vector<Texture_data*> m_textures;

    /// Map from texture paths to indices into m_textures.
    std::map<std::string, unsigned> m_texture_map;

    /// The string constant table, we allow it to grow by the user.
    mutable String_constant_table m_string_constant_table;
};


/// Helper class for target argument blocks.
class Argument_block
{
public:
    /// Constructor.
    /// The block will be initialized with the defaults from the material instance.
    ///
    /// \param mat_instance  the material instance for the parameter defaults
    /// \param layout        the layout of the target argument block to initialize
    /// \param res_col       the resource collection to resolve resource IDs
    Argument_block(
        mi::mdl::IMaterial_instance const* mat_instance,
        mi::mdl::IGenerated_code_value_layout const* layout,
        Resource_collection& res_col)
        : m_is_valid(false)
    {
        m_data.resize(layout->get_size());

        for (size_t i = 0, n = mat_instance->get_parameter_count(); i < n; ++i) {
            mi::mdl::IGenerated_code_value_layout::State state = layout->get_nested_state(i);
            if (layout->set_value(
                m_data.data(),
                mat_instance->get_parameter_default(i),
                &res_col,
                state) < 0)
            {
                m_data.clear();
                return;
            }
        }

        m_is_valid = true;
    }

    /// Copy constructor.
    Argument_block(Argument_block const& other)
        : m_data(other.m_data)
        , m_is_valid(other.m_is_valid)
    {
    }

    ~Argument_block() {};

    /// Get the writable argument block data.
    char* get_data() { return m_data.data(); }

    /// Get the argument block data.
    char const* get_data() const { return m_data.data(); }

    /// Get the size of the argument block data.
    size_t get_size() const { return m_data.size(); }

    /// Return whether the initialization of the argument block data was successful.
    bool is_valid() const { return m_is_valid; }

private:
    std::vector<char> m_data;
    bool m_is_valid;
};


/// Class managing the generated code of a link unit and the used resources and materials.
class Target_code
{
public:
    /// Constructor.
    Target_code(
        mi::mdl::IGenerated_code_executable*  code,
        mi::mdl::ILink_unit*                  link_unit,
        Resource_collection const&            res_col,
        std::vector<Material_instance> const& mat_instances,
        std::vector<size_t> const&            arg_block_indexes,
        std::vector<Argument_block> const&    arg_blocks)
        : m_code(code, mi::base::DUP_INTERFACE)
        , m_link_unit(link_unit, mi::base::DUP_INTERFACE)
        , m_res_col(res_col)
        , m_mat_instances(mat_instances)
        , m_arg_block_indexes(arg_block_indexes)
        , m_arg_blocks(arg_blocks)
    {
        m_code_lambda = code->get_interface<mi::mdl::IGenerated_code_lambda_function>();

        size_t code_size = 0;
        char const* code_data = code->get_source_code(code_size);
        m_source_code.assign(code_data, code_data + code_size);
    }

    /// Get the generated source code.
    char const* get_src_code() const { return m_source_code.c_str(); }

    /// Get the length of the generated source code (including last '\0')
    size_t get_src_code_size() const { return m_source_code.size() + 1; }

    /// Return the number of callable functions inside the generated code.
    size_t get_callable_function_count() const {
        return m_link_unit->get_function_count();
    }

    /// Get the kind of the i'th callable function.
    mi::mdl::IGenerated_code_executable::Function_kind
        get_callable_function_kind(size_t index) const {
        return m_link_unit->get_function_kind(index);
    }

    /// Get the DF kind of the i'th callable function.
    mi::mdl::IGenerated_code_executable::Distribution_kind
        get_callable_function_df_kind(size_t index) const {
        return m_link_unit->get_distribution_kind(index);
    }

    /// Get the name of the i'th callable function.
    const char* get_callable_function(size_t index) const {
        return m_link_unit->get_function_name(index);
    }

    /// Get the size of the argument block of the i'th callable function.
    size_t get_callable_function_argument_block_index(size_t index) const {
        return m_link_unit->get_function_arg_block_layout_index(index);
    }

    /// Get the prototype of the i'th callable function in the given language.
    std::string get_callable_function_prototype(
        size_t index,
        mi::mdl::IGenerated_code_executable::Prototype_language lang) const
    {
        return std::string(m_link_unit->get_function_prototype(index, lang));
    }

    /// Get the list of argument block indices per material.
    std::vector<size_t> const& get_argument_block_indices() const {
        return m_arg_block_indexes;
    }

    /// Get the number of argument blocks.
    size_t get_argument_block_count() const {
        return m_arg_blocks.size();
    }

    /// Get the argument block at the given index.
    Argument_block const* get_argument_block(size_t index) const {
        return &m_arg_blocks[index];
    }

    /// Get the layout of the argument block at the given index.
    mi::mdl::IGenerated_code_value_layout const* get_argument_block_layout(size_t index) const {
        return m_link_unit->get_arg_block_layout(index);
    }

    /// Get the number of material instances.
    size_t get_material_instance_count() const {
        return m_mat_instances.size();
    }

    /// Get the material instance at the given index.
    Material_instance const& get_material_instance(size_t index) const {
        return m_mat_instances[index];
    }

    /// Get the data for the read-only data segment if available.
    ///
    /// \param size  will be assigned to the length of the RO data segment
    /// \returns the data segment or nullptr if no RO data segment is available.
    char const* get_ro_data_segment(size_t& size) const {
        mi::mdl::IGenerated_code_executable::Segment const* desc = m_code->get_data_segment(0);
        if (desc != nullptr) {
            size = desc->size;
            return (char const*)desc->data;
        }
        size = 0u;
        return nullptr;
    }

    /// Get the number of textures in the resource collection.
    size_t get_texture_count() const {
        return m_res_col.get_textures().size();
    }

    /// Get the texture at the given index.
    Texture_data const* get_texture(size_t index) const {
        return m_res_col.get_textures()[index];
    }

    /// Get the IGenerated_code_executable object.
    mi::base::Handle<mi::mdl::IGenerated_code_executable const> get_code() const {
        return m_code;
    }

    /// Get the IGenerated_code_lambda function object.
    mi::base::Handle<mi::mdl::IGenerated_code_lambda_function> get_code_lambda() const {
        return m_code_lambda;
    }
    /// Get the string constant table.
    String_constant_table& get_string_constant_table() {
        return m_res_col.get_string_constant_table();
    }

private:
    mi::base::Handle<mi::mdl::IGenerated_code_executable const>       m_code;
    mi::base::Handle<mi::mdl::IGenerated_code_lambda_function>        m_code_lambda;
    mi::base::Handle<mi::mdl::ILink_unit const>                       m_link_unit;
    Resource_collection const& m_res_col;
    std::vector<Material_instance>                               m_mat_instances;
    std::vector<size_t>                                          m_arg_block_indexes;
    std::vector<Argument_block>                                  m_arg_blocks;
    std::string                                                  m_source_code;
};

//------------------------------------------------------------------------------
//
// MDL material compilation code
//
//------------------------------------------------------------------------------

struct Target_function_description
{
    Target_function_description(
        const char* expression_path    = nullptr,
        const char* base_function_name = nullptr)
        : path(expression_path)
        , base_fname(base_function_name)
        , argument_block_index(~0)
        , function_index(~0)
        , distribution_kind(mi::mdl::IGenerated_code_executable::DK_INVALID)
        , return_code(~0) // not processed
    {
    }

    /// The path from the material root to the expression that should be translated,
    /// e.g., \c "surface.scattering".
    const char* path;

    /// The base name of the generated functions.
    /// If \c nullptr is passed, the function name will be 'lambda' followed by an increasing
    /// counter. Note, that this counter is tracked per link unit. That means, you need to
    /// provide functions names when using multiple link units in order to avoid collisions.
    const char* base_fname;

    /// The index of argument block that belongs to the compiled material the function is
    /// generated from or ~0 if none of the added function required arguments.
    /// It allows to get the layout and a writable pointer to argument data. This is an output
    /// parameter which is available after adding the function to the link unit.
    size_t argument_block_index;

    /// The index of the generated function for accessing the callable function information of
    /// the link unit or ~0 if the selected function is an invalid distribution function.
    /// ~0 is not an error case, it just means, that evaluating the function will result in 0.
    /// In case the function is a distribution function, the returned index will be the
    /// index of the \c init function, while \c sample, \c evaluate, and \c pdf will be
    /// accessible by the consecutive indices, i.e., function_index + 1, function_index + 2,
    /// function_index + 3. This is an output parameter which is available after adding the
    /// function to the link unit.
    size_t function_index;

    /// Return the distribution kind of this function (or NONE in case expressions). This is
    /// an output parameter which is available after adding the function to the link unit.
    mi::mdl::IGenerated_code_executable::Distribution_kind distribution_kind;

    /// Return code of the processing of the function:
    /// -   0  Success
    /// -  ~0  Function not processed
    /// -  -1  An error occurred while processing the function.
    int return_code;
};

enum Backend_options
{
    BACKEND_OPTIONS_NONE = 0,
    // if true, the generated code will expect the renderer to provide
    // a state and a texture runtime with derivatives.
    BACKEND_OPTIONS_ENABLE_DERIVATIVES = 1 << 0,
    // if true, it disables the generation of separate PDF functions.
    BACKEND_OPTIONS_DISABLE_PDF = 1 << 1,
    // if true, it generates auxiliary methods on distribution functions.
    BACKEND_OPTIONS_ENABLE_AUX = 1 << 2,
    // enable a warning if a spectrum color is converted into an RGB.
    BACKEND_OPTIONS_WARN_SPECTRUM_CONVERSION = 1 << 3,
    // enable using a renderer provided function to adapt normals.
    BACKEND_OPTIONS_ADAPT_NORMAL = 1 << 4,
    // enable using a renderer provided function to adapt microfacet
    // roughness.
    BACKEND_OPTIONS_ADAPT_MICROFACET_ROUGHNESS = 1 << 5,
    // if true, it enables the creation of the read-only data segments for bigger constants
    BACKEND_OPTIONS_ENABLE_RO_SEGMENT = 1 << 6,
    // if enabled, the generated code will use the optional "flags" field in the BSDF data structs
    BACKEND_OPTIONS_ENABLE_BSDF_FLAGS = 1 << 7,
};

class Material_backend_compiler : public Material_compiler {
public:
    /// Constructor.
    ///
    /// \param mdl_compiler         the MDL compiler interface
    /// \param target_backend       the target backend for the generated code
    ///                             (PTX, HLSL, GLSL, LLVM_IR)
    /// \param num_texture_results  the size of a renderer provided array for texture results
    ///                             in the MDL shading state in number of float4 elements
    ///                             processed by the init() function of distribution functions
    /// \param backend_options      material backend compiler flag options
    /// \param df_handle_mode       controls how the handles of distribution functions can be used
    /// \param lambda_return_mode   selects how results are returned by lambda functions
    Material_backend_compiler(
        mi::mdl::IMDL* mdl_compiler,
        mi::mdl::ICode_generator::Target_language           target_backend,
        unsigned                                            num_texture_results,
        mi::Uint32                                          backend_options,
        const std::string&                                  df_handle_mode,
        const std::string&                                  lambda_return_mode,
        const std::unordered_map<std::string, std::string>& additional_be_options = {})
        : Material_compiler(mdl_compiler)
        , m_target_backend(target_backend)
        , m_jit_be(mi::base::make_handle(mdl_compiler->load_code_generator("jit"))
            .get_interface<mi::mdl::ICode_generator_jit>())
        , m_link_unit()
        , m_res_col(mdl_compiler, m_jit_be.get(), m_module_manager)
        , m_enable_derivatives(backend_options & BACKEND_OPTIONS_ENABLE_DERIVATIVES)
        , m_gen_base_name_suffix_counter(0)
    {
        // Set the JIT backend options: we use a private code generator here, so it is safe to
        // modify the backend options and ignore thread contexts
        mi::mdl::Options& options = m_jit_be->access_options();

        if (backend_options & BACKEND_OPTIONS_ENABLE_RO_SEGMENT) {
            // Option "enable_ro_segment": Default is disabled.
            // If you have a lot of big arrays, enabling this might speed up compilation.
            options.set_option(MDL_JIT_OPTION_ENABLE_RO_SEGMENT, "true");
        }

        if (backend_options & BACKEND_OPTIONS_ENABLE_BSDF_FLAGS) {
            // Option "enable_bsdf_flags": Default is disabled.
            // Add a flags field in the BSDF data structures in libbsdf.
            options.set_option(MDL_JIT_OPTION_LIBBSDF_FLAGS_IN_BSDF_DATA, "true");
        }

        if (backend_options & BACKEND_OPTIONS_ENABLE_DERIVATIVES) {
            // Option "jit_tex_runtime_with_derivs": Default is disabled.
            // We enable it to get coordinates with derivatives for texture lookup functions.
            options.set_option(MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES, "true");
        }

        if (backend_options & BACKEND_OPTIONS_DISABLE_PDF) {
            // Option "jit_enable_pdf": Default is enabled.
            options.set_option(MDL_JIT_OPTION_ENABLE_PDF, "false");
        }

        if (backend_options & BACKEND_OPTIONS_ENABLE_AUX) {
            // Option "jit_enable_auxiliary": Default is disabled.
            options.set_option(MDL_JIT_OPTION_ENABLE_AUXILIARY, "true");
        }

        if (backend_options & BACKEND_OPTIONS_WARN_SPECTRUM_CONVERSION) {
            // Option "jit_warn_spectrum_conversion": Default is disabled.
            options.set_option(MDL_JIT_WARN_SPECTRUM_CONVERSION, "true");
        }

        if (backend_options & BACKEND_OPTIONS_ADAPT_NORMAL) {
            // Option "jit_use_renderer_adapt_normal": Default is disabled.
            options.set_option(MDL_JIT_OPTION_USE_RENDERER_ADAPT_NORMAL, "true");
        }

        if (backend_options & BACKEND_OPTIONS_ADAPT_MICROFACET_ROUGHNESS) {
            // Option "jit_use_renderer_adapt_microfacet_roughness": Default is disabled.
            options.set_option(MDL_JIT_OPTION_USE_RENDERER_ADAPT_MICROFACET_ROUGHNESS, "true");
        }

        // Option "jit_tex_lookup_call_mode": Default mode is vtable mode.
        if (target_backend == mi::mdl::ICode_generator::Target_language::TL_NATIVE)
        {
            // df_native uses default vtable mode and user defined resources handler
            options.set_option(MDL_JIT_USE_BUILTIN_RESOURCE_HANDLER_CPU, "false");
        }
        else
        {
            // You can switch to the slower vtable mode by commenting out the next line.
            options.set_option(MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE, "direct_call");
        }


        // Option "jit_map_strings_to_ids": Default is off.
        options.set_option(MDL_JIT_OPTION_MAP_STRINGS_TO_IDS, "true");

        // Option "df_handle_slot_mode": Default is "none".
        // When using light path expressions, individual parts of the distribution functions can be
        // selected using "handles". The contribution of each of those parts has to be evaluated
        // during rendering. This option controls how many parts are evaluated with each call into
        // the generated "evaluate" and "auxiliary" functions and how the data is passed.
        options.set_option(MDL_JIT_OPTION_LINK_LIBBSDF_DF_HANDLE_SLOT_MODE, df_handle_mode.c_str());

        // Option "jit_lambda_return_mode": Default is "default".
        // Selects how generated lambda functions return their results. For PTX, the default
        // is equivalent to "sret" mode, where all values are returned in a buffer provided as
        // first argument. In "value" mode, base types and vector types are directly returned
        // by the functions, other types are returned as for the "sret" mode.
        options.set_option(MDL_JIT_OPTION_LAMBDA_RETURN_MODE, lambda_return_mode.c_str());

        for (const auto& option : additional_be_options)
            options.set_option(option.first.c_str(), option.second.c_str());

        // After we set the options, we can create a link unit
        m_link_unit = mi::base::make_handle(m_jit_be->create_link_unit(
            /*ctx=*/nullptr,
            m_target_backend,
            /*enable_simd=*/ m_target_backend == mi::mdl::ICode_generator::TL_LLVM_IR,
            /*sm_version=*/ 30,
            /*num_texture_spaces=*/ 1,
            /*num_texture_results=*/ num_texture_results));
    }

    /// Add a subexpression of a given material to the link unit.
    ///
    /// \param path               the path of the sub-expression
    /// \param fname              the name of the generated function from the added expression
    /// \param class_compilation  if true, use class compilation
    bool add_material_subexpr(
        std::string const& material_name,
        char const*        path,
        char const*        fname,
        bool               class_compilation = false);

    /// Add a distribution function of a given material to the link unit.
    ///
    /// \param path               the path of the sub-expression
    /// \param fname              the name of the generated function from the added expression
    /// \param class_compilation  if true, use class compilation
    bool add_material_df(
        std::string const& material_name,
        char const*        path,
        char const*        fname,
        bool               class_compilation = false);

    /// Add (multiple) MDL distribution function and expressions of a material to this link unit.
    /// For each distribution function it results in four functions, suffixed with \c "_init",
    /// \c "_sample", \c "_evaluate", and \c "_pdf". Functions can be selected by providing a
    /// a list of \c Target_function_descriptions. Each of them needs to define the \c path, the root
    /// of the expression that should be translated. After calling this function, each element of
    /// the list will contain information for later usage in the application,
    /// e.g., the \c argument_block_index and the \c function_index.
    ///
    /// \param material_name                mdl path of the material to generate from
    /// \param function_descriptions        description of functions to generate
    /// \param description_count            number of descriptions passed
    /// \param class_compilation            if true, use class compilation
    /// \param flags                        material instantiation flags
    bool add_material(
        std::string const&           material_name,
        Target_function_description* function_descriptions,
        size_t                       description_count,
        bool                         class_compilation = false,
        mi::Uint32                   flags = 0);

    bool add_material(
        mi::base::Handle <mi::mdl::IMaterial_instance> imat_instance,
        Target_function_description* function_descriptions,
        size_t                       description_count,
        bool                         class_compilation = false,
        mi::Uint32                   flags = 0);

    /// Generate target code for the current link unit.
    /// Note, the Target_code is only valid as long as this Material_backend_compiler exists.
    ///
    /// \return nullptr on failure
    Target_code* generate_target_code();

    /// Get the list of material instances.
    /// There will be one entry per add_* call.
    std::vector<Material_instance>& get_material_instances()
    {
        return m_mat_instances;
    }

    /// Create an instance of the given material and initialize it with default parameters.
    ///
    /// \param material_name      a fully qualified MDL material name
    /// \param class_compilation  true, if class_compilation should be used
    /// \param flags              material instantiation flags
    mi::base::Handle<mi::mdl::IMaterial_instance>
        create_and_init_material_instance(
            std::string const& material_name,
            bool              class_compilation,
            mi::Uint32        flags)
    {
        Material_instance mat_inst = create_material_instance(material_name);
        if (!mat_inst)
            return mi::base::Handle<mi::mdl::IMaterial_instance>();

        mi::mdl::Dag_error_code err = initialize_material_instance(
            mat_inst, {}, class_compilation, flags);
        // TODO: does not generate a message
        check_success(err == mi::mdl::EC_NONE);

        m_mat_instances.push_back(mat_inst);

        return mat_inst.get_material_instance();
    }
private:

    bool add_material_single_init(
        mi::mdl::IMaterial_instance* material_instance,
        Target_function_description* function_descriptions,
        size_t                       description_count);

    /// Collect the resources in the arguments of a material instance and registers them with
    /// a lambda function.
    void collect_material_argument_resources(
        mi::mdl::IMaterial_instance* mat_instance,
        mi::mdl::ILambda_function*   lambda);

protected:
    mi::mdl::ICode_generator::Target_language      m_target_backend;
    mi::base::Handle<mi::mdl::ICode_generator_jit> m_jit_be;

    mi::base::Handle<mi::mdl::ILink_unit>  m_link_unit;
    Resource_collection                    m_res_col;
    std::vector<Material_instance>         m_mat_instances;
    std::vector<size_t>                    m_arg_block_indexes;
    std::vector<Argument_block>            m_arg_blocks;
    bool                                   m_enable_derivatives;
    size_t                                 m_gen_base_name_suffix_counter;
};

// Generates target code for the current link unit.
Target_code* Material_backend_compiler::generate_target_code()
{
    // ctx should be the same value used when the unit was created
    mi::base::Handle<mi::mdl::IGenerated_code_executable> code(
        m_jit_be->compile_unit(
            /*ctx=*/nullptr,
            /*module_cache=*/nullptr,
            m_link_unit.get(),
            /*llvm_ir_output=*/ m_target_backend == mi::mdl::ICode_generator::TL_LLVM_IR));
    check_success(code);
    print_messages(code->access_messages());

#ifdef DUMP_PTX
    size_t s;
    std::cout << "Dumping ptx code:\n\n"
        << code->get_source_code(s) << std::endl;
#endif

    // collect all strings from the generated code and populate the string constant table
    m_res_col.collect_strings(code);

    // create all target argument blocks
    for (size_t i = 0, n = m_arg_block_indexes.size(); i < n; ++i) {
        size_t arg_block_index = m_arg_block_indexes[i];

        mi::base::Handle<mi::mdl::IMaterial_instance> mat_instance(
            m_mat_instances[i].get_material_instance());

        // Create a class-compilation argument block if necessary
        if (arg_block_index != size_t(~0)) {
            check_success(arg_block_index == m_arg_blocks.size() &&
                "Link unit not in sync with our Material_compiler");
            mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> layout(
                m_link_unit->get_arg_block_layout(arg_block_index));
            m_arg_blocks.push_back(
                Argument_block(mat_instance.get(), layout.get(), m_res_col));
            if (!m_arg_blocks.back().is_valid())
                return nullptr;
        }
    }

    return new Target_code(
        code.get(),
        m_link_unit.get(),
        m_res_col,
        m_mat_instances,
        m_arg_block_indexes,
        m_arg_blocks);
}

// Collect the resources in the arguments of a material instance and registers them with a lambda
// function.
void Material_backend_compiler::collect_material_argument_resources(
    mi::mdl::IMaterial_instance* mat_instance,
    mi::mdl::ILambda_function* lambda)
{
    m_res_col.set_current_lambda(lambda);

    mi::mdl::IValue_factory* vf = lambda->get_value_factory();

    for (size_t i = 0, n = mat_instance->get_parameter_count(); i < n; ++i) {
        mi::mdl::IValue const* value = mat_instance->get_parameter_default(i);

        switch (value->get_kind()) {
        case mi::mdl::IValue::VK_TEXTURE:
            m_res_col.texture(vf->import(value), 0);
            break;
        case mi::mdl::IValue::VK_LIGHT_PROFILE:
            m_res_col.light_profile(vf->import(value));
            break;
        case mi::mdl::IValue::VK_BSDF_MEASUREMENT:
            m_res_col.bsdf_measurement(vf->import(value));
            break;
        default:
            break;
        }
    }

    m_res_col.set_current_lambda(nullptr);
}

// Add a subexpression of a given material to the link unit.
bool Material_backend_compiler::add_material_subexpr(
    const std::string& material_name,
    char const* path,
    const char* fname,
    bool        class_compilation)
{
    Target_function_description desc;
    desc.path = path;
    desc.base_fname = fname;
    add_material(material_name, &desc, 1, class_compilation);
    return desc.return_code == 0;
}

// Add a distribution function of a given material to the link unit.
bool Material_backend_compiler::add_material_df(
    std::string const& material_name,
    char const* path,
    char const* base_fname,
    bool        class_compilation)
{
    Target_function_description desc;
    desc.path = path;
    desc.base_fname = base_fname;
    add_material(material_name, &desc, 1, class_compilation);
    return desc.return_code == 0;
}


namespace
{
    std::vector<std::string> split_path_tokens(const std::string& input)
    {
        std::vector<std::string> chunks;

        size_t offset(0);
        size_t pos(0);
        while (pos != std::string::npos)
        {
            pos = input.find('.', offset);

            if (pos == std::string::npos)
            {
                chunks.push_back(input.substr(offset).c_str());
                break;
            }

            chunks.push_back(input.substr(offset, pos - offset));
            offset = pos + 1;
        }
        return chunks;
    }

}

bool Material_backend_compiler::add_material_single_init(
    mi::mdl::IMaterial_instance* material_instance,
    Target_function_description* function_descriptions,
    size_t                       description_count)
{
    // increment once for each add_material invocation
    m_gen_base_name_suffix_counter++;

    // generate all function names to be able to take pointers afterswards
    std::vector<std::string> base_fname_list;
    for (size_t i = 0; i < description_count; ++i)
    {
        if (!function_descriptions[i].path)
        {
            function_descriptions[i].return_code = -1;
            return false;
        }

        // use the provided base name or generate one
        std::string base_fname;
        if (function_descriptions[i].base_fname && function_descriptions[i].base_fname[0])
            base_fname = function_descriptions[i].base_fname;
        else
        {
            std::stringstream sstr;
            sstr << "lambda_" << m_gen_base_name_suffix_counter
                 << "__" << function_descriptions[i].path;
            base_fname = sstr.str();
        }

        std::replace(base_fname.begin(), base_fname.end(), '.', '_');
        base_fname_list.push_back(std::move(base_fname));
    }

    std::vector<mi::mdl::IDistribution_function::Requested_function> func_list;
    for (size_t i = 1; i < description_count; i++)
        func_list.emplace_back(function_descriptions[i].path, base_fname_list[i].c_str());

    mi::base::Handle<mi::mdl::IDistribution_function> dist_func(
        m_mdl_compiler->create_distribution_function());

    // set init function name
    mi::base::Handle<mi::mdl::ILambda_function> root_lambda(dist_func->get_root_lambda());
    root_lambda->set_name(base_fname_list[0].c_str());

    // add all material parameters to the lambda function
    for (size_t i = 0, n = material_instance->get_parameter_count(); i < n; ++i)
    {
        const mi::mdl::IValue* value = material_instance->get_parameter_default(i);

        size_t idx = root_lambda->add_parameter(
            value->get_type(),
            material_instance->get_parameter_name(i));

        // map the i'th material parameter to this new parameter
        root_lambda->set_parameter_mapping(i, idx);
    }
    
    // import full material into the main lambda
    if (dist_func->initialize(
        material_instance,
        func_list.data(),
        func_list.size(),
        /*include_geometry_normal=*/ true,
        /*calc_derivatives=*/ m_enable_derivatives,
        /*allow_double_expr_lambdas=*/ false,
        &m_module_manager) != mi::mdl::IDistribution_function::EC_NONE)
        return false;

    // collect the resources of the distribution function and the material arguments
    m_res_col.collect(dist_func.get());
    collect_material_argument_resources(material_instance, root_lambda.get());

    // argument block index for the entire material
    size_t arg_block_index = size_t(~0);

    std::vector<size_t> main_func_indices(func_list.size() + 1); // +1 for init function
    if (!m_link_unit->add(
        dist_func.get(),
        nullptr,
        &m_module_manager,
        &arg_block_index,
        main_func_indices.data(),
        main_func_indices.size()))
    {
        for (size_t i = 0; i < description_count; ++i)
            function_descriptions[i].return_code = -1;
        return false;
    }
    
    m_arg_block_indexes.push_back(arg_block_index);

    // fill output field for the init function, which is always added first
    function_descriptions[0].function_index = main_func_indices[0];
    function_descriptions[0].distribution_kind = mi::mdl::IGenerated_code_executable::DK_NONE;
    function_descriptions[0].argument_block_index = arg_block_index;
    function_descriptions[0].return_code = 0;

    // fill output fields for the other main functions
    for (mi::Size i = 1; i < description_count; ++i)
    {
        function_descriptions[i].function_index = main_func_indices[i];
        function_descriptions[i].argument_block_index = arg_block_index;
        function_descriptions[i].return_code = 0;

        mi::base::Handle<mi::mdl::ILambda_function> main_func(
            dist_func->get_main_function(i - 1));
        mi::mdl::DAG_node const* main_node = main_func->get_body();
        mi::mdl::IType const* main_type = main_node->get_type()->skip_type_alias();
        switch (main_type->get_kind())
        {
        case mi::mdl::IType::TK_BSDF:
            function_descriptions[i].distribution_kind
                = mi::mdl::IGenerated_code_executable::DK_BSDF;
            break;
        case mi::mdl::IType::TK_HAIR_BSDF:
            function_descriptions[i].distribution_kind
                = mi::mdl::IGenerated_code_executable::DK_HAIR_BSDF;
            break;
        case mi::mdl::IType::TK_EDF:
            function_descriptions[i].distribution_kind
                = mi::mdl::IGenerated_code_executable::DK_EDF;
            break;
        case mi::mdl::IType::TK_VDF:
            function_descriptions[i].distribution_kind
                = mi::mdl::IGenerated_code_executable::DK_INVALID;
            break;
        default:
            function_descriptions[i].distribution_kind
                = mi::mdl::IGenerated_code_executable::DK_NONE;
            break;
        }
    }

    return true;
}

// Add (multiple) MDL distribution function and expressions of a material to this link unit.
bool Material_backend_compiler::add_material(
    const std::string&           material_name,
    Target_function_description* function_descriptions,
    size_t                       description_count,
    bool                         class_compilation,
    mi::Uint32                   flags)
{
    // Load the given module and create a material instance
    mi::base::Handle<mi::mdl::IMaterial_instance> mat_instance(
        create_and_init_material_instance(material_name.c_str(), class_compilation, flags));
    if (!mat_instance)
        return false;

    if (description_count > 0
        && function_descriptions[0].path
        && strcmp(function_descriptions[0].path, "init") == 0)
    {
        return add_material_single_init(
            mat_instance.get(), function_descriptions, description_count);
    }

    // argument block index for the entire material
    // (initialized by the first function that requires material arguments)
    size_t arg_block_index = size_t(~0);

    // increment once for each add_material invocation
    m_gen_base_name_suffix_counter++;

    // iterate over functions to generate
    for (size_t i = 0; i < description_count; ++i)
    {
        if (!function_descriptions[i].path)
        {
            function_descriptions[i].return_code = -1;
            return false;
        }

        // parse path into . separated tokens
        auto tokens = split_path_tokens(function_descriptions[i].path);
        std::vector<const char*> tokens_c;
        for (auto&& t : tokens)
            tokens_c.push_back(t.c_str());

        // Access the requested material expression node
        const mi::mdl::DAG_node* expr_node = get_dag_arg(
            mat_instance->get_constructor(), tokens_c, mat_instance.get());
        if (!expr_node)
        {
            function_descriptions[i].return_code = -1;
            return false;
        }

        // use the provided base name or generate one
        std::stringstream sstr;
        if (function_descriptions[i].base_fname && function_descriptions[i].base_fname[0])
            sstr << function_descriptions[i].base_fname;
        else
        {
            sstr << "lambda_" << m_gen_base_name_suffix_counter;
            sstr << "__" << function_descriptions[i].path;
        }

        std::string function_name = sstr.str();
        std::replace(function_name.begin(), function_name.end(), '.', '_');

        switch (expr_node->get_type()->get_kind())
        {
        case mi::mdl::IType::TK_BSDF:
        case mi::mdl::IType::TK_EDF:
            //case mi::mdl::IType::TK_VDF:
        {
            // set further infos that are passed back
            switch (expr_node->get_type()->get_kind())
            {
            case mi::mdl::IType::TK_BSDF:
                function_descriptions[i].distribution_kind
                    = mi::mdl::IGenerated_code_executable::DK_BSDF;
                break;

            case mi::mdl::IType::TK_EDF:
                function_descriptions[i].distribution_kind
                    = mi::mdl::IGenerated_code_executable::DK_EDF;
                break;

                // case mi::mdl::IType::TK_VDF:
                //     function_descriptions[i].distribution_kind
                //       = mi::mdl::IGenerated_code_executable::DK_VDF;
                //     break;

            default:
                function_descriptions[i].distribution_kind =
                    mi::mdl::IGenerated_code_executable::DK_INVALID;
                function_descriptions[i].return_code = -1;
                return false;
            }

            // check if the distribution function is the default one, e.g. 'bsdf()'
            // if that's the case we don't need to translate as the evaluation of the function
            // will result in zero
            if (expr_node->get_kind() == mi::mdl::DAG_node::EK_CONSTANT &&
                mi::mdl::as<mi::mdl::DAG_constant>(expr_node)->get_value()->get_kind()
                == mi::mdl::IValue::VK_INVALID_REF)
            {
                function_descriptions[i].function_index = ~0;
                break;
            }

            // Create new distribution function object and access the main lambda
            mi::base::Handle<mi::mdl::IDistribution_function> dist_func(
                m_dag_be->create_distribution_function());
            mi::base::Handle<mi::mdl::ILambda_function> root_lambda(
                dist_func->get_root_lambda());

            // set the name of the init function
            std::string init_name = function_name + "_init";
            root_lambda->set_name(init_name.c_str());

            // Add all material parameters to the lambda function
            for (size_t i = 0, n = mat_instance->get_parameter_count(); i < n; ++i)
            {
                mi::mdl::IValue const* value = mat_instance->get_parameter_default(i);

                size_t idx = root_lambda->add_parameter(
                    value->get_type(),
                    mat_instance->get_parameter_name(i));

                // Map the i'th material parameter to this new parameter
                root_lambda->set_parameter_mapping(i, idx);
            }

            // Import full material into the main lambda
            mi::mdl::IDistribution_function::Requested_function req_func(
                function_descriptions[i].path, function_name.c_str());

            // Initialize the distribution function
            if (dist_func->initialize(
                mat_instance.get(),
                &req_func,
                1,
                /*include_geometry_normal=*/ true,
                /*calc_derivatives=*/ m_enable_derivatives,
                /*allow_double_expr_lambdas=*/ false,
                &m_module_manager) != mi::mdl::IDistribution_function::EC_NONE)
                return false;

            // Collect the resources of the distribution function and the material arguments
            m_res_col.collect(dist_func.get());
            collect_material_argument_resources(mat_instance.get(), root_lambda.get());

            // Add the lambda function to the link unit. Note that it is save to pass
            // no module cache here, as we do not use dropt_import_entries() in the Core API
            size_t main_func_indices[2];
            if (!m_link_unit->add(
                dist_func.get(),
                /*module_cache=*/nullptr,
                &m_module_manager,
                &arg_block_index,
                main_func_indices,
                2))
            {
                function_descriptions[i].return_code = -1;
                continue;
            }

            // for distribution functions, let function_index point to the init function
            function_descriptions[i].function_index = main_func_indices[0];
            break;
        }

        default:
        {
            // Create a lambda function
            mi::base::Handle<mi::mdl::ILambda_function> lambda(
                m_dag_be->create_lambda_function(mi::mdl::ILambda_function::LEC_CORE));

            lambda->set_name(function_name.c_str());

            // Add all material parameters to the lambda function
            for (size_t i = 0, n = mat_instance->get_parameter_count(); i < n; ++i)
            {
                mi::mdl::IValue const* value = mat_instance->get_parameter_default(i);

                size_t idx = lambda->add_parameter(
                    value->get_type(),
                    mat_instance->get_parameter_name(i));

                // Map the i'th material parameter to this new parameter
                lambda->set_parameter_mapping(i, idx);
            }

            // Copy the expression into the lambda function
            // (making sure the expression is owned by it).
            expr_node = lambda->import_expr(mat_instance->get_dag_unit(), expr_node);
            lambda->set_body(expr_node);

            if (m_enable_derivatives)
                lambda->initialize_derivative_infos(&m_module_manager);

            // Collect the resources of the lambda and the material arguments
            m_res_col.collect(lambda.get());
            collect_material_argument_resources(mat_instance.get(), lambda.get());

            // set further infos that are passed back
            function_descriptions[i].distribution_kind =
                mi::mdl::IGenerated_code_executable::DK_NONE;

            // Add the lambda function to the link unit. Note that it is save to pass
            // no module cache here, as we do not use dropt_import_entries() in the Core API
            if (!m_link_unit->add(
                lambda.get(),
                /*module_cache=*/nullptr,
                &m_module_manager,
                mi::mdl::IGenerated_code_executable::FK_LAMBDA,
                &arg_block_index,
                &function_descriptions[i].function_index))
            {
                function_descriptions[i].return_code = -1;
                continue;
            }
            break;
        }
        }
    }

    m_arg_block_indexes.push_back(arg_block_index);

    // pass out the block index
    for (size_t i = 0; i < description_count; ++i)
    {
        function_descriptions[i].argument_block_index = arg_block_index;
        function_descriptions[i].return_code = 0;
    }

    return true;
}

// Add (multiple) MDL distribution function and expressions of a material to this link unit.
bool Material_backend_compiler::add_material(
    mi::base::Handle <mi::mdl::IMaterial_instance> imat_instance,
    Target_function_description* function_descriptions,
    size_t                       description_count,
    bool                         class_compilation,
    mi::Uint32                   flags)
{
    m_mat_instances[0].set_material_instance(imat_instance);

    if (description_count > 0
        && function_descriptions[0].path
        && strcmp(function_descriptions[0].path, "init") == 0)
    {
        return add_material_single_init(
            imat_instance.get(), function_descriptions, description_count);
    }

    // argument block index for the entire material
    // (initialized by the first function that requires material arguments)
    size_t arg_block_index = size_t(~0);

    // increment once for each add_material invocation
    m_gen_base_name_suffix_counter++;

    // iterate over functions to generate
    for (size_t i = 0; i < description_count; ++i)
    {
        if (!function_descriptions[i].path)
        {
            function_descriptions[i].return_code = -1;
            return false;
        }

        // parse path into . separated tokens
        auto tokens = split_path_tokens(function_descriptions[i].path);
        std::vector<const char*> tokens_c;
        for (auto&& t : tokens)
            tokens_c.push_back(t.c_str());

        // Access the requested material expression node
        const mi::mdl::DAG_node* expr_node = get_dag_arg(
            imat_instance->get_constructor(), tokens_c, imat_instance.get());
        if (!expr_node)
        {
            function_descriptions[i].return_code = -1;
            return false;
        }

        // use the provided base name or generate one
        std::stringstream sstr;
        if (function_descriptions[i].base_fname && function_descriptions[i].base_fname[0])
            sstr << function_descriptions[i].base_fname;
        else
        {
            sstr << "lambda_" << m_gen_base_name_suffix_counter;
            sstr << "__" << function_descriptions[i].path;
        }

        std::string function_name = sstr.str();
        std::replace(function_name.begin(), function_name.end(), '.', '_');

        switch (expr_node->get_type()->get_kind())
        {
        case mi::mdl::IType::TK_BSDF:
        case mi::mdl::IType::TK_EDF:
            //case mi::mdl::IType::TK_VDF:
        {
            // set further infos that are passed back
            switch (expr_node->get_type()->get_kind())
            {
            case mi::mdl::IType::TK_BSDF:
                function_descriptions[i].distribution_kind
                    = mi::mdl::IGenerated_code_executable::DK_BSDF;
                break;

            case mi::mdl::IType::TK_EDF:
                function_descriptions[i].distribution_kind
                    = mi::mdl::IGenerated_code_executable::DK_EDF;
                break;

                // case mi::mdl::IType::TK_VDF:
                //     function_descriptions[i].distribution_kind
                //       = mi::mdl::IGenerated_code_executable::DK_VDF;
                //     break;

            default:
                function_descriptions[i].distribution_kind =
                    mi::mdl::IGenerated_code_executable::DK_INVALID;
                function_descriptions[i].return_code = -1;
                return false;
            }

            // check if the distribution function is the default one, e.g. 'bsdf()'
            // if that's the case we don't need to translate as the evaluation of the function
            // will result in zero
            if (expr_node->get_kind() == mi::mdl::DAG_node::EK_CONSTANT &&
                mi::mdl::as<mi::mdl::DAG_constant>(expr_node)->get_value()->get_kind()
                == mi::mdl::IValue::VK_INVALID_REF)
            {
                function_descriptions[i].function_index = ~0;
                break;
            }

            // Create new distribution function object and access the main lambda
            mi::base::Handle<mi::mdl::IDistribution_function> dist_func(
                m_dag_be->create_distribution_function());
            mi::base::Handle<mi::mdl::ILambda_function> root_lambda(
                dist_func->get_root_lambda());

            // set the name of the init function
            std::string init_name = function_name + "_init";
            root_lambda->set_name(init_name.c_str());

            // Add all material parameters to the lambda function
            for (size_t i = 0, n = imat_instance->get_parameter_count(); i < n; ++i)
            {
                mi::mdl::IValue const* value = imat_instance->get_parameter_default(i);

                size_t idx = root_lambda->add_parameter(
                    value->get_type(),
                    imat_instance->get_parameter_name(i));

                // Map the i'th material parameter to this new parameter
                root_lambda->set_parameter_mapping(i, idx);
            }

            // Import full material into the main lambda
            mi::mdl::IDistribution_function::Requested_function req_func(
                function_descriptions[i].path, function_name.c_str());

            // Initialize the distribution function
            if (dist_func->initialize(
                imat_instance.get(),
                &req_func,
                1,
                /*include_geometry_normal=*/ true,
                /*calc_derivatives=*/ m_enable_derivatives,
                /*allow_double_expr_lambdas=*/ false,
                &m_module_manager) != mi::mdl::IDistribution_function::EC_NONE)
                return false;

            // Collect the resources of the distribution function and the material arguments
            m_res_col.collect(dist_func.get());
            collect_material_argument_resources(imat_instance.get(), root_lambda.get());

            // Add the lambda function to the link unit. Note that it is save to pass
            // no module cache here, as we do not use dropt_import_entries() in the Core API
            size_t main_func_indices[2];
            if (!m_link_unit->add(
                dist_func.get(),
                /*module_cache=*/nullptr,
                &m_module_manager,
                &arg_block_index,
                main_func_indices,
                2))
            {
                function_descriptions[i].return_code = -1;
                continue;
            }

            // for distribution functions, let function_index point to the init function
            function_descriptions[i].function_index = main_func_indices[0];
            break;
        }

        default:
        {
            // Create a lambda function
            mi::base::Handle<mi::mdl::ILambda_function> lambda(
                m_dag_be->create_lambda_function(mi::mdl::ILambda_function::LEC_CORE));

            lambda->set_name(function_name.c_str());

            // Add all material parameters to the lambda function
            for (size_t i = 0, n = imat_instance->get_parameter_count(); i < n; ++i)
            {
                mi::mdl::IValue const* value = imat_instance->get_parameter_default(i);

                size_t idx = lambda->add_parameter(
                    value->get_type(),
                    imat_instance->get_parameter_name(i));

                // Map the i'th material parameter to this new parameter
                lambda->set_parameter_mapping(i, idx);
            }

            // Copy the expression into the lambda function
            // (making sure the expression is owned by it).
            expr_node = lambda->import_expr(imat_instance->get_dag_unit(), expr_node);
            lambda->set_body(expr_node);

            if (m_enable_derivatives)
                lambda->initialize_derivative_infos(&m_module_manager);

            // Collect the resources of the lambda and the material arguments
            m_res_col.collect(lambda.get());
            collect_material_argument_resources(imat_instance.get(), lambda.get());

            // set further infos that are passed back
            function_descriptions[i].distribution_kind =
                mi::mdl::IGenerated_code_executable::DK_NONE;

            // Add the lambda function to the link unit. Note that it is save to pass
            // no module cache here, as we do not use dropt_import_entries() in the Core API
            if (!m_link_unit->add(
                lambda.get(),
                /*module_cache=*/nullptr,
                &m_module_manager,
                mi::mdl::IGenerated_code_executable::FK_LAMBDA,
                &arg_block_index,
                &function_descriptions[i].function_index))
            {
                function_descriptions[i].return_code = -1;
                continue;
            }
            break;
        }
        }
    }

    m_arg_block_indexes.push_back(arg_block_index);

    // pass out the block index
    for (size_t i = 0; i < description_count; ++i)
    {
        function_descriptions[i].argument_block_index = arg_block_index;
        function_descriptions[i].return_code = 0;
    }

    return true;
}

//------------------------------------------------------------------------------
//
// Utility functions
//
//------------------------------------------------------------------------------

// Export the given RGBF data to the given path.
// The file format is determined by the path (must be supported by OpenImageIO).
bool export_image_rgbf(
    char const* path, mi::Uint32 width, mi::Uint32 height, mi::mdl::tct_float3 const* data)
{
    size_t n = width * height;
    std::vector<unsigned char> tmp(3 * n);
    for (size_t i = 0, j = 0; i < n; ++i, j += 3) {
        tmp[j] = static_cast<unsigned char>(std::max(0.f, std::min(data[i].x, 1.f)) * 255.0f);
        tmp[j + 1] = static_cast<unsigned char>(std::max(0.f, std::min(data[i].y, 1.f)) * 255.0f);
        tmp[j + 2] = static_cast<unsigned char>(std::max(0.f, std::min(data[i].z, 1.f)) * 255.0f);
    }

    OIIO::ROI roi(0, width, 0, height, 0, 1, 0, 3);
    OIIO::ImageSpec spec(roi, OIIO::TypeDesc::UINT8);

    std::unique_ptr<OIIO::ImageOutput> image(OIIO::ImageOutput::create(path));
    if (!image)
        return false;

    mi::Sint32 bytes_per_row = 3 * width * sizeof(unsigned char);

    image->open(path, spec);
    bool success = image->write_image(
        OIIO::TypeDesc::UINT8,
        tmp.data() + (height - 1) * 3 * width,
        /*xstride*/ OIIO::AutoStride,
        /*ystride*/ -bytes_per_row,
        /*zstride*/ OIIO::AutoStride);
    image->close();
    return success;
}

#endif // MI_EXAMPLE_SHARED_BACKENDS_H
