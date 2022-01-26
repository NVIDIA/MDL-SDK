/******************************************************************************
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

// Code shared by CUDA MDL Core examples

#ifndef EXAMPLE_CUDA_SHARED_H
#define EXAMPLE_CUDA_SHARED_H

#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "example_shared.h"

#include <cuda.h>
#ifdef OPENGL_INTEROP
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cudaGL.h>
#endif
#include <cuda_runtime.h>
#include <vector_functions.h>

#include <FreeImage.h>

// anonymous namespace for FreeImage IO interface functions
namespace {

unsigned DLL_CALLCONV read_handler(void* buffer, unsigned size, unsigned count, fi_handle handle)
{
    if (handle == nullptr)
        return 0;

    mi::mdl::IMDL_resource_reader *reader =
        reinterpret_cast<mi::mdl::IMDL_resource_reader *>(handle);

    return unsigned(reader->read(buffer, size * count) / size);
}

int DLL_CALLCONV seek_handler(fi_handle handle, long offset, int origin)
{
    if (handle == nullptr)
        return 0;

    mi::mdl::IMDL_resource_reader *reader =
        reinterpret_cast<mi::mdl::IMDL_resource_reader *>(handle);

    return reader->seek(offset, mi::mdl::IMDL_resource_reader::Position(origin)) ? 0 : -1;
}

long DLL_CALLCONV tell_handler(fi_handle handle)
{
    if (handle == nullptr)
        return 0;

    mi::mdl::IMDL_resource_reader *reader =
        reinterpret_cast<mi::mdl::IMDL_resource_reader *>(handle);

    return long(reader->tell());
}

}  // anonymous namespace


/// Representation of textures, holding meta and image data.
class Texture_data
{
public:
    /// Constructor for an invalid texture.
    Texture_data()
    : m_path()
    , m_gamma_mode(mi::mdl::IValue_texture::gamma_default)
    , m_shape(mi::mdl::IType_texture::TS_2D)
    , m_dib(nullptr)
    , m_bsdf_data(nullptr)
    , m_width(0)
    , m_height(0)
    , m_depth(0)
    {
    }

    /// Constructor from an MDL IValue_texture.
    Texture_data(
        mi::mdl::IValue_texture const *tex,
        mi::mdl::ICode_generator_jit *jit_be,
        mi::mdl::IEntity_resolver *resolver)
    : m_path(tex->get_string_value())
    , m_gamma_mode(tex->get_gamma_mode())
    , m_shape(tex->get_type()->get_shape())
    , m_dib(nullptr)
    , m_bsdf_data(nullptr)
    , m_width(0)
    , m_height(0)
    , m_depth(0)
    {
        if (tex->get_bsdf_data_kind() != mi::mdl::IValue_texture::BDK_NONE) {
            load_bsdf_data(tex, jit_be);
            return;
        }

        // FreeImage only supports 2D textures
        if (m_shape != mi::mdl::IType_texture::TS_2D)
            return;

        load_image(resolver);
    }

    /// Constructor from a file path.
    Texture_data(
        char const *path,
        mi::mdl::IEntity_resolver *resolver)
    : m_path(path)
    , m_gamma_mode(mi::mdl::IValue_texture::gamma_default)
    , m_shape(mi::mdl::IType_texture::TS_2D)
    , m_dib(nullptr)
    , m_bsdf_data(nullptr)
    , m_width(0)
    , m_height(0)
    , m_depth(0)
    {
        load_image(resolver);
    }

    /// Destructor.
    ~Texture_data()
    {
        if (m_dib) {
            FreeImage_Unload(m_dib);
            m_dib = nullptr;
        }
    }

    /// Returns true, if the texture was loaded successfully.
    bool is_valid() const { return m_dib != nullptr || m_bsdf_data != nullptr; }

    /// Get the texture width.
    mi::Uint32 get_width() const { return m_width; }

    /// Get the texture height.
    mi::Uint32 get_height() const { return m_height; }

    /// Get the texture depth. Currently only depth 1 is supported.
    mi::Uint32 get_depth() const { return m_depth; }

    /// Get the gamma mode.
    mi::mdl::IValue_texture::gamma_mode get_gamma_mode() const { return m_gamma_mode; }

    /// Get the texture shape.
    mi::mdl::IType_texture::Shape get_shape() const { return m_shape; }

    /// Get the texture image data.
    FIBITMAP *get_dib() const { return m_dib; }

    /// Get the bsdf texture data if present.
    unsigned char const *get_bsdf_data() const { return m_bsdf_data; }

private:
    /// Load the image and get the meta data.
    ///
    /// \param resolver  the entity resolver allowing to find resources in the configured
    ///                  MDL search path and supporting MDL archives
    void load_image(mi::mdl::IEntity_resolver *resolver)
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

        mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(elem->open_reader(0));

        FreeImageIO io;
        io.read_proc = read_handler;
        io.write_proc = 0;
        io.seek_proc = seek_handler;
        io.tell_proc = tell_handler;

        FREE_IMAGE_FORMAT fif = FreeImage_GetFileTypeFromHandle(&io, reader.get());
        if (fif == FIF_UNKNOWN) {
            // no signature? try to get type by file name
            fif = FreeImage_GetFIFFromFilename(m_path.c_str());
        }
        // unknown or unsupported type?
        if (fif == FIF_UNKNOWN || !FreeImage_FIFSupportsReading(fif))
            return;

        m_dib = FreeImage_LoadFromHandle(fif, &io, reader.get());
        if (m_dib == nullptr)
            return;

        m_width = FreeImage_GetWidth(m_dib);
        m_height = FreeImage_GetHeight(m_dib);
        m_depth = 1;
    }

    /// Load the bsdf data.
    void load_bsdf_data(mi::mdl::IValue_texture const *tex, mi::mdl::ICode_generator_jit *jit_be)
    {
        size_t res_theta = 0, res_roughness = 0, res_ior = 0;
        if (!jit_be->get_libbsdf_multiscatter_data_resolution(
                tex->get_bsdf_data_kind(),
                res_theta, res_roughness, res_ior))
            return;

        size_t data_size = 0;
        m_bsdf_data = jit_be->get_libbsdf_multiscatter_data(
            tex->get_bsdf_data_kind(), data_size);
        if (m_bsdf_data == nullptr)
            return;

        m_width = res_theta;
        m_height = res_roughness;
        m_depth = res_ior;
    }

private:
    std::string m_path;
    mi::mdl::IValue_texture::gamma_mode m_gamma_mode;
    mi::mdl::IType_texture::Shape m_shape;
    FIBITMAP *m_dib;
    unsigned char const *m_bsdf_data;
    mi::Uint32 m_width;
    mi::Uint32 m_height;
    mi::Uint32 m_depth;
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
    unsigned lookup_id_for_string(const char *name) const {
        String_map::const_iterator it(m_string_constants_map.find(name));
        if (it != m_string_constants_map.end())
            return it->second;

        return 0u;
    }

    /// Get the ID for a given string, register it if it does not exist, yet.
    unsigned get_id_for_string(const char *name) {
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

    /// Get the string for a given ID, or nullptr if this ID does not exist.
    const char *get_string(unsigned id) {
        if (id == 0 || id - 1 >= m_strings.size())
            return nullptr;
        return m_strings[id - 1].c_str();
    }

    /// Get all string constants used inside a target code and their maximum length.
    void get_all_strings(
        const mi::mdl::IGenerated_code_executable *target_code)
    {
        m_max_len = 0;
        // ignore the 0, it is the "Not-a-known-string" entry
        m_strings.reserve(target_code->get_string_constant_count());
        for (size_t i = 1, n = target_code->get_string_constant_count(); i < n; ++i) {
            const char *s = target_code->get_string_constant(i);
            size_t l = strlen(s);
            if (l > m_max_len)
                m_max_len = l;
            m_string_constants_map[s] = (unsigned)i;
            m_strings.push_back(s);
        }
    }

private:
    String_constant_table(String_constant_table const &) = delete;
    String_constant_table &operator=(String_constant_table const &) = delete;

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
        mi::mdl::IMDL                *mdl,
        mi::mdl::ICode_generator_jit *jit_be,
        mi::mdl::ICall_name_resolver &call_resolver)
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
    virtual void texture(mi::mdl::IValue const *t, Texture_usage tex_usage) override
    {
        texture_impl(t);
    }

    /// Called for an enumerated light profile resource.
    ///
    /// \param t  the light profile resource or an invalid_ref
    virtual void light_profile(mi::mdl::IValue const *t) override
    {
        auto lp = mi::mdl::as<mi::mdl::IValue_light_profile>(t);

        // not supported in this example
        std::cerr << "warning: Light profiles are not supported by the MDL Core examples.\n"
                     "         However, the loaded material references the light profile:\n"
                  << "         " << lp->get_string_value() << "\n";
        (void) t;
    }

    /// Called for an enumerated bsdf measurement resource.
    ///
    /// \param t  the bsdf measurement resource or an invalid_ref
    virtual void bsdf_measurement(mi::mdl::IValue const *t) override
    {
        auto bm = mi::mdl::as<mi::mdl::IValue_bsdf_measurement>(t);

        // not supported in this example
        std::cerr << "warning: Measured BSDFs are not supported by the MDL Core examples.\n"
                     "         However, the loaded material references the BSDF measurement:\n"
                  << "         " << bm->get_string_value() << "\n";
        (void) t;
    }

    /// Sets the current lambda used for registering the resources.
    void set_current_lambda(mi::mdl::ILambda_function *lambda)
    {
        m_cur_lambda = mi::base::make_handle_dup(lambda);
    }

    /// Collect and map all resources used by the given lambda function.
    void collect(mi::mdl::ILambda_function *lambda)
    {
        mi::base::Handle<mi::mdl::ILambda_function> old_lambda = m_cur_lambda;
        m_cur_lambda = mi::base::make_handle_dup(lambda);

        lambda->enumerate_resources(m_call_resolver, *this, lambda->get_body());

        m_cur_lambda = old_lambda;
    }

    /// Collect and map all resources used by the given distribution function.
    void collect(mi::mdl::IDistribution_function *dist_func)
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
    std::vector<Texture_data *> const &get_textures() const {
        return m_textures;
    }

    /// Returns the resource index for the given resource value usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// \param resource  the resource value
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    virtual mi::Uint32 get_resource_index(mi::mdl::IValue_resource const *resource) override
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
    virtual mi::Uint32 get_string_index(mi::mdl::IValue_string const *s) override
    {
        return m_string_constant_table.lookup_id_for_string(s->get_value());
    }

    /// Collect all string from the given executable code.
    void collect_strings(mi::base::Handle<const mi::mdl::IGenerated_code_executable> code)
    {
        m_string_constant_table.get_all_strings(code.get());
    }

    /// Get the string table, so it can grow.
    String_constant_table &get_string_constant_table() const {
        return m_string_constant_table;
    }

private:
    /// Called for an enumerated texture resource.
    /// Registers the texture in this collection and in the current lambda, if set.
    ///
    /// \param t  the texture resource or an invalid_ref
    ///
    /// \returns the resource index or 0 if no resource index can be returned
    mi::Uint32 texture_impl(mi::mdl::IValue const *t)
    {
        mi::mdl::IValue_texture const *tex_val = mi::mdl::as<mi::mdl::IValue_texture>(t);
        if (tex_val == nullptr)
            return 0;

        // add the invalid texture, if this is the first texture
        if (m_textures.empty()) {
            m_textures.push_back(new Texture_data());
        }

        Texture_data *tex;
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
        } else {
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
            } else {
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

    Resource_collection(Resource_collection const &) = delete;
    Resource_collection &operator=(Resource_collection const &) = delete;

private:
    /// The MDL JIT backend to retrieve BSDF texture data.
    mi::base::Handle<mi::mdl::ICode_generator_jit> m_jit_be;

    /// The MDL entity resolver for accessing resources.
    mi::base::Handle<mi::mdl::IEntity_resolver> m_entity_resolver;

    /// The MDL call name resolver.
    mi::mdl::ICall_name_resolver &m_call_resolver;

    /// The current lambda for which resources should be registered.
    mi::base::Handle<mi::mdl::ILambda_function> m_cur_lambda;

    /// List of loaded textures.
    std::vector<Texture_data *> m_textures;

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
        mi::mdl::IGenerated_code_dag::IMaterial_instance const *mat_instance,
        mi::mdl::IGenerated_code_value_layout const *layout,
        Resource_collection &res_col)
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
    Argument_block(Argument_block const &other)
    : m_data(other.m_data)
    , m_is_valid(other.m_is_valid)
    {
    }

    /// Get the writable argument block data.
    char *get_data() { return m_data.data(); }

    /// Get the argument block data.
    char const *get_data() const { return m_data.data(); }

    /// Get the size of the argument block data.
    size_t get_size() const { return m_data.size(); }

    /// Return whether the initialization of the argument block data was successful.
    bool is_valid() const { return m_is_valid; }

private:
    std::vector<char> m_data;
    bool m_is_valid;
};


/// Class managing the generated code of a link unit and the used resources and materials.
class Ptx_code
{
public:
    /// Constructor.
    Ptx_code(
        mi::mdl::IGenerated_code_executable  *code,
        mi::mdl::ILink_unit                  *link_unit,
        Resource_collection const            &res_col,
        std::vector<Material_instance> const &mat_instances,
        std::vector<size_t> const            &arg_block_indexes,
        std::vector<Argument_block> const    &arg_blocks)
    : m_code(code, mi::base::DUP_INTERFACE)
    , m_link_unit(link_unit, mi::base::DUP_INTERFACE)
    , m_res_col(res_col)
    , m_mat_instances(mat_instances)
    , m_arg_block_indexes(arg_block_indexes)
    , m_arg_blocks(arg_blocks)
    {
        size_t code_size = 0;
        char const *code_data = code->get_source_code(code_size);
        m_source_code.assign(code_data, code_data + code_size);
    }

    /// Get the generated PTX source code.
    char const *get_src_code() const { return m_source_code.c_str(); }

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
    const char *get_callable_function(size_t index) const {
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
    std::vector<size_t> const &get_argument_block_indices() const {
        return m_arg_block_indexes;
    }

    /// Get the number of argument blocks.
    size_t get_argument_block_count() const {
        return m_arg_blocks.size();
    }

    /// Get the argument block at the given index.
    Argument_block const *get_argument_block(size_t index) const {
        return &m_arg_blocks[index];
    }

    /// Get the layout of the argument block at the given index.
    mi::mdl::IGenerated_code_value_layout const *get_argument_block_layout(size_t index) const {
        return m_link_unit->get_arg_block_layout(index);
    }

    /// Get the number of material instances.
    size_t get_material_instance_count() const {
        return m_mat_instances.size();
    }

    /// Get the material instance at the given index.
    Material_instance const &get_material_instance(size_t index) const {
        return m_mat_instances[index];
    }

    /// Get the data for the read-only data segment if available.
    ///
    /// \param size  will be assigned to the length of the RO data segment
    /// \returns the data segment or nullptr if no RO data segment is available.
    char const *get_ro_data_segment(size_t &size) const {
        return m_code->get_ro_data_segment(size);
    }

    /// Get the number of textures in the resource collection.
    size_t get_texture_count() const {
        return m_res_col.get_textures().size();
    }

    /// Get the texture at the given index.
    Texture_data const *get_texture(size_t index) const {
        return m_res_col.get_textures()[index];
    }

    /// Get the IGenerated_code_executable object.
    mi::base::Handle<mi::mdl::IGenerated_code_executable const> get_code() const {
        return m_code;
    }

    /// Get the string constant table.
    String_constant_table &get_string_constant_table() {
        return m_res_col.get_string_constant_table();
    }

private:
    mi::base::Handle<mi::mdl::IGenerated_code_executable const>  m_code;
    mi::base::Handle<mi::mdl::ILink_unit const>                  m_link_unit;
    Resource_collection const                                   &m_res_col;
    std::vector<Material_instance>                               m_mat_instances;
    std::vector<size_t>                                          m_arg_block_indexes;
    std::vector<Argument_block>                                  m_arg_blocks;
    std::string                                                  m_source_code;
};


// Structure representing an MDL texture, containing filtered and unfiltered CUDA texture
// objects and the size of the texture.
struct Texture
{
    Texture(cudaTextureObject_t  filtered_object,
            cudaTextureObject_t  unfiltered_object,
            uint3                size)
        : filtered_object(filtered_object)
        , unfiltered_object(unfiltered_object)
        , size(size)
        , inv_size(make_float3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z))
    {}

    cudaTextureObject_t  filtered_object;    // uses filter mode cudaFilterModeLinear
    cudaTextureObject_t  unfiltered_object;  // uses filter mode cudaFilterModePoint
    uint3                size;               // size of the texture, needed for texel access
    float3               inv_size;           // the inverse values of the size of the texture
};

// Structure representing the resources used by the generated code of a target code.
struct Target_code_data
{
    Target_code_data(size_t num_textures, CUdeviceptr textures, CUdeviceptr ro_data_segment)
        : num_textures(num_textures)
        , textures(textures)
        , ro_data_segment(ro_data_segment)
    {}

    size_t      num_textures;      // number of elements in the textures field
    CUdeviceptr textures;          // a device pointer to a list of Texture objects, if used
    CUdeviceptr ro_data_segment;   // a device pointer to the read-only data segment, if used
};


//------------------------------------------------------------------------------
//
// Helper functions
//
//------------------------------------------------------------------------------

// Return a textual representation of the given value.
template <typename T>
std::string to_string(T val)
{
    std::ostringstream stream;
    stream << val;
    return stream.str();
}


//------------------------------------------------------------------------------
//
// CUDA helper functions
//
//------------------------------------------------------------------------------

// Helper macro. Checks whether the expression is cudaSuccess and if not prints a message and
// resets the device and exits.
#define check_cuda_success(expr)                                            \
    do {                                                                    \
        int err = (expr);                                                   \
        if (err != 0) {                                                     \
            fprintf(stderr, "CUDA error %d in file %s, line %u: \"%s\".\n", \
                err, __FILE__, __LINE__, #expr);                            \
            keep_console_open();                                            \
            cudaDeviceReset();                                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (false)


// Initialize CUDA.
CUcontext init_cuda(
#ifdef OPENGL_INTEROP
    const bool opengl_interop
#endif
    )
{
    CUdevice cu_device;
    CUcontext cu_context;

    check_cuda_success(cuInit(0));
#if defined(OPENGL_INTEROP) && !defined(__APPLE__)
    if (opengl_interop) {
        // Use first device used by OpenGL context
        unsigned int num_cu_devices;
        check_cuda_success(cuGLGetDevices(&num_cu_devices, &cu_device, 1, CU_GL_DEVICE_LIST_ALL));
    }
    else
#endif
        // Use first device
        check_cuda_success(cuDeviceGet(&cu_device, 0));

    check_cuda_success(cuCtxCreate(&cu_context, 0, cu_device));

    // For this example, increase printf CUDA buffer size to support a larger number
    // of MDL debug::print() calls per CUDA kernel launch
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 16 * 1024 * 1024);

    return cu_context;
}

// Uninitialize CUDA.
void uninit_cuda(CUcontext cuda_context)
{
    check_cuda_success(cuCtxDestroy(cuda_context));
}

// Allocate memory on GPU and copy the given data to the allocated memory.
CUdeviceptr gpu_mem_dup(void const *data, size_t size)
{
    CUdeviceptr device_ptr;
    check_cuda_success(cuMemAlloc(&device_ptr, size));
    check_cuda_success(cuMemcpyHtoD(device_ptr, data, size));
    return device_ptr;
}

// Allocate memory on GPU and copy the given data to the allocated memory.
template<typename T>
CUdeviceptr gpu_mem_dup(std::vector<T> const &data)
{
    return gpu_mem_dup(&data[0], data.size() * sizeof(T));
}


//------------------------------------------------------------------------------
//
// Material_gpu_context class
//
//------------------------------------------------------------------------------

// Helper class responsible for making textures and read-only data available to the GPU
// by generating and managing a list of Target_code_data objects.
class Material_gpu_context
{
public:
    // Constructor.
    Material_gpu_context(bool enable_derivatives)
        : m_enable_derivatives(enable_derivatives)
        , m_device_target_code_data_list(0)
        , m_device_target_argument_block_list(0)
    {
        // Use first entry as "not-used" block
        m_target_argument_block_list.push_back(0);
    }

    // Free all acquired resources.
    ~Material_gpu_context();

    // Prepare the needed data of the given target code.
    bool prepare_target_code_data(Ptx_code const *target_code);

    // Get a device pointer to the target code data list.
    CUdeviceptr get_device_target_code_data_list();

    // Get a device pointer to the target argument block list.
    CUdeviceptr get_device_target_argument_block_list();

    // Get a device pointer to the i'th target argument block.
    CUdeviceptr get_device_target_argument_block(size_t i)
    {
        // First entry is the "not-used" block, so start at index 1.
        if (i + 1 >= m_target_argument_block_list.size()) return 0;
        return m_target_argument_block_list[i + 1];
    }

    // Get the number of target argument blocks.
    size_t get_argument_block_count() const
    {
        return m_own_arg_blocks.size();
    }

    // Get the argument block of the i'th BSDF.
    // If the BSDF has no target argument block, size_t(~0) is returned.
    size_t get_bsdf_argument_block_index(size_t i) const
    {
        if (i >= m_bsdf_arg_block_indices.size()) return size_t(~0);
        return m_bsdf_arg_block_indices[i];
    }

    // Get a writable copy of the i'th target argument block.
    Argument_block *get_argument_block(size_t i)
    {
        if (i >= m_own_arg_blocks.size())
            return nullptr;
        return &m_own_arg_blocks[i];
    }

    // Get the layout of the i'th target argument block.
    mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> get_argument_block_layout(
        size_t i)
    {
        if (i >= m_arg_block_layouts.size())
            return mi::base::Handle<mi::mdl::IGenerated_code_value_layout const>();
        return m_arg_block_layouts[i];
    }

    // Update the i'th target argument block on the device with the data from the corresponding
    // block returned by get_argument_block().
    void update_device_argument_block(size_t i);
private:
    // Generate the next mip map level data, returns a pointer to a newly allocated buffer
    // and updates the width and height parameters with the new data.
    float4 *gen_mip_level(
        float4 *prev_data,
        unsigned &cur_width,
        unsigned &cur_height);

    // Prepare the texture identified by the texture_index for use by the texture access functions
    // on the GPU.
    bool prepare_texture(
        Ptx_code const       *code_ptx,
        size_t                texture_index,
        std::vector<Texture> &textures);

    // If true, mipmaps will be generated for all 2D textures.
    bool m_enable_derivatives;

    // The device pointer of the target code data list.
    CUdeviceptr m_device_target_code_data_list;

    // List of all target code data objects owned by this context.
    std::vector<Target_code_data> m_target_code_data_list;

    // The device pointer of the target argument block list.
    CUdeviceptr m_device_target_argument_block_list;

    // List of all target argument blocks owned by this context.
    std::vector<CUdeviceptr> m_target_argument_block_list;

    // List of all local, writable copies of the target argument blocks.
    std::vector<Argument_block> m_own_arg_blocks;

    // List of argument block indices per material BSDF.
    std::vector<size_t> m_bsdf_arg_block_indices;

    // List of all target argument block layouts.
    std::vector<mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> > m_arg_block_layouts;

    // List of all Texture objects owned by this context.
    std::vector<Texture> m_all_textures;

    // List of all CUDA arrays owned by this context.
    std::vector<cudaArray_t> m_all_texture_arrays;

    // List of all CUDA mipmapped arrays owned by this context.
    std::vector<cudaMipmappedArray_t> m_all_texture_mipmapped_arrays;
};

// Free all acquired resources.
Material_gpu_context::~Material_gpu_context()
{
    for (std::vector<cudaArray_t>::iterator it = m_all_texture_arrays.begin(),
            end = m_all_texture_arrays.end(); it != end; ++it) {
        check_cuda_success(cudaFreeArray(*it));
    }
    for (std::vector<cudaMipmappedArray_t>::iterator it = m_all_texture_mipmapped_arrays.begin(),
            end = m_all_texture_mipmapped_arrays.end(); it != end; ++it) {
        check_cuda_success(cudaFreeMipmappedArray(*it));
    }
    for (std::vector<Texture>::iterator it = m_all_textures.begin(),
            end = m_all_textures.end(); it != end; ++it) {
        check_cuda_success(cudaDestroyTextureObject(it->filtered_object));
        check_cuda_success(cudaDestroyTextureObject(it->unfiltered_object));
    }
    for (std::vector<Target_code_data>::iterator it = m_target_code_data_list.begin(),
            end = m_target_code_data_list.end(); it != end; ++it) {
        if (it->textures)
            check_cuda_success(cuMemFree(it->textures));
        if (it->ro_data_segment)
            check_cuda_success(cuMemFree(it->ro_data_segment));
    }
    for (std::vector<CUdeviceptr>::iterator it = m_target_argument_block_list.begin(),
            end = m_target_argument_block_list.end(); it != end; ++it) {
        if (*it != 0)
            check_cuda_success(cuMemFree(*it));
    }
    check_cuda_success(cuMemFree(m_device_target_code_data_list));
}

// Get a device pointer to the target code data list.
CUdeviceptr Material_gpu_context::get_device_target_code_data_list()
{
    if (!m_device_target_code_data_list)
        m_device_target_code_data_list = gpu_mem_dup(m_target_code_data_list);
    return m_device_target_code_data_list;
}

// Get a device pointer to the target argument block list.
CUdeviceptr Material_gpu_context::get_device_target_argument_block_list()
{
    if (!m_device_target_argument_block_list)
        m_device_target_argument_block_list = gpu_mem_dup(m_target_argument_block_list);
    return m_device_target_argument_block_list;
}

// Generate the next mip map level data, returns a pointer to a newly allocated buffer
// and updates the width and height parameters with the new data.
float4 *Material_gpu_context::gen_mip_level(
    float4 *prev_data,
    unsigned &cur_width,
    unsigned &cur_height)
{
    unsigned offsets_x[4] = { 0, 1, 0, 1 };
    unsigned offsets_y[4] = { 0, 0, 1, 1 };

    unsigned prev_width = cur_width, prev_height = cur_height;
    unsigned width  = std::max(prev_width / 2, 1u);
    unsigned height = std::max(prev_height / 2, 1u);

    float4 *data = static_cast<float4 *>(malloc(width * height * sizeof(float4)));

    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            // The current pixel (x,y) corresponds to the four pixels
            // [prev_x, prev_x+1] x [prev_y,prev_y+1] in the the previous layer.
            unsigned prev_x = 2 * x, prev_y = 2 * y;

            unsigned num_summands = 0;
            float4 sum = { 0 };

            // Loop over the at most four pixels corresponding to pixel (x,y)
            for (unsigned i = 0; i < 4; ++i) {
                unsigned cur_prev_x = prev_x + offsets_x[i];
                unsigned cur_prev_y = prev_y + offsets_y[i];

                if (cur_prev_x >= prev_width || cur_prev_y >= prev_height)
                    continue;

                float4 *prev_ptr = prev_data + cur_prev_y * prev_width + cur_prev_x;
                sum.x += prev_ptr->x;
                sum.y += prev_ptr->y;
                sum.z += prev_ptr->z;
                sum.w += prev_ptr->w;
                ++num_summands;
            }
            float4 *data_ptr = data + y * width + x;
            data_ptr->x = sum.x / num_summands;
            data_ptr->y = sum.y / num_summands;
            data_ptr->z = sum.z / num_summands;
            data_ptr->w = sum.w / num_summands;
        }
    }

    // update width and height
    cur_width = width;
    cur_height = height;

    return data;
}

// Prepare the texture identified by the texture_index for use by the texture access functions
// on the GPU.
// Note: Currently no support for 3D textures, Udim textures, PTEX and cubemaps.
bool Material_gpu_context::prepare_texture(
    Ptx_code const       *code_ptx,
    size_t                texture_index,
    std::vector<Texture> &textures)
{
    Texture_data const *tex = code_ptx->get_texture(texture_index);
    if (!tex->is_valid()) {
        fprintf(stderr, "Error: Requested texture is invalid\n");
        return false;
    }

    unsigned width = tex->get_width(), height = tex->get_height();

    if (tex->get_shape() == mi::mdl::IType_texture::TS_BSDF_DATA) {
        unsigned char const *bsdf_data = tex->get_bsdf_data();
        if (bsdf_data == nullptr) {
            fprintf(stderr, "Error: bsdf data missing for requested texture\n");
            return false;
        }

        unsigned depth = tex->get_depth();

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));

        cudaArray_t device_tex_data;
        cudaExtent extent = make_cudaExtent(width, height, depth);
        check_cuda_success(cudaMalloc3DArray(&device_tex_data, &channel_desc, extent));

        cudaMemcpy3DParms copy_params = { 0 };
        copy_params.srcPtr = make_cudaPitchedPtr(
            const_cast<unsigned char *>(bsdf_data), width * sizeof(float), width, height);
        copy_params.dstArray = device_tex_data;
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyHostToDevice;
        check_cuda_success(cudaMemcpy3D(&copy_params));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_tex_data;

        m_all_texture_arrays.push_back(device_tex_data);

        // Create filtered texture object
        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0]   = cudaAddressModeClamp;
        tex_desc.addressMode[1]   = cudaAddressModeClamp;
        tex_desc.addressMode[2]   = cudaAddressModeClamp;
        tex_desc.filterMode       = cudaFilterModeLinear;
        tex_desc.readMode         = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        cudaTextureObject_t tex_obj = 0;
        check_cuda_success(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

        // Create unfiltered texture object and use a black border for access outside of the texture
        cudaTextureObject_t tex_obj_unfilt = 0;
        tex_desc.addressMode[0]   = cudaAddressModeBorder;
        tex_desc.addressMode[1]   = cudaAddressModeBorder;
        tex_desc.addressMode[2]   = cudaAddressModeBorder;
        tex_desc.filterMode       = cudaFilterModePoint;

        check_cuda_success(cudaCreateTextureObject(
            &tex_obj_unfilt, &res_desc, &tex_desc, nullptr));

        // Store texture infos in result vector
        textures.push_back(Texture(
            tex_obj,
            tex_obj_unfilt,
            make_uint3(tex->get_width(), tex->get_height(), tex->get_depth())));
        m_all_textures.push_back(textures.back());
        return true;
    }


    // For simplicity, the texture access functions are only implemented for float4 and gamma
    // is pre-applied here (all images are converted to linear space).

    FIBITMAP *dib = tex->get_dib();
    FIBITMAP *own_dib = nullptr;
    if (FreeImage_GetImageType(dib) != FIT_RGBAF) {
        own_dib = FreeImage_ConvertToRGBAF(dib);
        dib = own_dib;
    }

    // This example expects, that there is no additional padding per image line
    if (FreeImage_GetPitch(dib) != unsigned(width * 4 * sizeof(float))) {
        if (own_dib)
            FreeImage_Unload(own_dib);
        return false;
    }

    // Convert to linear color space if necessary
    if (tex->get_gamma_mode() != mi::mdl::IValue_texture::gamma_linear) {
        if (own_dib == nullptr) {
            own_dib = FreeImage_Clone(dib);
            dib = own_dib;
        }

        // FreeImage_AdjustGamma does not support floating point data,
        // so we need to do it ourselves

        float *data = reinterpret_cast<float *>(FreeImage_GetBits(dib));
        for (size_t i = 0, n = width * height * 4; i < n; i += 4) {
            // Only adjust r, g and b, not alpha
            data[i] = std::pow(data[i], 2.2f);
            data[i + 1] = std::pow(data[i + 1], 2.2f);
            data[i + 2] = std::pow(data[i + 2], 2.2f);
        }
    }

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));

    // Copy image data to GPU array depending on texture shape (currently only 2D textures)
    if (tex->get_shape() != mi::mdl::IType_texture::TS_2D) {
        std::cerr << "Currently only 2D textures are supported!" << std::endl;
        return false;
    } else if (m_enable_derivatives) {
        // mipmapped textures use CUDA mipmapped arrays
        unsigned num_levels = 1 + unsigned(std::log2f(float(std::min(width, height))));
        cudaExtent extent = make_cudaExtent(width, height, 0);
        cudaMipmappedArray_t device_tex_miparray;
        check_cuda_success(cudaMallocMipmappedArray(
            &device_tex_miparray, &channel_desc, extent, num_levels));

        unsigned cur_width = width, cur_height = height;
        float4 *prev_data = nullptr;

        // create all mipmap levels and copy them to the CUDA arrays in the mipmapped array
        for (unsigned level = 0; level < num_levels; ++level) {
            float4 *cur_data;
            if (level == 0)
                cur_data = reinterpret_cast<float4 *>(FreeImage_GetBits(dib));
            else
                cur_data = gen_mip_level(prev_data, cur_width, cur_height);

            cudaArray_t device_cur_level_array;
            cudaGetMipmappedArrayLevel(&device_cur_level_array, device_tex_miparray, level);
            check_cuda_success(cudaMemcpy2DToArray(device_cur_level_array, 0, 0, cur_data,
                cur_width * sizeof(float4), cur_width * sizeof(float4), cur_height,
                cudaMemcpyHostToDevice));

            if (level >= 2)
                free(prev_data);
            prev_data = cur_data;
        }
        if (num_levels >= 2)
            free(prev_data);

        res_desc.resType = cudaResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = device_tex_miparray;

        m_all_texture_mipmapped_arrays.push_back(device_tex_miparray);
    } else {
        // 2D texture objects use CUDA arrays
        cudaArray_t device_tex_data;
        check_cuda_success(cudaMallocArray(&device_tex_data, &channel_desc, width, height));

        BYTE const *data = FreeImage_GetBits(dib);
        check_cuda_success(cudaMemcpy2DToArray(device_tex_data, 0, 0, data,
            width * sizeof(float4), width * sizeof(float4), height,
            cudaMemcpyHostToDevice));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_tex_data;

        m_all_texture_arrays.push_back(device_tex_data);
    }

    // For cube maps we need clamped address mode to avoid artifacts in the corners
    cudaTextureAddressMode addr_mode =
        tex->get_shape() == mi::mdl::IType_texture::TS_CUBE
        ? cudaAddressModeClamp : cudaAddressModeWrap;

    // Create filtered texture object
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0]   = addr_mode;
    tex_desc.addressMode[1]   = addr_mode;
    tex_desc.addressMode[2]   = addr_mode;
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    if (res_desc.resType == cudaResourceTypeMipmappedArray) {
        tex_desc.mipmapFilterMode = cudaFilterModeLinear;
        tex_desc.maxAnisotropy = 16;
        tex_desc.minMipmapLevelClamp = 0.f;
        tex_desc.maxMipmapLevelClamp = 1000.f;  // default value in OpenGL
    }

    cudaTextureObject_t tex_obj = 0;
    check_cuda_success(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    // Create unfiltered texture object if necessary (cube textures have no texel functions)
    cudaTextureObject_t tex_obj_unfilt = 0;
    if (tex->get_shape() != mi::mdl::IType_texture::TS_CUBE) {
        // Use a black border for access outside of the texture
        tex_desc.addressMode[0]   = cudaAddressModeBorder;
        tex_desc.addressMode[1]   = cudaAddressModeBorder;
        tex_desc.addressMode[2]   = cudaAddressModeBorder;
        tex_desc.filterMode       = cudaFilterModePoint;

        check_cuda_success(cudaCreateTextureObject(
            &tex_obj_unfilt, &res_desc, &tex_desc, nullptr));
    }

    // Store texture infos in result vector
    textures.push_back(Texture(
        tex_obj,
        tex_obj_unfilt,
        make_uint3(tex->get_width(), tex->get_height(), tex->get_depth())));
    m_all_textures.push_back(textures.back());

    if (own_dib)
        FreeImage_Unload(own_dib);

    return true;
}

// Prepare the needed target code data of the given target code.
bool Material_gpu_context::prepare_target_code_data(Ptx_code const *target_code)
{
    // Target code data list may not have been retrieved already
    check_success(!m_device_target_code_data_list);

    // Handle the read-only data segments if necessary.
    // They are only created, if the "enable_ro_segment" backend option was set to "on".
    CUdeviceptr device_ro_data = 0;
    size_t ro_data_size = 0;
    if (char const *ro_data = target_code->get_ro_data_segment(ro_data_size)) {
        device_ro_data = gpu_mem_dup(ro_data, ro_data_size);
    }

    // Copy textures to GPU if the code has more than just the invalid texture
    CUdeviceptr device_textures = 0;
    size_t num_textures = target_code->get_texture_count();
    if (num_textures > 1) {
        std::vector<Texture> textures;

        // Loop over all textures skipping the first texture,
        // which is always the invalid texture
        for (size_t i = 1; i < num_textures; ++i) {
            if (!prepare_texture(target_code, i, textures))
                return false;
        }

        // Copy texture list to GPU
        device_textures = gpu_mem_dup(textures);
    }

    m_target_code_data_list.push_back(
        Target_code_data(num_textures, device_textures, device_ro_data));

    for (size_t i = 0, num = target_code->get_argument_block_count(); i < num; ++i) {
        Argument_block const *arg_block = target_code->get_argument_block(i);
        CUdeviceptr dev_block = gpu_mem_dup(arg_block->get_data(), arg_block->get_size());
        m_target_argument_block_list.push_back(dev_block);
        m_own_arg_blocks.push_back(Argument_block(*arg_block));
        m_arg_block_layouts.push_back(
            mi::base::make_handle(target_code->get_argument_block_layout(i)));
    }

    for (size_t arg_block_index : target_code->get_argument_block_indices()) {
        m_bsdf_arg_block_indices.push_back(arg_block_index);
    }

    return true;
}

// Update the i'th target argument block on the device with the data from the corresponding
// block returned by get_argument_block().
void Material_gpu_context::update_device_argument_block(size_t i)
{
    CUdeviceptr device_ptr = get_device_target_argument_block(i);
    if (device_ptr == 0) return;

    Argument_block *arg_block = get_argument_block(i);
    check_cuda_success(cuMemcpyHtoD(
        device_ptr, arg_block->get_data(), arg_block->get_size()));
}


//------------------------------------------------------------------------------
//
// MDL material compilation code
//
//------------------------------------------------------------------------------

struct Target_function_description
{
    Target_function_description(const char* expression_path = NULL,
                                const char* base_function_name = NULL)
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
    /// If \c NULL is passed, the function name will be 'lambda' followed by an increasing
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


class Material_ptx_compiler : public Material_compiler {
public:
    /// Constructor.
    ///
    /// \param mdl_compiler         the MDL compiler interface
    /// \param num_texture_results  the size of a renderer provided array for texture results
    ///                             in the MDL shading state in number of float4 elements
    ///                             processed by the init() function of distribution functions
    /// \param enable_derivatives   If true, the generated code will expect the renderer to provide
    ///                             a state and a texture runtime with derivatives
    Material_ptx_compiler(
        mi::mdl::IMDL     *mdl_compiler,
        unsigned          num_texture_results,
        bool              enable_derivatives,
        const std::string df_handle_mode)
    : Material_compiler(mdl_compiler)
    , m_jit_be(mi::base::make_handle(mdl_compiler->load_code_generator("jit"))
        .get_interface<mi::mdl::ICode_generator_jit>())
    , m_link_unit()
    , m_res_col(mdl_compiler, m_jit_be.get(), m_module_manager)
    , m_enable_derivatives(enable_derivatives)
    , m_gen_base_name_suffix_counter(0)
    {
        // Set the JIT backend options: e, we use a private code generator here, so it is safe to
        // modify the backend options and ignore thread contexts
        mi::mdl::Options &options = m_jit_be->access_options();

        // Option "enable_ro_segment": Default is disabled.
        // If you have a lot of big arrays, enabling this might speed up compilation.
        options.set_option(MDL_JIT_OPTION_ENABLE_RO_SEGMENT, "true");

        if (enable_derivatives) {
            // Option "jit_tex_runtime_with_derivs": Default is disabled.
            // We enable it to get coordinates with derivatives for texture lookup functions.
            options.set_option(MDL_JIT_OPTION_TEX_RUNTIME_WITH_DERIVATIVES, "true");
        }

        // Option "jit_tex_lookup_call_mode": Default mode is vtable mode.
        // You can switch to the slower vtable mode by commenting out the next line.
        options.set_option(MDL_JIT_OPTION_TEX_LOOKUP_CALL_MODE, "direct_call");

        // Option "jit_map_strings_to_ids": Default is off.
        options.set_option(MDL_JIT_OPTION_MAP_STRINGS_TO_IDS, "true");

        // Option "df_handle_slot_mode": Default is "none".
        // When using light path expressions, individual parts of the distribution functions can be
        // selected using "handles". The contribution of each of those parts has to be evaluated 
        // during rendering. This option controls how many parts are evaluated with each call into 
        // the generated "evaluate" and "auxiliary" functions and how the data is passed.
        options.set_option(MDL_JIT_OPTION_LINK_LIBBSDF_DF_HANDLE_SLOT_MODE, df_handle_mode.c_str());

        // Option "jit_scene_data_names": Default is "".
        // Uncomment the line below to enable calling the scene data runtime functions
        // for any scene data names or specify a comma-separated list of names for which
        // you may provide scene data. The example runtime functions always return the
        // default values, which is the same as not supporting any scene data.
        //     options.set_option(MDL_JIT_OPTION_SCENE_DATA_NAMES, "*");

        // After we set the options, we can create a link unit
        m_link_unit = mi::base::make_handle(m_jit_be->create_link_unit(
            /*ctx=*/nullptr,
            mi::mdl::ICode_generator_jit::CM_PTX,
            /*enable_simd=*/ false,
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
        std::string const             &material_name,
        char const                    *path,
        char const                    *fname,
        bool                           class_compilation = false);

    /// Add a distribution function of a given material to the link unit.
    ///
    /// \param path               the path of the sub-expression
    /// \param fname              the name of the generated function from the added expression
    /// \param class_compilation  if true, use class compilation
    bool add_material_df(
        const std::string             &material_name,
        char const                    *path,
        char const                    *fname,
        bool                           class_compilation = false);

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
    bool add_material(
        const std::string             &material_name,
        Target_function_description   *function_descriptions,
        size_t                         description_count,
        bool                           class_compilation = false);


    /// Generate CUDA PTX target code for the current link unit.
    /// Note, the Ptx_code is only valid as long as this Material_ptx_compiler exists.
    ///
    /// \return nullptr on failure
    Ptx_code *generate_cuda_ptx();

    /// Get the list of material instances.
    /// There will be one entry per add_* call.
    std::vector<Material_instance> &get_material_instances()
    {
        return m_mat_instances;
    }

private:
    /// Create an instance of the given material and initialize it with default parameters.
    ///
    /// \param material_name      a fully qualified MDL material name
    /// \param class_compilation  true, if class_compilation should be used
    mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance>
    create_and_init_material_instance(
        std::string const &material_name,
        bool              class_compilation)
    {
        Material_instance mat_inst = create_material_instance(material_name);
        if (!mat_inst)
            return mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance>();

        mi::mdl::IGenerated_code_dag::Error_code err = initialize_material_instance(
            mat_inst, {}, class_compilation);
        // TODO: does not generate a message
        check_success(err == mi::mdl::IGenerated_code_dag::EC_NONE);

        m_mat_instances.push_back(mat_inst);

        return mat_inst.get_material_instance();
    }

    /// Collect the resources in the arguments of a material instance and registers them with
    /// a lambda function.
    void collect_material_argument_resources(
        mi::mdl::IGenerated_code_dag::IMaterial_instance *mat_instance,
        mi::mdl::ILambda_function                        *lambda);

protected:
    mi::base::Handle<mi::mdl::ICode_generator_jit> m_jit_be;

    mi::base::Handle<mi::mdl::ILink_unit>  m_link_unit;
    Resource_collection                    m_res_col;
    std::vector<Material_instance>         m_mat_instances;
    std::vector<size_t>                    m_arg_block_indexes;
    std::vector<Argument_block>            m_arg_blocks;
    bool                                   m_enable_derivatives;
    size_t                                 m_gen_base_name_suffix_counter;
};

// Generates CUDA PTX target code for the current link unit.
Ptx_code *Material_ptx_compiler::generate_cuda_ptx()
{
    // ctx should be the same value used when the unit was created
    mi::base::Handle<mi::mdl::IGenerated_code_executable> code_ptx(
        m_jit_be->compile_unit(/*ctx=*/nullptr, m_link_unit.get()));
    check_success(code_ptx);

#ifdef DUMP_PTX
    size_t s;
    std::cout << "Dumping CUDA PTX code:\n\n"
        << code_ptx->get_source_code(s) << std::endl;
#endif

    // collect all strings from the generated code and populate the string constant table
    m_res_col.collect_strings(code_ptx);

    // create all target argument blocks
    for (size_t i = 0, n = m_arg_block_indexes.size(); i < n; ++i) {
        size_t arg_block_index = m_arg_block_indexes[i];

        mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance> mat_instance(
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

    return new Ptx_code(
        code_ptx.get(),
        m_link_unit.get(),
        m_res_col,
        m_mat_instances,
        m_arg_block_indexes,
        m_arg_blocks);
}

// Collect the resources in the arguments of a material instance and registers them with a lambda
// function.
void Material_ptx_compiler::collect_material_argument_resources(
    mi::mdl::IGenerated_code_dag::IMaterial_instance *mat_instance,
    mi::mdl::ILambda_function                        *lambda)
{
    m_res_col.set_current_lambda(lambda);

    mi::mdl::IValue_factory *vf = lambda->get_value_factory();

    for (size_t i = 0, n = mat_instance->get_parameter_count(); i < n; ++i) {
        mi::mdl::IValue const *value = mat_instance->get_parameter_default(i);

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
bool Material_ptx_compiler::add_material_subexpr(
    const std::string             &material_name,
    char const                    *path,
    const char                    *fname,
    bool                           class_compilation)
{
    Target_function_description desc;
    desc.path = path;
    desc.base_fname = fname;
    add_material(material_name, &desc, 1, class_compilation);
    return desc.return_code == 0;
}

// Add a distribution function of a given material to the link unit.
bool Material_ptx_compiler::add_material_df(
    std::string const             &material_name,
    char const                    *path,
    char const                    *base_fname,
    bool                           class_compilation)
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

// Add (multiple) MDL distribution function and expressions of a material to this link unit.
bool Material_ptx_compiler::add_material(
    const std::string             &material_name,
    Target_function_description   *function_descriptions,
    size_t                         description_count,
    bool                           class_compilation)
{
    // Load the given module and create a material instance
    mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance> mat_instance(
        create_and_init_material_instance(material_name.c_str(), class_compilation));
    if (!mat_instance)
        return false;

    // argument block index for the entire material
    // (initialized by the first function that requires material arguments)
    size_t arg_block_index = size_t(~0);

    // increment once for each add_material invocation
    m_gen_base_name_suffix_counter++;

    // iterate over functions to generate
    for (size_t i = 0; i < description_count; ++i)
    {
        // parse path into . separated tokens
        auto tokens = split_path_tokens(function_descriptions[i].path);
        std::vector<const char*> tokens_c;
        for (auto&& t : tokens)
            tokens_c.push_back(t.c_str());

        // Access the requested material expression node
        const mi::mdl::DAG_node* expr_node = get_dag_arg(mat_instance->get_constructor(),
                                                         tokens_c, mat_instance.get());
        if (!expr_node)
        {
            function_descriptions[i].return_code = -1;
            return false;
        }

        // use the provided base name or generate one
        std::stringstream sstr;
        if (function_descriptions[i].base_fname)
            sstr << function_descriptions[i].base_fname;
        else
        {
            sstr << "lambda_" << m_gen_base_name_suffix_counter;
        }
        sstr << "__" << function_descriptions[i].path;

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
                    mi::mdl::IValue const *value = mat_instance->get_parameter_default(i);

                    size_t idx = root_lambda->add_parameter(
                        value->get_type(),
                        mat_instance->get_parameter_name(i));

                    // Map the i'th material parameter to this new parameter
                    root_lambda->set_parameter_mapping(i, idx);
                }

                // Import full material into the main lambda
                mi::mdl::DAG_node const *material_constructor =
                    root_lambda->import_expr(mat_instance->get_constructor());

                mi::mdl::IDistribution_function::Requested_function req_func(
                    function_descriptions[i].path, function_name.c_str());

                // Initialize the distribution function
                if (dist_func->initialize(
                        material_constructor,
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
                size_t main_func_index;
                if (!m_link_unit->add(
                    dist_func.get(),
                    /*module_cache=*/NULL,
                    &m_module_manager,
                    &arg_block_index,
                    &main_func_index,
                    1))
                {
                    function_descriptions[i].return_code = -1;
                    continue;
                }

                // for distribution functions, let function_index point to the init function,
                // which does not count as main function
                function_descriptions[i].function_index = main_func_index - 1;
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
                    mi::mdl::IValue const *value = mat_instance->get_parameter_default(i);

                    size_t idx = lambda->add_parameter(
                        value->get_type(),
                        mat_instance->get_parameter_name(i));

                    // Map the i'th material parameter to this new parameter
                    lambda->set_parameter_mapping(i, idx);
                }

                // Copy the expression into the lambda function
                // (making sure the expression is owned by it).
                expr_node = lambda->import_expr(expr_node);
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
                    /*module_cache=*/NULL,
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
// Material execution code
//
//------------------------------------------------------------------------------

// Helper function to create PTX source code for a non-empty 32-bit value array.
void print_array_u32(
    std::string &str, std::string const &name, unsigned count, std::string const &content)
{
    str += ".visible .const .align 4 .u32 " + name + "[";
    if (count == 0) {
        // PTX does not allow empty arrays, so use a dummy entry
        str += "1] = { 0 };\n";
    } else {
        str += to_string(count) + "] = { " + content + " };\n";
    }
}

// Helper function to create PTX source code for a non-empty function pointer array.
void print_array_func(
    std::string &str, std::string const &name, unsigned count, std::string const &content)
{
    str += ".visible .const .align 8 .u64 " + name + "[";
    if (count == 0) {
        // PTX does not allow empty arrays, so use a dummy entry
        str += "1] = { dummy_func };\n";
    } else {
        str += to_string(count) + "] = { " + content + " };\n";
    }
}

// Generate PTX array containing the references to all generated functions.
std::string generate_func_array_ptx(
    std::vector<std::unique_ptr<Ptx_code>> const &target_codes)
{
    // Create PTX header and mdl_expr_functions_count constant
    std::string src =
        ".version 4.0\n"
        ".target sm_20\n"
        ".address_size 64\n";

    // Workaround needed to let CUDA linker resolve the function pointers in the arrays.
    // Also used for "empty" function arrays.
    src += ".func dummy_func() { ret; }\n";

    std::string tc_offsets;
    std::string function_names;
    std::string tc_indices;
    std::string ab_indices;
    unsigned f_count = 0;

    // Iterate over all target codes
    for (size_t tc_index = 0, num = target_codes.size(); tc_index < num; ++tc_index)
    {
        Ptx_code const *target_code = target_codes[tc_index].get();

        // in case of multiple target codes, we need to address the functions by a pair of
        // target_code_index and function_index.
        // the elements in the resulting function array can then be index by offset + func_index.
        if (!tc_offsets.empty())
            tc_offsets += ", ";
        tc_offsets += to_string(f_count);

        // Collect all names and prototypes of callable functions within the current target code
        for (size_t func_index = 0, func_count = target_code->get_callable_function_count();
             func_index < func_count; ++func_index)
        {
            // add to function list
            if (!tc_indices.empty())
            {
                tc_indices += ", ";
                function_names += ", ";
                ab_indices += ", ";
            }

            // target code index in case of multiple link units
            tc_indices += to_string(tc_index);

            // name of the function
            function_names += target_code->get_callable_function(func_index);

            // Get argument block index and translate to 1 based list index (-> 0 = not-used)
            mi::Size ab_index = target_code->get_callable_function_argument_block_index(func_index);
            ab_indices += to_string(ab_index == mi::Size(~0) ? 0 : (ab_index + 1));
            f_count++;

            // Add prototype declaration
            src += target_code->get_callable_function_prototype(
                func_index, mi::mdl::IGenerated_code_executable::PL_PTX);
            src += '\n';
        }
    }

    // infos per target code (link unit)
    src += std::string(".visible .const .align 4 .u32 mdl_target_code_count = ")
        + to_string(target_codes.size()) + ";\n";
    print_array_u32(
        src, std::string("mdl_target_code_offsets"), unsigned(target_codes.size()), tc_offsets);

    // infos per function
    src += std::string(".visible .const .align 4 .u32 mdl_functions_count = ")
        + to_string(f_count) + ";\n";
    print_array_func(src, std::string("mdl_functions"), f_count, function_names);
    print_array_u32(src, std::string("mdl_arg_block_indices"), f_count, ab_indices);
    print_array_u32(src, std::string("mdl_target_code_indices"), f_count, tc_indices);

    return src;
}

// Build a linked CUDA kernel containing our kernel and all the generated code, making it
// available to the kernel via an added "mdl_expr_functions" array.
CUmodule build_linked_kernel(
    std::vector<std::unique_ptr<Ptx_code>> const &target_codes,
    const char *ptx_file,
    const char *kernel_function_name,
    CUfunction *out_kernel_function)
{
    // Generate PTX array containing the references to all generated functions.
    // The linker will resolve them to addresses.

    std::string ptx_func_array_src = generate_func_array_ptx(target_codes);
#ifdef DUMP_PTX
    std::cout << "Dumping CUDA PTX code for the \"mdl_expr_functions\" array:\n\n"
        << ptx_func_array_src << std::endl;
#endif

    // Link all generated code, our generated PTX array and our kernel together

    CUlinkState   cuda_link_state;
    CUmodule      cuda_module;
    void         *linked_cubin;
    size_t        linked_cubin_size;
    char          error_log[8192], info_log[8192];
    CUjit_option  options[4];
    void         *optionVals[4];

    // Setup the linker

    // Pass a buffer for info messages
    options[0] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[0] = info_log;
    // Pass the size of the info buffer
    options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[1] = reinterpret_cast<void *>(uintptr_t(sizeof(info_log)));
    // Pass a buffer for error messages
    options[2] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[2] = error_log;
    // Pass the size of the error buffer
    options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[3] = reinterpret_cast<void *>(uintptr_t(sizeof(error_log)));

    check_cuda_success(cuLinkCreate(4, options, optionVals, &cuda_link_state));

    CUresult link_result = CUDA_SUCCESS;
    do {
        // Add all code generated by the MDL PTX backend
        for (size_t i = 0, num_target_codes = target_codes.size(); i < num_target_codes; ++i) {
            link_result = cuLinkAddData(
                cuda_link_state, CU_JIT_INPUT_PTX,
                const_cast<char *>(target_codes[i]->get_src_code()),
                target_codes[i]->get_src_code_size(),
                nullptr, 0, nullptr, nullptr);
            if (link_result != CUDA_SUCCESS) break;
        }
        if (link_result != CUDA_SUCCESS) break;

        // Add the "mdl_expr_functions" array PTX module
        link_result = cuLinkAddData(
            cuda_link_state, CU_JIT_INPUT_PTX,
            const_cast<char *>(ptx_func_array_src.c_str()),
            ptx_func_array_src.size(),
            nullptr, 0, nullptr, nullptr);
        if (link_result != CUDA_SUCCESS) break;

        // Add our kernel
        link_result = cuLinkAddFile(
            cuda_link_state, CU_JIT_INPUT_PTX,
            ptx_file, 0, nullptr, nullptr);
        if (link_result != CUDA_SUCCESS) break;

        // Link everything to a cubin
        link_result = cuLinkComplete(cuda_link_state, &linked_cubin, &linked_cubin_size);
    } while (false);
    if (link_result != CUDA_SUCCESS) {
        std::cerr << "PTX linker error:\n" << error_log << std::endl;
        check_cuda_success(link_result);
    }

    std::cout << "CUDA link completed. Linker output:\n" << info_log << std::endl;

    // Load the result and get the entrypoint of our kernel
    check_cuda_success(cuModuleLoadData(&cuda_module, linked_cubin));
    check_cuda_success(cuModuleGetFunction(
        out_kernel_function, cuda_module, kernel_function_name));

    int regs = 0;
    check_cuda_success(
        cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, *out_kernel_function));
    int lmem = 0;
    check_cuda_success(
        cuFuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, *out_kernel_function));
    std::cout << "kernel uses " << regs << " registers and " << lmem << " lmem" << std::endl;

    // Cleanup
    check_cuda_success(cuLinkDestroy(cuda_link_state));

    return cuda_module;
}


//------------------------------------------------------------------------------
//
// Utility functions
//
//------------------------------------------------------------------------------

// Export the given 8-bit per channel RGBA data to the given path.
// The file format is determined by the path (must be supported by FreeImage).
bool export_image_rgba(
    char const *path, mi::Uint32 width, mi::Uint32 height, mi::Uint32 const *data)
{
    FIBITMAP *dib = FreeImage_AllocateT(
        FIT_BITMAP,
        int(width),
        int(height),
        /*bpp=*/ 32,
        /*red_mask=*/   0x00ff0000,
        /*green_mask=*/ 0x0000ff00,
        /*blue_mask=*/  0x000000ff);
    if (dib == nullptr)
        return false;

    if (FreeImage_GetPitch(dib) != unsigned(width * sizeof(mi::Uint32))) {
        std::cout << "Unexpected pitch" << std::endl;
        FreeImage_Unload(dib);
        return false;
    }

    memcpy(FreeImage_GetBits(dib), data, width * height * sizeof(mi::Uint32));

    FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(path);
    // unknown or unsupported type?
    if (fif == FIF_UNKNOWN || !FreeImage_FIFSupportsWriting(fif)) {
        FreeImage_Unload(dib);
        return false;
    }

    bool res = FreeImage_Save(fif, dib, path) != 0;
    FreeImage_Unload(dib);
    return res;
}

// Export the given RGBF data to the given path.
// The file format is determined by the path (must be supported by FreeImage).
bool export_image_rgbf(
    char const *path, mi::Uint32 width, mi::Uint32 height, float3 const *data)
{
    FIBITMAP *dib = FreeImage_AllocateT(
        FIT_RGBF,
        int(width),
        int(height));
    if (dib == nullptr)
        return false;

    if (FreeImage_GetPitch(dib) != unsigned(width * sizeof(float3))) {
        std::cout << "Unexpected pitch" << std::endl;
        FreeImage_Unload(dib);
        return false;
    }

    memcpy(FreeImage_GetBits(dib), data, width * height * sizeof(float3));

    FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(path);
    // unknown or unsupported type?
    if (fif == FIF_UNKNOWN || !FreeImage_FIFSupportsWriting(fif)) {
        FreeImage_Unload(dib);
        return false;
    }

    bool res = false;
    switch (fif)
    {
        // conversion required? Note, this list is not complete
        case FIF_PNG:
        case FIF_JPEG:
        case FIF_BMP:
        {
            FIBITMAP *dst = FreeImage_AllocateT(
                FIT_BITMAP,
                int(width),
                int(height),
                /*bpp=*/ 32,
                /*red_mask=*/   0x00ff0000,
                /*green_mask=*/ 0x0000ff00,
                /*blue_mask=*/  0x000000ff);


            // calculate the number of bytes per pixel (3 for 24-bit or 4 for 32-bit)
            const unsigned bytespp = FreeImage_GetLine(dst) / FreeImage_GetWidth(dst);

            const unsigned src_pitch = FreeImage_GetPitch(dib);
            const unsigned dst_pitch = FreeImage_GetPitch(dst);

            const BYTE *dst_bits = (BYTE*) FreeImage_GetBits(dst);
            BYTE *src_bits = (BYTE*) FreeImage_GetBits(dib);

            for (unsigned y = 0; y < height; y++)
            {
                BYTE *dst_pixel = (BYTE*) dst_bits;
                const FIRGBF *src_pixel = (FIRGBF*) src_bits;
                for (unsigned x = 0; x < width; x++)
                {
                    // convert and scale to the range [0..255]
                    dst_pixel[FI_RGBA_RED] =
                        (char)(std::max(0.f, std::min(src_pixel->red, 1.f)) * 255.0f);
                    dst_pixel[FI_RGBA_GREEN] =
                        (char)(std::max(0.f, std::min(src_pixel->green, 1.f)) * 255.0f);
                    dst_pixel[FI_RGBA_BLUE] =
                        (char)(std::max(0.f, std::min(src_pixel->blue, 1.f)) * 255.0f);
                    dst_pixel[FI_RGBA_ALPHA] = 255;

                    dst_pixel += bytespp;
                    src_pixel++;
                }
                dst_bits += dst_pitch;
                src_bits += src_pitch;
            }

            res = FreeImage_Save(fif, dst, path) != 0;
            FreeImage_Unload(dst);
            break;
        }

        // no conversion required
        default:
            res = FreeImage_Save(fif, dib, path) != 0;
            break;
    }

    FreeImage_Unload(dib);
    return res;
}

#endif // EXAMPLE_CUDA_SHARED_H
