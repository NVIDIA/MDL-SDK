/******************************************************************************
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
 *****************************************************************************/

// examples/mdl_sdk/optix7/optix7_mdl_helper.h
//
// Contains a helper class for handling MDL in OptiX 7.


#ifndef OPTIX7_MDL_HELPER_H
#define OPTIX7_MDL_HELPER_H

#include <string>
#include <vector>

#include <mi/mdl_sdk.h>
#include "material_info_helper.h"
#include "texture_support_cuda.h"
#include "example_shared.h"

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            exit_failure("CUDA error %d \"%s\" in file %s, line %u: \"%s\".\n", \
                err, cudaGetErrorString(err), __FILE__, __LINE__, #expr); \
        } \
    } while (false)


/// A helper class for handling MDL in OptiX 7.
class Mdl_helper
{
public:
    /// Result of a material compilation.
    struct Compile_result
    {
        /// The compiled material.
        mi::base::Handle<mi::neuraylib::ICompiled_material const> compiled_material;

        /// The generated target code object.
        mi::base::Handle<mi::neuraylib::ITarget_code const> target_code;

        /// The argument block for the compiled material.
        mi::base::Handle<mi::neuraylib::ITarget_argument_block const> argument_block;

        /// Information required to load a texture.
        struct Texture_info
        {
            std::string                                db_name;
            mi::neuraylib::ITarget_code::Texture_shape shape;

            Texture_info()
            : shape(mi::neuraylib::ITarget_code::Texture_shape_invalid)
            {}

            Texture_info(
                char const *db_name,
                mi::neuraylib::ITarget_code::Texture_shape shape)
            : db_name(db_name)
            , shape(shape)
            {}
        };

        /// Information required to load a light profile.
        struct Light_profile_info
        {
            std::string db_name;

            Light_profile_info()
            {}

            Light_profile_info(char const *db_name)
            : db_name(db_name)
            {}
        };

        /// Information required to load a BSDF measurement.
        struct Bsdf_measurement_info
        {
            std::string db_name;

            Bsdf_measurement_info()
            {}

            Bsdf_measurement_info(char const *db_name)
            : db_name(db_name)
            {}
        };

        /// Textures used by the compile result.
        std::vector<Texture_info> textures;

        /// Textures used by the compile result.
        std::vector<Light_profile_info> light_profiles;

        /// Textures used by the compile result.
        std::vector<Bsdf_measurement_info> bsdf_measurements;

        /// Constructor.
        Compile_result()
        {
            // add invalid resources
            textures.emplace_back();
            light_profiles.emplace_back();
            bsdf_measurements.emplace_back();
        }
    };

    /// Constructs an Mdl_helper object.
    /// \param configure_options     options used to configure the MDL SDK
    /// \param num_texture_spaces    the number of texture spaces provided in the MDL
    ///                              Shading_state_material fields text_coords, tangent_t and
    ///                              tangent_v by the renderer
    ///                              If invalid texture spaces are requested in the MDL materials,
    ///                              null values will be returned.
    /// \param num_texture_results   the size of the text_results array in the MDL
    ///                              Shading_state_material text_results field provided by
    ///                              the renderer.
    ///                              The text_results array will be filled in the init function
    ///                              created for a distribution function and used by the sample,
    ///                              evaluate and pdf functions, if the size is non-zero.
    /// \param enable_derivatives    If true, texture lookup functions with derivatives will
    ///                              be called and mipmaps will be generated for all 2D textures.
    Mdl_helper(
        mi::examples::mdl::Configure_options const &configure_options,
        unsigned num_texture_spaces = 1,
        unsigned num_texture_results = 0,
        bool enable_derivatives = false);

    /// Destructor shutting down Neuray.
    ~Mdl_helper();

    /// Get the last error message reported by the MDL SDK.
    std::string const &get_last_error_message() const {
        return m_last_mdl_error;
    }

    /// Sets the renderer module to be linked with the generated code.
    ///
    /// \param renderer_module_path  Path to the LLVM 7 module
    /// \param visible_functions     Functions which should not be optimized away
    void set_renderer_module(
        const std::string &renderer_module_path,
        const std::string &visible_functions);

    /// Adds a path to search for MDL modules and resources used by those modules.
    void add_mdl_path(const std::string &mdl_path);

    /// Helper function to extract the module name from a fully-qualified material name.
    std::string get_module_name(const std::string& material_name) const;

    /// Helper function to extract the material name from a fully-qualified material name.
    std::string get_material_name(const std::string& material_name) const;

    /// Prints the messages of the given context.
    /// Returns true, if the context does not contain any error messages, false otherwise.
    bool log_messages(mi::neuraylib::IMdl_execution_context* context);

    /// Create an MDL SDK transaction.
    mi::base::Handle<mi::neuraylib::ITransaction> create_transaction() {
        return mi::base::make_handle<mi::neuraylib::ITransaction>(
            m_global_scope->create_transaction());
    }

    /// Get the image API component.
    mi::base::Handle<mi::neuraylib::IImage_api> get_image_api() {
        return m_image_api;
    }

    /// Get the import export API component.
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> get_impexp_api() {
        return mi::base::Handle<mi::neuraylib::IMdl_impexp_api>(
            m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    }

    /// Compile the MDL material into target code.
    Compile_result compile_mdl_material(
        mi::neuraylib::ITransaction *transaction,
        std::string const &material_name,
        std::vector<mi::neuraylib::Target_function_description> &descs,
        bool class_compilation,
        Material_info **out_mat_info = nullptr);

    /// Copy the image data of a canvas to a CUDA array.
    void copy_canvas_to_cuda_array(
        cudaArray_t device_array,
        mi::neuraylib::ICanvas const *canvas);

    /// Prepare the given texture for use by the texture access functions on the GPU.
    bool prepare_texture(
        mi::neuraylib::ITransaction                *transaction,
        char const                                 *texture_db_name,
        mi::neuraylib::ITarget_code::Texture_shape  texture_shape,
        std::vector<Texture>                       &textures);

private:
    /// If true, mipmaps will be generated for all 2D textures.
    bool m_enable_derivatives;

    /// The Neuray interface of the MDL SDK.
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;

    /// The MDL configuration.
    mi::base::Handle<mi::neuraylib::IMdl_configuration> m_mdl_config;

    /// The MDL compiler.
    mi::base::Handle<mi::neuraylib::IMdl_compiler> m_mdl_compiler;

    /// The Neuray database of the MDL SDK.
    mi::base::Handle<mi::neuraylib::IDatabase> m_database;

    /// The global scope of the data base used to create transactions.
    mi::base::Handle<mi::neuraylib::IScope> m_global_scope;

    /// The MDL factory.
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_mdl_factory;

    /// Can be used to query status information like errors and warnings.
    /// The context is also used to set options for module loading, MDL export,
    /// material compilation, and for the code generation.
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_execution_context;

    /// The CUDA PTX backend of the MDL compiler.
    mi::base::Handle<mi::neuraylib::IMdl_backend> m_be_cuda_ptx;

    /// The Image API for converting image to other formats.
    mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;

    /// The last error message from MDL SDK.
    std::string m_last_mdl_error;

    typedef std::map<mi::base::Uuid, mi::base::Handle<mi::neuraylib::ITarget_code const>>
        Target_code_cache;

    /// Maps a compiled material hash to a target code object to avoid generation of duplicate code.
    Target_code_cache m_target_code_cache;

    typedef std::map<std::string, Texture> Texture_cache;

    /// Maps a texture database name and shape to a Texture object to avoid the texture
    /// being loaded and converted multiple times.
    Texture_cache m_texture_cache;

    /// List of CUDA texture arrays.
    std::vector<cudaArray_t> m_texture_arrays;

    /// List of CUDA texture mipmapped arrays
    std::vector<cudaMipmappedArray_t> m_texture_mipmapped_arrays;
};

// Constructs an Mdl_helper object.
Mdl_helper::Mdl_helper(
    mi::examples::mdl::Configure_options const &configure_options,
    unsigned num_texture_spaces,
    unsigned num_texture_results,
    bool enable_derivatives)
: m_enable_derivatives(enable_derivatives)
{
    m_neuray = mi::examples::mdl::load_and_get_ineuray();
    if (!m_neuray.is_valid_interface())
        exit_failure("ERROR: Initialization of MDL SDK failed: libmdl_sdk" MI_BASE_DLL_FILE_EXT
            " not found or wrong version.");

    m_mdl_compiler = m_neuray->get_api_component<mi::neuraylib::IMdl_compiler>();
    if (!m_mdl_compiler)
        exit_failure("ERROR: Initialization of MDL compiler failed!");

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(m_neuray.get(), configure_options))
        exit_failure("ERROR: Configuration of MDL SDK failed.");

    m_mdl_config = m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>();
    if (!m_mdl_config)
        exit_failure("ERROR: Retrieving MDL configuration failed!");

    // Load additionally required optional plugins for texture support
    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plug_config(
        m_neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());

    // Consider the dds plugin as optional
    plug_config->load_plugin_library("dds" MI_BASE_DLL_FILE_EXT);

    if (m_neuray->start() != 0)
        exit_failure("ERROR: Starting MDL SDK failed!");

    m_database = m_neuray->get_api_component<mi::neuraylib::IDatabase>();
    m_global_scope = m_database->get_global_scope();

    m_mdl_factory = m_neuray->get_api_component<mi::neuraylib::IMdl_factory>();

    // Configure the execution context, that is used for various configurable operations and for
    // querying warnings and error messages.
    // It is possible to have more than one in order to use different settings.
    m_execution_context = m_mdl_factory->create_execution_context();

    m_execution_context->set_option("internal_space", "coordinate_world");  // equals default
    m_execution_context->set_option("bundle_resources", false);             // equals default
    m_execution_context->set_option("meters_per_scene_unit", 1.0f);         // equals default
    m_execution_context->set_option("mdl_wavelength_min", 380.0f);          // equals default
    m_execution_context->set_option("mdl_wavelength_max", 780.0f);          // equals default
    m_execution_context->set_option("include_geometry_normal", true);       // equals default

    mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
        m_neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());
    m_be_cuda_ptx = mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX);
    if (m_be_cuda_ptx->set_option(
            "num_texture_spaces", std::to_string(num_texture_spaces).c_str()) != 0)
        exit_failure("ERROR: Setting PTX option num_texture_spaces failed");
    if (m_be_cuda_ptx->set_option(
            "num_texture_results", std::to_string(num_texture_results).c_str()) != 0)
        exit_failure("ERROR: Setting PTX option num_texture_results failed");
    if (m_be_cuda_ptx->set_option("sm_version", "30") != 0)
        exit_failure("ERROR: Setting PTX option sm_version failed");
    if (m_be_cuda_ptx->set_option("tex_lookup_call_mode", "direct_call") != 0)
        exit_failure("ERROR: Setting PTX option tex_lookup_call_mode failed");
    if (enable_derivatives) {
        // Option "texture_runtime_with_derivs": Default is disabled.
        // We enable it to get coordinates with derivatives for texture lookup functions.
        if (m_be_cuda_ptx->set_option("texture_runtime_with_derivs", "on") != 0)
            exit_failure("ERROR: Setting PTX option texture_runtime_with_derivs failed");
    }
    if (m_be_cuda_ptx->set_option("inline_aggressively", "on") != 0)
        exit_failure("ERROR: Setting PTX option inline_aggressively failed");
    if (m_be_cuda_ptx->set_option("scene_data_names", "*") != 0)
        exit_failure("ERROR: Setting PTX option scene_data_names failed");

    m_image_api = m_neuray->get_api_component<mi::neuraylib::IImage_api>();
}

// Destructor shutting down Neuray.
Mdl_helper::~Mdl_helper()
{
    for (auto it = m_texture_cache.begin(); it != m_texture_cache.end(); ++it) {
        cudaDestroyTextureObject(it->second.filtered_object);
        cudaDestroyTextureObject(it->second.unfiltered_object);
    }

    for (cudaArray_t array : m_texture_arrays) {
        cudaFreeArray(array);
    }

    for (cudaMipmappedArray_t array : m_texture_mipmapped_arrays) {
        cudaFreeMipmappedArray(array);
    }

    // Clear target code cache before shutting down MDL SDK
    m_target_code_cache.clear();

    m_image_api.reset();
    m_be_cuda_ptx.reset();
    m_execution_context.reset();
    m_mdl_factory.reset();
    m_global_scope.reset();
    m_database.reset();
    m_mdl_compiler.reset();

    m_neuray->shutdown();
}

// Sets the renderer module to be linked with the generated code.
void Mdl_helper::set_renderer_module(
    const std::string &renderer_module_path,
    const std::string &visible_functions)
{
    std::vector<char> renderer_module(
        mi::examples::io::read_binary_file(renderer_module_path));
    if (renderer_module.empty())
        exit_failure("ERROR: %s could not be opened", renderer_module_path.c_str());

    if (m_be_cuda_ptx->set_option_binary(
            "llvm_renderer_module", renderer_module.data(), renderer_module.size()) != 0)
        exit_failure("ERROR: Setting PTX option llvm_renderer_module failed");

    // limit functions for which PTX code is generated to the entry functions
    if (m_be_cuda_ptx->set_option("visible_functions", visible_functions.c_str()) != 0)
        exit_failure("ERROR: Setting PTX option visible_functions failed");
}

// Adds a path to search for MDL modules and resources used by those modules.
void Mdl_helper::add_mdl_path(const std::string &mdl_path)
{
    // Set module path for MDL file and resources
    if (m_mdl_config->add_mdl_path(mdl_path.c_str()) != 0)
        exit_failure("ERROR: Adding module path failed!");
}

// Helper function to extract the module name from a fully-qualified material name.
std::string Mdl_helper::get_module_name(const std::string& material_name) const
{
    size_t p = material_name.rfind("::");
    return material_name.substr(0, p);
}

// Helper function to extract the material name from a fully-qualified material name.
std::string Mdl_helper::get_material_name(const std::string& material_name) const
{
    size_t p = material_name.rfind("::");
    if (p == std::string::npos)
        return material_name;
    return material_name.substr(p + 2, material_name.size() - p);
}

// Returns a string-representation of the given message severity
inline const char* message_severity_to_string(mi::base::Message_severity severity)
{
    switch (severity) {
    case mi::base::MESSAGE_SEVERITY_ERROR:
        return "error";
    case mi::base::MESSAGE_SEVERITY_WARNING:
        return "warning";
    case mi::base::MESSAGE_SEVERITY_INFO:
        return "info";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
        return "verbose";
    case mi::base::MESSAGE_SEVERITY_DEBUG:
        return "debug";
    default:
        break;
    }
    return "";
}

// Returns a string-representation of the given message category
inline const char* message_kind_to_string(mi::neuraylib::IMessage::Kind message_kind)
{
    switch (message_kind) {
    case mi::neuraylib::IMessage::MSG_INTEGRATION:
        return "MDL SDK";
    case mi::neuraylib::IMessage::MSG_IMP_EXP:
        return "Importer/Exporter";
    case mi::neuraylib::IMessage::MSG_COMILER_BACKEND:
        return "Compiler Backend";
    case mi::neuraylib::IMessage::MSG_COMILER_CORE:
        return "Compiler Core";
    case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
        return "Compiler Archive Tool";
    case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
        return "Compiler DAG generator";
    default:
        break;
    }
    return "";
}

// Prints the messages of the given context.
// Returns true, if the context does not contain any error messages, false otherwise.
bool Mdl_helper::log_messages(mi::neuraylib::IMdl_execution_context* context)
{
    m_last_mdl_error.clear();

    for (mi::Size i = 0; i < context->get_messages_count(); ++i) {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
        m_last_mdl_error += message_kind_to_string(message->get_kind());
        m_last_mdl_error += " ";
        m_last_mdl_error += message_severity_to_string(message->get_severity());
        m_last_mdl_error += ": ";
        m_last_mdl_error += message->get_string();
        m_last_mdl_error += "\n";
    }
    return context->get_error_messages_count() == 0;
}

/// Callback that notifies the application about new resources when generating an
/// argument block for an existing target code.
class Resource_callback
    : public mi::base::Interface_implement<mi::neuraylib::ITarget_resource_callback>
{
public:
    /// Constructor.
    Resource_callback(
        mi::neuraylib::ITransaction *transaction,
        mi::neuraylib::ITarget_code const *target_code,
        Mdl_helper::Compile_result &compile_result)
    : m_transaction(mi::base::make_handle_dup(transaction))
    , m_target_code(mi::base::make_handle_dup(target_code))
    , m_compile_result(compile_result)
    {
    }

    /// Destructor.
    virtual ~Resource_callback() = default;

    /// Returns a resource index for the given resource value usable by the target code
    /// resource handler for the corresponding resource type.
    ///
    /// \param resource  the resource value
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    mi::Uint32 get_resource_index(mi::neuraylib::IValue_resource const *resource) override
    {
        // check whether we already know the resource index
        auto it = m_resource_cache.find(resource);
        if (it != m_resource_cache.end())
            return it->second;

        // handle resources already known by the target code
        mi::Uint32 res_idx = m_target_code->get_known_resource_index(m_transaction.get(), resource);
        if (res_idx != 0) {
            // only accept body resources
            switch (resource->get_kind()) {
            case mi::neuraylib::IValue::VK_TEXTURE:
                if (res_idx < m_target_code->get_body_texture_count())
                    return res_idx;
                break;
            case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
                if (res_idx < m_target_code->get_body_light_profile_count())
                    return res_idx;
                break;
            case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
                if (res_idx < m_target_code->get_body_bsdf_measurement_count())
                    return res_idx;
                break;
            default:
                return 0u;  // invalid kind
            }
        }

        switch (resource->get_kind()) {
        case mi::neuraylib::IValue::VK_TEXTURE:
            {
                mi::base::Handle<mi::neuraylib::IValue_texture const> val_texture(
                    resource->get_interface<mi::neuraylib::IValue_texture const>());
                if (!val_texture)
                    return 0u;  // unknown resource

                mi::base::Handle<const mi::neuraylib::IType_texture> texture_type(
                    val_texture->get_type());

                mi::neuraylib::ITarget_code::Texture_shape shape =
                    mi::neuraylib::ITarget_code::Texture_shape(texture_type->get_shape());

                m_compile_result.textures.emplace_back(resource->get_value(), shape);
                res_idx = m_compile_result.textures.size() - 1;
                break;
            }
        case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
            m_compile_result.light_profiles.emplace_back(resource->get_value());
            res_idx = m_compile_result.light_profiles.size() - 1;
            break;
        case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
            m_compile_result.bsdf_measurements.emplace_back(resource->get_value());
            res_idx = m_compile_result.bsdf_measurements.size() - 1;
            break;
        default:
            return 0u;  // invalid kind
        }

        m_resource_cache[resource] = res_idx;
        return res_idx;
    }

    /// Returns a string identifier for the given string value usable by the target code.
    ///
    /// The value 0 is always the "not known string".
    ///
    /// \param s  the string value
    mi::Uint32 get_string_index(mi::neuraylib::IValue_string const *s) override
    {
        char const *str_val = s->get_value();
        if (str_val == nullptr)
            return 0u;

        for (mi::Size i = 0, n = m_target_code->get_string_constant_count(); i < n; ++i) {
            if (strcmp(m_target_code->get_string_constant(i), str_val) == 0)
                return mi::Uint32(i);
        }

        // string not known by code
        return 0u;
    }

private:
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;

    std::map<mi::neuraylib::IValue_resource const *, mi::Uint32> m_resource_cache;
    Mdl_helper::Compile_result &m_compile_result;
};

// Compile the MDL material into target code.
Mdl_helper::Compile_result Mdl_helper::compile_mdl_material(
    mi::neuraylib::ITransaction *transaction,
    std::string const &material_name,
    std::vector<mi::neuraylib::Target_function_description> &descs,
    bool class_compilation,
    Material_info **out_mat_info)
{
    Compile_result res;

    // Access needed API components
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    // Create an execution context for options and error message handling
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // Load the module
    std::string module_name = get_module_name(material_name);
    mdl_impexp_api->load_module(transaction, module_name.c_str(), context.get());
    if (!log_messages(context.get()))
        return res;

    // Create a material instance from the material definition
    // with the default arguments.
    const char *prefix = (material_name.find("::") == 0) ? "mdl" : "mdl::";

    std::string material_db_name = prefix + material_name;
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        transaction->access<mi::neuraylib::IMaterial_definition>(
            material_db_name.c_str()));
    if (!material_definition) {
        m_last_mdl_error = "Material \"" + material_name + "\" not found";
        return res;
    }

    mi::Sint32 ret = 0;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
        material_definition->create_material_instance(0, &ret));
    if (ret != 0) {
        m_last_mdl_error = "Instantiating material \"" + material_name + "\" failed";
        return res;
    }

    // Create a compiled material
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    res.compiled_material = material_instance->create_compiled_material(flags, context.get());
    if (!log_messages(context.get()))
        return res;

    // Reuse old target code, if possible
    mi::base::Uuid material_hash = res.compiled_material->get_hash();
    auto it = m_target_code_cache.find(material_hash);
    if (it != m_target_code_cache.end()) {
        res.target_code = it->second;

        // initialize with body resources always being required
        if (res.target_code->get_body_texture_count() > 0) {
            for (mi::Size i = 1, n = res.target_code->get_body_texture_count(); i < n; ++i) {
                res.textures.emplace_back(
                    res.target_code->get_texture(i),
                    res.target_code->get_texture_shape(i));
            }
        }

        if (res.target_code->get_body_light_profile_count() > 0) {
            for (mi::Size i = 1, n = res.target_code->get_body_light_profile_count(); i < n; ++i) {
                res.light_profiles.emplace_back(
                    res.target_code->get_light_profile(i));
            }
        }

        if (res.target_code->get_body_bsdf_measurement_count() > 0) {
            for (mi::Size i = 1, n = res.target_code->get_body_bsdf_measurement_count(); i < n;
                    ++i) {
                res.bsdf_measurements.emplace_back(
                    res.target_code->get_bsdf_measurement(i));
            }
        }

        if (res.target_code->get_argument_block_count() > 0) {
            // Create argument block for the new compiled material and additional resources
            mi::base::Handle<Resource_callback> res_callback(
                new Resource_callback(transaction, res.target_code.get(), res));
            res.argument_block = res.target_code->create_argument_block(
                0, res.compiled_material.get(), res_callback.get());
        }
    }
    else {
        // Generate target code for the compiled material
        mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(m_be_cuda_ptx->create_link_unit(
            transaction, context.get()));
        link_unit->add_material(
            res.compiled_material.get(), descs.data(), descs.size(), context.get());
        if (!log_messages(context.get()))
            return res;

        res.target_code = mi::base::Handle<const mi::neuraylib::ITarget_code>(
            m_be_cuda_ptx->translate_link_unit(link_unit.get(), context.get()));
        if (!log_messages(context.get()))
            return res;

        m_target_code_cache[material_hash] = res.target_code;

        // add all used resources
        for (mi::Size i = 1, n = res.target_code->get_texture_count(); i < n; ++i) {
            res.textures.emplace_back(
                res.target_code->get_texture(i),
                res.target_code->get_texture_shape(i));
        }

        if (res.target_code->get_light_profile_count() > 0) {
            for (mi::Size i = 1, n = res.target_code->get_light_profile_count(); i < n; ++i) {
                res.light_profiles.emplace_back(
                    res.target_code->get_light_profile(i));
            }
        }

        if (res.target_code->get_bsdf_measurement_count() > 0) {
            for (mi::Size i = 1, n = res.target_code->get_bsdf_measurement_count(); i < n; ++i) {
                res.bsdf_measurements.emplace_back(
                    res.target_code->get_bsdf_measurement(i));
            }
        }
        if (res.target_code->get_argument_block_count() > 0)
            res.argument_block = res.target_code->get_argument_block(0);
    }

    if (out_mat_info != nullptr) {
        mi::base::Handle<mi::neuraylib::ITarget_value_layout const> arg_layout;

        if (res.target_code->get_argument_block_count() > 0)
            arg_layout = res.target_code->get_argument_block_layout(0);

        *out_mat_info = new Material_info(
            material_definition.get(),
            res.compiled_material.get(),
            arg_layout.get(),
            res.argument_block.get());
    }

    return res;
}

// Copy the image data of a canvas to a CUDA array.
void Mdl_helper::copy_canvas_to_cuda_array(
    cudaArray_t device_array,
    mi::neuraylib::ICanvas const *canvas)
{
    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
    mi::Float32 const *data = static_cast<mi::Float32 const *>(tile->get_data());
    CUDA_CHECK(cudaMemcpy2DToArray(
        device_array, 0, 0, data,
        canvas->get_resolution_x() * sizeof(float) * 4,
        canvas->get_resolution_x() * sizeof(float) * 4,
        canvas->get_resolution_y(),
        cudaMemcpyHostToDevice));
}

// Prepare the texture identified by the texture_index for use by the texture access functions
// on the GPU.
bool Mdl_helper::prepare_texture(
    mi::neuraylib::ITransaction                *transaction,
    char const                                 *texture_db_name,
    mi::neuraylib::ITarget_code::Texture_shape  texture_shape,
    std::vector<Texture>                       &textures)
{
    // Get access to the texture data by the texture database name from the target code.
    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        transaction->access<mi::neuraylib::ITexture>(texture_db_name));

    // First check the texture cache
    std::string entry_name = std::string(texture_db_name) + "_"
        + std::to_string(unsigned(texture_shape));
    auto it = m_texture_cache.find(entry_name);
    if (it != m_texture_cache.end()) {
        textures.push_back(it->second);
        return true;
    }

    // Access image and canvas via the texture object
    mi::base::Handle<const mi::neuraylib::IImage> image(
        transaction->access<mi::neuraylib::IImage>(texture->get_image()));
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas());
    mi::Uint32 tex_width = canvas->get_resolution_x();
    mi::Uint32 tex_height = canvas->get_resolution_y();
    mi::Uint32 tex_layers = canvas->get_layers_size();
    char const *image_type = image->get_type();

    if (image->is_uvtile()) {
        std::cerr << "The example does not support uvtile textures!" << std::endl;
        return false;
    }

    if (canvas->get_tiles_size_x() != 1 || canvas->get_tiles_size_y() != 1) {
        std::cerr << "The example does not support tiled images!" << std::endl;
        return false;
    }

    // For simplicity, the texture access functions are only implemented for float4 and gamma
    // is pre-applied here (all images are converted to linear space).

    // Convert to linear color space if necessary
    if (texture->get_effective_gamma() != 1.0f) {
        // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
        mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
            m_image_api->convert(canvas.get(), "Color"));
        gamma_canvas->set_gamma(texture->get_effective_gamma());
        m_image_api->adjust_gamma(gamma_canvas.get(), 1.0f);
        canvas = gamma_canvas;
    } else if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0) {
        // Convert to expected format
        canvas = m_image_api->convert(canvas.get(), "Color");
    }

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));

    // Copy image data to GPU array depending on texture shape
    if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube ||
            texture_shape == mi::neuraylib::ITarget_code::Texture_shape_3d ||
            texture_shape == mi::neuraylib::ITarget_code::Texture_shape_bsdf_data) {
        // Cubemap and 3D texture objects require 3D CUDA arrays

        if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube &&
            tex_layers != 6) {
            std::cerr << "Invalid number of layers (" << tex_layers
                << "), cubemaps must have 6 layers!" << std::endl;
            return false;
        }

        // Allocate a 3D array on the GPU
        cudaExtent extent = make_cudaExtent(tex_width, tex_height, tex_layers);
        cudaArray_t device_tex_array;
        CUDA_CHECK(cudaMalloc3DArray(
            &device_tex_array, &channel_desc, extent,
            texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube ?
            cudaArrayCubemap : 0));

        // Prepare the memcpy parameter structure
        cudaMemcpy3DParms copy_params;
        memset(&copy_params, 0, sizeof(copy_params));
        copy_params.dstArray = device_tex_array;
        copy_params.extent = make_cudaExtent(tex_width, tex_height, 1);
        copy_params.kind = cudaMemcpyHostToDevice;

        // Copy the image data of all layers (the layers are not consecutive in memory)
        for (mi::Uint32 layer = 0; layer < tex_layers; ++layer) {
            mi::base::Handle<const mi::neuraylib::ITile> tile(
                canvas->get_tile(0, 0, layer));
            float const *data = static_cast<float const *>(tile->get_data());

            copy_params.srcPtr = make_cudaPitchedPtr(
                const_cast<float *>(data), tex_width * sizeof(float) * 4,
                tex_width, tex_height);
            copy_params.dstPos = make_cudaPos(0, 0, layer);

            CUDA_CHECK(cudaMemcpy3D(&copy_params));
        }

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_tex_array;

        m_texture_arrays.push_back(device_tex_array);
    } else if (m_enable_derivatives) {
        // mipmapped textures use CUDA mipmapped arrays
        mi::Uint32 num_levels = image->get_levels();
        cudaExtent extent = make_cudaExtent(tex_width, tex_height, 0);
        cudaMipmappedArray_t device_tex_miparray;
        CUDA_CHECK(cudaMallocMipmappedArray(
            &device_tex_miparray, &channel_desc, extent, num_levels));

        // create all mipmap levels and copy them to the CUDA arrays in the mipmapped array
        mi::base::Handle<mi::IArray> mipmaps(m_image_api->create_mipmaps(canvas.get(), 1.0f));

        for (mi::Uint32 level = 0; level < num_levels; ++level) {
            mi::base::Handle<mi::neuraylib::ICanvas const> level_canvas;
            if (level == 0)
                level_canvas = canvas;
            else {
                mi::base::Handle<mi::IPointer> mipmap_ptr(
                    mipmaps->get_element<mi::IPointer>(level - 1));
                level_canvas = mipmap_ptr->get_pointer<mi::neuraylib::ICanvas>();
            }
            cudaArray_t device_level_array;
            cudaGetMipmappedArrayLevel(&device_level_array, device_tex_miparray, level);
            copy_canvas_to_cuda_array(device_level_array, level_canvas.get());
        }

        res_desc.resType = cudaResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = device_tex_miparray;

        m_texture_mipmapped_arrays.push_back(device_tex_miparray);
    } else {
        // 2D texture objects use CUDA arrays
        cudaArray_t device_tex_array;
        CUDA_CHECK(cudaMallocArray(
            &device_tex_array, &channel_desc, tex_width, tex_height));

        copy_canvas_to_cuda_array(device_tex_array, canvas.get());

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_tex_array;

        m_texture_arrays.push_back(device_tex_array);
    }

    // For cube maps we need clamped address mode to avoid artifacts in the corners
    cudaTextureAddressMode addr_mode =
        texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube
        ? cudaAddressModeClamp
        : cudaAddressModeWrap;

    // Create filtered texture object
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = addr_mode;
    tex_desc.addressMode[1] = addr_mode;
    tex_desc.addressMode[2] = addr_mode;
    tex_desc.filterMode     = cudaFilterModeLinear;
    tex_desc.readMode       = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    if (res_desc.resType == cudaResourceTypeMipmappedArray) {
        tex_desc.mipmapFilterMode = cudaFilterModeLinear;
        tex_desc.maxAnisotropy = 16;
        tex_desc.minMipmapLevelClamp = 0.f;
        tex_desc.maxMipmapLevelClamp = 1000.f;  // default value in OpenGL
    }

    cudaTextureObject_t tex_obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    // Create unfiltered texture object if necessary (cube textures have no texel functions)
    cudaTextureObject_t tex_obj_unfilt = 0;
    if (texture_shape != mi::neuraylib::ITarget_code::Texture_shape_cube) {
        // Use a black border for access outside of the texture
        tex_desc.addressMode[0] = cudaAddressModeBorder;
        tex_desc.addressMode[1] = cudaAddressModeBorder;
        tex_desc.addressMode[2] = cudaAddressModeBorder;
        tex_desc.filterMode     = cudaFilterModePoint;

        CUDA_CHECK(cudaCreateTextureObject(
            &tex_obj_unfilt, &res_desc, &tex_desc, nullptr));
    }

    // Store texture infos in result vector
    textures.push_back(Texture(
        tex_obj,
        tex_obj_unfilt,
        make_uint3(tex_width, tex_height, tex_layers)));

    m_texture_cache[entry_name] = textures.back();

    return true;
}

#endif // OPTIX7_MDL_HELPER_H
