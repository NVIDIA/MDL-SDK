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

 // examples/mdl_core/code_gen/example_code_gen.cpp
 //
 // Loads an MDL material and processes it up to code generation and beyond

#include <iostream>
#include <string>
#include <vector>

#include "example_shared_backends.h"

using TD = Target_function_description;  // only for readability

// The options parsed from the command-line
class Options
{
public:
    /// Prints the usage information to the stream \p s.
    static void print_usage(std::ostream& s);

    /// Parses the command-line given by \p argc and \p argv.
    ///
    /// \return   \c true in case of success and \c false otherwise.
    bool parse(int argc, char* argv[]);

    /// General options
    bool m_help = false;
    std::vector<std::string> m_mdl_paths;
    std::string m_output_file;

    /// Code compilation and generation options
    bool m_use_class_compilation = true;
    bool m_fold_ternary_on_df = false;
    bool m_fold_all_bool_parameters = false;
    bool m_fold_all_enum_parameters = false;
    bool m_single_init = false;
    bool m_ignore_noinline = true;
    bool m_disable_pdf = false;
    bool m_enable_aux = false;
    std::string m_lambda_return_mode;
    bool m_warn_spectrum_conv = false;
    std::string m_backend = "hlsl";
    bool m_use_derivatives = false;
    mi::Uint32 m_num_texture_results = 16;
    bool m_dump_metadata = false;
    bool m_adapt_normal = false;
    bool m_adapt_microfacet_roughness = false;
    bool m_experimental = false;
    bool m_run_material_analysis = false;

    /// MDL qualified material name to generate code for
    std::string m_qualified_material_name = "::nvidia::sdk_examples::tutorials::example_material";

    /// The expressions to generate code for
    std::vector<TD> m_descs;
    std::vector<std::unique_ptr<std::string> > m_desc_strs;  // Collection for storing the strings
};

// Dump generated meta data tables.
void dump_metadata(Target_code *code, std::ostream& out)
{
    out << "/* String table\n";
    String_constant_table& string_table = code->get_string_constant_table();
    for (unsigned i = 0, n = string_table.get_number_of_strings(); i < n; ++i)
    {
        const char* c = string_table.get_string(i);
        out << "   " << i << ": \"" << c << "\"\n";
    }
    out << "*/\n\n";

    out << "/* Texture table\n";
    for (mi::Size i = 0, n = code->get_texture_count(); i < n; ++i)
    {
        const Texture_data* tex = code->get_texture(i);

        const char* c = tex->get_url();
        const char* b = ( tex->get_bsdf_data() != nullptr ) ? "(body)" : "(non-body)";
        out << "   " << i << ": \"" << c << "\" " << b << "\n";
    }
    out << "*/\n\n";
}

static char const *opacity(mi::mdl::IMaterial_instance::Opacity o)
{
    switch (o) {
    case mi::mdl::IMaterial_instance::OPACITY_OPAQUE:
        return "OPAQUE";
    case mi::mdl::IMaterial_instance::OPACITY_TRANSPARENT:
        return "TRANSPARENT";
    default:
        return "UNKNOWN";
    }
}

// The main function for this example
void code_generation(mi::mdl::IMDL* mdl_compiler, Options& options)
{
    // Enable experimental features if requested
    if (options.m_experimental)
    {
        mi::mdl::Options& options = mdl_compiler->access_options();
        options.set_option(MDL_OPTION_EXPERIMENTAL_FEATURES, "true");
    }

    // DAG backend?
    if (options.m_backend == "dag")
    {
        // Initialize the material compiler
        Material_compiler mc(
            mdl_compiler,
            // Dumping output to file or stdout
            !options.m_output_file.empty() ? options.m_output_file.c_str() : nullptr);

        // Set user defined paths
        for (auto path : options.m_mdl_paths)
            mc.add_module_path(path.c_str());

        // Create a material instance which will be used to create DAG nodes for the material arguments
        Material_instance mat_instance(
            mc.create_material_instance(options.m_qualified_material_name));
        check_success(mat_instance);

        // Instantiation flags
        mi::Uint32 flags = 0;
        if (options.m_ignore_noinline)
            flags |= mi::mdl::IMaterial_instance::IGNORE_NOINLINE;

        // For class compilation only
        if (options.m_use_class_compilation)
        {
            if (options.m_fold_all_bool_parameters)
                flags |= mi::mdl::IMaterial_instance::NO_BOOL_PARAMS;
            if (options.m_fold_all_enum_parameters)
                flags |= mi::mdl::IMaterial_instance::NO_ENUM_PARAMS;
            if (options.m_fold_ternary_on_df)
                flags |= mi::mdl::IMaterial_instance::NO_TERNARY_ON_DF;
        }

        // Initialize the material instance
        mi::mdl::Dag_error_code err =
            mc.initialize_material_instance(
                mat_instance,
                {},
                options.m_use_class_compilation,
                flags);
        check_success(err == mi::mdl::EC_NONE);

        // Dump the generated code
        mc.get_printer()->print(mat_instance.get_material_instance().get());

        // The target code can also be serialized for later reuse.
        // This can make sense to reduce startup time for larger scenes that have been loaded
        // before. A key for this kind of cache is the hash of a compiled material:
        const mi::mdl::DAG_hash* dag_hash = mat_instance.get_material_instance()->get_hash();
        const unsigned char* hash = dag_hash->data();

        // Converting to 128-bit UUID representation
        mi::base::Uuid uuid;
        uuid.m_id1 = (hash[0] << 24) | (hash[1] << 16) | (hash[2] << 8) | hash[3];
        uuid.m_id2 = (hash[4] << 24) | (hash[5] << 16) | (hash[6] << 8) | hash[7];
        uuid.m_id3 = (hash[8] << 24) | (hash[9] << 16) | (hash[10] << 8) | hash[11];
        uuid.m_id4 = (hash[12] << 24) | (hash[13] << 16) | (hash[14] << 8) | hash[15];

        std::cout << "\n\n\n";
        std::cout << "Compiled material hash: \n" << std::hex << uuid.m_id1 << " " << uuid.m_id2
            << " " << uuid.m_id3 << " " << uuid.m_id4 << std::dec << "\n";

        if (options.m_run_material_analysis) {
            std::cout << "\n\n\n";
            std::cout << "Might depend on transform state functions: " <<
                (mat_instance->depends_on_transform() ? "YES" : "NO") << "\n";
            std::cout << "Might depend on state::object_id(): " <<
                (mat_instance->depends_on_object_id() ? "YES" : "NO") << "\n";
            std::cout << "Might depend on global distribution: " <<
                (mat_instance->depends_on_global_distribution() ? "YES" : "NO") << "\n";
            std::cout << "Might depend on uniform scene data: " <<
                (mat_instance->depends_on_uniform_scene_data() ? "YES" : "NO") << "\n";
            std::cout << "Opacity of this instance: " <<
                opacity(mat_instance->get_opacity()) << "\n";
            std::cout << "Surface opacity of this instance: " <<
                opacity(mat_instance->get_surface_opacity()) << "\n";
            const mi::mdl::IValue_float* cutout_opacity = mat_instance->get_cutout_opacity();
            std::cout << "Has constant cutout opacity: " << (cutout_opacity ? "YES" : "NO") << "\n";
            if (cutout_opacity)
                std::cout << "Cutout opacity of this instance: " << cutout_opacity->get_value() << "\n";
        }
    }
    else // For all other backends
    {
        // Select backend
        mi::mdl::ICode_generator::Target_language target_backend;
        if (options.m_backend == "ptx")
            target_backend = mi::mdl::ICode_generator::TL_PTX;
        else if(options.m_backend == "hlsl")
            target_backend = mi::mdl::ICode_generator::TL_HLSL;
        else if (options.m_backend == "glsl")
            target_backend = mi::mdl::ICode_generator::TL_GLSL;
        else // "llvm"
            target_backend = mi::mdl::ICode_generator::TL_LLVM_IR;

        // Select material compiler backend options
        mi::Uint32 backend_options = BACKEND_OPTIONS_NONE;
        if (options.m_use_derivatives)
            backend_options |= BACKEND_OPTIONS_ENABLE_DERIVATIVES;
        if (options.m_disable_pdf)
            backend_options |= BACKEND_OPTIONS_DISABLE_PDF;
        if (options.m_enable_aux)
            backend_options |= BACKEND_OPTIONS_ENABLE_AUX;
        if (options.m_warn_spectrum_conv)
            backend_options |= BACKEND_OPTIONS_WARN_SPECTRUM_CONVERSION;
        if (options.m_adapt_normal)
            backend_options |= BACKEND_OPTIONS_ADAPT_NORMAL;
        if (options.m_adapt_microfacet_roughness)
            backend_options |= BACKEND_OPTIONS_ADAPT_MICROFACET_ROUGHNESS;

        // Initialize the material compiler
        Material_backend_compiler mc(
            mdl_compiler,
            target_backend,
            /*num_texture_results=*/ options.m_num_texture_results,
            backend_options,
            /*df_handle_mode=*/ "none",
            /*handle_return_mode=*/ options.m_lambda_return_mode.empty() ?
                "default" : options.m_lambda_return_mode);

        bool success = true;

        // Add user defined paths
        for (std::size_t i = 0; i < options.m_mdl_paths.size(); ++i)
            mc.add_module_path(options.m_mdl_paths[i].c_str());

        Target_function_description desc;

        // Select some default expressions to generate code for
        // The functions to select depend on the renderer
        // To get started, generating 'surface.scattering' would be enough
        if (options.m_descs.empty())
        {
            auto& descs = options.m_descs;
            descs.push_back(TD("ior", "ior"));
            descs.push_back(TD("thin_walled", "thin_walled"));
            descs.push_back(TD("surface.scattering", "surface_scattering"));
            descs.push_back(TD("surface.emission.emission", "surface_emission_emission"));
            descs.push_back(TD("surface.emission.intensity", "surface_emission_intensity"));
            descs.push_back(TD("surface.emission.mode", "surface_emission_mode"));
            descs.push_back(TD("backface.scattering", "backface_scattering"));
            descs.push_back(TD("backface.emission.emission", "backface_emission_emission"));
            descs.push_back(TD("backface.emission.intensity", "backface_emission_intensity"));
            descs.push_back(TD("backface.emission.mode", "backface_emission_mode"));
            descs.push_back(TD("volume.absorption_coefficient", "volume_absorption_coefficient"));
            descs.push_back(TD("volume.scattering_coefficient", "volume_scattering_coefficient"));
            descs.push_back(TD("geometry.normal", "geometry_normal"));
            descs.push_back(TD("geometry.cutout_opacity", "geometry_cutout_opacity"));
            descs.push_back(TD("geometry.displacement", "geometry_displacement"));
        }

        // Use single init
        if (options.m_single_init)
            options.m_descs.insert(options.m_descs.begin(), TD("init", "init"));

        // Instantiation flags
        mi::Uint32 flags = 0;
        if (options.m_ignore_noinline)
            flags |= mi::mdl::IMaterial_instance::IGNORE_NOINLINE;

        // For class compilation only
        if (options.m_use_class_compilation)
        {
            if (options.m_fold_all_bool_parameters)
                flags |= mi::mdl::IMaterial_instance::NO_BOOL_PARAMS;
            if (options.m_fold_all_enum_parameters)
                flags |= mi::mdl::IMaterial_instance::NO_ENUM_PARAMS;
            if (options.m_fold_ternary_on_df)
                flags |= mi::mdl::IMaterial_instance::NO_TERNARY_ON_DF;
        }

        // Add MDL material distribution functions and expressions to the link unit
        success = mc.add_material(
            options.m_qualified_material_name,
            &options.m_descs[0],
            options.m_descs.size(),
            options.m_use_class_compilation,
            flags);

        if (!success)
        {
            // Print any compiler messages, if available
            mc.print_messages();
        }
        else
        {
            // Generate code for selected backend in the link unit
            std::vector<std::unique_ptr<Target_code> > target_code;
            target_code.push_back(std::unique_ptr<Target_code>(mc.generate_target_code()));

            // Dump generated code to file or stdout
            if (!options.m_output_file.empty())
            {
                std::ofstream file_stream;
                file_stream.open(options.m_output_file.c_str());
                if (file_stream)
                {
                    file_stream << target_code[0]->get_src_code();
                    if (options.m_dump_metadata)
                        dump_metadata(target_code[0].get(), file_stream);
                    file_stream.close();
                }
            }
            else
            {
                std::cout << "\n\n\n";
                std::cout << target_code[0]->get_src_code() << "\n\n\n";
                if (options.m_dump_metadata)
                    dump_metadata(target_code[0].get(), std::cout);
                std::cout << "\n\n\n";
            }

            // The target code can also be serialized for later reuse.
            // This can make sense to reduce startup time for larger scenes that have been loaded
            // before. A key for this kind of cache is the hash of a compiled material:
            const mi::mdl::DAG_hash* dag_hash = mc.get_material_instances()[0].get_material_instance()->get_hash();
            const unsigned char* hash = dag_hash->data();

            // Converting to 128-bit UUID representation
            mi::base::Uuid uuid;
            uuid.m_id1 = (hash[0] << 24) | (hash[1] << 16) | (hash[2] << 8) | hash[3];
            uuid.m_id2 = (hash[4] << 24) | (hash[5] << 16) | (hash[6] << 8) | hash[7];
            uuid.m_id3 = (hash[8] << 24) | (hash[9] << 16) | (hash[10] << 8) | hash[11];
            uuid.m_id4 = (hash[12] << 24) | (hash[13] << 16) | (hash[14] << 8) | hash[15];

            std::cout << "Compiled material hash: \n" << std::hex << uuid.m_id1 << " " << uuid.m_id2
                << " " << uuid.m_id3 << " " << uuid.m_id4 << std::dec << "\n";
        }
    }
}

int MAIN_UTF8(int argc, char* argv[])
{
    // Parse command line options
    Options options;
    if (!options.parse(argc, argv))
    {
        options.print_usage(std::cout);
        exit(EXIT_FAILURE);
    }
    
    // Print description of command line options
    if (options.m_help)
    {
        options.print_usage(std::cout);
        exit(EXIT_SUCCESS);
    }

    // Access the MDL core compiler
    mi::base::Handle<mi::mdl::IMDL> mdl_compiler(load_mdl_compiler());
    check_success(mdl_compiler);

    // Create a textured material
    code_generation(mdl_compiler.get(), options);

    // Free MDL core compiler before shutting down
    mdl_compiler = 0;

    // Unload MDL Core
    check_success(unload());

    keep_console_open();
    return EXIT_SUCCESS;
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8

// Print command line options description
void Options::print_usage(std::ostream& s)
{
    s << R"(
code_gen [options] <qualified_material_name>

options:

  -h|--help                     Print this usage message and exit.
  -p|--mdl_path <path>          Add the given path to the MDL search path.
  -o|--output <file>            Export the module to this file. Default: stdout
  -b|--backend <backend>        Select the back-end to generate code for. One of
                                {DAG, GLSL, HLSL, PTX, LLVM}. Default: HLSL
  -e|--expr_path <path>         Add an MDL expression path to generate, like \"surface.scattering\".
                                Defaults to a set of expression paths.
  -d|--derivatives              Generate code with derivative support.
  -i|--instance_compilation     Use instance compilation instead of class compilation.
  -t|--text_results <num>       Number of float4 texture result slots in the state. Default: 16
  -M|--dump-meta-data           Print all generated meta data tables.
  --ft                          Fold ternary operators when used on distribution functions.
  --fb                          Fold boolean parameters.
  --fe                          Fold enum parameters.
  --single-init                 Compile in single init mode.
  --dian                        Disable ignoring anno::noinline() annotations.
  --disable_pdf                 Disable generation of separate PDF function.
  --enable_aux                  Enable generation of auxiliary function.
  --lambda_return_mode <mode>   Set how base types and vector types are returned for PTX and LLVM
                                backends. One of {default, sret, value}.
  --adapt_normal                Enable renderer callback to adapt the normal.
  --adapt_microfacet_roughness  Enable renderer callback to adapt the roughness for
                                microfacet BSDFs.
  --experimental                Enable experimental compiler features (for internal testing).
  --warn-spectrum-conv          Warn if a spectrum constructor is converted into RGB.)
  --analyze                     Run backend analysis)";

    s << std::endl;
}

/// Parse command line options
bool Options::parse(int argc, char* argv[])
{
    m_mdl_paths.push_back(get_samples_mdl_root());

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg[0] == '-') {
            if (arg == "-h" || arg == "--help")
                m_help = true;
            else if (arg == "-d" || arg == "--derivatives")
                m_use_derivatives = true;
            else if (arg == "-i" || arg == "--instance_compilation")
                m_use_class_compilation = false;
            else if (arg == "--ft")
                m_fold_ternary_on_df = true;
            else if (arg == "--fb")
                m_fold_all_bool_parameters = true;
            else if (arg == "--fe")
                m_fold_all_enum_parameters = true;
            else if (arg == "--single-init")
                m_single_init = true;
            else if (arg == "--dian")
                m_ignore_noinline = false;
            else if (arg == "--disable_pdf")
                m_disable_pdf = true;
            else if (arg == "--enable_aux")
                m_enable_aux = true;
            else if (arg == "--lambda_return_mode") {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for --lambda_return_mode missing." << std::endl;
                    return false;
                }
                m_lambda_return_mode = argv[++i];
            }
            else if (arg == "--adapt_normal")
                m_adapt_normal = true;
            else if (arg == "--adapt_microfacet_roughness")
                m_adapt_microfacet_roughness = true;
            else if (arg == "--analyze")
                m_run_material_analysis = true;
            else if (arg == "--experimental")
                m_experimental = true;
            else if (arg == "--warn-spectrum-conv")
                m_warn_spectrum_conv = true;
            else if (arg == "-p" || arg == "--mdl_path")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -p|--mdl_path missing." << std::endl;
                    return false;
                }
                m_mdl_paths.push_back(argv[++i]);
            }
            else if (arg == "-o" || arg == "--output")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -o|--output missing." << std::endl;
                    return false;
                }
                m_output_file = argv[++i];
            }
            else if (arg == "-b" || arg == "--backend")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -b|--backend missing." << std::endl;
                    return false;
                }
                m_backend = argv[++i];
                std::transform(m_backend.begin(), m_backend.end(), m_backend.begin(),
                    [](unsigned char c) { return std::tolower(c); });
            }
            else if (arg == "-e" || arg == "--expr_path")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -e|--expr_path missing." << std::endl;
                    return false;
                }
                std::string expr = argv[++i];
                std::string func_name = expr;
                std::transform(func_name.begin(), func_name.end(), func_name.begin(),
                    [](unsigned char c) { return char(c) == '.' ? '_' : c; });
                m_desc_strs.push_back(std::unique_ptr<std::string>(new std::string(expr)));
                m_desc_strs.push_back(std::unique_ptr<std::string>(new std::string(func_name)));
                m_descs.push_back(TD(
                    m_desc_strs[m_desc_strs.size() - 2].get()->c_str(),
                    m_desc_strs.back().get()->c_str()));
            }
            else if (arg == "-t" || arg == "--text_results")
            {
                if (i == argc - 1)
                {
                    std::cerr << "error: Argument for -t|--text_results missing." << std::endl;
                    return false;
                }
                m_num_texture_results = std::max(atoi(argv[++i]), 0);
            }
            else if (arg == "-M" || arg == "--dump_meta_data")
                m_dump_metadata = true;
            else
            {
                std::cerr << "error: Unknown option \"" << arg << "\"." << std::endl;
                return false;
            }
        }
        else
        {
            if (i != argc - 1)
            {
                std::cerr << "error: Unexpected argument \"" << arg << "\"." << std::endl;
                return false;
            }
            m_qualified_material_name = arg;
        }
    }

    if (!m_help)
    {
        if (m_qualified_material_name.empty())
        {
            std::cerr << "error: Qualified material name missing." << std::endl;
            return false;
        }

        if (m_backend != "dag" && m_backend != "glsl" && m_backend != "hlsl" &&
            m_backend != "ptx" && m_backend != "llvm")
        {
            std::cerr << "error: Back-end is missing or invalid." << std::endl;
            return false;
        }
    }
    return true;
}