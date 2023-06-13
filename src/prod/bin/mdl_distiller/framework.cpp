/******************************************************************************
 * Copyright (c) 2017-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \file framework.cpp

#include <mi/mdl_sdk.h>

#include <fstream>
#include <iostream>

#include <set>
#include <iterator>
#include <algorithm>

#include "framework.h"

// Helper classes
#include "mdl_assert.h"
#include "mdl_distiller_utils.h"

// MDL projection classes
#include "options.h"
#include "user_timer.h"
#include "mdl_printer.h"

using mi::base::Handle;
using mi::neuraylib::INeuray;
using mi::neuraylib::ITransaction;
using mi::neuraylib::ICompiled_material;
using mi::neuraylib::IExpression;
using mi::neuraylib::IExpression_constant;
using mi::neuraylib::IExpression_direct_call;
using mi::neuraylib::IExpression_list;
using mi::neuraylib::IExpression_parameter;
using mi::neuraylib::IExpression_temporary;
using mi::neuraylib::IValue;
using mi::neuraylib::IMessage;
using mi::neuraylib::IValue_bool;
using mi::neuraylib::IMdl_factory;
using mi::neuraylib::IMdl_impexp_api;

// Constructs internal MDL expression representation from compiled material.
Mdl_projection_framework::Mdl_projection_framework()
{
}

/// Return a unique ID, just an incrementing int in string form.
std::string uniq_id() {
    static int s_uniq_id = 1;
    std::stringstream s;
    s << 'n' << s_uniq_id++;
    return s.str();
}

/// Test function that compiles the newly created material from the string
/// representation of the MDL module of the simplified MDL material.
/// Creates a default instance of the material under the name 'instance_name'
/// and compiles it under the name 'compiled_instance_name'
int test_module( IMdl_impexp_api* mdl_impexp_api,
                 ITransaction* transaction,
                 IMdl_factory* mdl_factory,
                 const char* module_src,              // the module src
                 const char* material_name,           // what to instantiate
                 const char* compiled_instance_name,  // the result in the DB
                 Options* options)
{
    std::string module_mdl_name = std::string("::test_") + uniq_id();
    std::string module_db_name = std::string("mdl") + module_mdl_name;
    std::string instance_name = module_mdl_name + "_instance";
    {
        // Load module and create a material instance
        std::string material_db_name = std::string("mdl") + module_mdl_name + "::" + material_name;

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        // Load module.
        int result = mdl_impexp_api->load_module_from_string( transaction, module_mdl_name.c_str(),
                                                              module_src, context.get());
        if ( result) {
            std::cerr << module_src << "\n";
            std::cerr << "Error: material '" << module_mdl_name << "::" << material_name
                      << "' failed to compile with code " << result << "\n";
            mi::Size err_cnt = context->get_error_messages_count();
            for (mi::Size idx = 0; idx < err_cnt; idx++) {
                Handle<const IMessage> msg(context->get_error_message(idx));
                std::cerr << "  " << msg->get_string() << "\n";
            }
            return 1;
        }

        // Access module
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<const mi::neuraylib::IModule>(module_db_name.c_str()));
        if (!module) {
            std::cerr << "ERROR: failed to access module '" << module_mdl_name.c_str() << "'\n";
            return -1;
        }

        // Use overload resolution to find the exact signature.
        mi::base::Handle<const mi::IArray> overloads(
            module->get_function_overloads(material_db_name.c_str()));
        if (!overloads || overloads->get_length() != 1) {
            std::cerr << "ERROR: failed to find signature for material '" << material_db_name
                      << "'\n";
            return -1;
        }
        mi::base::Handle<const mi::IString> overload(overloads->get_element<mi::IString>(0));
        material_db_name = overload->get_c_str();

        // create instance with the default arguments.
        Handle<const mi::neuraylib::IFunction_definition> material_definition(
            transaction->access<mi::neuraylib::IFunction_definition>( material_db_name.c_str()));
        if (! material_definition.is_valid_interface()) {
            std::cerr << "ERROR: Material '" << material_name << "' does not exist.\n";
            return 1;
        }
        Handle<mi::neuraylib::IFunction_call> material_instance(
            material_definition->create_function_call( 0, &result));
        switch ( result) {
        case -3:
            std::cerr << "ERROR: Material '" << material_name << "' cannot be default initialized.\n";
            return 1;
        case -4:
            std::cerr << "ERROR: Material '" << material_name << "' is not exported.\n";
            return 1;
        }
        transaction->store( material_instance.get(), instance_name.c_str());
    }
    {
        // Compile the material instance in instance compilation mode
        Handle<const mi::neuraylib::IFunction_call> material_instance(
            transaction->access<mi::neuraylib::IFunction_call>( instance_name.c_str()));
        mi::Uint32 flags = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        if (options->target_material_model_mode) {
            context->set_option("target_material_model_mode", true);
        }

        Handle<const mi::neuraylib::IMaterial_instance> material_instance2(
            material_instance->get_interface<mi::neuraylib::IMaterial_instance>());
        Handle<ICompiled_material> compiled_material(
            material_instance2->create_compiled_material( flags, context.get()));
        if (context->get_error_messages_count() != 0) {
            std::cerr << "ERROR: Instance compilation failed when testing material '"
                      << material_name << "'.\n";
            return 1;
        }
        transaction->store( compiled_material.get(), compiled_instance_name);
    }
    return 0;
}

///
///
void export_normal_map_canvas(
    const char* filename,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::ICanvas* canvas
    )
{
    // Convert normal values from the interval [-1.0,1.0] to [0.0, 1.0]
    mi::base::Handle<mi::neuraylib::ITile> tile (canvas->get_tile());
    mi::Float32* data = static_cast<mi::Float32*>(tile->get_data());

    const mi::Uint32 n = canvas->get_resolution_x() * canvas->get_resolution_y() * 3;
    for(mi::Uint32 i=0; i<n; ++i) {
        data[i] = (data[i] + 1.f) * 0.5f;
    }

    // Export canvas
    mdl_impexp_api->export_canvas(filename, canvas);
}

/// Set an option string-value pair in an IMap.
void set_option( Handle<mi::IMap> options,
                 Handle<mi::neuraylib::IFactory> factory,
                 const char* name, const char* value) {
    Handle<mi::IString> istring(factory->create<mi::IString>());
    istring->set_c_str( value);
    options->insert(name, istring.get());
}

void set_option( Handle<mi::IMap> options,
                 Handle<mi::neuraylib::IFactory> factory,
                 const char* name, std::string value) {
    set_option( options, factory, name, value.c_str());
}

/// Set an option string-value pair in an IMap.
void set_option( Handle<mi::IMap> options,
                 Handle<mi::neuraylib::IFactory> factory,
                 const char* name, bool value) {
    Handle<mi::IBoolean> ival(factory->create<mi::IBoolean>());
    ival->set_value( value);
    options->insert(name, ival.get());
}

/// Set an option string-value pair in an IMap.
void set_option( Handle<mi::IMap> options,
                 Handle<mi::neuraylib::IFactory> factory,
                 const char* name, int value) {
    Handle<mi::ISint32> ival(factory->create<mi::ISint32>());
    ival->set_value( value);
    options->insert(name, ival.get());
}

/// Main function to project an MDL material.  Uses an instance or
/// class compiled material in an ICompiled_material as input,
/// converts it to an MDL expression, applies selected rule sets,
/// optionally bakes function call graphs into textures, and returns
/// the result as a new ICompiled_material. Writes the resulting MDL
/// modules to 'out' if non-null and writes generated textures to
/// files.  It also checks whether the generated modules compile.
///
/// May require adaption to particular use cases.
///
/// \return 0 in case of a failure or if options->test_module is set
///           to false. Otherwise it returns the new ICompiled_material
///           with a reference count of 1.
///
const ICompiled_material* mdl_distill( INeuray* neuray,
                                       IMdl_impexp_api* mdl_impexp_api,
                                       ITransaction* transaction,
                                       const ICompiled_material* compiled_material,
                                       const char* material_name,
                                       const char* target,
                                       Options* options,
                                       double add_to_total_time,
                                       std::ostream* out)
{
    // Return value
    const ICompiled_material* result_material = 0;

    // collect all paths we want to bake, or leave empty if options->all_textures is true
    Bake_paths bake_paths;

    User_timer total_time;
    total_time.start();
    User_timer project_time;
    project_time.start();

    // Distiller
    Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());

    Handle<IMdl_factory> mdl_factory(
        neuray->get_api_component<IMdl_factory>());

    Handle<mi::neuraylib::IFactory> factory(
        neuray->get_api_component<mi::neuraylib::IFactory>());
    Handle<mi::IMap> distiller_options( factory->create<mi::IMap>("Map<Interface>"));

    set_option( distiller_options, factory, "layer_normal", options->layer_normal);
    set_option( distiller_options, factory, "top_layer_weight", options->top_layer_weight);
    set_option( distiller_options, factory, "merge_metal_and_base_color",
                options->merge_metal_and_base_color);
    set_option( distiller_options, factory, "merge_transmission_and_base_color",
                options->merge_transmission_and_base_color);
    set_option( distiller_options, factory, "target_material_model_mode",
                options->target_material_model_mode);
    set_option( distiller_options, factory, "_dbg_quiet", options->quiet);
    set_option( distiller_options, factory, "_dbg_verbosity", options->verbosity);
    set_option( distiller_options, factory, "_dbg_trace", options->trace);

    mi::Sint32 result = 0;
    Handle<const ICompiled_material> distilled_material(
        distiller_api->distill_material( compiled_material,
                                         target,
                                         distiller_options.get(),
                                         &result));
    if ((result != 0) || ! distilled_material.is_valid_interface()) {
        std::cerr << "ERROR: Distilling failed with error code " << result << " for material '"
                  << material_name << "'.\n";
        return 0;
    }
    project_time.stop();

    std::string new_material_name( derived_material_name(material_name));
    Mdl_projection_framework framework;

    mdl_spec required_mdl_version = options->export_spec;
    std::stringstream new_material;

    {
        Mdl_printer mdl_printer(neuray, new_material, options, target, material_name,
                                new_material_name, distilled_material.get(),
                                NULL, // bake_paths
                                false, // do_bake
                                transaction);
        mdl_printer.set_emit_messages(false);
        mdl_printer.analyze_material();
        if (mdl_printer.get_required_mdl_version() > required_mdl_version)
            required_mdl_version = mdl_printer.get_required_mdl_version();
    }

    if (options->verbosity > 3) {
        std::cerr << "Info: MDL auto export specification version resolves to ";
        switch (required_mdl_version) {
        case mdl_spec_1_3: std::cerr << "1.3.\n"; break;
        case mdl_spec_1_6: std::cerr << "1.6.\n"; break;
        case mdl_spec_1_7: std::cerr << "1.7.\n"; break;
        case mdl_spec_1_8: std::cerr << "1.8.\n"; break;
        default: std::cerr << "<UNKNOWN>.\n";  break; // cannot happen
        }
    }

    if ( ! options->all_textures) {
        // TODO: Texture baking paths are hard coded for two targets and not yet
        //       customizable for new distilling targets in plugins
        if ( 0 == strcmp( target, "diffuse")) {
            bake_paths.push_back( Bake_path( "color", "surface.scattering.tint"));
        }
    }

    // Write module to a stringstream for re-testing with the compiler
    if (options->verbosity > 3)
        std::cerr << "Info: write MDL module.\n";

    std::stringstream new_material_baked;
    {
        Mdl_printer mdl_printer(neuray, new_material, options, target, material_name,
                                new_material_name, distilled_material.get(),
                                NULL, // bake_paths
                                false, // do_bake
                                transaction);
        mdl_printer.set_emit_messages(false);
        mdl_printer.print_module();
    }
    User_timer recompile_time;
    std::string compiled_instance_name = std::string("compiled_test_instance_") + uniq_id();
    if ( options->test_module || options->bake) {
        if (options->verbosity > 3)
            std::cerr << "Info: test MDL module with compiler\n";
        recompile_time.start();

        if ( 0 != test_module( mdl_impexp_api, transaction, mdl_factory.get(),
                               new_material.str().c_str(),
                               new_material_name.c_str(),
                               compiled_instance_name.c_str(),
                               options)) {
            if (options->verbosity > 2) {
                std::cout << new_material.str() << "\n";
                std::cerr << "Error: the (above) module on stdout did not compile!\n";
            }
            return 0;
        }
        if ( ! options->bake)
            result_material = transaction->access<ICompiled_material>(
                compiled_instance_name.c_str());
        recompile_time.stop();
    }

    User_timer bake_time;
    User_timer baked_export_time;

    if ( options->bake) {
        if (options->verbosity > 3)
            std::cerr << "Info: bake textures\n";
        {
            Mdl_printer mdl_printer(neuray, new_material_baked, options,
                                    target, material_name, new_material_name,
                                    distilled_material.get(),
                                    &bake_paths,
                                    true, // do_bake
                                    transaction);
            mdl_printer.print_module();
        }

        for ( unsigned i = 0; i < bake_paths.size(); ++i) {
            if (options->verbosity > 3)
                std::cerr << "Bake texture path '" << bake_paths[i].path << "\'\n";

            // Handle<mi::neuraylib::IMdl_distiller_api> distiller_api(
            //     neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
            bake_time.start();
            Handle<const mi::neuraylib::IBaker> baker(
                distiller_api->create_baker(
                    compiled_material, bake_paths[i].path.c_str()));

            if ( baker) {
                Handle<mi::neuraylib::IImage_api> image_api(
                    neuray->get_api_component<mi::neuraylib::IImage_api>());
                // Or use baker->get_pixel_type().
                std::string pixel_type =
                    bake_paths[i].type == "float" ? "Float32" :
                    bake_paths[i].type == "float3" ? "Float32<3>" : "Rgb_fp";
                Handle<mi::neuraylib::ICanvas> tex_data( image_api->create_canvas(
                                                             pixel_type.c_str(), options->bake, options->bake));

                int bake_res = baker->bake_texture( tex_data.get());
                bake_time.stop();

                if (bake_res) {
                    std::cerr << "Error: bake_texture returned: " << bake_res
                              << "\n";
                }

                baked_export_time.start();
                std::string filename = options->texture_dir + "/" +
                    baked_texture_file_name( new_material_name.c_str(), bake_paths[i].path);
                if(is_normal_map_path(bake_paths[i].path))
                    export_normal_map_canvas(filename.c_str(), mdl_impexp_api, tex_data.get());
                else
                    mdl_impexp_api->export_canvas( filename.c_str(), tex_data.get());
                baked_export_time.stop();
            } else {
                std::cerr << "Error: unable to create baker for path " 
                          << bake_paths[i].path << "\n";
                bake_time.stop();
            }
        }

        if ( options->test_module) {
            if (options->verbosity > 3)
                std::cerr << "Info: test baked MDL module with compiler\n";
            std::string compiled_instance_name =
                std::string("baked_compiled_test_instance_") + uniq_id();
            if ( 0 != test_module( mdl_impexp_api, transaction, mdl_factory.get(),
                                   new_material_baked.str().c_str(),
                                   new_material_name.c_str(),
                                   compiled_instance_name.c_str(),
                                   options))
                return 0;
            result_material = transaction->access<ICompiled_material>(
                compiled_instance_name.c_str());
        }
    }

    total_time.stop();
    if (options->verbosity > 3)
        std::cerr << "Info: write MDL module to stdout.\n";
    if ( out) {
        (*out) << "// Generated by mdl_distiller\n"
               << "// target: " << target << '\n'
               << "// total_time:     " << (total_time.time()*1000+add_to_total_time) << " ms\n"
               << "// project_time:   " << (project_time.time() * 1000) << " ms\n";
        if ( options->test_module || options->bake)
            (*out) << "// recompile_time: " << (recompile_time.time() * 1000) << " ms\n";
        if ( options->bake) {
            (*out) << "// bake_time:      " << (bake_time.time() * 1000) << " ms\n";
            (*out) << "// baked_export_time: " << (baked_export_time.time() * 1000) << " ms\n";
        }
        if ( options->bake) {
            std::cerr << new_material_baked.str();
            (*out) << new_material_baked.str();
        } else {
            {
                Mdl_printer mdl_printer(neuray, *out, options, target, material_name,
                                        new_material_name, distilled_material.get(),
                                        NULL, // bake_paths
                                        false, // do_bake
                                        transaction);
                if ( options->outline) {
                    mdl_printer.set_outline_mode(true);
                }
//                    mdl_printer.set_suppress_default_parameters(true);
                mdl_printer.print_module();
            }
        }
        (*out) << "// end of generated material\n";
    }
    return result_material;
}
