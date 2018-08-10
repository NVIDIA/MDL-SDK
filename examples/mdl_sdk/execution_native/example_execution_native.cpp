/******************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

// examples/example_execution_native.cpp
//
// Introduces the execution of generated code for compiled materials for
// the native (CPU) backend and shows how to manually bake a material
// sub-expression to a texture.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <mi/mdl_sdk.h>

#include "example_shared.h"


// Creates an instance of "mdl::example_execution::execution_material".
void create_material_instance(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_compiler* mdl_compiler,
    const char* instance_name)
{
    // Load the "example_execution" module.
    check_success(mdl_compiler->load_module(transaction, "::nvidia::sdk_examples::tutorials") >= 0);

    // Create a material instance from the material definition
    // "mdl::example_execution::execution_material" with the default arguments.
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        transaction->access<mi::neuraylib::IMaterial_definition>(
            "mdl::nvidia::sdk_examples::tutorials::example_execution1"));
    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
        material_definition->create_material_instance(0, &result));
    check_success(result == 0);
    transaction->store(material_instance.get(), instance_name);
}

// Compiles the given material instance in the given compilation modes and stores it in the DB.
void compile_material_instance(
    mi::neuraylib::ITransaction* transaction,
    const char* instance_name,
    const char* compiled_material_name,
    bool class_compilation)
{
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance(
       transaction->access<mi::neuraylib::IMaterial_instance>(instance_name));
    mi::Sint32 result = 0;
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance->create_compiled_material(flags, 1.0f, 380.0f, 780.0f, &result));
    check_success(result == 0);

    transaction->store(compiled_material.get(), compiled_material_name);
}

// Generate and execute native CPU code for a subexpression of a given compiled material.
mi::neuraylib::ITarget_code const *generate_native(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_compiler* mdl_compiler,
    const char* compiled_material_name,
    const char* path,
    const char* fname)
{
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        transaction->edit<mi::neuraylib::ICompiled_material>(compiled_material_name));

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_native(
        mdl_compiler->get_backend(mi::neuraylib::IMdl_compiler::MB_NATIVE));
    check_success(be_native->set_option("num_texture_spaces", "1") == 0);

    // Generate the native code
    mi::Sint32 result = -1;
    mi::base::Handle<const mi::neuraylib::ITarget_code> code_native(
        be_native->translate_material_expression(
            transaction, compiled_material.get(), path, fname, &result));
    check_success(result == 0);
    check_success(code_native);

    code_native->retain();
    return code_native.get();
}

// Bake the material expression created with the native backend into a canvas with the given
// resolution.
mi::neuraylib::ICanvas *bake_expression_native(
    mi::neuraylib::IImage_api         *image_api,
    mi::neuraylib::ITarget_code const *code_native,
    mi::Uint32                         width,
    mi::Uint32                         height)
{
    // Create a canvas (with only one tile)
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        image_api->create_canvas("Rgb_fp", width, height));

    // Setup MDL material state (with only one texture space)
    mi::Float32_3_struct   texture_coords[1]    = { { 0.0f, 0.0f, 0.0f } };
    mi::Float32_3_struct   texture_tangent_u[1] = { { 1.0f, 0.0f, 0.0f } };
    mi::Float32_3_struct   texture_tangent_v[1] = { { 0.0f, 1.0f, 0.0f } };
    mi::Float32_4_4 identity(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
    mi::neuraylib::Shading_state_material mdl_state = {
        /*normal=*/           { 0.0f, 0.0f, 1.0f },
        /*geom_normal=*/      { 0.0f, 0.0f, 1.0f },
        /*position=*/         { 0.0f, 0.0f, 0.0f },
        /*animation_time=*/   0.0f,
        /*texture_coords=*/   texture_coords,
        /*tangent_u=*/        texture_tangent_u,
        /*tangent_v=*/        texture_tangent_v,
        /*text_results=*/     NULL,
        /*ro_data_segment=*/  NULL,
        /*world_to_object=*/  &identity[0],
        /*object_to_world=*/  &identity[0],
        /*object_id=*/        0
    };

    // Provide a large enough buffer for any result type.
    // In this case, we know, we will get a color which is a float3, so this is overkill.
    union
    {
        int                     int_val;
        float                   float_val;
        double                  double_val;
        mi::Float32_3_struct    float3_val;
        mi::Float32_4_struct    float4_val;
        mi::Float32_4_4_struct  float4x4_val;
        mi::Float64_3_struct    double3_val;
        mi::Float64_4_struct    double4_val;
        mi::Float64_4_4_struct  double4x4_val;
    } execute_result = { 0 };

    // Calculate all expression values for a 2x2 quad around the center of the world
    // and write them to the canvas.
    mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
    mi::Float32_3_struct *data = static_cast<mi::Float32_3_struct *>(tile->get_data());
    for (mi::Uint32 y = 0; y < height; ++y)
    {
        for (mi::Uint32 x = 0; x < width; ++x)
        {
            // Update state for the current pixel
            float rel_x = float(x) / float(width);
            float rel_y = float(y) / float(height);
            mdl_state.position.x = 2.0f * rel_x - 1;  // [-1, 1)
            mdl_state.position.y = 2.0f * rel_y - 1;  // [-1, 1)
            texture_coords[0].x  = rel_x;             // [0, 1)
            texture_coords[0].y  = rel_y;             // [0, 1)

            // Evaluate sub-expression
            check_success(code_native->execute(0, mdl_state, NULL, &execute_result) == 0);

            // Apply gamma correction
            execute_result.float3_val.x = powf(execute_result.float3_val.x, 1.f / 2.2f);
            execute_result.float3_val.y = powf(execute_result.float3_val.y, 1.f / 2.2f);
            execute_result.float3_val.z = powf(execute_result.float3_val.z, 1.f / 2.2f);

            // Store result in texture
            data[y * width + x] = execute_result.float3_val;
        }
    }

    canvas->retain();
    return canvas.get();
}

int main(int /*argc*/, char* /*argv*/[])
{
    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(load_and_get_ineuray());
    check_success(neuray.is_valid_interface());

    // Configure the MDL SDK
    configure(neuray.get());

    // Start the MDL SDK
    mi::Sint32 result = neuray->start();
    check_start_success(result);

    {
        // Create a transaction
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        { 
            // Access the MDL SDK compiler component
            mi::base::Handle<mi::neuraylib::IMdl_compiler> mdl_compiler(
                neuray->get_api_component<mi::neuraylib::IMdl_compiler>());

            // Load the "example_execution" module and create a material instance
            std::string instance_name = "instance of execution_material";
            create_material_instance(transaction.get(), mdl_compiler.get(), instance_name.c_str());

            // Compile the material instance in instance compilation mode
            std::string instance_compilation_name
                = std::string("instance compilation of ") + instance_name;
            compile_material_instance(
                transaction.get(), instance_name.c_str(),
                instance_compilation_name.c_str(), false);


            // Generate target code for some material expression
            mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
                generate_native(
                    transaction.get(), mdl_compiler.get(), instance_compilation_name.c_str(),
                    "surface.scattering.tint", "tint"));

            // Acquire image api needed to create a canvas for baking
            mi::base::Handle<mi::neuraylib::IImage_api> image_api(
                neuray->get_api_component<mi::neuraylib::IImage_api>());

            // Bake the expression into a 700x520 canvas
            mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                bake_expression_native(
                    image_api.get(), target_code.get(), 700, 520));

            // Export the canvas to an image on disk
            mdl_compiler->export_canvas("example_native.png", canvas.get());
        }

        transaction->commit();
    }

    // Shut down the MDL SDK
    check_success(neuray->shutdown() == 0);
    neuray = 0;

    // Unload the MDL SDK
    check_success(unload());

    keep_console_open();
    return EXIT_SUCCESS;
}
