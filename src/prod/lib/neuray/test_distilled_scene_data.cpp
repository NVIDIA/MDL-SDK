/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

/** \file
 ** \brief Unit test for MDL-1780: referenced scene data after distilling.
 **/


#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <cstring>

#include <mi/base/handle.h>
#include <mi/neuraylib/icompiled_material.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/imaterial_instance.h>
#include <mi/neuraylib/imdl_distiller_api.h>
#include <mi/neuraylib/imdl_execution_context.h>
#include <mi/neuraylib/imdl_factory.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/iplugin_configuration.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/itransaction.h>

#include "test_shared.h"

#define DIR_PREFIX "output_test_distilled_scene_data"

#include "test_shared_mdl.h"

namespace {

const char* const TEST_MODULE = "test_mdl_distilled_scene_data";
const char* const MATERIAL_DEFINITION =
    "mdl::test_mdl_distilled_scene_data::md_geomprop()";
const char* const MATERIAL_INSTANCE =
    "mdl::test_mdl_distilled_scene_data::mi_geomprop";
const char* const SCENE_DATA_NAME = "CUSTOM_GEOMPROP";

void check_referenced_scene_data(
    const mi::neuraylib::ICompiled_material* cm, const char* expected_name)
{
    MI_CHECK( cm);
    MI_CHECK_EQUAL( 1, cm->get_referenced_scene_data_count());
    MI_CHECK_EQUAL_CSTR( expected_name, cm->get_referenced_scene_data_name( 0));
}

void check_distilled_referenced_scene_data(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::INeuray* neuray,
    bool class_compilation,
    const char* distiller_target)
{
    mi::base::Handle<mi::neuraylib::IMdl_distiller_api> mdl_distiller_api(
        neuray->get_api_component<mi::neuraylib::IMdl_distiller_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    mi::base::Handle<const mi::neuraylib::IMaterial_instance> mi(
        transaction->access<mi::neuraylib::IMaterial_instance>( MATERIAL_INSTANCE));
    MI_CHECK( mi);

    mi::base::Handle<const mi::neuraylib::ICompiled_material> cm(
        mi->create_compiled_material( flags, context.get()));
    MI_CHECK_CTX( context);
    MI_CHECK( cm);
    check_referenced_scene_data( cm.get(), SCENE_DATA_NAME);

    mi::Sint32 errors = -1;
    mi::base::Handle<const mi::neuraylib::ICompiled_material> distilled_cm(
        mdl_distiller_api->distill_material(
            cm.get(), distiller_target, /*distiller_options*/ nullptr, &errors));
    MI_CHECK_EQUAL( 0, errors);
    MI_CHECK( distilled_cm);
    check_referenced_scene_data( distilled_cm.get(), SCENE_DATA_NAME);
}

} // namespace

MI_TEST_AUTO_FUNCTION( test_distilled_scene_data )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray);

    {
        std::string path = MI::TEST::mi_src_path( "prod/lib/neuray");
        set_mdl_paths( neuray.get(), {path});

        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_configuration(
            neuray->get_api_component<mi::neuraylib::IPlugin_configuration>());
        MI_CHECK_EQUAL(
            0, plugin_configuration->load_plugin_library( plugin_path_mdl_distiller));

        MI_CHECK_EQUAL( 0, neuray->start());

        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());

        install_external_resolver( neuray.get());

        std::string module_name = std::string( "::") + TEST_MODULE;
        import_mdl_module( transaction.get(), neuray.get(), module_name.c_str(), 0);
        do_create_function_call( transaction.get(), MATERIAL_DEFINITION, MATERIAL_INSTANCE);

        const char* distiller_targets[] = { "diffuse", "transmissive_pbr" };
        for( const char* target : distiller_targets) {
            check_distilled_referenced_scene_data(
                transaction.get(), neuray.get(), /*class_compilation=*/false, target);
            check_distilled_referenced_scene_data(
                transaction.get(), neuray.get(), /*class_compilation=*/true, target);
        }

        uninstall_external_resolver( neuray.get());
        MI_CHECK_EQUAL( 0, transaction->commit());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());

    neuray.reset();
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

