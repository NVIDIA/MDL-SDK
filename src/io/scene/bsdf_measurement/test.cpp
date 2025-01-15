/******************************************************************************
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for io/scene/bsdf_measurement"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/handle.h>
#include <mi/neuraylib/bsdf_isotropic_data.h>

#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/lib/config/config.h>
#include <base/lib/mem/mem.h>
#include <base/lib/path/i_path.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_database.h>
#include <base/data/db/i_db_transaction.h>

#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/mdl_elements/test_shared.h>

using namespace MI;

void check_bsdf_data(
    const mi::neuraylib::IBsdf_isotropic_data* bsdf_data,
    mi::neuraylib::Bsdf_type type,
    mi::Uint32 resolution_theta,
    mi::Uint32 resolution_phi)
{
    MI_CHECK_EQUAL( resolution_theta, bsdf_data->get_resolution_theta());
    MI_CHECK_EQUAL( resolution_phi, bsdf_data->get_resolution_phi());
    MI_CHECK_EQUAL( type, bsdf_data->get_type());

    mi::Size size = resolution_theta * resolution_theta * resolution_phi;
    if( type == mi::neuraylib::BSDF_RGB)
        size *= 3;
    mi::base::Handle<const mi::neuraylib::IBsdf_buffer> bsdf_buffer( bsdf_data->get_bsdf_buffer());
    const mi::Float32* data = bsdf_buffer->get_data();
    for( mi::Size i = 0; i < size; ++i)
        MI_CHECK_EQUAL( static_cast<mi::Float32>( i), data[i]);
}

void check_bsdf_measurement(
    DB::Transaction* transaction,
    DB::Access<BSDFM::Bsdf_measurement>& bsdf_measurement,
    bool has_reflection,
    mi::neuraylib::Bsdf_type reflection_type,
    mi::Uint32 reflection_resolution_theta,
    mi::Uint32 reflection_resolution_phi,
    bool has_transmission,
    mi::neuraylib::Bsdf_type transmission_type,
    mi::Uint32 transmission_resolution_theta,
    mi::Uint32 transmission_resolution_phi)
{
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> reflection(
        bsdf_measurement->get_reflection<mi::neuraylib::IBsdf_isotropic_data>( transaction));
    MI_CHECK( !has_reflection || reflection.is_valid_interface());
    if( has_reflection)
        check_bsdf_data( reflection.get(),
            reflection_type, reflection_resolution_theta, reflection_resolution_phi);
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> transmission(
        bsdf_measurement->get_transmission<mi::neuraylib::IBsdf_isotropic_data>( transaction));
    MI_CHECK( !has_transmission || transmission.is_valid_interface());
    if( has_transmission)
        check_bsdf_data( transmission.get(),
            transmission_type, transmission_resolution_theta, transmission_resolution_phi);
}

void import_check_export_reimport_check(
    DB::Transaction* transaction,
    const char* filename,
    bool has_reflection,
    mi::neuraylib::Bsdf_type reflection_type,
    mi::Uint32 reflection_resolution_theta,
    mi::Uint32 reflection_resolution_phi,
    bool has_transmission,
    mi::neuraylib::Bsdf_type transmission_type,
    mi::Uint32 transmission_resolution_theta,
    mi::Uint32 transmission_resolution_phi)
{
    DB::Tag tag;

    {
        // Import the BSDF measurement as DB element from file
        auto* bsdf_measurement = new BSDFM::Bsdf_measurement();
        std::string path = TEST::mi_src_path( "io/scene/bsdf_measurement/") + filename;
        mi::Sint32 result = bsdf_measurement->reset_file( transaction, path);
        MI_CHECK_EQUAL( 0, result);
        bsdf_measurement->dump();
        tag = transaction->store_for_reference_counting( bsdf_measurement);
    }

    {
        // Access and check
        DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement( tag, transaction);
        MI_CHECK( !bsdf_measurement->get_filename().empty());
        check_bsdf_measurement( transaction, bsdf_measurement,
            has_reflection, reflection_type, reflection_resolution_theta, reflection_resolution_phi,
            has_transmission, transmission_type, transmission_resolution_theta, transmission_resolution_phi);

        // Export the BSDF data
        mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> reflection(
            bsdf_measurement->get_reflection<mi::neuraylib::IBsdf_isotropic_data>( transaction));
        mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> transmission(
            bsdf_measurement->get_transmission<mi::neuraylib::IBsdf_isotropic_data>( transaction));
        bool result = BSDFM::export_to_file( reflection.get(), transmission.get(), filename);
        MI_CHECK( result);
    }

    {
        // Re-import the exported BSDF measurement as DB element
        auto* bsdf_measurement = new BSDFM::Bsdf_measurement();
        mi::Sint32 result = bsdf_measurement->reset_file( transaction, filename);
        MI_CHECK_EQUAL( 0, result);
        tag = transaction->store_for_reference_counting( bsdf_measurement);
    }

    {
        // Access and check
        DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement( tag, transaction);
        MI_CHECK( !bsdf_measurement->get_filename().empty());
        check_bsdf_measurement( transaction, bsdf_measurement,
            has_reflection, reflection_type, reflection_resolution_theta, reflection_resolution_phi,
            has_transmission, transmission_type, transmission_resolution_theta, transmission_resolution_phi);
    }

    {
        // Import the BSDF measurement as DB element via a reader
        auto* bsdf_measurement = new BSDFM::Bsdf_measurement();
        std::string path = TEST::mi_src_path( "io/scene/bsdf_measurement/") + filename;
        DISK::File_reader_impl reader;
        MI_CHECK( reader.open( path.c_str()));
        mi::Sint32 result = bsdf_measurement->reset_reader( transaction, &reader);
        bsdf_measurement->dump();
        MI_CHECK_EQUAL( 0, result);
        tag = transaction->store_for_reference_counting( bsdf_measurement);
    }

    {
        // Access and check
        DB::Access<BSDFM::Bsdf_measurement> bsdf_measurement( tag, transaction);
        MI_CHECK( bsdf_measurement->get_filename().empty());
        check_bsdf_measurement( transaction, bsdf_measurement,
            has_reflection, reflection_type, reflection_resolution_theta, reflection_resolution_phi,
            has_transmission, transmission_type, transmission_resolution_theta, transmission_resolution_phi);
    }
}

DB::Tag get_impl_tag( DB::Transaction* transaction, DB::Tag proxy_tag)
{
    DB::Access<BSDFM::Bsdf_measurement> proxy( proxy_tag, transaction);
    return proxy->get_impl_tag();

}

void check_sharing( DB::Transaction* transaction, const char* bsdfm_filename)
{
    std::string filename = TEST::mi_src_path( "io/scene/bsdf_measurement/") + bsdfm_filename;

    DISK::File_reader_impl reader;
    MI_CHECK( reader.open( filename.c_str()));

    mi::base::Uuid invalid_hash{0,0,0,0};
    mi::base::Uuid some_hash1{1,1,1,1};
    mi::base::Uuid some_hash2{2,2,2,2};
    mi::Sint32 result;

    // Load twice with invalid hash (proxy not shared)
    DB::Tag tag1_proxy_invalid_notshared = BSDFM::load_mdl_bsdf_measurement(
        transaction,
        &reader,
        /*dummies for all paths*/ "",
        "",
        "",
        "",
        invalid_hash,
        /*shared_proxy*/ false,
        result);
    DB::Tag tag1_impl_invalid_notshared = get_impl_tag( transaction, tag1_proxy_invalid_notshared);
    reader.rewind();

    DB::Tag tag2_proxy_invalid_notshared = BSDFM::load_mdl_bsdf_measurement(
        transaction,
        &reader,
        /*dummies for all paths*/ "",
        "",
        "",
        "",
        invalid_hash,
        /*shared_proxy*/ false,
        result);
    DB::Tag tag2_impl_invalid_notshared = get_impl_tag( transaction, tag2_proxy_invalid_notshared);
    reader.rewind();

    // Check that there is no implementation class sharing with invalid hashes
    MI_CHECK_NOT_EQUAL( tag1_proxy_invalid_notshared.get_uint(), tag2_proxy_invalid_notshared.get_uint());
    MI_CHECK_NOT_EQUAL( tag1_impl_invalid_notshared.get_uint(), tag2_impl_invalid_notshared.get_uint());

    // Load twice with valid hash (proxy not shared)
    DB::Tag tag1_proxy_hash1_notshared = BSDFM::load_mdl_bsdf_measurement(
        transaction,
        &reader,
        /*dummies for all paths*/ "",
        "",
        "",
        "",
        some_hash1,
        /*shared_proxy*/ false,
        result);
    DB::Tag tag1_impl_hash1_notshared = get_impl_tag( transaction, tag1_proxy_hash1_notshared);
    reader.rewind();

    DB::Tag tag2_proxy_hash1_notshared = BSDFM::load_mdl_bsdf_measurement(
        transaction,
        &reader,
        /*dummies for all paths*/ "",
        "",
        "",
        "",
        some_hash1,
        /*shared_proxy*/ false,
        result);
    DB::Tag tag2_impl_hash1_notshared = get_impl_tag( transaction, tag2_proxy_hash1_notshared);
    reader.rewind();

    // Check that the implementation class is shared for equal hashes
    MI_CHECK_NOT_EQUAL( tag1_proxy_hash1_notshared.get_uint(), tag2_proxy_hash1_notshared.get_uint());
    MI_CHECK_EQUAL( tag1_impl_hash1_notshared.get_uint(), tag2_impl_hash1_notshared.get_uint());

    // Load again with different hash (proxy not shared)
    DB::Tag tag1_proxy_hash2_notshared = BSDFM::load_mdl_bsdf_measurement(
        transaction,
        &reader,
        /*dummies for all paths*/ "",
        "",
        "",
        "",
        some_hash2,
        /*shared_proxy*/ false,
        result);
    DB::Tag tag1_impl_hash2_notshared = get_impl_tag( transaction, tag1_proxy_hash2_notshared);
    reader.rewind();

    // Check that the implementation class is not shared for unequal hashes
    MI_CHECK_NOT_EQUAL( tag1_proxy_hash1_notshared.get_uint(), tag1_proxy_hash2_notshared.get_uint());
    MI_CHECK_NOT_EQUAL( tag1_impl_hash1_notshared.get_uint(), tag1_impl_hash2_notshared.get_uint());

    // Check naming scheme
    const char* name_impl_invalid_notshared = transaction->tag_to_name( tag1_impl_invalid_notshared);
    MI_CHECK( !name_impl_invalid_notshared);
    const char* name_impl_hash1_notshared = transaction->tag_to_name( tag1_impl_hash1_notshared);
    MI_CHECK_EQUAL_CSTR(
        name_impl_hash1_notshared, "MI_default_bsdf_measurement_impl_0x00000001000000010000000100000001");
    const char* name_impl_hash2_notshared = transaction->tag_to_name( tag1_impl_hash2_notshared);
    MI_CHECK_EQUAL_CSTR(
        name_impl_hash2_notshared, "MI_default_bsdf_measurement_impl_0x00000002000000020000000200000002");

    // Load again with first hash (proxy shared, non-empty filename to allow shared proxies)
    DB::Tag tag1_proxy_hash1_shared = BSDFM::load_mdl_bsdf_measurement(
        transaction,
        &reader,
        /*dummies for all paths*/ "f",
        "",
        "",
        "",
        some_hash1,
        /*shared_proxy*/ true,
        result);
    DB::Tag tag1_impl_hash1_shared = get_impl_tag( transaction, tag1_proxy_hash1_shared);
    reader.rewind();

    // Check that proxy class is not shared with unshared proxies
    MI_CHECK_NOT_EQUAL( tag1_proxy_hash1_shared.get_uint(), tag1_proxy_hash1_notshared.get_uint());
    MI_CHECK_NOT_EQUAL( tag1_proxy_hash1_shared.get_uint(), tag1_proxy_hash1_notshared.get_uint());

    // Check that implementation class is shared with unshared proxies
    MI_CHECK_EQUAL( tag1_impl_hash1_shared.get_uint(), tag1_impl_hash1_notshared.get_uint());
    MI_CHECK_EQUAL( tag1_impl_hash1_shared.get_uint(), tag1_impl_hash1_notshared.get_uint());

    // Load again with first hash (proxy shared, non-empty filename to allow shared proxies)
    DB::Tag tag2_proxy_hash1_shared = BSDFM::load_mdl_bsdf_measurement(
        transaction,
        &reader,
        /*dummies for all paths*/ "f",
        "",
        "",
        "",
        some_hash1,
        /*shared_proxy*/ true,
        result);
    DB::Tag tag2_impl_hash1_shared = get_impl_tag( transaction, tag2_proxy_hash1_shared);
    reader.rewind();

    // Check that proxy class is shared with shared proxies
    MI_CHECK_EQUAL( tag2_proxy_hash1_shared.get_uint(), tag1_proxy_hash1_shared.get_uint());
    MI_CHECK_EQUAL( tag2_proxy_hash1_shared.get_uint(), tag1_proxy_hash1_shared.get_uint());

    // Check that implementation class is shared with shared proxies
    MI_CHECK_EQUAL( tag2_impl_hash1_shared.get_uint(), tag1_impl_hash1_shared.get_uint());
    MI_CHECK_EQUAL( tag2_impl_hash1_shared.get_uint(), tag1_impl_hash1_shared.get_uint());

}

void check_failures( DB::Transaction* transaction)
{
    // Failures (-5 difficult to test)
    {
        //  Invalid filename extension
        std::string file_path = TEST::mi_src_path( "io/scene/mdl_elements/resources/test.png");
        auto bsdf_measurement = std::make_unique<BSDFM::Bsdf_measurement>();
        mi::Sint32 result = bsdf_measurement->reset_file( transaction, file_path);
        MI_CHECK_EQUAL( result, -3);
    }
    {
        //  File does not exist
        std::string file_path = TEST::mi_src_path(
            "io/scene/bsdf_measurement/test_not_existing.mbsdf");
        auto bsdf_measurement = std::make_unique<BSDFM::Bsdf_measurement>();
        mi::Sint32 result = bsdf_measurement->reset_file( transaction, file_path);
        MI_CHECK_EQUAL( result, -4);
    }
    {
        //  File format eror
        std::string file_path = TEST::mi_src_path(
            "io/scene/bsdf_measurement/test_file_format_error.mbsdf");
        auto bsdf_measurement = std::make_unique<BSDFM::Bsdf_measurement>();
        mi::Sint32 result = bsdf_measurement->reset_file( transaction, file_path);
        MI_CHECK_EQUAL( result, -7);
    }
}

MI_TEST_AUTO_FUNCTION( test_bsdf_measurement )
{
    Unified_database_access db_access;

    SYSTEM::Access_module<CONFIG::Config_module> m_config_module( false);
    m_config_module->override( "check_serializer_store=1");
    m_config_module->override( "check_serializer_edit=1");

    SYSTEM::Access_module<PATH::Path_module> m_path_module( false);
    m_path_module->add_path( PATH::RESOURCE, ".");

    DB::Database* database = db_access.get_database();
    DB::Scope* scope = database->get_global_scope();
    DB::Transaction* transaction = scope->start_transaction();

    import_check_export_reimport_check(
        transaction, "test_scalar_theta_2_phi_3.mbsdf",
        true, mi::neuraylib::BSDF_SCALAR, 2, 3, false, mi::neuraylib::BSDF_SCALAR, 0, 0);
    import_check_export_reimport_check(
        transaction, "test_rgb_theta_2_phi_3.mbsdf",
        true, mi::neuraylib::BSDF_RGB, 2, 3, false, mi::neuraylib::BSDF_RGB, 0, 0);
    import_check_export_reimport_check(
        transaction, "test_reflection_scalar_theta_2_phi_3_transmission_scalar_theta_2_phi_4.mbsdf",
        true, mi::neuraylib::BSDF_SCALAR, 2, 3, true, mi::neuraylib::BSDF_SCALAR, 2, 4);
    import_check_export_reimport_check(
        transaction, "test_reflection_scalar_theta_2_phi_3.mbsdf",
        true, mi::neuraylib::BSDF_SCALAR, 2, 3, false, mi::neuraylib::BSDF_SCALAR, 0, 0);
    import_check_export_reimport_check(
        transaction, "test_transmission_scalar_theta_2_phi_4.mbsdf",
        false, mi::neuraylib::BSDF_SCALAR, 0, 0, true, mi::neuraylib::BSDF_SCALAR, 2, 4);
    import_check_export_reimport_check(
        transaction, "test_none.mbsdf",
        false, mi::neuraylib::BSDF_SCALAR, 0, 0, false, mi::neuraylib::BSDF_SCALAR, 0, 0);

    check_sharing( transaction, "test_scalar_theta_2_phi_3.mbsdf");

    {
        // Create a default-constructed bsdf_measurement DB element
        auto* bsdf_measurement = new BSDFM::Bsdf_measurement();
        DB::Tag tag = transaction->store_for_reference_counting( bsdf_measurement);

        // Access the DB element
        DB::Access<BSDFM::Bsdf_measurement> access( tag, transaction);
        MI_CHECK( access->get_filename().empty());

        // Edit the DB element
        DB::Edit<BSDFM::Bsdf_measurement> edit( tag, transaction);
        MI_CHECK( edit->get_filename().empty());
    }

    check_failures( transaction);

    transaction->commit();
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
