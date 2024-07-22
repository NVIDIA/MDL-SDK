/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for io/image/image"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <filesystem>

#include "i_image.h"

#include <mi/base/handle.h>
#include <mi/math.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/istring.h>

#include <base/system/main/access_module.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>
#include <base/lib/plug/i_plug.h>
#include <base/data/idata/i_idata_factory.h>
#include <base/data/serial/i_serial_buffer_serializer.h>

#include "test_shared.h"
#include <prod/lib/neuray/test_shared.h>

namespace fs = std::filesystem;

const std::string DIR_PREFIX = "output_test_import_export";

using namespace MI;

class Mdl_container_callback : public mi::base::Interface_implement<IMAGE::IMdl_container_callback>
{
public:
    /// Sets the expected filenames and the reader to return
    void set_reader(
        const char* container_filename, const char* member_filename, mi::neuraylib::IReader* reader)
    {
        m_container_filename = container_filename;
        m_member_filename = member_filename;
        m_reader = make_handle_dup( reader);
    }

    mi::neuraylib::IReader* get_reader(
        const char* container_filename, const char* member_filename) final
    {
        MI_CHECK_EQUAL_CSTR( m_container_filename.c_str(), container_filename);
        MI_CHECK_EQUAL_CSTR( m_member_filename.c_str(), member_filename);
        MI_CHECK( m_reader);
        m_reader->rewind();
        m_reader->retain();
        return m_reader.get();
    }

private:
    std::string m_container_filename;
    std::string m_member_filename;
    mi::base::Handle<mi::neuraylib::IReader> m_reader;
};

SYSTEM::Access_module<IMAGE::Image_module> g_image_module;
mi::base::Handle<Mdl_container_callback> g_mdl_container_callback;
mi::base::Handle<mi::IMap> g_export_options;

mi::neuraylib::ICanvas* serialize_deserialize( mi::neuraylib::ICanvas* canvas)
{
    SERIAL::Buffer_serializer serializer;
    g_image_module->serialize_canvas( &serializer, canvas);
    SERIAL::Buffer_deserializer deserializer;
    deserializer.reset( serializer.get_buffer(), serializer.get_buffer_size());
    return g_image_module->deserialize_canvas( &deserializer);
}

void test_import(
    const char* file, const char* export_format, const char* reference_format = nullptr)
{
    std::string root_path  = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    if( !reference_format)
        reference_format = export_format;

    {
        std::cout << "\nTesting lazy file-based import of " << file << std::endl;

        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            g_image_module->create_canvas( IMAGE::File_based(), input_path, /*selector*/ nullptr));
        MI_CHECK( canvas);

        canvas = serialize_deserialize( canvas.get());
        MI_CHECK( canvas);

        std::string output_path = DIR_PREFIX + "/export1_of_" + file + "." + export_format;

        bool result = g_image_module->export_canvas(
            canvas.get(), output_path.c_str(), g_export_options.get());
        MI_CHECK( result);

        std::string reference_path
            = root_path + "reference/export_of_" + file + "." + reference_format;
        MI_CHECK_IMG_DIFF( output_path, reference_path);
    }
    {
        std::cout << "\nTesting non-lazy reader-based import of " << file << std::endl;

        DISK::File_reader_impl reader;
        bool result = reader.open( input_path.c_str());
        MI_CHECK( result);

        mi::base::Handle<mi::neuraylib::ICanvas> canvas( g_image_module->create_canvas(
            IMAGE::Container_based(), &reader, "dummy container", input_path,
            /*selector*/ nullptr));
        MI_CHECK( canvas);

        canvas = serialize_deserialize( canvas.get());
        MI_CHECK( canvas);

        std::string output_path = DIR_PREFIX + "/export2_of_" + file + "." + export_format;

        result = g_image_module->export_canvas(
            canvas.get(), output_path.c_str(), g_export_options.get());
        MI_CHECK( result);

        std::string reference_path
            = root_path + "reference/export_of_" + file + "." + reference_format;
        MI_CHECK_IMG_DIFF( output_path, reference_path);
    }
    {
        std::cout << "\nTesting lazy reader-based import of " << file << std::endl;

        DISK::File_reader_impl reader;
        bool result = reader.open( input_path.c_str());
        MI_CHECK( result);

        g_mdl_container_callback->set_reader( "dummy container", input_path.c_str(), &reader);
        g_image_module->set_mdl_container_callback( g_mdl_container_callback.get());

        mi::base::Handle<mi::neuraylib::ICanvas> canvas( g_image_module->create_canvas(
            IMAGE::Container_based(), &reader, "dummy container", input_path,
            /*selector*/ nullptr));
        MI_CHECK( canvas);

        canvas = serialize_deserialize( canvas.get());
        MI_CHECK( canvas);

        std::string output_path = DIR_PREFIX + "/export3_of_" + file + "." + export_format;

        result = g_image_module->export_canvas(
            canvas.get(), output_path.c_str(), g_export_options.get());
        MI_CHECK( result);

        std::string reference_path
            = root_path + "reference/export_of_" + file + "." + reference_format;
        MI_CHECK_IMG_DIFF( output_path, reference_path);

        g_image_module->set_mdl_container_callback( nullptr);
        g_mdl_container_callback->set_reader( "", "", nullptr);
    }
}

void test_export( const char* file, const char* export_format)
{
    std::cout << "\nTesting export of " << file << " to " << export_format << std::endl;

    std::string root_path  = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        g_image_module->create_canvas( IMAGE::File_based(), input_path, /*selector*/ nullptr));
    MI_CHECK( canvas);

    canvas = serialize_deserialize( canvas.get());
    MI_CHECK( canvas);

    std::string output_path = DIR_PREFIX + "/export_of_" + file + "." + export_format;

    bool result = g_image_module->export_canvas(
        canvas.get(), output_path.c_str(), g_export_options.get());
    MI_CHECK( result);

    // re-import and export as PNG

    canvas = g_image_module->create_canvas(
        IMAGE::File_based(), output_path, /*selector*/ nullptr);
    MI_CHECK( canvas);

    std::string png_path
        = DIR_PREFIX + "/export_of_export_of_" + file + "." + export_format + ".png";

    result = g_image_module->export_canvas(
        canvas.get(), png_path.c_str(), g_export_options.get());
    MI_CHECK( result);

    std::string reference_path
        = root_path + "reference/export_of_export_of_" + file + "." + export_format + ".png";
    MI_CHECK_IMG_DIFF( png_path, reference_path);
}

void test_selector( const char* file, const char* selector)
{
    std::string root_path  = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    const char* safe_selector = selector ? selector : "null";

    {
        std::cout << "\nTesting selector \"" << safe_selector << "\" for " << file << std::endl;

        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            g_image_module->create_canvas( IMAGE::File_based(), input_path, selector));
        MI_CHECK( canvas);
        MI_CHECK_NOT_EQUAL( canvas->get_resolution_x(), 1);

        std::string output_path = DIR_PREFIX + "/export_of_" + file + "_" + safe_selector + ".png";

        bool result = g_image_module->export_canvas(
            canvas.get(), output_path.c_str(), g_export_options.get());
        MI_CHECK( result);

        std::string reference_path
            = root_path + "reference/export_of_" + file + "_" + safe_selector + ".png";
        MI_CHECK_IMG_DIFF( output_path, reference_path);
    }
}

void test_selector_fail( const char* file, const char* selector)
{
    std::string root_path  = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    {
        std::cout << "\nTesting selector \"" << selector << "\" for " << file
                  << " (failure expected)" << std::endl;

        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            g_image_module->create_canvas( IMAGE::File_based(), input_path, selector));
        MI_CHECK( canvas);
        MI_CHECK_EQUAL( canvas->get_resolution_x(), 1);
    }
}

void test_exr_data_type()
{
    const char* file = "test_simple_oiio.exr";

    std::string root_path  = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    IDATA::Factory factory( nullptr);
    mi::base::Handle exr_data_type( factory.create<mi::IString>());
    exr_data_type->set_c_str( "Float16");

    mi::base::Handle local_options( factory.clone<mi::IMap>( g_export_options.get()));
    local_options->insert( "exr:data_type", exr_data_type.get());

    {
        std::cout << "\nTesting exr:data_type" << std::endl;

        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            g_image_module->create_canvas( IMAGE::File_based(), input_path, /*selector*/ nullptr));
        MI_CHECK( canvas);
        MI_CHECK_NOT_EQUAL( canvas->get_resolution_x(), 1);

        // Set red channel of lower left pixel to a value that cannot be represent in Float16
        // (65504 is the largest value that can be represented).
        mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( 0));
        float* data = static_cast<float*>( tile->get_data());
        data[0] = 65505.0f;

        std::string output_path = DIR_PREFIX + "/export_of_" + file + "_exr_data_type.exr";

        bool result = g_image_module->export_canvas(
            canvas.get(), output_path.c_str(), local_options.get());
        MI_CHECK( result);

        std::string reference_path
            = root_path + "reference/export_of_" + file + "_exr_data_type.exr";
        MI_CHECK_IMG_DIFF( output_path, reference_path);
    }
}

void test_exr_create_multipart_for_alpha()
{
    const char* file = "test_simple_alpha_oiio.exr";

    std::string root_path  = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    IDATA::Factory factory( nullptr);
    mi::base::Handle exr_create_multipart_for_alpha( factory.create<mi::IBoolean>());
    exr_create_multipart_for_alpha->set_value( true);

    mi::base::Handle local_options( factory.clone<mi::IMap>( g_export_options.get()));
    local_options->insert( "exr:create_multipart_for_alpha", exr_create_multipart_for_alpha.get());

    {
        std::cout << "\nTesting exr:create_multipart_for_alpha" << std::endl;

        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            g_image_module->create_canvas( IMAGE::File_based(), input_path, /*selector*/ nullptr));
        MI_CHECK( canvas);
        MI_CHECK_NOT_EQUAL( canvas->get_resolution_x(), 1);

        std::string output_path
            = DIR_PREFIX + "/export_of_" + file + "_exr_create_multipart_for_alpha.exr";

        bool result = g_image_module->export_canvas(
            canvas.get(), output_path.c_str(), local_options.get());
        MI_CHECK( result);

        std::string reference_path
            = root_path + "reference/export_of_" + file + "_exr_create_multipart_for_alpha.exr";
        MI_CHECK_IMG_DIFF( output_path, reference_path);

        // no selector defaults to first subimage
        mi::base::Handle<mi::neuraylib::ICanvas> reimport(
            g_image_module->create_canvas( IMAGE::File_based(), output_path, /*selector*/ nullptr));
        MI_CHECK( reimport);
        MI_CHECK_NOT_EQUAL( reimport->get_resolution_x(), 1);

        mi::base::Handle<mi::neuraylib::ICanvas> subimage00(
            g_image_module->create_canvas( IMAGE::File_based(), output_path, "rgb"));
        MI_CHECK( subimage00);
        MI_CHECK_NOT_EQUAL( subimage00->get_resolution_x(), 1);

        mi::base::Handle<mi::neuraylib::ICanvas> subimage01(
            g_image_module->create_canvas( IMAGE::File_based(), output_path, "alpha"));
        MI_CHECK( subimage01);
        MI_CHECK_NOT_EQUAL( subimage01->get_resolution_x(), 1);
    }
}

void test_exr_data_type_and_create_multipart_for_alpha()
{
    const char* file = "test_simple_alpha_oiio.exr";

    std::string root_path  = TEST::mi_src_path( "io/image/image/tests/");
    std::string input_path = root_path + file;

    IDATA::Factory factory( nullptr);
    mi::base::Handle exr_data_type( factory.create<mi::IString>());
    exr_data_type->set_c_str( "Float16");
    mi::base::Handle exr_create_multipart_for_alpha( factory.create<mi::IBoolean>());
    exr_create_multipart_for_alpha->set_value( true);

    mi::base::Handle local_options( factory.clone<mi::IMap>( g_export_options.get()));
    local_options->insert( "exr:data_type", exr_data_type.get());
    local_options->insert( "exr:create_multipart_for_alpha", exr_create_multipart_for_alpha.get());

    {
        std::cout << "\nTesting exr:data_type and exr:create_multipart_for_alpha" << std::endl;

        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            g_image_module->create_canvas( IMAGE::File_based(), input_path, /*selector*/ nullptr));
        MI_CHECK( canvas);
        MI_CHECK_NOT_EQUAL( canvas->get_resolution_x(), 1);

        // Set red channel of lower left pixel to a value that cannot be represent in Float16
        // (65504 is the largest value that can be represented).
        mi::base::Handle<mi::neuraylib::ITile> tile( canvas->get_tile( 0));
        float* data = static_cast<float*>( tile->get_data());
        data[0] = 65505.0f;

        std::string output_path = DIR_PREFIX + "/export_of_" + file
            + "_exr_data_type_and_create_multipart_for_alpha.exr";

        bool result = g_image_module->export_canvas(
            canvas.get(), output_path.c_str(), local_options.get());
        MI_CHECK( result);

        std::string reference_path = root_path + "reference/export_of_" + file
            + "_exr_data_type_and_create_multipart_for_alpha.exr";
        MI_CHECK_IMG_DIFF( output_path, reference_path);

        // no selector defaults to first subimage
        mi::base::Handle<mi::neuraylib::ICanvas> reimport(
            g_image_module->create_canvas( IMAGE::File_based(), output_path, /*selector*/ nullptr));
        MI_CHECK( reimport);
        MI_CHECK_NOT_EQUAL( reimport->get_resolution_x(), 1);

        mi::base::Handle<mi::neuraylib::ICanvas> subimage00(
            g_image_module->create_canvas( IMAGE::File_based(), output_path, "rgb"));
        MI_CHECK( subimage00);
        MI_CHECK_NOT_EQUAL( subimage00->get_resolution_x(), 1);

        mi::base::Handle<mi::neuraylib::ICanvas> subimage01(
            g_image_module->create_canvas( IMAGE::File_based(), output_path, "alpha"));
        MI_CHECK( subimage01);
        MI_CHECK_NOT_EQUAL( subimage01->get_resolution_x(), 1);
    }
}

MI_TEST_AUTO_FUNCTION( test_import_export )
{
    fs::remove_all( fs::u8path( DIR_PREFIX));
    fs::create_directory( fs::u8path( DIR_PREFIX));

    SYSTEM::Access_module<MEM::Mem_module> mem_module( false);
    SYSTEM::Access_module<LOG::Log_module> log_module( false);

    SYSTEM::Access_module<PLUG::Plug_module> plug_module( false);
    std::string plugin_path;

    MI_CHECK( plug_module->load_library( plugin_path_dds));
    MI_CHECK( plug_module->load_library( plugin_path_openimageio));

    g_image_module.set();

    g_mdl_container_callback = new Mdl_container_callback();

    IDATA::Factory factory( nullptr);
    mi::base::Handle force_default_gamma( factory.create<mi::IBoolean>());
    force_default_gamma->set_value( true);

    g_export_options = factory.create<mi::IMap>( nullptr, "Map<Interface>");
    g_export_options->insert( "force_default_gamma", force_default_gamma.get());

    // Note that there are different tests for importing and exporting a certain file format (even
    // though the export test re-imports that file format again)
    // - The import test originally only tested those file formats which were supported by GIMP at
    //   that time (which was used to create those test images).
    // - The export test only tests those file formats that are supported for exporting by our
    //   plugins.

    test_import( "test_simple.bmp", "bmp", "png");
    test_import( "test_simple_fi.exr", "png");
    test_import( "test_simple_imf.exr", "png");
    test_import( "test_simple_oiio.exr", "png");
    test_import( "test_simple.gif", "png");
    test_import( "test_simple_fi.hdr", "png");
    test_import( "test_simple.jpg", "png"); // not lossless
    test_import( "test_simple.png", "png");
    test_import( "test_simple.tga", "tga", "png");
    test_import( "test_simple_deflate.tif", "tif", "png");
    test_import( "test_simple_jpeg.tif", "tif", "png");
    test_import( "test_simple_lzw.tif", "tif", "png");
    test_import( "test_simple_none.tif", "tif", "png");
    test_import( "test_simple_packbits.tif", "tif", "png");

    test_import( "test_simple_alpha.bmp", "bmp");
    test_import( "test_simple_alpha_fi.exr", "tif");
    test_import( "test_simple_alpha_imf.exr", "tif");
    test_import( "test_simple_alpha_oiio.exr", "tif");
    test_import( "test_simple_alpha.png", "png");
    test_import( "test_simple_alpha.tga", "tga");
    test_import( "test_simple_alpha_deflate.tif", "tif");
    test_import( "test_simple_alpha_jpeg.tif", "tif");
    test_import( "test_simple_alpha_lzw.tif", "tif");
    test_import( "test_simple_alpha_none.tif", "tif");
    test_import( "test_simple_alpha_packbits.tif", "tif");

    test_export( "test_simple.png", "bmp");
    test_export( "test_simple.png", "dds");
    test_export( "test_simple.png", "j2k"); // not lossless
    test_export( "test_simple.png", "jp2"); // not lossless
    test_export( "test_simple.png", "jpg"); // not lossless
    test_export( "test_simple.png", "pbm");
    test_export( "test_simple.png", "png");
    test_export( "test_simple.png", "tga");
    test_export( "test_simple.png", "tif");
    test_export( "test_simple.png", "webp"); // not lossless

    test_export( "test_simple_alpha.png", "bmp");
    test_export( "test_simple_alpha.png", "dds");
    test_export( "test_simple_alpha.png", "exr");
    test_export( "test_simple_alpha.png", "png");
    test_export( "test_simple_alpha.png", "tga");
    test_export( "test_simple_alpha.png", "tif");

    // Test export with default gamma enabled and exported pixel type gamma (2.2 for Rgb_16 in
    // output file) different from canvas gamma (1.0 for Rgb_fp in input file).
    test_export( "test_gamma.exr", "png");

    // Test an EXR image with luminance-chroma channels with different sampling rates.
    test_import( "test_exr_luminance_chroma.exr", "png");
    test_import( "test_exr_luminance_chroma_alpha.exr", "png");


    // Test images with a gray and alpha channel.
    test_import( "test_gray_alpha_bpc_2.png", "png");


    // Test that a patched OIIO/libjpeg can still read progressive JPEGs correctly. Export as BMP
    // to avoid expensive PNG compression for this large texture.
    test_import( "test_jpg_progressive.jpg", "bmp", "png");


    // Test pixel type Uint16 (not one of our pixel types, mapped to Float32).
    test_import( "test_pt_uint16.png", "tif");


    // Test pixel type Sint32.
    test_import( "test_pt_sint32.tif", "tif");

    // Test various pixel types for DDS. Note the compression artifacts in the compressed formats.
    test_import( "test_dds_a16b16g16r16f.dds", "tif");
    test_import( "test_dds_a32b32g32r32f.dds", "tif");
    test_import( "test_dds_a8b8g8r8.dds", "tif");
    test_import( "test_dds_a8r8g8b8.dds", "tif");
    test_import( "test_dds_dxt1.dds", "tif");
    test_import( "test_dds_dxt3.dds", "tif");
    test_import( "test_dds_dxt5.dds", "tif");
    test_import( "test_dds_dxt1_alpha.dds", "tif");
    // Alpha values == 0 cause problems when associating/deassociating
    // test_import( "test_dds_dxt3_alpha.dds", "tif");
    // test_import( "test_dds_dxt5_alpha.dds", "tif");
    test_import( "test_dds_r32f.dds", "tif");
    test_import( "test_dds_r8g8b8.dds", "tif");
    test_import( "test_dds_x8b8g8r8.dds", "tif");
    test_import( "test_dds_x8r8g8b8.dds", "tif");

    test_import( "test_dds_l8.dds", "tif");

    // Similar as before, but with a different test image (intensity 0x80 instead of 0xff) for
    // testing gamma handling.
    test_import( "test_dds_gamma_dxt1.dds", "tif");
    test_import( "test_dds_gamma_dxt3.dds", "tif");
    test_import( "test_dds_gamma_dxt5.dds", "tif");

    // Test RGBA selectors for pixel type "Color" (with alpha).
    test_selector( "test_pt_color.tif", "R");
    test_selector( "test_pt_color.tif", "G");
    test_selector( "test_pt_color.tif", "B");
    test_selector( "test_pt_color.tif", "A");

    // Test RGBA selectors for pixel type "Rgb_fp" (without alpha).
    test_selector( "test_pt_rgb_fp.tif", "R");
    test_selector( "test_pt_rgb_fp.tif", "G");
    test_selector( "test_pt_rgb_fp.tif", "B");
    test_selector_fail( "test_pt_rgb_fp.tif", "A");

    // Test R selector on other RGB(A) pixel types.
    test_selector( "test_pt_rgba_16.tif", "R"); // with alpha, via "Color"
    test_selector( "test_pt_rgb_16.tif", "R");  // without alpha, via "Rgb_fp"
    test_selector( "test_pt_rgba.tif", "R");
    test_selector( "test_pt_rgb.tif", "R");

    // Test R selector on non-RGB(A) pixel types.
    test_selector_fail( "test_pt_float32.tif", "R");
    test_selector_fail( "test_pt_sint32.tif", "R");

    // Test non-RGBA selectors.
    test_selector_fail( "test_pt_color.tif", "r");
    test_selector_fail( "test_pt_color.tif", "X");
    test_selector_fail( "test_pt_color.tif", "diffuse");

    // EXR texture with "left" and "right" layers.
    test_selector( "test_simple_left_right.exr", "left");
    test_selector( "test_simple_left_right.exr", "right");
    test_selector( "test_simple_left_right.exr", "left.B");

    // EXR texture with "diffuse.left" and "diffuse.right" layers.
    test_selector( "test_simple_diffuse_left_right.exr", "diffuse.left");
    test_selector( "test_simple_diffuse_left_right.exr", "diffuse.right");
    test_selector( "test_simple_diffuse_left_right.exr", "diffuse.left.B");
    // "diffuse" it not a "most specific" layer name
    test_selector_fail( "test_simple_diffuse_left_right.exr", "diffuse");

    // EXR texture with "left" and "right" layers (interleaved on creation, sorted during
    // export/import).
    test_selector( "test_simple_left_right_BGR_interleaved.exr", "left");
    test_selector( "test_simple_left_right_BGR_interleaved.exr", "right");
    test_selector( "test_simple_left_right_BGR_interleaved.exr", "left.B");

    // EXR multiview texture with "left" and "right" views.
    test_selector( "test_simple_multiview.exr", nullptr);
    test_selector( "test_simple_multiview.exr", "right");
    test_selector( "test_simple_multiview.exr", "B");
    test_selector( "test_simple_multiview.exr", "right.B");
    // "left" is a view name, but not a layer name
    test_selector_fail( "test_simple_multiview.exr", "left");
    test_selector_fail( "test_simple_multiview.exr", "left.B");

    // EXR multipart texture with "left" and "right" parts, unique channel names.
    test_selector( "test_simple_multipart_unique.exr", "left.left");
    test_selector( "test_simple_multipart_unique.exr", "right.right");
    test_selector( "test_simple_multipart_unique.exr", "left.left.B");

    // EXR multipart texture with "left" and "right" parts, non-unique channel names.
    test_selector( "test_simple_multipart_non_unique.exr", "left");
    test_selector( "test_simple_multipart_non_unique.exr", "right");
    test_selector( "test_simple_multipart_non_unique.exr", "left.B");

    // EXR export options exr:data_type and/or exr:create_multipart_for_alpha.
    test_exr_data_type();

    g_mdl_container_callback = nullptr;

    g_image_module.reset();
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

