/***************************************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief The .axf importer internals.


// debug: dump texture files and preview images
//#define DUMP_TEXTURES

#include "axf_importer_reader.h"
#include "axf_importer_state.h"

#include "axf_importer_clearcoat_brdf_utils.h"

#include "example_shared.h"
#include "utils/strings.h"
#include "utils/ospath.h"

#include <AxF/decoding/AxF_basic_io.h>
#include <AxF/decoding/Sampler.h>
#include <AxF/decoding/TextureDecoder.h>

#include <cassert>
#include <cctype>
#include <sstream>
#include <vector>

#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN_NT
 #include <unistd.h>
#endif

using namespace mi::examples::strings;
using namespace mi::examples::ospath;

namespace mi {
namespace examples {
namespace impaxf {

using namespace std;
namespace AXF = axf::decoding;

/// Formats error message with context and appends it to the error messages in the result.
void Axf_importer::report_message(
    mi::Sint32 error_number,
    mi::base::Message_severity error_severity,
    const std::string& error_message,
    const Axf_impexp_state* import_state)
{
    std::ostringstream message;
    message << import_state->get_uri()
        << ':' << import_state->get_line_number() << ": "
        << error_message;

    printf("%s\n", message.str().c_str());
};

Axf_reader::Axf_reader(
    mi::neuraylib::INeuray *neuray,
    mi::neuraylib::ITransaction *transaction)
  : m_neuray(neuray),
    m_transaction(transaction),
    m_target_color_space(NULL),
    m_glossy_brdf_type(BRDF_TYPE_UNKNOWN),
    m_isotropic(true),
    m_fresnel_type(FRESNEL_NONE)
{}

namespace {

static const char *s_axf_supported_color_spaces[] = {
        AXF_COLORSPACE_CIE_1931_XYZ,
        AXF_COLORSPACE_LINEAR_SRGB_E,
        AXF_COLORSPACE_LINEAR_ADOBE_RGB_E,
        AXF_COLORSPACE_LINEAR_ADOBE_WIDEGAMUT_RGB_E,
        AXF_COLORSPACE_LINEAR_PROPHOTO_RGB_E,
        NULL
};

static const char *s_prototype_names[3] = {
    "mdl::nvidia::axf_importer::axf_importer::svbrdf("
        "texture_2d,texture_2d,texture_2d,texture_2d,texture_2d,texture_2d,texture_2d,"
        "::nvidia::axf_importer::axf_importer::brdf_type,bool,"
        "::nvidia::axf_importer::axf_importer::fresnel_type,texture_2d,bool,texture_2d,texture_2d,"
        "texture_2d,::base::texture_coordinate_info,bool,float,float,float,float,float,float,"
        "::tex::wrap_mode,float,float3,float,texture_2d,texture_2d,texture_2d,texture_2d)",
    "mdl::nvidia::axf_importer::axf_importer::carpaint(color[brdf_colors_size],float,float,float,"
        "float,float3,float3,float3,texture_2d,::base::texture_coordinate_info,bool,float,float,"
        "float,float,float,float,::tex::wrap_mode,float3,bool,float3,"
        "int[flake_importace_data_size],float,float,float,float,bool,bsdf_measurement,"
        "float[flake_orientation_falloff_size],texture_2d,bool)",
    "mdl::nvidia::axf_importer::axf_importer::volumetric(color,color,float,float)"
};

static void compute_spectral_texture_metadata(
    std::vector<char> &meta_data_buf,
    const std::vector<float> &wavelengths)
{
    // use unity matrix (each coefficient just represents a single wavelength sample)
    const size_t num_lambda = wavelengths.size();
    std::vector<float> matrix(num_lambda * num_lambda, 0.0f);
    for (unsigned int i = 0; i < num_lambda; ++i) {
        matrix[i * num_lambda + i] = 1.0f;
    }

    const size_t meta_data_size = sizeof(Texture_spectral_metadata) +
        (matrix.size() - 1) * sizeof(float); // -1 as Texture_spectral_metadata already has float matrix[1]

    meta_data_buf.resize(meta_data_size);
    Texture_spectral_metadata *meta_data =
        (Texture_spectral_metadata *)meta_data_buf.data();

    memcpy(
        meta_data->marker,
        TEXTURE_SPECTRAL_METADATA_MARKER, TEXTURE_SPECTRAL_METADATA_MARKER_SIZE);

    meta_data->lambda_min = wavelengths[0];
    meta_data->lambda_max = wavelengths[num_lambda - 1];
    meta_data->num_lambda = num_lambda;

    memcpy(meta_data->matrix, matrix.data(), matrix.size() * sizeof(float));
}

static size_t compute_num_spectral_meta_layers(
    size_t &offset0,
    const size_t layer_size_bytes,
    const size_t num_layers,
    const size_t meta_data_size)
{
    offset0 = num_layers * layer_size_bytes;
    // ensure 4 byte alignment for meta data start (assuming data was aligned to begin with)
    if (offset0 % 4)
        offset0 = 4 - (offset0 % 4);
    else
        offset0 = 0;
    return (meta_data_size + offset0 + layer_size_bytes - 1) / layer_size_bytes;
}

// spectra returned by the AxF SDK often immediately drop to zero below a certain minimum
// wavelength and above a certain maximum maximum wavelength (presumably because there is no data)...
// this is problematic for volume coefficients (since it suddenly leads to zero extinction)
// -> repeat the last non-zero entries instead of dropping to zero
static void expand_spectrum(std::vector<float> &spectrum)
{
#if 1
    // shouldn't be necessary anymore, as we now can query which data is there
    return;
#else
    const int num = (int)spectrum.size();
    int idx = -1;
    for (int i = 0; i < num; ++i)
        if (spectrum[i] > 0.0f) {
            idx = i;
            break;
        }
    for (int i = 0; i < idx; ++i)
        spectrum[i] = spectrum[idx];

    idx = num;
    for (int i = num - 1; i >= 0; --i)
        if (spectrum[i] > 0.0f) {
            idx = i;
            break;
        }
    for (int i = num - 1; i > idx; --i)
        spectrum[i] = spectrum[idx];
#endif
}


/// Little helper utility.
struct Scope_axf_file_handler// : private boost::noncopyable
{
    Scope_axf_file_handler(AXF::AXF_FILE_HANDLE file) : m_axf_file(file) {}
    ~Scope_axf_file_handler() { if (m_axf_file) AXF::axfCloseFile(&m_axf_file); }

    AXF::AXF_FILE_HANDLE m_axf_file;
};

/// Little helper utility.
template <typename T>
class Scope_handler// : private boost::noncopyable
{
public:
    Scope_handler(T* inst) : m_inst(inst) {}
    ~Scope_handler() { T::destroy(&m_inst); }

    operator bool() const { return m_inst != 0; }
    T* operator->() { return m_inst; }
    const T* operator->() const { return m_inst; }

    T* get() { return m_inst; }
    const T* get() const { return m_inst; }
private:
    T* m_inst;
};

/// Find out whether \p path is a file or not
/// \return true if \p path is a file
bool is_file(
    const char* const path)
{
    struct stat buf;
    if (::stat(path, &buf) == 0)
        return (buf.st_mode & S_IFREG) != 0;
    return false;
}

/// Retrieve the hardcoded prefix used for storing axf imported textures.
string get_axf_texture_prefix() { return "__AxF_imported_tex_"; }
/// Retrieve the hardcodede prefix used for storing axf imported images.
string get_axf_image_prefix() { return "__AxF_imported_img_"; }
/// Retrieve the hardcodede prefix used for storing axf imported preview images.
string get_axf_pimage_prefix() { return "__AxF_imported_pimg_"; }
/// Retrieve the hardcoded prefix used for storing axf imported measured BRDFs.
string get_axf_mbsdf_prefix() { return "__AxF_imported_mbsdf__"; }
/// Retrieve the hardcoded prefix used for storing axf spectra (color constructors)
string get_axf_spectrum_prefix() { return "__AxF_imported_spectrum__"; }

}

bool Axf_reader::read(
    const char* file_name,
    Axf_impexp_state* impexp_state)
{
    AXF::axfDisableLogging(); //!! TODO: implement logging callback?

    const char *cs = impexp_state->get_target_color_space();
    // check if color space is exposed by AxF SDK, default to linear sRGB if not (and warn)
    for (int i = 0; ; ++i) {
        const char *s = s_axf_supported_color_spaces[i];
        if (!s) {
            const string msg = string("Unsupported target color space \"") + cs
                    + string("\", defaulting to \"" AXF_COLORSPACE_LINEAR_SRGB_E "\"");
            Axf_importer::report_message(6026, mi::base::MESSAGE_SEVERITY_WARNING,
                msg, impexp_state);
            m_target_color_space = AXF_COLORSPACE_LINEAR_SRGB_E;
            break;
        }
        if (strcmp(cs, s) == 0) {
            m_target_color_space = s;
            break;
        }
    }

    if (!file_name || !is_file(file_name))
    {
        string msg = string("Could not find AxF file: ") + file_name;
        Axf_importer::report_message(6002, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return false;
    }
    m_filename = file_name;

    // access .axf file
    Scope_axf_file_handler file_h(AXF::axfOpenFile(file_name, true));
    if (!file_h.m_axf_file) {
        string msg = string("Failed to open AxF file ") + m_filename;
        Axf_importer::report_message(6002, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return false;
    }

    // import corresponding MDL module and access the materials
    if (!access_mdl_material_definitions(impexp_state))
        return false;

    const int num_materials = AXF::axfGetNumberOfMaterials(file_h.m_axf_file);
    for (int m = 0; m < num_materials; ++m) {
        bool success = read_material(file_h.m_axf_file, m, impexp_state);
        if (!success)
            return false;
    }

    const unsigned int num_variants = handle_collected_variants(impexp_state);
    if (num_variants > 0)
    {
        Axf_importer::report_message(6010, mi::base::MESSAGE_SEVERITY_INFO,
            std::string("Loaded ") + std::to_string(num_variants) +
            std::string(" materials from file ") + m_filename, impexp_state);
        return true;
    }
    else
        return false;
}

bool Axf_reader::read_material(
    AXF::AXF_FILE_HANDLE file,
    const int material_idx,
    Axf_impexp_state* impexp_state)
{
    AXF::AXF_MATERIAL_HANDLE material_h = AXF::axfGetMaterial(file, material_idx);

    char name_buf[AXF::AXF_MAX_KEY_SIZE];
    if (!AXF::axfGetMaterialIDString(material_h, name_buf, AXF::AXF_MAX_KEY_SIZE)) {
        string msg = string("Cannot retrieve material name from file ") + m_filename
            + string(" for ") + string(std::to_string(material_idx)) + string(". material.");
        Axf_importer::report_message(6003, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return false;
    }
    m_material_name = name_buf;

    m_material_display_name.clear();
    const size_t display_name_buf_size = AXF::axfGetMaterialDisplayName(material_h, nullptr, 0);
    if (display_name_buf_size > 0) {
        std::vector<wchar_t> display_name_buf(display_name_buf_size);
        if (AXF::axfGetMaterialDisplayName(material_h, display_name_buf.data(), display_name_buf_size))
            m_material_display_name = wchar_to_utf8(display_name_buf.data());
    }

    // we check the profiles one by one, as we do support different clearcoat coatings
    // depending on representation type

    // for carpaint we support non-refracting and refracting clearcoat (via baking)
    const AXF::VersionedCompatibilityProfile supported_base_profiles_carpaint[] = {
        {AXF_COMPAT_PROF_CARPAINT2, 1}
    };
    const AXF::VersionedCompatibilityProfile supported_coating_profiles_carpaint[] = {
        {AXF_COMPAT_PROF_CLEARCOAT_NO_REFRACT, 1},
        {AXF_COMPAT_PROF_CLEARCOAT_REFRACT, 1}
    };

    // volumetric has a refracing hull, which interestingly seems to be covered by
    // the refracting clearcoat profile
    const AXF::VersionedCompatibilityProfile supported_base_profiles_volumetric[] = {
        {AXF_COMPAT_PROF_VOLUMETRIC, 1}
    };
    const AXF::VersionedCompatibilityProfile supported_coating_profiles_volumetric[] = {
        {AXF_COMPAT_PROF_CLEARCOAT_REFRACT, 1}
    };

    // for SVBRDF we only support non-refracting clearcoat (and trigger conversion
    // for the rest)
    const AXF::VersionedCompatibilityProfile supported_base_profiles_svbrdf[] = {
        {AXF_COMPAT_PROF_SVBRDF_WARD, 1},
        {AXF_COMPAT_PROF_SVBRDF_WARD, 2},   // adds cutout and displacement
        {AXF_COMPAT_PROF_SVBRDF_WARD, 3},   // adds transmission color
        {AXF_COMPAT_PROF_SVBRDF_GGX, 1},
        {AXF_COMPAT_PROF_SVBRDF_GGX, 2}     // adds transmission color
    };
    const AXF::VersionedCompatibilityProfile supported_coating_profiles_svbrdf[] = {
        {AXF_COMPAT_PROF_CLEARCOAT_NO_REFRACT, 1}
    };

#define NUMEL(v) (sizeof(v) / sizeof(v[0]))

    int conversion_flags;
    // 1. check for carpaint
    AXF::AXF_REPRESENTATION_HANDLE rep_h = AXF::axfGetBestCompatibleRepresentation(
        material_h,
        supported_base_profiles_carpaint, NUMEL(supported_base_profiles_carpaint),
        supported_coating_profiles_carpaint, NUMEL(supported_coating_profiles_carpaint),
        conversion_flags);
    if (!rep_h) {
        // 2. check for volumetric
        rep_h = AXF::axfGetBestCompatibleRepresentation(
            material_h,
            supported_base_profiles_volumetric, NUMEL(supported_base_profiles_volumetric),
            supported_coating_profiles_volumetric, NUMEL(supported_coating_profiles_volumetric),
            conversion_flags);
    }
    if (!rep_h) {
        // 3. check for svbrdf
        rep_h = AXF::axfGetBestCompatibleRepresentation(
            material_h,
            supported_base_profiles_svbrdf, NUMEL(supported_base_profiles_svbrdf),
            supported_coating_profiles_svbrdf, NUMEL(supported_coating_profiles_svbrdf),
            conversion_flags);
    }

#undef NUMEL

    bool success = false;
    if (rep_h)
        success = handle_representation(rep_h, conversion_flags, impexp_state);

    if (!success) {
        const string msg = string("No supported representation found for AxF material ") + m_material_name;
        Axf_importer::report_message(6005, mi::base::MESSAGE_SEVERITY_ERROR, msg, impexp_state);
    }
    return success;
}

static void split_texture(
    std::vector<float>& out_a,
    std::vector<float>& out_b,
    const std::vector<float>& tex_buffer)
{
    const size_t num_pixels = out_a.size();
    assert(out_b.size() == num_pixels);
    assert(tex_buffer.size() == 2 * num_pixels);

    for (size_t i = 0; i < num_pixels; ++i)
    {
        out_a[i] = tex_buffer[2 * i];
        out_b[i] = tex_buffer[2 * i + 1];
    }
}

static int get_property_index(
    AXF::TextureDecoder *tex_decoder,
    const char *const name)
{
#if 0
    // appears to be unimplemented as of version 1.11...
    const int idx = tex_decoder->getPropertyIndexFromName(name);
#else
    // using linear search, the number of properties is small (keeping a map with string keys
    // is likely much more overhead)
    int idx = -1;
    for (int p = 0; p < tex_decoder->getNumProperties(); ++p)
    {
        char buf[AXF::AXF_MAX_KEY_SIZE];
        if (!tex_decoder->getPropertyName(p, buf, AXF::AXF_MAX_KEY_SIZE))
            continue;

        if (strcmp(buf, name) == 0)
        {
            idx = p;
            break;
        }
    }
#endif

    return idx;
}

// obtain texture decoder property of known size
static bool get_property(
    void *target,
    const size_t target_size,
    AXF::TextureDecoder *tex_decoder,
    const AXF::PropertyType type,
    const char *name,
    Axf_impexp_state* impexp_state,
    const bool report_error)
{
    memset(target, 0, target_size);

    const char *error = NULL;
    const int idx = get_property_index(tex_decoder, name);
    if (idx < 0)
        error = "not found";
    if (!error && tex_decoder->getPropertyType(idx) != type)
        error = "wrong type";

    if (!error && tex_decoder->getPropertySize(idx) != target_size)
        error = "invalid size";

    if (!error && !tex_decoder->getProperty(idx, target, type, target_size))
        error = "retrieval failed";

    if(error)
    {
        if (report_error)
            Axf_importer::report_message(
                6022, mi::base::MESSAGE_SEVERITY_ERROR,
                std::string("property \"") + std::string(name) +
                std::string("\": ") + std::string(error),
                impexp_state);
        return false;
    }

    return true;
}


// obtain texture decoder array property of unknown size
template <typename T>
static bool get_array_property(
    std::vector<T> &data,
    AXF::TextureDecoder *tex_decoder,
    const AXF::PropertyType type,
    const char *name,
    Axf_impexp_state* impexp_state,
    const bool report_error)
{
    const char *error = NULL;
    const int idx = get_property_index(tex_decoder, name);
    if (idx < 0)
        error = "not found";
    if (!error && tex_decoder->getPropertyType(idx) != type)
        error = "wrong type";

    if (!error)
    {
        const int size = tex_decoder->getPropertySize(idx);
        assert(size % sizeof(T) == 0);
        assert(size >= 0);
        data.resize(size / sizeof(T));
        if (size > 0)
        {
            if (!tex_decoder->getProperty(idx, data.data(), type, size))
                error = "retrieval failed";
        }
    }

    if(error)
    {
        if (report_error)
            Axf_importer::report_message(
                6022, mi::base::MESSAGE_SEVERITY_ERROR,
                std::string("property \"") + std::string(name) +
                std::string("\": ") + std::string(error),
                impexp_state);
        return false;
    }

    return true;
}

static bool flake_sanitize(float &r, float &g, float &b, const float threshold)
{
    r = std::max(r, 0.0f);
    g = std::max(g, 0.0f);
    b = std::max(b, 0.0f);
    return (std::max(r, std::max(g, b)) >= threshold);
}

static unsigned int hist_idx(const float x, const unsigned int size)
{
    int idx = (int)(x * (float)size);
    return (unsigned int)std::max(0, std::min((int)size - 1, idx));
}

static void initialize_carpaint_flakes(
    std::vector<unsigned int> &flake_importance_data,
    float &flake_intensity_scale,
    float flake_uvw_scale[3],
    std::vector<float> &flake_orientation_falloff,
    const unsigned int width,
    const unsigned int height,
    const std::vector<float> &flake_btf_data,
    const int flake_btf_num_theta_i,
    const int flake_btf_num_theta_f,
    const std::vector<int> &flake_btf_slice_lut)
{
    flake_importance_data.clear();
    flake_intensity_scale = 0.0f;
    flake_uvw_scale[0] = flake_uvw_scale[1] = flake_uvw_scale[2] = 0.01f;

    if (flake_btf_num_theta_i <= 0 || flake_btf_num_theta_f <= 0 || flake_btf_data.size() == 0)
        return;

    //!! constants need be kept in sync with axf_importer.mdl
    const unsigned int l_hist_size = 64;
    const unsigned int c_hist_size = 16;
    flake_importance_data.resize(l_hist_size + c_hist_size * c_hist_size, 0);

    unsigned int *hist_l = &flake_importance_data[0];
    unsigned int *hist_c = &flake_importance_data[l_hist_size];

    // we are using the first slice only to compute statistics of flake intensity and coloring
    const float *slice_data = &flake_btf_data[flake_btf_slice_lut[0] * width * height * 3];

    // determine maximum 'intensity'
    float max_l = 0.0f;
    float sum_l = 0.0f;
    for (unsigned int i = 0; i < width * height; ++i) {
        float r = slice_data[3 * i];
        float g = slice_data[3 * i + 1];
        float b = slice_data[3 * i + 2];
        flake_sanitize(r, g, b, 0.0f);
        const float l = r + g + b;
        sum_l += l;
        max_l = fmaxf(l, max_l);
    }

    // compute histogram for intensity and chromaticity
    for (unsigned int i = 0; i < width * height; ++i) {
        float r = slice_data[3 * i];
        float g = slice_data[3 * i + 1];
        float b = slice_data[3 * i + 2];
        const bool is_flake = flake_sanitize(r, g, b, 0.1f * max_l);
        float l = r + g + b;
        r /= l;
        g /= l;
        l /= max_l;

        ++hist_l[hist_idx(l, l_hist_size)];
        if (is_flake)
            ++hist_c[hist_idx(r, c_hist_size) * c_hist_size + hist_idx(g, c_hist_size)];
    }

    // compare with other slices' average intensity to approximate intensity falloff
    // based on theta_f
    flake_orientation_falloff.resize(flake_btf_num_theta_f, 1.0f);
    if (sum_l > 0.0f) {
        for (int i_f = 0; i_f < flake_btf_num_theta_f; ++i_f) {
            const float *f = slice_data + i_f * width * height * 3;

            float sum_l_i = 0.0f;
            for (unsigned int i = 0; i < width * height; ++i) {
                float r = f[3 * i];
                float g = f[3 * i + 1];
                float b = f[3 * i + 2];
                flake_sanitize(r, g, b, 0.0f);
                const float l = r + g + b;
                sum_l_i += l;
            }
            flake_orientation_falloff[i_f] = sum_l_i / sum_l;
        }
    }

    // compute cdf
    for (unsigned int i = 1; i < l_hist_size; ++i)
        hist_l[i] += hist_l[i - 1];
    for (unsigned int i = 1; i < c_hist_size * c_hist_size; ++i)
        hist_c[i] += hist_c[i - 1];

    // approximate scale factor from going from flake BTF function to highly glossy flake BRDF
    const float intensity_scale_factor = 0.4f;
    flake_intensity_scale = max_l * intensity_scale_factor;

    // a flake is approximately one and a half pixels in size, but blurred out to some extend
    const float flake_size_btf_u = 0.75f / (float)width;
    const float flake_size_btf_v = 0.75f / (float)height;
    const float flake_size_procedural = 1.3f; // approximate size of a procedural flake
    flake_uvw_scale[0] = flake_size_procedural / flake_size_btf_u;
    flake_uvw_scale[1] = flake_size_procedural / flake_size_btf_v;
    flake_uvw_scale[2] = 0.5f * (flake_uvw_scale[0] + flake_uvw_scale[1]);
}

// upscale a texture buffer to a given target resolution
static void upscale_tex_buffer(
    std::vector<float> &tex,
    unsigned int &rx, unsigned int &ry,
    const unsigned int t_rx, const unsigned int t_ry,
    const unsigned int num_channels)
{
    if (rx == t_rx && ry == t_ry)
        return;

    assert(t_rx > 1 && t_ry > 1);

    std::vector<float> t_tex(t_rx * t_ry * num_channels);
    if (rx == 1 && ry == 1)
    {
        // common case: 1x1 Fresnel texture
        for (unsigned int i = 0; i < t_rx * t_ry * num_channels; i += num_channels) {
            for (unsigned int j = 0; j < num_channels; ++j)
                t_tex[i + j] = tex[j];
        }
    }
    else
    {
        const float step_y = 1.0f / (float)(t_ry - 1);
        const float step_x = 1.0f / (float)(t_rx - 1);

        for (unsigned int y = 0; y < t_ry; ++y) {
            const float v = (float)y * step_y;
            for (unsigned int x = 0; x < t_rx; ++x) {
                const float u = (float)x * step_x;

                const unsigned int t_idx = (y * t_rx + x) * num_channels;
                tex_lookup(num_channels, &t_tex[t_idx], u, v, rx, ry, &tex[0]);
            }
        }
    }

    tex.swap(t_tex);
    rx = t_rx;
    ry = t_ry;
}

enum Clearcoat_model_variant {
    CLEARCOAT_DEFAULT,                      // AXF_TRANSMISSION_VARIANT_DEFAULT
    CLEARCOAT_NO_SOLID_ANGLE_COMPRESSION,   // AXF_TRANSMISSION_VARIANT_NO_SOLID_ANGLE_COMPRESSION
    CLEARCOAT_DSPBR2020X                    // AXF_TRANSMISSION_VARIANT_DSPBR2020X
};

// returns false in case there is no clearcoat
static bool get_clearcoat_params(
    bool &refractive,
    Clearcoat_model_variant &variant,
    const AXF::TextureDecoder *const tex_decoder)
{
    refractive = false;
    variant = CLEARCOAT_DEFAULT;

    const bool has_clearcoat = tex_decoder->hasClearCoat();
    if (has_clearcoat) {
        char type_key[AXF::AXF_MAX_KEY_SIZE];
        if (!tex_decoder->getClearCoatTransmissionModelType(type_key, AXF::AXF_MAX_KEY_SIZE)) {
            assert(!"clearcoat type retrieval should not fail");
            return false;
        }

        refractive = strcmp(type_key, AXF_TYPEKEY_TRANSMISSION_REFRACTIVE_DIRAC) == 0;

        if (!tex_decoder->getClearCoatTransmissionModelVariant(type_key, AXF::AXF_MAX_KEY_SIZE)) {
            assert(!"clearcoat variant retrieval should not fail");
            return false;
        }

        if (strcmp(type_key, AXF_TRANSMISSION_VARIANT_NO_SOLID_ANGLE_COMPRESSION) == 0)
            variant = CLEARCOAT_NO_SOLID_ANGLE_COMPRESSION;
        else if (strcmp(type_key, AXF_TRANSMISSION_VARIANT_DSPBR2020X) == 0)
            variant = CLEARCOAT_DSPBR2020X;
        // else variant = CLEARCOAT_DEFAULT;
    }

    return has_clearcoat;
}

bool Axf_reader::check_texture_type(
    const unsigned int channels,
    const Input_texture_type type,
    const char *tex_name,
    Axf_impexp_state* impexp_state)
{
    const char *channel_error = nullptr;
    if (type == INPUT_TEXTURE_COLOR) {
        if (m_wavelengths.empty()) {
            if (channels != 3)
                channel_error = "RGB";
        } else {
            if (channels != m_wavelengths.size())
                channel_error = "spectrum";
        }
    } else if (type == INPUT_TEXTURE_NORMAL) {
        if (channels != 3)
            channel_error = "normalmap";
    } else if (type == INPUT_TEXTURE_FLOAT) {
        if (channels != 1)
            channel_error = "float";
    } else if (type == INPUT_TEXTURE_FLOAT12) {
        if (channels > 2)
            channel_error = "float/float2";
    }

    if (channel_error)
    {
        ostringstream str;
        str << "Encountered texture \"" << tex_name
            << "\" with unexpected number of channels in AxF material " << m_material_name
            << ", expected type: " << channel_error;
        Axf_importer::report_message(
            6021,
            mi::base::MESSAGE_SEVERITY_ERROR,
            str.str(), impexp_state);
        return false;
    }
    return true;
}

static bool is_unmodified_normal(
    const std::vector<float> tex_buffer)
{
    return
        (tex_buffer.size() == 3) &&
        (tex_buffer[0] == 0.0f) &&
        (tex_buffer[1] == 0.0f) &&
        (tex_buffer[2] == 1.0f);
}

bool Axf_reader::handle_carpaint_representation(
    AXF::AXF_REPRESENTATION_HANDLE rep_h,
    const int conversion_flags,
    AXF::TextureDecoder *tex_decoder,
    Axf_impexp_state* impexp_state)
{
    // initialize some fields (they should _always_ be set later on, since all properties
    // and textures should be present in the AxF, but let's play safe)
    m_ior = 1.5f;
    m_ct_diffuse = 0.0f;
    for (int i = 0; i < 3; ++i) {
        m_ct_coeffs[i] = 0.0f;
        m_ct_f0s[i] = 0.0f;
        m_ct_spreads[i] = 0.0f;
    }
    m_brdf_colors_scale = 1.0f;
    m_brdf_colors_2d.clear();
    m_brdf_colors_2d_rx = m_brdf_colors_2d_ry = 1;
    m_flake_intensity_scale = 0.0f;
    m_flake_uvw_scale[0] = m_flake_uvw_scale[1] = m_flake_uvw_scale[2] = 1.0f;

    // query refractiveness of clearcoat
    Clearcoat_model_variant clearcoat_variant;
    const bool has_clearcoat = get_clearcoat_params(m_refractive_clearcoat, clearcoat_variant, tex_decoder);
    if (has_clearcoat) {
        // refractive clearcoat should never have the correct eta scaling, Dassault mode should not occur
        assert((m_refractive_clearcoat && (clearcoat_variant == CLEARCOAT_NO_SOLID_ANGLE_COMPRESSION)) ||
               (!m_refractive_clearcoat && (clearcoat_variant == CLEARCOAT_DEFAULT)));
        get_property(
            &m_ior, sizeof(m_ior),
            tex_decoder, AXF::TYPE_FLOAT,
            AXF_CARPAINT2_PROPERTY_CC_IOR, impexp_state,
            /*report_error=*/true);
    } else
        m_ior = 1.0f;

    get_property(
        &m_ct_diffuse, sizeof(m_ct_diffuse),
        tex_decoder, AXF::TYPE_FLOAT,
        AXF_CARPAINT2_PROPERTY_BRDF_CT_DIFFUSE, impexp_state,
        /*report_error=*/true);
    std::vector<float> fvalues;
    get_array_property(
        fvalues,
        tex_decoder, AXF::TYPE_FLOAT_ARRAY,
        AXF_CARPAINT2_PROPERTY_BRDF_CT_COEFFS, impexp_state,
        /*report_error=*/true);
    for (size_t i = 0; i < std::min(fvalues.size(), size_t(3)); ++i)
        m_ct_coeffs[i] = fvalues[i];

    const size_t size = fvalues.size();
    if (size > 3)
        Axf_importer::report_message(6010, mi::base::MESSAGE_SEVERITY_ERROR,
            "more than 3 BRDF lobes for carpaint are not supported", impexp_state);

    // the Cook-Torrance model used by X-Rite is not normalized (missing a division by four)
    m_ct_coeffs[0] *= 4.0f;
    m_ct_coeffs[1] *= 4.0f;
    m_ct_coeffs[2] *= 4.0f;

    get_array_property(
        fvalues,
        tex_decoder, AXF::TYPE_FLOAT_ARRAY,
        AXF_CARPAINT2_PROPERTY_BRDF_CT_F0S, impexp_state,
        /*report_error=*/true);
    for (size_t i = 0; i < std::min(fvalues.size(), size_t(3)); ++i)
        m_ct_f0s[i] = fvalues[i];
    assert(size == fvalues.size());

    get_array_property(
        fvalues,
        tex_decoder, AXF::TYPE_FLOAT_ARRAY,
        AXF_CARPAINT2_PROPERTY_BRDF_CT_SPREADS, impexp_state,
        /*report_error=*/true);
    for (size_t i = 0; i < std::min(fvalues.size(), size_t(3)); ++i)
        m_ct_spreads[i] = fvalues[i];
    assert(size == fvalues.size());

#if 0
    int max_theta_i = 0;
    get_property(
        &max_theta_i, sizeof(max_theta_i),
        tex_decoder, AXF::TYPE_INT,AXF_CARPAINT2_PROPERTY_FLAKES_MAX_THETAI , impexp_state,
        /*report_error=*/true);
#endif
    int num_theta_i = 0;
    get_property(
        &num_theta_i, sizeof(num_theta_i),
        tex_decoder, AXF::TYPE_INT,AXF_CARPAINT2_PROPERTY_FLAKES_NUM_THETAI , impexp_state,
        /*report_error=*/true);
    int num_theta_f = 0;
    get_property(
        &num_theta_f, sizeof(num_theta_f),
        tex_decoder, AXF::TYPE_INT, AXF_CARPAINT2_PROPERTY_FLAKES_NUM_THETAF , impexp_state,
        /*report_error=*/true);

    std::vector<int> slice_lut;
    get_array_property(
        slice_lut,
        tex_decoder, AXF::TYPE_INT_ARRAY, AXF_CARPAINT2_PROPERTY_FLAKES_THETAFI_SLICE_LUT,
        impexp_state, /*report_error=*/true);

    assert(slice_lut.size() >= (size_t)num_theta_f);

    std::map<std::string, std::string> tex_to_param;
    for (int i=0, cnt=tex_decoder->getNumTextures(); i<cnt; ++i) {
        char tex_name[AXF::AXF_MAX_KEY_SIZE];
        if (!tex_decoder->getTextureName(i, tex_name, AXF::AXF_MAX_KEY_SIZE)) {
            string msg = string("Cannot retrieve texture name from AxF material ") + m_material_name;
            Axf_importer::report_message(6011, mi::base::MESSAGE_SEVERITY_ERROR,
                msg, impexp_state);
            return false;
        }

        const bool is_brdf_colors = strcmp(tex_name, AXF_CARPAINT2_TEXTURE_NAME_BRDF_COLORS) == 0;
        const bool is_normal = strcmp(tex_name, AXF_CARPAINT2_TEXTURE_NAME_CLEARCOAT_NORMAL) == 0;
        const bool is_flake_btf = strcmp(tex_name, AXF_CARPAINT2_TEXTURE_NAME_BTF_FLAKES) == 0;


        if (!is_brdf_colors && !is_normal && !is_flake_btf)
            continue;

        const int mip_level = 0;
        int width, height, depth, channels, datatype;
        tex_decoder->getTextureSize(i, mip_level, width, height, depth, channels, datatype);

        std::vector<float> tex_buffer(width * height * depth * channels);
        tex_decoder->getTextureData(i, mip_level, AXF::TYPE_FLOAT, tex_buffer.data());

        Input_texture_type expected_type;
        if (is_normal)
            expected_type = INPUT_TEXTURE_NORMAL;
        else if (is_flake_btf)
            expected_type = INPUT_TEXTURE_RGB; // flake BTF should always be RGB, even for spectral data
        else
            expected_type = INPUT_TEXTURE_COLOR;

        if (!check_texture_type(channels, expected_type, tex_name, impexp_state))
            return false;

        if (is_normal)
        {
            if (is_unmodified_normal(tex_buffer))
                continue;

            // transform normal map values from [-1,1] to [0,1]
            for (size_t i = 0; i < tex_buffer.size(); ++i)
                tex_buffer[i] = tex_buffer[i] * 0.5f + 0.5f;

            const string texture_name = write_texture(
                impexp_state,
                tex_name,
                tex_buffer,
                width,
                height,
                channels,
                TEXTURE_NORMAL);

            if (texture_name.empty())
            {
                const string msg =
                    string("Error while creating textures from AxF material ") + m_material_name;
                Axf_importer::report_message(
                    6030, mi::base::MESSAGE_SEVERITY_ERROR,
                    msg, impexp_state);
                return false;
            }
            else
                tex_to_param.insert(make_pair(texture_name, "clearcoat_normal_texture"));
        }
        else if (is_flake_btf)
        {
#ifdef DUMP_TEXTURES
            write_texture(
                    impexp_state, tex_name, tex_buffer, width, height * depth,
                    channels, TEXTURE_RGB);
#endif
            initialize_carpaint_flakes(
                m_flake_importance_data,
                m_flake_intensity_scale,
                m_flake_uvw_scale,
                m_flake_orientation_falloff,
                width, height,
                tex_buffer, num_theta_i, num_theta_f, slice_lut);
        }
        else
        {
            assert(is_brdf_colors);

            // for highest compatibility, we can bake the BRDF coloring into the measured BRDF data
            // block (for 1:1 match with refracting clear coat, where the conversion of the
            // coloring data cannot be perfect), but for spectral data this is not possible
            const bool bake_brdf_colors = m_wavelengths.empty();

            if (bake_brdf_colors) {
                // copy data for later 1:1 use in sub-clearcoat bsdf baking
                m_brdf_colors_2d = tex_buffer;
                m_brdf_colors_2d_rx = width;
                m_brdf_colors_2d_ry = height;
            } else {
                m_brdf_colors_2d.clear();
                m_brdf_colors_2d_rx = m_brdf_colors_2d_ry = 0;
            }

            // for refractive clearcoat we need to change the domain to non-refracted directions
            if (m_refractive_clearcoat)
                recode_brdf_colors(tex_buffer, width, height, channels, m_ior);

            // AxF carpaint uses a 2d texture to tint the sub-clearcoat BRDF, parameterized by
            // x = acos(dot(normal, half)) * (2/pi)
            // and
            // y = acos(dot(direction, half)) * (2/pi).
            //
            // For MDL measured_factor's texture input:
            // - we need swap x and y (as the usage is exactly opposite to above)
            // - we need to turn the image upside down
            {
                std::vector<float> tex_buffer2(tex_buffer.size());
                size_t pos = 0;
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {

                        const size_t idx = ((width - x - 1) * height + y) * channels;
                        for (int c = 0; c < channels; ++c) {
                            tex_buffer2[idx + c] = tex_buffer[pos++];
                        }
                    }
                }
                tex_buffer.swap(tex_buffer2);
                std::swap(width, height);
            }

            //!! sometimes the BRDFcolors texture is not <= 1.0...
            //   according to X-Rite the colors are "normalized" and should always be valid in
            //   combination with the BRDF weights -> so we can apply the scale there to enforce
            //   a "valid" [0,1] texture
            //   (we only do that if we don't bake the BRDF colors into a measured BRDF, as there
            //    the original data copy in m_brdf_colors_2d is used for baking)
            m_brdf_colors_scale = 1.0f;
            if (!bake_brdf_colors) {
                float max_val = 0.0f;
                for (size_t i = 0; i < tex_buffer.size(); ++i)
                    max_val = std::max(tex_buffer[i], max_val);
                if (max_val > 1.0f)
                {
                    m_brdf_colors_scale = max_val;

                    const float scale = 1.0f / max_val;
                    for (size_t i = 0; i < tex_buffer.size(); ++i)
                        tex_buffer[i] *= scale;
                }
            }

            const string texture_name = write_texture(
                impexp_state, tex_name, tex_buffer, width, height,
                channels, m_wavelengths.empty() ? TEXTURE_RGB : TEXTURE_SPECTRAL);
            tex_to_param.insert(make_pair(texture_name, "brdf_colors_2d"));
        }
    }

    create_variant(
            impexp_state, "carpaint",
            tex_decoder->getWidthMM(), tex_decoder->getHeightMM(),
            tex_to_param);

    return true;
}

bool Axf_reader::handle_volumetric_representation(
    AXF::AXF_REPRESENTATION_HANDLE rep_h,
    const int conversion_flags,
    AXF::TextureDecoder *tex_decoder,
    Axf_impexp_state* impexp_state)
{
    // initialize some fields (they should _always_ be set later on, since all properties
    // and textures should be present in the AxF, but let's play safe)
    m_ior = 1.0f;
    const unsigned int col_size = m_wavelengths.empty() ? 3 : m_wavelengths.size();
    m_sigma_a.resize(col_size, 0.0f);
    m_sigma_s.resize(col_size, 0.0f);
    m_phasefunc_g = 0.0f;

    // get IOR
    get_property(
        &m_ior, sizeof(m_ior),
        tex_decoder, AXF::TYPE_FLOAT,
        AXF_VOLUMETRIC_PROPERTY_NAME_IOR, impexp_state,
        /*report_error=*/true);


    for (int i=0, cnt=tex_decoder->getNumTextures(); i<cnt; ++i) {
        char tex_name[AXF::AXF_MAX_KEY_SIZE];
        if (!tex_decoder->getTextureName(i, tex_name, AXF::AXF_MAX_KEY_SIZE)) {
            string msg = string("Cannot retrieve texture name from AxF material ") + m_material_name;
            Axf_importer::report_message(6051, mi::base::MESSAGE_SEVERITY_ERROR,
                msg, impexp_state);
            return false;
        }

        const bool is_sigma_a = strcmp(tex_name, AXF_VOLUMETRIC_TEXTURE_NAME_SIGMAA) == 0;
        const bool is_sigma_s = strcmp(tex_name, AXF_VOLUMETRIC_TEXTURE_NAME_SIGMAS) == 0;
        const bool is_phasefunc_g = strcmp(tex_name, AXF_VOLUMETRIC_TEXTURE_NAME_PHASEFUNCG) == 0;

        if (!is_sigma_a && !is_sigma_s && !is_phasefunc_g)
            continue;

        const int mip_level = 0;
        int width, height, depth, channels, datatype;
        tex_decoder->getTextureSize(i, mip_level, width, height, depth, channels, datatype);

        std::vector<float> tex_buffer(width * height * depth * channels);
        tex_decoder->getTextureData(i, mip_level, AXF::TYPE_FLOAT, tex_buffer.data());

        if (is_phasefunc_g) {
            assert(channels == 1);
            m_phasefunc_g = tex_buffer[0];
        }
        else {
            assert(channels == col_size);
            memcpy(is_sigma_a ? m_sigma_a.data() : m_sigma_s.data(), tex_buffer.data(), col_size * sizeof(float));
        }
    }

    // data from AxF is per millimeter, we use per meter
    for (unsigned int i = 0; i < col_size; ++i) {
        m_sigma_a[i] *= 1000.0f;
        m_sigma_s[i] *= 1000.0f;
    }

    // avoid zero-padding for spectral coefficients
    if (!m_wavelengths.empty()) {
        expand_spectrum(m_sigma_a);
        expand_spectrum(m_sigma_s);
    }

    std::map<std::string, std::string> tex_to_param;
    create_variant(
            impexp_state, "volumetric",
            tex_decoder->getWidthMM(), tex_decoder->getHeightMM(),
            tex_to_param);

    return true;
}


bool Axf_reader::handle_representation(
    AXF::AXF_REPRESENTATION_HANDLE rep_h,
    const int conversion_flags,
    Axf_impexp_state* impexp_state)
{
    char typekey[AXF::AXF_MAX_KEY_SIZE];
    if (!AXF::axfGetRepresentationClass(rep_h, typekey, AXF::AXF_MAX_KEY_SIZE)) {
        const string msg = string("No representation type key found for AxF material ") + m_material_name;
        Axf_importer::report_message(6006, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return false;
    }

    if (strcmp(typekey, AXF_REPRESENTATION_CLASS_CARPAINT2) == 0)
        m_type = REP_CARPAINT;
    else if (strcmp(typekey, AXF_REPRESENTATION_CLASS_SVBRDF) == 0)
        m_type = REP_SVBRDF;
    else if (strcmp(typekey, AXF_REPRESENTATION_CLASS_VOLUMETRIC) == 0)
        m_type = REP_VOLUMETRIC;
    else if (strcmp(typekey, AXF_REPRESENTATION_CLASS_EPSVBRDF) == 0) {
        // EPSVBRDF not directly handled yet, need conversion to regular SVBRDF
        assert((conversion_flags & AXF::CONVERT_EPSVBRDF_TO_SVBRDF) != 0);
        m_type = REP_SVBRDF;
    }
    else
        // no error if there are other representations, we just cannot handle these
        return false;

    // handle preview image
    handle_preview_images(rep_h, impexp_state);

    const bool has_spectral_representation = AXF::axfGetRepresentationIsSpectral(rep_h);
    const Color_representation color_rep = impexp_state->get_color_rep();

    bool success = true;

    // first retrieve RGB representation, then spectral one
    unsigned int reps_converted = 0;
    for (unsigned int i = 0; (i < 2) && success; ++i) {

        if (i == 0) {
            // only want spectral (and we have that)? -> skip RGB
            if (has_spectral_representation && (color_rep == COLOR_REP_SPECTRAL))
                continue;

            m_wavelengths.clear();
            m_spectral_tex_meta_data.clear();

        } else {
            // only want RGB (or no spectral available)? -> skip spectral
            if (!has_spectral_representation || (color_rep == COLOR_REP_RGB))
                break;

            float lambda_min, lambda_max;
            int num_lambda;
            if (!axfGetRepresentationWavelengthRange(rep_h, lambda_min, lambda_max, num_lambda) ||
                num_lambda <= 0) {
                success = false;
                break;
            }

            m_wavelengths.resize(num_lambda);
            const float lambda_step = (num_lambda > 1) ? ((lambda_max - lambda_min) / float(num_lambda - 1)) : 0.0f;
            for (int j = 0; j < num_lambda; ++j)
                m_wavelengths[j] = lambda_min + j * lambda_step;

            compute_spectral_texture_metadata(
                m_spectral_tex_meta_data, m_wavelengths);
        }

        Scope_handler<AXF::TextureDecoder> tex_decoder(
            (i == 1) ?
            AXF::TextureDecoder::createSpectral(
                rep_h, m_wavelengths.data(), m_wavelengths.size(), AXF::ORIGIN_TOPLEFT, conversion_flags) :
            AXF::TextureDecoder::create(
                rep_h, m_target_color_space, AXF::ORIGIN_TOPLEFT, conversion_flags));

        if (!tex_decoder) {
            const string msg = string("Cannot create texture decoder for AxF material ") + m_material_name;
            Axf_importer::report_message(
                6010, mi::base::MESSAGE_SEVERITY_ERROR, msg, impexp_state);
            return false;
        }

        assert(success);
        m_skip_non_color_maps = reps_converted > 0;
        if (m_type == REP_CARPAINT)
            success = handle_carpaint_representation(
                rep_h, conversion_flags, tex_decoder.get(), impexp_state);
        else if (m_type == REP_VOLUMETRIC)
            success = handle_volumetric_representation(
                rep_h, conversion_flags, tex_decoder.get(), impexp_state);
        else {
            assert(m_type == REP_SVBRDF);
            success = handle_svbrdf_representation(
                rep_h, conversion_flags, tex_decoder.get(), impexp_state);
        }
        if (success)
            ++reps_converted;
    }
    return success && (reps_converted > 0);
}

bool Axf_reader::handle_svbrdf_representation(
    AXF::AXF_REPRESENTATION_HANDLE rep_h,
    const int conversion_flags,
    AXF::TextureDecoder *tex_decoder,
    Axf_impexp_state* impexp_state)
{
    int rep_version_major, rep_version_minor, rep_version_rev;
    axfGetRepresentationVersion( rep_h, rep_version_major, rep_version_minor, rep_version_rev);


    AXF::AXF_REPRESENTATION_HANDLE spec_model_h =
            AXF::axfGetSvbrdfSpecularModelRepresentation(rep_h);
    AXF::AXF_REPRESENTATION_HANDLE diff_model_h =
            AXF::axfGetSvbrdfDiffuseModelRepresentation(rep_h);

    if (!spec_model_h && !diff_model_h) {
        const string msg = string("Failed to retrieve SVBRDF model for AxF material ") + m_material_name;
        Axf_importer::report_message(6007, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return false;
    }

    // handle specular (really: glossy) component
    if (spec_model_h) {

        // get the typekey of specular SVBRDF
        char typekey[AXF::AXF_MAX_KEY_SIZE];
        if (!AXF::axfGetRepresentationTypeKey(spec_model_h, typekey, AXF::AXF_MAX_KEY_SIZE)) {
            ostringstream str;
            str << "No specular SVBRDF representation type key found for AxF material " << m_material_name;
            Axf_importer::report_message(6008, mi::base::MESSAGE_SEVERITY_ERROR,
                                         str.str(), impexp_state);
            return false;
        }

        m_glossy_brdf_type = BRDF_TYPE_UNKNOWN;
        m_fresnel_type = FRESNEL_NONE;
        bool is_anisotropic = false;
        bool has_fresnel = false;
        char variant[AXF::AXF_MAX_KEY_SIZE];
        if (!AXF::axfGetSvbrdfSpecularModelVariant(
                spec_model_h,variant,AXF::AXF_MAX_KEY_SIZE,is_anisotropic,has_fresnel)) {
            ostringstream str;
            str << "Failed to retrieve specular model variant for AxF material " << m_material_name;
            Axf_importer::report_message(6016,mi::base::MESSAGE_SEVERITY_ERROR,
                                         str.str(),impexp_state);
            return false;
        }
        m_isotropic = !is_anisotropic;

        if (!strcmp(typekey,AXF_TYPEKEY_SVBRDF_SPECULAR_WARD)) {
            if (!strcmp(variant,AXF_SVBRDF_SPECULAR_WARD_VARIANT_GEISLERMORODER))
                m_glossy_brdf_type = BRDF_WARD_GEISLERMORODER;
            // else if (!strcmp(variant,AXF_SVBRDF_SPECULAR_WARD_VARIANT_DUER))
            // else if (!strcmp(variant,AXF_SVBRDF_SPECULAR_WARD_VARIANT_WARD))
            else {
                m_glossy_brdf_type = BRDF_WARD_GEISLERMORODER;
                ostringstream str;
                str << "Unsupported or unknown specular BRDF model variant " << typekey
                    << " (" << variant << ')'
                    << " for AxF material " << m_material_name;
                Axf_importer::report_message(6017,
                                             mi::base::MESSAGE_SEVERITY_WARNING,
                                             str.str(),impexp_state);
            }
        }
        else if (!strcmp(typekey,AXF_TYPEKEY_SVBRDF_SPECULAR_COOKTORRANCE)) {
            // no variants
            m_glossy_brdf_type = BRDF_COOKTORRANCE;
        }
        else if (!strcmp(typekey,AXF_TYPEKEY_SVBRDF_SPECULAR_GGX)) {
            // no variants
            m_glossy_brdf_type = BRDF_GGX;
        }

        if (m_glossy_brdf_type == BRDF_TYPE_UNKNOWN) {
            ostringstream str;
            str << "Unsupported or unknown specular BRDF model " << typekey
                << " (" << variant << ')'
                << " for AxF material " << m_material_name;
            Axf_importer::report_message(6020, mi::base::MESSAGE_SEVERITY_ERROR,
                                         str.str(), impexp_state);
            return false;
        }

        // handle Fresnel type
        if (has_fresnel) {

            char fresnel_type[AXF::AXF_MAX_KEY_SIZE];
            if (!axfGetSvbrdfSpecularFresnelVariant(
                    spec_model_h, fresnel_type, AXF::AXF_MAX_KEY_SIZE))
            {
                ostringstream str;
                str << "Failed to retrieve Fresnel type " << " for AxF material " << m_material_name;
                Axf_importer::report_message(6022,
                                             mi::base::MESSAGE_SEVERITY_WARNING,
                                             str.str(),impexp_state);
                return false;
            }
            if (strcmp(fresnel_type, AXF_SVBRDF_FRESNEL_VARIANT_SCHLICK) == 0)
                m_fresnel_type = FRESNEL_SCHLICK;
            else if (strcmp(fresnel_type, AXF_SVBRDF_FRESNEL_VARIANT_SCHLICK_COLORED) == 0) {
                // should only appear for EPSVBRDF and then be converted to non-colored SVBRDF Schlick
                assert((conversion_flags & AXF::CONVERT_EPSVBRDF_TO_SVBRDF) != 0);
                m_fresnel_type = FRESNEL_SCHLICK;
            }
#if 0
            //!! according to X-Rite, this mode is not used
            else if (strcmp(fresnel_type, AXF_SVBRDF_FRESNEL_VARIANT_FRESNEL) == 1)
                m_fresnel_type = FRESNEL_FRESNEL;
#endif
            else
            {
                ostringstream str;
                str << "Unsupported Fresnel type " << fresnel_type
                    << " for AxF material " << m_material_name;
                Axf_importer::report_message(6023,
                                             mi::base::MESSAGE_SEVERITY_WARNING,
                                             str.str(),impexp_state);
                return false;
            }
        }
    }

    // retrieve clearcoat info
    bool refractive_clearcoat;
    Clearcoat_model_variant clearcoat_variant;
    m_has_clearcoat = get_clearcoat_params(refractive_clearcoat, clearcoat_variant, tex_decoder);
    if (m_has_clearcoat && refractive_clearcoat) {
        //!! we should no longer encounter this case, since the texture decoder supports
        //   conversion by now and we only ask for representations without refraction
        assert(!"should not get refractive clearcoat");
        ostringstream str;
        str << "Refractive clearcoat in SVBRDF representation is not supported"
            << " for AxF material " << m_material_name;
        Axf_importer::report_message(
            6024, mi::base::MESSAGE_SEVERITY_ERROR,
            str.str(), impexp_state);
        return false;
    }
    // we don't support this yet and don't ask for representations that need it
    assert(clearcoat_variant != CLEARCOAT_DSPBR2020X);

    // remember some textures, to check for energy conservation of badly-fitted materials
    // and "repair" that below
    // (high specular color value with low Fresnel, the X-Rite fitter does not handle this nicely)
    unsigned int diffuse_tex_rx = 0, diffuse_tex_ry = 0;
    std::vector<float> diffuse_tex;
    unsigned int fresnel_tex_rx = 0, fresnel_tex_ry = 0;
    std::vector<float> fresnel_tex;
    std::string fresnel_tex_name;
    unsigned int specular_color_tex_rx = 0, specular_color_tex_ry = 0;
    std::vector<float> specular_color_tex;
    std::string specular_color_tex_name;
    unsigned int trans_color_tex_rx = 0, trans_color_tex_ry = 0;
    std::vector<float> trans_color_tex;

    // map from AxF tex to MDL param name
    map<string, string> tex_to_param;
    for (int i=0, cnt=tex_decoder->getNumTextures(); i<cnt; ++i) {
        char tex_name[AXF::AXF_MAX_KEY_SIZE];
        if (!tex_decoder->getTextureName(i, tex_name, AXF::AXF_MAX_KEY_SIZE)) {
            string msg = string("Cannot retrieve texture name from AxF material ") + m_material_name;
            Axf_importer::report_message(6011, mi::base::MESSAGE_SEVERITY_ERROR,
                msg, impexp_state);
            return false;
        }

        const bool is_normal =
            strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_NORMAL) == 0;
        const bool is_clearcoat_normal =
                strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_CLEARCOAT_NORMAL) == 0;
        const bool is_clearcoat_ior =
                strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_CLEARCOAT_IOR) == 0;
#if 0 // disappeared with AxF 1.8 SDK
        const bool is_clearcoat_color =
                strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_CLEARCOAT_COLOR) == 0;
#else
        const bool is_clearcoat_color = false;
#endif

        const bool is_transmission_color =
                strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_TRANSMISSION_COLOR) == 0;

        const int mip_level = 0;
        int width, height, depth, channels, datatype;
        tex_decoder->getTextureSize(i, mip_level, width, height, depth, channels, datatype);
        assert(depth == 1);

        // height map has become a feature in AxF 1.4
        // (but we have seen it present in older AxF files, too - where we simply ignore it)
        const bool is_displacement = !strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_HEIGHT);
        if (is_displacement && (rep_version_major < 2) && (rep_version_minor < 4))
            continue;

        // same for "alpha"
        const bool is_alpha = !strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_ALPHA);
        if (is_alpha && (rep_version_major < 2) && (rep_version_minor < 4))
            continue;

        // anisotropic lobe (rgb) texture needs to be split (and
        // potentially transformed) into several scalar textures
        const bool is_specular_lobe = !strcmp(tex_name,AXF_SVBRDF_TEXTURE_NAME_SPECULAR_LOBE);
        const bool needs_separation = (is_specular_lobe && channels == 2);

        const bool is_fresnel = !strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_FRESNEL);
        const bool is_specular_color = !strcmp(tex_name, AXF_SVBRDF_TEXTURE_NAME_SPECULAR_COLOR);
        const bool is_diffuse_color = !strcmp(tex_name,AXF_SVBRDF_TEXTURE_NAME_DIFFUSE_COLOR);
        const bool is_aniso_rotation = !strcmp(tex_name,AXF_SVBRDF_TEXTURE_NAME_ANISO_ROTATION);

        Input_texture_type expected_type = INPUT_TEXTURE_FLOAT;
        if (is_diffuse_color || is_specular_color || is_clearcoat_color || is_transmission_color)
            expected_type = INPUT_TEXTURE_COLOR;
        else if (is_clearcoat_normal || is_normal)
            expected_type = INPUT_TEXTURE_NORMAL;
        else if (is_alpha || is_displacement || is_aniso_rotation || is_clearcoat_ior ||
                 is_fresnel) // should not yet encounter color Fresnel
            expected_type = INPUT_TEXTURE_FLOAT;
        else if (is_specular_lobe)
            expected_type = INPUT_TEXTURE_FLOAT12;
        else
            assert(!"unhandled texture");

        if (!check_texture_type(channels, expected_type, tex_name, impexp_state))
            return false;

        std::vector<float> tex_buffer(width * height * depth * channels);
        tex_decoder->getTextureData(i, mip_level, AXF::TYPE_FLOAT, tex_buffer.data());


        if (m_fresnel_type == FRESNEL_SCHLICK && (is_fresnel || is_specular_color))
        {
            if (is_fresnel) {
                fresnel_tex_rx = width;
                fresnel_tex_ry = height;
                fresnel_tex.swap(tex_buffer);
                fresnel_tex_name = tex_name;
            } else {
                specular_color_tex_rx = width;
                specular_color_tex_ry = height;
                specular_color_tex.swap(tex_buffer);
                specular_color_tex_name = tex_name;
            }
            // don't store those textures yet, may need to repair energy conservation below
            continue;
        }

        if (is_transmission_color)
        {
            trans_color_tex_rx = width;
            trans_color_tex_ry = height;
            trans_color_tex.swap(tex_buffer);
            continue;
        }

        bool texture_error = false;
        if (!needs_separation)
        {
            Texture_type texture_type = TEXTURE_SCALAR;
            if (is_diffuse_color || is_specular_color || is_clearcoat_color) {
                texture_type = m_wavelengths.empty() ? TEXTURE_RGB : TEXTURE_SPECTRAL;
            }

            if (is_aniso_rotation) {
                // source values are in [-pi/2,pi/2], we want [0,1].
                for (size_t i = 0; i < tex_buffer.size(); ++i) {
                    const float angle =
                        tex_buffer[i] >= 0.0f ? tex_buffer[i] : (float)(2.0 * M_PI) + tex_buffer[i];
                    tex_buffer[i] = angle * (float)(0.5 / M_PI);
                }
            }

            if (is_normal || is_clearcoat_normal)
            {
                if (is_unmodified_normal(tex_buffer))
                    continue;

                // transform normal map values from [-1,1] to [0,1]
                for (size_t i = 0; i < tex_buffer.size(); ++i)
                    tex_buffer[i] = tex_buffer[i] * 0.5f + 0.5f;

                texture_type = TEXTURE_NORMAL;
            }

            const string texture_name = write_texture(
                impexp_state,
                tex_name,
                tex_buffer,
                width,
                height,
                channels,
                texture_type);

            texture_error = texture_name.empty();

            // we keep the tex db name together with its mdl param name
            if (is_normal)
                tex_to_param.insert(make_pair(texture_name, "normal_texture"));
            else if (is_diffuse_color)
                tex_to_param.insert(make_pair(texture_name, "diffuse_texture"));
            else if (is_specular_color)
                tex_to_param.insert(make_pair(texture_name, "specular_texture"));
            else if (is_aniso_rotation)
                tex_to_param.insert(make_pair(texture_name, "specular_brdf_texture_rotation"));
            else if (is_specular_lobe)
                tex_to_param.insert(make_pair(texture_name, "specular_brdf_texture_u"));
            else if (is_fresnel)
                tex_to_param.insert(make_pair(texture_name, "specular_brdf_texture_fresnel"));
            else if (is_clearcoat_normal)
                tex_to_param.insert(make_pair(texture_name, "clearcoat_normal_texture"));
            else if (is_clearcoat_ior)
                tex_to_param.insert(make_pair(texture_name, "clearcoat_ior_texture"));
            else if (is_clearcoat_color)
                tex_to_param.insert(make_pair(texture_name, "clearcoat_color_texture"));
            else if (is_displacement)
                tex_to_param.insert(make_pair(texture_name, "height_texture"));
            else if (is_alpha)
                tex_to_param.insert(make_pair(texture_name, "alpha_texture"));
            else {
                ostringstream str;
                str << "Ignoring unknown texture " << tex_name
                    << " in AxF material " << m_material_name;
                Axf_importer::report_message(6020,
                                             mi::base::MESSAGE_SEVERITY_WARNING,
                                             str.str(), impexp_state);
            }
        }
        else
        {
            // process and split incoming texture
            assert(channels == 2);

            std::vector<float> tex_buffer_a(width * height);
            std::vector<float> tex_buffer_b(width * height);
            const char* tex_input_a = "specular_brdf_texture_u";
            const char* const tex_input_b = "specular_brdf_texture_v";
            switch (m_glossy_brdf_type) {
                case BRDF_COOKTORRANCE:
                    // no conversion necessary
                    // TODO Fresnel superseded by dedicated texture?
                    split_texture(tex_buffer_a,tex_buffer_b,tex_buffer);
                    tex_input_a = "specular_brdf_texture_f0";
                    break;
                case BRDF_WARD_GEISLERMORODER:
                case BRDF_GGX:
                    split_texture(tex_buffer_a,tex_buffer_b,tex_buffer);
                    break;
                default: {
                    ostringstream str;
                    str << "Ignoring texture " << tex_name
                        << " for unsupported specular BRDF in material " << m_material_name;
                    Axf_importer::report_message(6022,
                                                 mi::base::MESSAGE_SEVERITY_WARNING,
                                                 str.str(), impexp_state);
                    }
                    continue;
            }

            // generate scalar result textures
            const string texture_name_a = write_texture(
                    impexp_state,
                    string(tex_name) + string("a"), tex_buffer_a, width, height, 1,
                    TEXTURE_SCALAR);
            const string texture_name_b = write_texture(
                    impexp_state,
                    string(tex_name) + string("b"), tex_buffer_b, width, height, 1,
                    TEXTURE_SCALAR);

            texture_error = texture_name_a.empty() || texture_name_b.empty();
            if (!texture_error) {
                tex_to_param.insert(make_pair(texture_name_a,tex_input_a));
                tex_to_param.insert(make_pair(texture_name_b,tex_input_b));
            }
        }

        if (m_fresnel_type == FRESNEL_SCHLICK && is_diffuse_color) {
            diffuse_tex.swap(tex_buffer);
            diffuse_tex_rx = width;
            diffuse_tex_ry = height;
        }


        if (texture_error) {
            const string msg =
                string("Error while creating textures from AxF material ") + m_material_name;
            Axf_importer::report_message(
                6012, mi::base::MESSAGE_SEVERITY_ERROR,
                msg, impexp_state);
            return false;
        }
    }

    // preprocessing below needs to ensure that we operate on the same resolution
    const unsigned int max_rx = std::max(
        std::max(fresnel_tex_rx, trans_color_tex_rx), std::max(diffuse_tex_rx, specular_color_tex_rx));
    const unsigned int max_ry = std::max(
        std::max(fresnel_tex_ry, trans_color_tex_ry), std::max(diffuse_tex_ry, specular_color_tex_ry));

    const size_t num_color_channels = m_wavelengths.empty() ? 3 : m_wavelengths.size();
    const size_t num_pixels = max_rx * max_ry;
    if (m_fresnel_type == FRESNEL_SCHLICK) {
        // check if we need to "repair" energy conservation

        // typically nothing will happen here, since resolution should be the same
        upscale_tex_buffer(
            diffuse_tex, diffuse_tex_rx, diffuse_tex_ry, max_rx, max_ry, num_color_channels);
        upscale_tex_buffer(
            specular_color_tex, specular_color_tex_rx, specular_color_tex_ry, max_rx, max_ry, num_color_channels);

        // check for energy violation, re-scale glossy if necessary
        bool energy_violation = false;
        std::vector<float> scale_buf(num_pixels, 1.0f);
        for (size_t p = 0; p < num_pixels; ++p) {
            const size_t idx = num_color_channels * p;
            float d = diffuse_tex[idx];
            float g = specular_color_tex[idx];
            for (size_t c = 1; c < num_color_channels; ++c) {
                g = std::max(g, specular_color_tex[idx + c]);
                d = std::max(d, diffuse_tex[idx + c]);
            }

            if (d + g > 1.0f && d < 1.0f && g > 0.0f) {
                const float scale = (1.0f - d) / g;
                for (size_t c = 0; c < num_color_channels; ++c)
                    specular_color_tex[idx + c] *= scale;

                scale_buf[p] = scale;
                energy_violation = true;
            }
        }

        if (energy_violation) {
            // adapt Fresnel to compensate for glossy scaling above
            upscale_tex_buffer(
                fresnel_tex, fresnel_tex_rx, fresnel_tex_ry, max_rx, max_ry, 1);
            for (unsigned int p = 0; p < max_rx * max_ry; ++p) {
                fresnel_tex[p] = std::min(1.0f, fresnel_tex[p] / scale_buf[p]);
            }
        }

        // commit textures
        for (unsigned int i = 0; i < 2; ++i) {
            std::vector<float> *tex_buffer;
            unsigned int width, height;
            unsigned int channels;
            const char *tex_name;
            const char *target_slot;
            Texture_type tex_type;
            if (i == 0) {
                tex_buffer = &fresnel_tex;
                width = fresnel_tex_rx;
                height = fresnel_tex_ry;
                channels = 1;
                tex_name = fresnel_tex_name.c_str();
                target_slot = "specular_brdf_texture_fresnel";
                tex_type = m_wavelengths.empty() ? TEXTURE_SCALAR : TEXTURE_SCALAR_SPECTRAL;
            } else {
                tex_buffer = &specular_color_tex;
                width = specular_color_tex_rx;
                height = specular_color_tex_ry;
                channels = num_color_channels;
                tex_name = specular_color_tex_name.c_str();
                target_slot = "specular_texture";
                tex_type = m_wavelengths.empty() ? TEXTURE_RGB : TEXTURE_SPECTRAL;
            }

            const string texture_name = write_texture(
                impexp_state,
                tex_name, *tex_buffer,
                width, height, channels, tex_type);

            if (texture_name.empty()) {
                const string msg =
                    string("Error while creating textures from AxF material ") + m_material_name;
                Axf_importer::report_message(
                    6012, mi::base::MESSAGE_SEVERITY_ERROR,
                    msg, impexp_state);
                return false;
            }

            tex_to_param.insert(make_pair(texture_name, target_slot));
        }
    }

    if (!trans_color_tex.empty())
    {
        assert(m_fresnel_type == FRESNEL_SCHLICK);

        upscale_tex_buffer(
            specular_color_tex, specular_color_tex_rx, specular_color_tex_ry,
            max_rx, max_ry, num_color_channels);
        upscale_tex_buffer(
            trans_color_tex, trans_color_tex_rx, trans_color_tex_ry,
            max_rx, max_ry, num_color_channels);
        upscale_tex_buffer(
            fresnel_tex, fresnel_tex_rx, fresnel_tex_ry, max_rx, max_ry, 1);

        // compute rho_t and s'
        std::vector<float> transmission_s(num_pixels);
        std::vector<float> transmission_rho(num_pixels * num_color_channels);
        for (size_t i = 0; i < num_pixels; ++i)
        {
            const size_t idx = i * num_color_channels;
            float max_rho_s = specular_color_tex[idx];
            float max_rho_t0 = trans_color_tex[idx];
            for (size_t c = 1; c < num_color_channels; ++c) {
                max_rho_s = std::max(max_rho_s, specular_color_tex[idx + c]);
                max_rho_t0 = std::max(max_rho_t0, trans_color_tex[idx + c]);
            }
            const float F0 = fresnel_tex[i];

            transmission_s[i] = std::min(max_rho_s, (1.0f - max_rho_t0) / F0);
            for (size_t c = 0; c < num_color_channels; ++c)
                transmission_rho[idx + c] = trans_color_tex[idx + c] / (1.0f - transmission_s[i] * F0);
        }

        const std::string texture_name_s = write_texture(
            impexp_state,
            "transmission_s", transmission_s,
            max_rx, max_ry, 1,
            m_wavelengths.empty() ? TEXTURE_SCALAR : TEXTURE_SCALAR_SPECTRAL);

        const std::string texture_name_rho = write_texture(
            impexp_state,
            "transmission_rho", transmission_rho,
            max_rx, max_ry, num_color_channels,
            m_wavelengths.empty() ? TEXTURE_RGB : TEXTURE_SPECTRAL);

        if (texture_name_s.empty() || texture_name_rho.empty()) {
            const string msg =
                string("Error while creating textures from AxF material ") + m_material_name;
            Axf_importer::report_message(
                6012, mi::base::MESSAGE_SEVERITY_ERROR,
                msg, impexp_state);
            return false;
        }

        tex_to_param.insert(make_pair(texture_name_s, "specular_amount_texture"));
        tex_to_param.insert(make_pair(texture_name_rho, "transmission_color_texture"));
    }

    create_variant(
        impexp_state, "svbrdf", tex_decoder->getWidthMM(), tex_decoder->getHeightMM(),
        tex_to_param);
    return true;
}

string Axf_reader::write_texture(
    Axf_impexp_state* impexp_state,
    const string& tex_name,
    const vector<float>& tex_buffer,
    const int width,
    const int height,
    const int channels,
    Texture_type type)
{
    const string error_str(""); // empty string signals error

    string img_name = get_axf_image_prefix() +
        impexp_state->get_module_prefix() + string("_") +
        m_material_name + string("_") + string(tex_name);
    string texture_name = get_axf_texture_prefix() +
        impexp_state->get_module_prefix() + string("_") +
        m_material_name + string("_") + string(tex_name);
    if (type == TEXTURE_SPECTRAL || type == TEXTURE_SCALAR_SPECTRAL) {
        img_name += "_spectral";
        texture_name += "_spectral";
    }

    if (m_skip_non_color_maps &&
        (type != TEXTURE_SPECTRAL) && (type != TEXTURE_SCALAR_SPECTRAL)) {
        // return already created texture (but check if it's actually there)
        mi::base::Handle<const mi::neuraylib::ITexture> tex(
            m_transaction->access<mi::neuraylib::ITexture>(texture_name.c_str()));
        if (tex)
            return texture_name;
        else {
            // creation failed previously, so it will probable fail again...
            assert(!"texture should have been created already");
        }
    }

    if (type == TEXTURE_SCALAR_SPECTRAL)
        type = TEXTURE_SCALAR;

    const char* input_pixel_type = 0;
    // we only need to support floating point RGB and scalar here
    if (type == TEXTURE_NORMAL || type == TEXTURE_RGB)
        input_pixel_type = "Rgb_fp";
    else
        input_pixel_type = "Float32";

    // now create the appropriate image/texture out of it
    mi::base::Handle<mi::neuraylib::IImage_api> image_api(
        m_neuray->get_api_component<mi::neuraylib::IImage_api>());
    assert(image_api);

    const char *pixel_type;
    unsigned int layers = 1;
    unsigned int meta_layers = 0;
    if (type == TEXTURE_SPECTRAL) {
        pixel_type = "Float32";
        layers = channels;
        size_t spectral_offset0;
        meta_layers = compute_num_spectral_meta_layers(
            spectral_offset0, width * height * sizeof(float), channels, m_spectral_tex_meta_data.size());
        assert(spectral_offset0 == 0); //!! TODO: would need to be handled if pixel type wouldn't be float
    } else if (type == TEXTURE_NORMAL) {
        // use Rgb_16 for normal maps, because that will typically yield best precision through the
        // usage of the RGBAD type in iray core
        pixel_type = "Rgb_16";
    } else {
        // use RGBE for color and Float32 for scalar
        assert(channels == 3 || channels == 1);
        pixel_type = (channels == 3) ? "Rgbe" : "Float32";
    }

    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        image_api->create_canvas(pixel_type, width, height, layers + meta_layers));

    if (!canvas) {
        string msg = string("Failed to create canvas for AxF material \"") + m_material_name;
        msg += string("\" of texture ") + string(tex_name);
        Axf_importer::report_message(6014, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return error_str;
    }

    std::vector<float> tmp_layer_buffer;
    if (layers != 1)
        tmp_layer_buffer.resize(width * height);
    for (unsigned int l = 0; l < layers; ++l) {
        if (layers != 1) {
            // for the multi-layer spectral target, we need to separate the channels
            // (interleaved -> single channel image)
            for (size_t y = 0; y < (size_t)height; ++y) {
                for (size_t x = 0; x < (size_t)width; ++x) {
                    const size_t idx = y * width + x;
                    tmp_layer_buffer[idx] = tex_buffer[idx * layers + l];
                }
            }
        }
        // copy the data into the canvas, the API takes care of required conversion
        image_api->write_raw_pixels(
            width,
            height,
            canvas.get(),
            0,
            0,
            l,
            (layers == 1) ? tex_buffer.data() : tmp_layer_buffer.data(),
            /*topdown=*/true,
            input_pixel_type);
    }

    // copy spectral metadata to extra layer(s)
    if (type == TEXTURE_SPECTRAL) {
        size_t copied = 0;
        size_t uncopied = m_spectral_tex_meta_data.size();
        for (size_t l = layers; l < layers + meta_layers; ++l) {
            mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile(l));
            const size_t to_copy = std::min(uncopied, (size_t)width * (size_t)height * sizeof(float));
            memcpy(tile->get_data(), m_spectral_tex_meta_data.data() + copied, to_copy);
            copied += to_copy;
            uncopied -= to_copy;
        }
        assert(uncopied == 0);
    }

#ifdef DUMP_TEXTURES
    mi::base::Handle<mi::neuraylib::IExport_api> export_api(
            m_plugin_api->get_api_component<mi::neuraylib::IExport_api>());
    assert(export_api);
    std::string filename = m_material_name + tex_name;
    bool can_export = true;
    if (type == TEXTURE_NORMAL)
        filename += ".png";
    else if (type == TEXTURE_RGB)
        filename += ".hdr";
    else {
        filename += ".aix";
        can_export = false; //!! TODO: aix plugin can not save files yet...
    }
    if (can_export)
        export_api->export_canvas(filename.c_str(), canvas.get());
#endif

    const mi::Uint8 privacy = mi::neuraylib::ITransaction::LOCAL_SCOPE;
    // image
    mi::base::Handle<mi::neuraylib::IImage> image(
        m_transaction->create<mi::neuraylib::IImage>("Image"));
    image->set_from_canvas(canvas.get());
    m_transaction->store(image.get(), img_name.c_str(), privacy);
    image = 0;

    // texture
    mi::base::Handle<mi::neuraylib::ITexture> tex(
        m_transaction->create<mi::neuraylib::ITexture>("Texture"));
    tex->set_image(img_name.c_str());
    tex->set_gamma(1.0f);
    m_transaction->store(tex.get(), texture_name.c_str(), privacy);
    tex = 0;

    return texture_name;
}

static bool get_int_expr(
    int &result,
    const mi::neuraylib::IExpression_list * expr_list,
    const char *name)
{
    mi::base::Handle<const mi::neuraylib::IExpression> expr(expr_list->get_expression(name));
    if (!expr)
        return false;
    mi::base::Handle<const mi::neuraylib::IExpression_constant> exprc(
        expr->get_interface<const mi::neuraylib::IExpression_constant>());
    if (!exprc)
        return false;

    mi::base::Handle<const mi::neuraylib::IValue> value(exprc->get_value());
    mi::base::Handle<const mi::neuraylib::IValue_int> value_int(value->get_interface<const mi::neuraylib::IValue_int>());
    if (!value_int)
        return false;

    result = value_int->get_value();
    return true;
}

bool Axf_reader::access_mdl_material_definitions(
    Axf_impexp_state* impexp_state)
{
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());

    // Load the module "axf_importer" and access it from the DB.
    check_success(mdl_impexp_api->load_module(
        m_transaction, "::nvidia::axf_importer::axf_importer", context.get()) >= 0);
    print_messages(context.get());

    mi::base::Handle<const mi::neuraylib::IModule> mdl_module(
        m_transaction->access<mi::neuraylib::IModule>("mdl::nvidia::axf_importer::axf_importer"));
    check_success(mdl_module.is_valid_interface());

    bool got_version = false;
    int major = -1, minor = -1, patch = -1;
    if (mdl_module) {
        assert(mdl_module);
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> annotation_block(mdl_module->get_annotations());

        if (annotation_block) {
            for (mi::Size i = 0; i < annotation_block->get_size(); ++i) {

                mi::base::Handle<const mi::neuraylib::IAnnotation> annotation(annotation_block->get_annotation(i));
                if (annotation) {
                    mi::base::Handle<const mi::neuraylib::IAnnotation_definition> def(annotation->get_definition());
                    if (def && def->get_semantic() == mi::neuraylib::IAnnotation_definition::AS_VERSION_ANNOTATION) {
                        mi::base::Handle<const mi::neuraylib::IExpression_list> expr_list(annotation->get_arguments());
                        if (expr_list) {
                            got_version = get_int_expr(major, expr_list.get(), "major");
                            got_version |= get_int_expr(minor, expr_list.get(), "minor");
                            got_version |= get_int_expr(patch, expr_list.get(), "patch");
                        }
                    }
                }
            }
        }
    }

    if (!got_version) {
        Axf_importer::report_message(
            6031, mi::base::MESSAGE_SEVERITY_ERROR,
            "Failed to retrieve version of nvidia::axf_importer::axf_importer, the installed version may be out of date",
            impexp_state);
        return false;
    }
    if (major < 1 || (major == 1 && minor < 7) || (major == 1 && minor == 7 && patch < 100)) {
        ostringstream str;
        str << "Insufficient version (" << major << '.' << minor << '.' << patch << ") of nvidia::axf_importer::axf_importer, required version is (1.7.99)";
        Axf_importer::report_message(
            6031, mi::base::MESSAGE_SEVERITY_ERROR,
            str.str(), impexp_state);
        return false;
    }

    m_svbrdf_material =
        m_transaction->access<mi::neuraylib::IFunction_definition>(s_prototype_names[0]);
    m_carpaint_material =
        m_transaction->access<mi::neuraylib::IFunction_definition>(s_prototype_names[1]);
    m_volumetric_material =
        m_transaction->access<mi::neuraylib::IFunction_definition>(s_prototype_names[2]);

    if (!m_svbrdf_material) {
        std::string msg = "Cannot access the definition of \"";
        msg += s_prototype_names[0];
        msg += "\". Has MDL module \"nvidia::axf_importer::axf_importer\" been properly imported?";
        Axf_importer::report_message(6031, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return false;
    }
    if (!m_carpaint_material) {
        std::string msg = "Cannot access the definition of \"";
        msg += s_prototype_names[1];
        msg += "\". Has MDL module \"nvidia::axf_importer::axf_importer\" been properly imported?";
        Axf_importer::report_message(6032, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return false;
    }
    if (!m_volumetric_material) {
        std::string msg = "Cannot access the definition of \"";
        msg += s_prototype_names[2];
        msg += "\". Has MDL module \"nvidia::axf_importer::axf_importer\" been properly imported?";
        Axf_importer::report_message(6044, mi::base::MESSAGE_SEVERITY_ERROR,
            msg, impexp_state);
        return false;
    }

    return true;
}

namespace {

template<class T>
T* get_parameter_type(
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name)
{
    // check if the parameter exists
    const mi::base::Handle<const mi::neuraylib::IType_list> type_list(
        material->get_parameter_types());
    const mi::base::Handle<const mi::neuraylib::IType> type(type_list->get_type(param_name));
    if (!type)
        return nullptr;

    // follow aliases and cast to T
    const mi::base::Handle<const mi::neuraylib::IType> actual_type(type->skip_all_type_aliases());
    const mi::base::Handle<T> requested_type(actual_type->get_interface<T>());

    if (!requested_type)
        return nullptr;

    // return T
    requested_type->retain();
    return requested_type.get();
}

void set_float_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    float value)
{
    const mi::base::Handle<const mi::neuraylib::IType_float> type(
        get_parameter_type<const mi::neuraylib::IType_float>(material, param_name));

    if (!type)
    {
        Axf_importer::report_message(6034, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_float(value));
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(val.get()));
    default_parameters->add_expression(param_name, expr.get());
}

void set_float3_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    const float* value)
{
    const mi::Size size = 3;

    const mi::base::Handle<const mi::neuraylib::IType_vector> type(
        get_parameter_type<const mi::neuraylib::IType_vector>(material, param_name));
    if (!type || type->get_size() != size)
    {
        Axf_importer::report_message(6035, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }
    const mi::base::Handle<const mi::neuraylib::IType> elem_type(type->get_element_type());
    if (elem_type->get_kind() != mi::neuraylib::IType::TK_FLOAT)
    {
        Axf_importer::report_message(6036, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    mi::base::Handle<mi::neuraylib::IValue_vector> vec(val_factory->create_vector(type.get()));

    for (mi::Size i = 0; i < size; ++i)
    {
        mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_float(value[i]));
        vec->set_value(i, val.get());
    }

    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(vec.get()));
    default_parameters->add_expression(param_name, expr.get());
}

void set_color_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    const float* rgb)
{
    const mi::base::Handle<const mi::neuraylib::IType_color> type(
        get_parameter_type<const mi::neuraylib::IType_color>(material, param_name));

    if (!type)
    {
        Axf_importer::report_message(6052, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_color(rgb[0], rgb[1], rgb[2]));
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(val.get()));
    default_parameters->add_expression(param_name, expr.get());
}

#ifdef UNUSED_CODE
void set_color_array_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IType_factory* type_factory,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    const std::vector<float> &values)
{
    const mi::Size size = values.size() / 3;

    // get and check parameter type
    const mi::base::Handle<const mi::neuraylib::IType_array> type(
        get_parameter_type<const mi::neuraylib::IType_array>(material, param_name));

    if (!type)
    {
        Axf_importer::report_message(6037, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    // get and check element type
    mi::base::Handle<const mi::neuraylib::IType> elem_type(type->get_element_type());
    elem_type = mi::base::Handle<const mi::neuraylib::IType>(elem_type->skip_all_type_aliases());
    if (elem_type->get_kind() != mi::neuraylib::IType::TK_COLOR)
    {
        Axf_importer::report_message(6038, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    // create a new type since the old one is a deferred sized array
    mi::base::Handle<const mi::neuraylib::IType_array> new_ar_type(
        type_factory->create_immediate_sized_array(elem_type.get(), size));

    // create and fill the array
    mi::base::Handle<mi::neuraylib::IValue_array> ar(val_factory->create_array(new_ar_type.get()));
    for (mi::Size i = 0; i < size; ++i)
    {
        mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_color(
            values[i * 3 + 0], values[i * 3 + 1], values[i * 3 + 2]));
        ar->set_value(i, val.get());
    }
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(ar.get()));
    default_parameters->add_expression(param_name, expr.get());
}
#endif

void set_int_array_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IType_factory* type_factory,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    const std::vector<unsigned int> &values)
{
    const mi::Size size = values.size();

    // get and check parameter type
    const mi::base::Handle<const mi::neuraylib::IType_array> type(
        get_parameter_type<const mi::neuraylib::IType_array>(material, param_name));

    if (!type)
    {
        Axf_importer::report_message(6039, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    // get and check element type
    mi::base::Handle<const mi::neuraylib::IType> elem_type(type->get_element_type());
    elem_type = mi::base::Handle<const mi::neuraylib::IType>(elem_type->skip_all_type_aliases());

    if (elem_type->get_kind() != mi::neuraylib::IType::TK_INT)
    {
        Axf_importer::report_message(6040, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    // create a new type since the old one is a deferred sized array
    mi::base::Handle<const mi::neuraylib::IType_array> new_ar_type(
        type_factory->create_immediate_sized_array(elem_type.get(), size));

    // create and fill the array
    mi::base::Handle<mi::neuraylib::IValue_array> ar(val_factory->create_array(new_ar_type.get()));
    for (mi::Size i = 0; i < size; ++i)
    {
        mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_int(values[i]));
        ar->set_value(i, val.get());
    }
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(ar.get()));
    default_parameters->add_expression(param_name, expr.get());
}

void set_float_array_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IType_factory* type_factory,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    const std::vector<float> &values)
{
    const mi::Size size = values.size();

    // get and check parameter type
    const mi::base::Handle<const mi::neuraylib::IType_array> type(
        get_parameter_type<const mi::neuraylib::IType_array>(material, param_name));

    if (!type)
    {
        Axf_importer::report_message(6039, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    // get and check element type
    mi::base::Handle<const mi::neuraylib::IType> elem_type(type->get_element_type());
    elem_type = mi::base::Handle<const mi::neuraylib::IType>(elem_type->skip_all_type_aliases());

    if (elem_type->get_kind() != mi::neuraylib::IType::TK_FLOAT)
    {
        Axf_importer::report_message(6040, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    // create a new type since the old one is a deferred sized array
    mi::base::Handle<const mi::neuraylib::IType_array> new_ar_type(
        type_factory->create_immediate_sized_array(elem_type.get(), size));

    // create and fill the array
    mi::base::Handle<mi::neuraylib::IValue_array> ar(val_factory->create_array(new_ar_type.get()));
    for (mi::Size i = 0; i < size; ++i)
    {
        mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_float(values[i]));
        ar->set_value(i, val.get());
    }
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(ar.get()));
    default_parameters->add_expression(param_name, expr.get());
}

void set_bool_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    bool value)
{
    const mi::base::Handle<const mi::neuraylib::IType_bool> type(
        get_parameter_type<const mi::neuraylib::IType_bool>(material, param_name));
    if(!type)
    {
        Axf_importer::report_message(6041, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_bool(value));
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(val.get()));
    default_parameters->add_expression(param_name, expr.get());
}

void set_brdf_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    int value)
{
    const mi::base::Handle<const mi::neuraylib::IType_enum> enum_type(
        get_parameter_type<const mi::neuraylib::IType_enum>(material, param_name));
    if (!enum_type
        || strcmp(enum_type->get_symbol(), "::nvidia::axf_importer::axf_importer::brdf_type") != 0)
    {
        Axf_importer::report_message(6042, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_enum(enum_type.get(), value));
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(val.get()));
    default_parameters->add_expression(param_name, expr.get());
}

void set_fresnel_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    int value)
{
    const mi::base::Handle<const mi::neuraylib::IType_enum> enum_type(
        get_parameter_type<const mi::neuraylib::IType_enum>(material, param_name));
    if (!enum_type ||
        strcmp(enum_type->get_symbol(), "::nvidia::axf_importer::axf_importer::fresnel_type") != 0)
    {
        Axf_importer::report_message(6043, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_enum(enum_type.get(), value));
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(val.get()));
    default_parameters->add_expression(param_name, expr.get());
}

bool set_texture_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    const char* value)
{
    const mi::base::Handle<const mi::neuraylib::IType_texture> type(
        get_parameter_type<const mi::neuraylib::IType_texture>(material, param_name));
    if (!type || type->get_shape() != mi::neuraylib::IType_texture::TS_2D) // always 2D
        return false;

    mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_texture(type.get(), value));
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(val.get()));
    default_parameters->add_expression(param_name, expr.get());
    return true;
}

bool set_bsdf_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* param_name,
    const char* value)
{
    const mi::base::Handle<const mi::neuraylib::IType_bsdf_measurement> type(
        get_parameter_type<const mi::neuraylib::IType_bsdf_measurement>(material, param_name));
    if (!type)
        return false;

    mi::base::Handle<mi::neuraylib::IValue> val(val_factory->create_bsdf_measurement(value));
    mi::base::Handle<mi::neuraylib::IExpression> expr(expr_factory->create_constant(val.get()));
    default_parameters->add_expression(param_name, expr.get());
    return true;
}

void set_spectral_param(
    Axf_impexp_state* impexp_state,
    mi::neuraylib::IExpression_list* default_parameters,
    mi::neuraylib::IType_factory* type_factory,
    mi::neuraylib::IValue_factory* val_factory,
    mi::neuraylib::IExpression_factory* expr_factory,
    const mi::neuraylib::IFunction_definition* material,
    const char* material_name,
    mi::neuraylib::ITransaction* transaction,
    const char* param_name,
    const std::vector<float> &wavelengths,
    const std::vector<float> &values)
{
    const mi::base::Handle<const mi::neuraylib::IType_color> type(
        get_parameter_type<const mi::neuraylib::IType_color>(material, param_name));

    if (!type)
    {
        Axf_importer::report_message(6052, mi::base::MESSAGE_SEVERITY_ERROR,
                                     string("Failed to set parameter \"") + param_name
                                     + string("\" for current AxF material."), impexp_state);
        return;
    }

    assert(wavelengths.size() == values.size());
    const size_t num_wavelengths = wavelengths.size();

    //
    // create a spectral color constructor call
    //

    mi::base::Handle<const mi::neuraylib::IFunction_definition> function_definition(
        transaction->access<mi::neuraylib::IFunction_definition>(
            "mdl::color(float[N],float[N])"));
    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        expr_factory->create_expression_list());
    mi::base::Handle<const mi::neuraylib::IType> elem_type(type_factory->create_float());
    mi::base::Handle<const mi::neuraylib::IType_array> array_type(
        type_factory->create_immediate_sized_array(elem_type.get(), num_wavelengths));

    // attach wavelengths and values arrays
    for (unsigned int i = 0; i < 2; ++i)
    {
        const char *array_name = function_definition->get_parameter_name(i);

        const float *f = (i == 0) ? wavelengths.data() : values.data();

        // create and fill the array
        mi::base::Handle<mi::neuraylib::IValue_array> ar(val_factory->create_array(array_type.get()));
        for (mi::Size j = 0; j < num_wavelengths; ++j)
        {
            mi::base::Handle<mi::neuraylib::IValue> val(
                val_factory->create_float(f[j]));
            ar->set_value(j, val.get());
        }

        mi::base::Handle<mi::neuraylib::IExpression> array_expr(expr_factory->create_constant(ar.get()));
        arguments->add_expression(array_name, array_expr.get());
    }

    mi::Sint32 res;
    mi::base::Handle<mi::neuraylib::IFunction_call> function_call(
        function_definition->create_function_call(arguments.get(), &res));
    const string color_constructor_name =
        get_axf_spectrum_prefix() +
        impexp_state->get_module_prefix() + string("_") +
        material_name + string("_") + param_name;
    transaction->store(function_call.get(), color_constructor_name.c_str());


    //
    // attach call to parameter
    //

    mi::base::Handle<mi::neuraylib::IExpression_call> call(expr_factory->create_call(color_constructor_name.c_str()));
    default_parameters->add_expression(param_name, call.get());
}

/// Valid MDL identifiers have to follow certain rules. For example, they may not contain spaces.
/// "An identifier is an alphabetic character followed by a possibly empty sequence of
/// alphabetic characters, decimal digits, and underscores, that is neither a typename
/// nor a reserved word."
string make_valid_mdl_id(
    mi::neuraylib::IMdl_factory* mdl_factory, const string& id)
{
    if (id.empty())
        return string();

    string result;
    result.reserve(id.size());

    // check that it starts with an alphabetic character - otherwise we replace it with 'X'
    size_t index = 0;
    result.push_back(isalpha(id[index])? id[index] : 'X');

    // check that now we have only alphanumeric characters or '_' - else replace it with '_'
    for (index=1; index < id.size(); ++index) {
        if (isalnum(id[index]) || id[index] == '_')
            result.push_back(id[index]);
        else
            result.push_back('_');
    }

    if (mdl_factory->is_valid_mdl_identifier(result.c_str()))
        return result;
    else
        return string("X") + result;
}

// turn into valid form ::module1::...::module2::
string make_valid_mdl_module_path(
    mi::neuraylib::IMdl_factory* mdl_factory, const string &path)
{
    std::string result;

    size_t start = 0;

    for (start = 0; start < path.size(); ++start) {
        if (path[start] != ':')
            break;
    }

    bool prev_colon = false;
    for (size_t pos = start; pos < path.size(); ++pos) {
        bool colon = (path[pos] == ':');
        if (colon && prev_colon) {
            result += "::" + make_valid_mdl_id(mdl_factory, path.substr(start, pos - start - 1));

            for (start = pos + 1; start < path.size(); ++start) {
                if (path[start] != ':') {
                    break;
                }
            }
            pos = start - 1;
            colon = false;
        }
        prev_colon = colon;
    }

    if (start < path.size()) {
        result += "::" + make_valid_mdl_id(mdl_factory, path.substr(start, path.size() - start));
    }

    return result + "::";
}
}

void Axf_reader::create_variant(
    Axf_impexp_state* impexp_state,
    const string& representation,
    const float widthMM,
    const float heightMM,
    const map<string, string>& tex_to_param)
{
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IValue_factory> val_factory(
        mdl_factory->create_value_factory(m_transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expr_factory(
        mdl_factory->create_expression_factory(m_transaction));
    mi::base::Handle<mi::neuraylib::IType_factory> type_factory(
        mdl_factory->create_type_factory(m_transaction));

    // create an expression list and required parameters
    mi::base::Handle<mi::neuraylib::IExpression_list> default_parameters(
        expr_factory->create_expression_list());

    // add the textures
    map<string, string>::const_iterator it, end=tex_to_param.end();
    for (it=tex_to_param.begin(); it != end; ++it) {
        const string& value = it->first;
        const string& param_name = it->second;

        const mi::neuraylib::IFunction_definition *mat_def = nullptr;
        switch (m_type) {
            case REP_VOLUMETRIC:
                mat_def = m_volumetric_material.get();
                break;
            case REP_SVBRDF:
                mat_def = m_svbrdf_material.get();
                break;
            case REP_CARPAINT:
                mat_def = m_carpaint_material.get();
                break;
        };

        if (!set_texture_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            mat_def, param_name.c_str(), value.c_str()))
        {
            string msg = string("Failed to appropriate texture parameter \"") + param_name
                + string("\" for current AxF material.");
            Axf_importer::report_message(6015, mi::base::MESSAGE_SEVERITY_ERROR,
                                         msg, impexp_state);
            continue;
        }
    }

    string material_name = make_valid_mdl_id(
        mdl_factory.get(), m_material_name + string("_") + representation);
    string display_name = m_material_display_name;
    if (!m_wavelengths.empty()) {
        material_name += "_spectral";
        display_name += " (" + representation + ", spectral)";
    } else {
        display_name +=  " (" + representation + ')';
    }

    // add the sample_size params: sample_size_u <-- width, sample_size_v <-- height
    const char* param_name;

    if (m_type != REP_VOLUMETRIC)
    {
        param_name = "sample_size_u";
        set_float_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_svbrdf_material.get(), param_name, widthMM / 1000.0f);

        param_name = "sample_size_v";
        set_float_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_svbrdf_material.get(), param_name, heightMM / 1000.0f);
    }

    if (m_type == REP_SVBRDF)
    {
        param_name = "isotropic";
        set_bool_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_svbrdf_material.get(), param_name, m_isotropic);

        param_name = "brdf_type";
        set_brdf_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_svbrdf_material.get(), param_name, static_cast<int>(m_glossy_brdf_type));

        param_name = "fresnel_type";
        set_fresnel_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_svbrdf_material.get(), param_name, static_cast<int>(m_fresnel_type));

        param_name = "has_clearcoat";
        set_bool_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_svbrdf_material.get(), param_name, m_has_clearcoat);

        m_variants.push_back(
            Variant_data(REP_SVBRDF, material_name, display_name, default_parameters));
    }
    else if (m_type == REP_VOLUMETRIC)
    {
        param_name = "phasefunc_g";
        set_float_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_volumetric_material.get(), param_name, m_phasefunc_g);

        param_name = "ior";
        set_float_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_volumetric_material.get(), param_name, m_ior);


        if (m_wavelengths.empty()) {
            param_name = "sigma_s";
            set_color_param(
                impexp_state,
                default_parameters.get(), val_factory.get(), expr_factory.get(),
                m_volumetric_material.get(), param_name, m_sigma_s.data());
            param_name = "sigma_a";
            set_color_param(
                impexp_state,
                default_parameters.get(), val_factory.get(), expr_factory.get(),
                m_volumetric_material.get(), param_name, m_sigma_a.data());
        } else {
            param_name = "sigma_s";
            set_spectral_param(
                impexp_state,
                default_parameters.get(), type_factory.get(), val_factory.get(), expr_factory.get(),
                m_volumetric_material.get(), material_name.c_str(), m_transaction, param_name, m_wavelengths, m_sigma_s);
            param_name = "sigma_a";
            set_spectral_param(
                impexp_state,
                default_parameters.get(), type_factory.get(), val_factory.get(), expr_factory.get(),
                m_volumetric_material.get(), material_name.c_str(), m_transaction, param_name, m_wavelengths, m_sigma_a);
        }

        m_variants.push_back(
            Variant_data(REP_VOLUMETRIC, material_name, display_name, default_parameters));
    }
    else
    {
        assert(m_type == REP_CARPAINT);

        const bool bake_brdf_colors = !m_brdf_colors_2d.empty();

        // create measured BRDF data block
        {

            mi::base::Handle<mi::neuraylib::IBsdf_measurement> bsdf_measurement(
                m_transaction->create<mi::neuraylib::IBsdf_measurement>("Bsdf_measurement"));

            const unsigned int res_theta = 45;
            const unsigned int res_phi = 90;

            mi::base::Handle<mi::neuraylib::Bsdf_isotropic_data> bsdf_data(
                new mi::neuraylib::Bsdf_isotropic_data(
                    res_theta, res_phi, bake_brdf_colors ? mi::neuraylib::BSDF_RGB : mi::neuraylib::BSDF_SCALAR));

            mi::base::Handle<mi::neuraylib::Bsdf_buffer> bsdf_data_buffer(
                static_cast<mi::neuraylib::Bsdf_buffer *>(bsdf_data->get_bsdf_buffer()));
            float *data = bsdf_data_buffer->get_data();

            // set up BRDF parameters
            Brdf_params params;
            params.diffuse = m_ct_diffuse;
            for (int i = 0; i < 3; ++i) {
                params.ct_weight[i] = m_ct_coeffs[i];
                params.ct_f0[i] = m_ct_f0s[i];
                params.ct_roughness[i] = std::max(m_ct_spreads[i], 1e-5f);
            }
            params.ior = m_ior;
            params.refr = m_refractive_clearcoat;
            if (bake_brdf_colors) {
                params.brdf_colors = m_brdf_colors_2d.data();
                params.brdf_colors_rx = m_brdf_colors_2d_rx;
                params.brdf_colors_ry = m_brdf_colors_2d_ry;
            } else {
                // color curve is handled outside, make sure to include scaling
                params.diffuse *= m_brdf_colors_scale;
                params.ct_weight[0] *= m_brdf_colors_scale;
                params.ct_weight[1] *= m_brdf_colors_scale;
                params.ct_weight[2] *= m_brdf_colors_scale;

                params.brdf_colors = nullptr;
                params.brdf_colors_rx = 0;
                params.brdf_colors_ry = 0;
            }
            // bake
            create_measured_subclearcoat(data, res_theta, res_phi, params);
            bsdf_measurement->set_reflection(bsdf_data.get());

            string bsdf_measurement_name =
                get_axf_mbsdf_prefix() +
                impexp_state->get_module_prefix() + string("_") +
                m_material_name + string("_sub_clearcoat_measurement");
            if (bake_brdf_colors)
                bsdf_measurement_name += "_color";
            m_transaction->store(
                bsdf_measurement.get(), bsdf_measurement_name.c_str(),
                mi::neuraylib::ITransaction::LOCAL_SCOPE);

            const char* param_name = "sub_clearcoat_measurement";
            if (!set_bsdf_param(
                impexp_state,
                default_parameters.get(), val_factory.get(), expr_factory.get(),
                m_carpaint_material.get(), param_name, bsdf_measurement_name.c_str()))
            {
                string msg =
                    string("Failed to appropriate bsdf measurement parameter \"") + param_name
                    + string("\" for current AxF material.");
                Axf_importer::report_message(
                    6033, mi::base::MESSAGE_SEVERITY_ERROR,
                    msg, impexp_state);
            }
        }

        // need to apply normalization of BRDFcolors curve
        m_ct_diffuse *= m_brdf_colors_scale;
        m_ct_coeffs[0] *= m_brdf_colors_scale;
        m_ct_coeffs[1] *= m_brdf_colors_scale;
        m_ct_coeffs[2] *= m_brdf_colors_scale;

        param_name = "precise_sub_clearcoat_component";
        set_bool_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, true);

        param_name = "ior";
        set_float_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_ior);

        param_name = "ct_diffuse";
        set_float_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_ct_diffuse);

        param_name = "ct_coeffs";
        set_float3_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_ct_coeffs);

        param_name = "ct_f0s";
        set_float3_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_ct_f0s);

        param_name = "ct_spreads";
        set_float3_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_ct_spreads);

        param_name = "enable_flakes";
        set_bool_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_flake_intensity_scale > 0.0f);

        param_name = "flake_importance_data";
        set_int_array_param(
            impexp_state,
            default_parameters.get(), type_factory.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_flake_importance_data);

        param_name = "flake_peak_intensity_scale";
        set_float_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_flake_intensity_scale);

        param_name = "flake_uvw_scale";
        set_float3_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_flake_uvw_scale);

        param_name = "flake_orientation_falloff";
        set_float_array_param(
            impexp_state,
            default_parameters.get(), type_factory.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, m_flake_orientation_falloff);

        param_name = "brdf_colors_2d_included_in_measurement";
        set_bool_param(
            impexp_state,
            default_parameters.get(), val_factory.get(), expr_factory.get(),
            m_carpaint_material.get(), param_name, bake_brdf_colors);

        m_variants.push_back(
            Variant_data(REP_CARPAINT, material_name, display_name, default_parameters));
    }
}

// handle preview images
void Axf_reader::handle_preview_images(
    AXF::AXF_REPRESENTATION_HANDLE rep_h,
    Axf_impexp_state* impexp_state
    ) const
{
    int num = AXF::axfGetNumPreviewImages(rep_h);
    for (int i=0; i<num; ++i) {
        int width, height, channels;
        float widthMM, heightMM;
        if (!AXF::axfGetPreviewImageInfo(rep_h, i, width, height, channels, widthMM, heightMM)) {
            const string msg =
                string("Failed to retrieve preview image info for material \"") + m_material_name + '\"';
            Axf_importer::report_message(6016, mi::base::MESSAGE_SEVERITY_WARNING,
                msg, impexp_state);
            continue;
        }
        string pixel_type;
        // we only support RGB and alpha here
        if (channels == 3)
            pixel_type = "Rgb_fp";
        else if (channels == 1)
            pixel_type = "Float32";
        else
            continue;

        vector<float> img(width * height * channels);
        if (!AXF::axfGetPreviewImage(rep_h,i,m_target_color_space,img.data(),width,height,channels)) {
            const string msg =
                string("Failed to retrieve preview image info for material \"") + m_material_name + '\"';
            Axf_importer::report_message(6017, mi::base::MESSAGE_SEVERITY_WARNING,
                msg, impexp_state);
            continue;
        }

        // now create the appropriate image out of it
        mi::base::Handle<mi::neuraylib::IImage_api> image_api(
            m_neuray->get_api_component<mi::neuraylib::IImage_api>());
        assert(image_api);

        mi::base::Handle<mi::neuraylib::ICanvas> canvas(
            image_api->create_canvas(pixel_type.c_str(), width, height)
        );
        if (!canvas) {
            const string msg = string("Failed to create preview canvas of AxF material \"")
                    + m_material_name + string("\".");
            Axf_importer::report_message(6018, mi::base::MESSAGE_SEVERITY_WARNING,
                msg, impexp_state);
            continue;
        }

        // copy the data into the canvas
        bool topdown = true;
        image_api->write_raw_pixels(
            width,
            height,
            canvas.get(),
            0,
            0,
            0,
            img.data(),
            topdown,
            canvas->get_type());

#ifdef DUMP_TEXTURES
        mi::base::Handle<mi::neuraylib::IExport_api> export_api(
            m_plugin_api->get_api_component<mi::neuraylib::IExport_api>());
        assert(export_api);
        std::string filename = m_material_name + "_preview.hdr";
        export_api->export_canvas(filename.c_str(), canvas.get());
#endif

        // create image
        const string img_name = get_axf_pimage_prefix() +
            impexp_state->get_module_prefix() + string("_") +
            m_material_name + string("_")
            + std::to_string(width) + string("x") + std::to_string(height);
        mi::base::Handle<mi::neuraylib::IImage> image(
            m_transaction->create<mi::neuraylib::IImage>("Image"));
        image->set_from_canvas(canvas.get());
        m_transaction->store(image.get(),img_name.c_str(),mi::neuraylib::ITransaction::LOCAL_SCOPE);
    }
}

static const mi::neuraylib::IAnnotation_block* copy_annotations(
    mi::neuraylib::ITransaction* trans,
    mi::neuraylib::IExpression_factory* expr_factory,
    mi::neuraylib::IValue_factory* value_factory,
    const char* prototype_name,
    const char* display_name)
{
    auto mat_def = mi::base::make_handle(trans->access<mi::neuraylib::IFunction_definition>(prototype_name));
    auto annos = mi::base::make_handle(mat_def->get_annotations());
    if (!annos) {
        return nullptr;
    }

    auto new_annos = mi::base::make_handle(expr_factory->create_annotation_block());

    for (mi::Size i = 0, n = annos->get_size(); i < n; ++i) {
        auto anno = mi::base::make_handle(annos->get_annotation(i));
        auto anno_def = mi::base::make_handle(anno->get_definition());
        // don't hide materials
        if (anno_def && (anno_def->get_semantic() == mi::neuraylib::IAnnotation_definition::AS_HIDDEN_ANNOTATION)) {
             continue;
        }

        // set a meaningful display name
        if (anno_def && (anno_def->get_semantic() == mi::neuraylib::IAnnotation_definition::AS_DISPLAY_NAME_ANNOTATION)) {
            auto name = mi::base::make_handle(value_factory->create_string(display_name));
            auto new_expr_constant = mi::base::make_handle(expr_factory->create_constant(name.get()));

            auto new_expr_list = mi::base::make_handle(expr_factory->create_expression_list());
            new_expr_list->add_expression("name", new_expr_constant.get());

            auto new_anno =  mi::base::make_handle(expr_factory->create_annotation(anno->get_name(), new_expr_list.get()));
            new_annos->add_annotation(new_anno.get());
            continue;
        }

        auto anno_args = mi::base::make_handle(anno->get_arguments());
        auto new_anno_args =  mi::base::make_handle(anno_args ? expr_factory->clone(anno_args.get()) : nullptr);
        auto new_anno =  mi::base::make_handle(expr_factory->create_annotation(anno->get_name(), new_anno_args.get()));

        new_annos->add_annotation(new_anno.get());
    }
    new_annos->retain();
    return new_annos.get();
}

unsigned int Axf_reader::handle_collected_variants(
    Axf_impexp_state* impexp_state)
{
    // create module name name
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
        m_neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    string axf_name = Ospath::basename(m_filename);
    string mat_name(axf_name), dummy;
    Ospath::splitext(mat_name, axf_name, dummy);
    string module_name
        = make_valid_mdl_module_path(mdl_factory.get(), impexp_state->get_module_prefix())
        + make_valid_mdl_id(mdl_factory.get(), axf_name);

    // create module builder
    mi::base::Handle<mi::neuraylib::IExpression_factory> expr_factory(
        mdl_factory->create_expression_factory(m_transaction));
    mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
        mdl_factory->create_value_factory(m_transaction));
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
        mdl_factory->create_execution_context());
    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(
        mdl_factory->create_module_builder(
            m_transaction,
            ("mdl" + module_name).c_str(),
            mi::neuraylib::MDL_VERSION_1_0,
            mi::neuraylib::MDL_VERSION_LATEST,
            context.get()));
    if (!module_builder) {
        Axf_importer::report_message(6019, mi::base::MESSAGE_SEVERITY_ERROR,
            "Failed to create module builder for \"" + module_name + "\".", impexp_state);
        return false;
    }

    unsigned int num_variants = 0;
    std::vector<bool> success(m_variants.size(), false);
    for (size_t i=0; i<m_variants.size(); ++i) {

        const char *prototype_name;
        switch (m_variants[i].m_type) {
            case REP_SVBRDF:
                prototype_name = s_prototype_names[0];
                break;
            case REP_CARPAINT:
                prototype_name = s_prototype_names[1];
                break;
            case REP_VOLUMETRIC:
                prototype_name = s_prototype_names[2];
                break;
            default:
                assert(false);
                continue;
        }

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> annos(
            copy_annotations(m_transaction, expr_factory.get(), value_factory.get(), prototype_name, m_variants[i].m_display_name.c_str()));

        mi::Sint32 result = module_builder->add_variant(
             m_variants[i].m_material_name.c_str(),
             prototype_name,
             m_variants[i].m_default_values.get(),
             /*annotations*/ annos.get(),
             /*return_annotations*/ nullptr,
             /*is_exported*/ true,
             context.get());
        if (result < 0) {
            Axf_importer::report_message(6019, mi::base::MESSAGE_SEVERITY_ERROR,
                "Failed to create variant \"" + m_variants[i].m_material_name + "\" for \"" + module_name + "\".",
                impexp_state);
            for (mi::Size i = 0; i < context->get_messages_count(); ++i) {
                mi::base::Handle<const mi::neuraylib::IMessage> msg(context->get_message(i));
                Axf_importer::report_message(6019, msg->get_severity(),
                    msg->get_string(), impexp_state);
            }
        }
        else {
            // Export the variant.
            mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
                m_neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
            check_success(mdl_impexp_api->export_module(
                m_transaction,
                ("mdl" + module_name).c_str(),
                impexp_state->get_mdl_output_filename()) == 0);
            ++num_variants;
            success[i] = true;
        }
    }

    for (mi::Size i = 0; i < context->get_messages_count(); ++i) {
        mi::base::Handle<const mi::neuraylib::IMessage> msg(context->get_message(i));
        Axf_importer::report_message(6019, msg->get_severity(),
            msg->get_string(), impexp_state);
    }

    return num_variants;
}

}
}
}