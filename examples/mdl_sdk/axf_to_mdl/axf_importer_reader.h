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

#ifndef EXAMPLE_AXF_TO_MDL_AXF_IMPORTER_READER_H
#define EXAMPLE_AXF_TO_MDL_AXF_IMPORTER_READER_H

#include <map>
#include <string>
#include <vector>

#include "example_shared.h"

#include <AxF/decoding/TextureDecoder.h>

namespace mi {
namespace examples {
namespace impaxf {

class Axf_impexp_state;

/// metadata for spectral textures
#define TEXTURE_SPECTRAL_METADATA_MARKER "spectralmetadata"
#define TEXTURE_SPECTRAL_METADATA_MARKER_SIZE 16
struct Texture_spectral_metadata {
    union {
        char marker[TEXTURE_SPECTRAL_METADATA_MARKER_SIZE];
        float padd[TEXTURE_SPECTRAL_METADATA_MARKER_SIZE / 4];
    };

    float lambda_min;
    float lambda_max;
    unsigned int num_lambda;

    float matrix[1];
};

class Axf_importer
{
public:
    Axf_importer() {};
    ~Axf_importer() {};

    /// Formats error message with context and appends it to the error messages in the result.
    static void report_message(
        mi::Sint32 error_number,
        mi::base::Message_severity error_severity,
        const std::string& error_message,
        const Axf_impexp_state* import_state);
};

/// The main class for doing the import.
class Axf_reader
{
public:
    /// Constructor.
    Axf_reader(
        mi::neuraylib::INeuray *neuray,
        mi::neuraylib::ITransaction *transaction);

    /// Read the given .axf file. This functions performs several tasks under the hood
    ///  - retrieving color space transformation, SVBRDF representation, preview from AxF file
    ///  - storing textures in neuray DB
    ///  - creating corresponding MDL material definition based on nvidia/axf_importer.mdl
    /// \return success
    bool read(
        const char* file_name,
        Axf_impexp_state* impexp_state);

private:
    std::string m_filename;                             ///< name of the current .axf file
    mi::neuraylib::INeuray *m_neuray; ///< access to MDL SDK
    mi::neuraylib::ITransaction* m_transaction;
    mi::base::Handle<const mi::neuraylib::IFunction_definition> m_svbrdf_material;
    mi::base::Handle<const mi::neuraylib::IFunction_definition> m_carpaint_material;
    mi::base::Handle<const mi::neuraylib::IFunction_definition> m_volumetric_material;

    enum Representation_type {
        REP_SVBRDF,
        REP_CARPAINT,
        REP_VOLUMETRIC
    };
    Representation_type m_type;

    const char *m_target_color_space;

    //
    // SVBRDF data
    //
    //!! keep in sync with axf_importer.mdl's enum "brdf_type"
    enum Brdf_type {
        // supported
        BRDF_WARD_GEISLERMORODER,
        BRDF_COOKTORRANCE,
        BRDF_GGX,
        // unsupported (and long deprecated by AxF SDK)
        //BRDF_WARD_DUER,
        //BRDF_WARD_ORIGINAL,
        //BRDF_ASHIKHMINSHIRLEY,
        //BRDF_BLINNPHONG,
        BRDF_TYPE_UNKNOWN
    };
    Brdf_type m_glossy_brdf_type;
    bool m_isotropic;
    //!! keep in sync with axf_importer.mdl's enum "fresnel_type"
    enum Fresnel_type {
        FRESNEL_NONE,
        FRESNEL_SCHLICK,
        FRESNEL_FRESNEL,
        FRESNEL_UNKNOWN
    };
    Fresnel_type m_fresnel_type;
    bool m_has_clearcoat;

    //
    // Carpaint data
    //
    float m_ior;
    float m_ct_diffuse;
    float m_ct_coeffs[3];
    float m_ct_f0s[3];
    float m_ct_spreads[3];
    float m_brdf_colors_scale;
    std::vector<float> m_brdf_colors_2d;
    unsigned int m_brdf_colors_2d_rx, m_brdf_colors_2d_ry;
    std::vector<unsigned int> m_flake_importance_data;
    std::vector<float> m_flake_orientation_falloff;
    float m_flake_intensity_scale;
    float m_flake_uvw_scale[3];
    bool m_refractive_clearcoat;

    //
    // Volumetric data
    //
    std::vector<float> m_sigma_a;
    std::vector<float> m_sigma_s;
    float m_phasefunc_g;

    //
    // for spectral data import
    //
    std::vector<float> m_wavelengths; // (empty for RGB decoding)
    bool m_skip_non_color_maps;
    std::vector<char> m_spectral_tex_meta_data;

    //
    // name / id of currently processed material
    //
    std::string m_material_name;
    std::string m_material_display_name;

    /// Keep all data for the variants together. This is required since the neuray API does not
    /// allow incremental addition of single variants to a module. In other words, all variants must
    /// be collected first and then be inserted at once.
    class Variant_data {
    public:
        Variant_data(
            const Representation_type type,
            const std::string &material_name,
            const std::string &display_name,
            const mi::base::Handle<mi::neuraylib::IExpression_list> &default_values)
            : m_type(type), m_material_name(material_name),
              m_display_name(display_name), m_default_values(default_values) {
        }
        Representation_type m_type;
        std::string m_material_name;
        std::string m_display_name;
        mi::base::Handle<mi::neuraylib::IExpression_list> m_default_values;

    };
    std::vector<Variant_data> m_variants;

    /// The internal read method.
    bool read_material(
        axf::decoding::AxFFileHandle* file,
        int material_idx,
        Axf_impexp_state* impexp_state);

    bool handle_representation(
        axf::decoding::AxFRepresentationHandle* rep_h,
        int conversion_flags,
        Axf_impexp_state* impexp_state);
   
    bool handle_carpaint_representation(
        axf::decoding::AxFRepresentationHandle* rep_h,
        int conversion_flags,
        axf::decoding::TextureDecoder *tex_decoder,
        Axf_impexp_state* impexp_state);
    bool handle_volumetric_representation(
        axf::decoding::AxFRepresentationHandle* rep_h,
        int conversion_flags,
        axf::decoding::TextureDecoder *tex_decoder,
        Axf_impexp_state* impexp_state);
    bool handle_svbrdf_representation(
        axf::decoding::AxFRepresentationHandle* rep_h,
        int conversion_flags,
        axf::decoding::TextureDecoder *tex_decoder,
        Axf_impexp_state* impexp_state);

    /// Handle the preview images of this representation.
    void handle_preview_images(
        axf::decoding::AxFRepresentationHandle* rep_h,
        Axf_impexp_state* impexp_state
        ) const;

    // texture types we expect from the AxF SDK
    enum Input_texture_type {
        INPUT_TEXTURE_COLOR,   // spectrum or RGB
        INPUT_TEXTURE_RGB,
        INPUT_TEXTURE_FLOAT,
        INPUT_TEXTURE_FLOAT12, // float or float2
        INPUT_TEXTURE_NORMAL
    };
    bool check_texture_type(
        unsigned int num_channels,
        Input_texture_type type,
        const char *tex_name,
        Axf_impexp_state* impexp_state);
    
    // texture types we store in iray
    enum Texture_type {
        TEXTURE_RGB,             // RGB color
        TEXTURE_SPECTRAL,        // spectral data
        TEXTURE_SCALAR,          // scalar
        TEXTURE_SCALAR_SPECTRAL, // scalar computed from spectral input
        TEXTURE_NORMAL           // normalmap
    };

    /// Create the passed in texture in the neuray DB.
    /// \return DB name of the saved texture
    std::string write_texture(
        Axf_impexp_state* impexp_state,
        const std::string& tex_name,
        const std::vector<float>& tex_buffer,
        int width,
        int height,
        int channels,
        Texture_type type);
    /// Load the corresponding MDL module and access the representations of the MDL definition.
    /// \param impexp_state the current io state
    /// \return false if fails
    bool access_mdl_material_definitions(
        Axf_impexp_state* impexp_state);
    /// Create the new variant.
    /// \param impexp_state the current io state
    /// \param representation name of the representation
    /// \param material name of the material
    /// \param widthMM width of the \p representation in millimeters
    /// \param heightMM height of the \p representation in millimeters
    /// \param tex_to_param mapping from texture param names to parameter names
    void create_variant(
        Axf_impexp_state* impexp_state,
        const std::string& representation,
        float widthMM,
        float heightMM,
        const std::map<std::string, std::string>& tex_to_param);
    /// Insert now all collected variants into one module.
    /// \param impexp_state the current io state
    unsigned int handle_collected_variants(
        Axf_impexp_state* impexp_state);
};

}
}
}
#endif
