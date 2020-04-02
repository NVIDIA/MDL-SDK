/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief  Spectral tables.

#ifndef SPECTRAL_TABLES_H_
#define SPECTRAL_TABLES_H_

namespace mi {
namespace mdl {
namespace spectral {

// CIE standard observer color matching functions (1931)
static const unsigned int SPECTRAL_XYZ_RES = 81;
static const float SPECTRAL_XYZ_LAMBDA_MIN = 380.000000f;
static const float SPECTRAL_XYZ_LAMBDA_MAX = 780.000000f;
static const float SPECTRAL_XYZ_LAMBDA_STEP = 5.0f;
extern const float SPECTRAL_XYZ1931_X[];
extern const float SPECTRAL_XYZ1931_Y[];
extern const float SPECTRAL_XYZ1931_Z[];

#if 0
// CIE standard observer color matching functions (1964)
extern const float SPECTRAL_XYZ1964_X[];
extern const float SPECTRAL_XYZ1964_Y[];
extern const float SPECTRAL_XYZ1964_Z[];
#endif

// standard illuminants
extern const float D60[];
extern const float D65[];

// tables for Smits-styple sRGB to spectrum conversion
extern const float srgb_relative_s001[];
extern const float srgb_relative_s010[];
extern const float srgb_relative_s011[];
extern const float srgb_relative_s100[];
extern const float srgb_relative_s101[];
extern const float srgb_relative_s110[];

// tables for chromaticity grid color to spectrum conversion
// for different illuminants
struct Chroma_cell
{
    unsigned int idx[4]; // indices to spectral data for all points (first index additionally
                         // encodes cell type)
    float xy[8];         // chromaticity of all points
};
struct Chroma_grid_info
{
    unsigned int res;           // resolution //!! == 16 for all our data
    unsigned int white_idx;     // index of whitepoint in spectral data array
    float p0[2];                // translation
    float tf[4];                // rotation
    unsigned int num_cells;
    unsigned int num_spectra;
};
extern const Chroma_grid_info chroma_grid_info_e;
extern const Chroma_cell chroma_cells_e[225];
extern const float chroma_spectra_e[13366];
extern const Chroma_grid_info chroma_grid_info_d60;
extern const Chroma_cell chroma_cells_d60[225];
extern const float chroma_spectra_d60[14432];
extern const Chroma_grid_info chroma_grid_info_d65;
extern const Chroma_cell chroma_cells_d65[225];
extern const float chroma_spectra_d65[14104];

} // namespace spectral
} // namespace mdl
} // namespace mi

#endif /* SPECTRAL_TABLES_H_ */ 
