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
/// \brief  Low level utilities for conversion and creation of light spectra.
///

#ifndef MDL_RUNTIME_SPECTRAL_I_SPECTRAL_H_
#define MDL_RUNTIME_SPECTRAL_I_SPECTRAL_H_

#include "spectral_tables.h"

namespace mi {
namespace mdl {
namespace spectral {

// convention for spectra:
// - num_samples point samples between lambda_min and lambda_max
// - first sample is at lambda_min, last at lambda max
// - equidistant point samples, step size is (lambda_max - lambda_min) / (num_samples - 1)   
// - num_samples >= 2

// get value from a spectrum using linear interpolation
float get_value_lerp(
     const float *table_values,
     unsigned int num_values,
     float lambda_min,
     float lambda_max,
     float lambda);

// convert a spectrum to XYZ, transforming radiometric to photometric quantities
// (i.e. if spectral intensities are absolute in W / (m^2 * nm), luminance output is lum / m^2,
// if it is radiance in W / (m^2 * nm * sr) output is cd / m^2)
void spectrum_to_XYZ(
    float XYZ[3], const float *spectrum, unsigned int num_values, 
    float lambda_min, float lambda_max);


// compute CIE XYZ Y-component (== luminance)
float spectrum_to_Y(
    const float *spectrum, const unsigned int num_values,
    const float lambda_min, const float lambda_max);

#if 0
// scale to a given luminance, i.e. conversion to XYZ has the specified luminance
void spectrum_scale_to_Y(
    float * const spectrum, unsigned int num_values,
    float lambda_min, float lambda_max, float Y);
#endif

// change number of point samples and range
//!! TODO: add some kind of oversampling to avoid aliasing
void spectrum_resample(
    float *target, unsigned int target_num_values, float target_lambda_min, float target_lambda_max,
    const float *source, unsigned int source_num_values, float source_lambda_min, 
    float source_lambda_max);

// change number of point samples, use arbitrarily placed input samples
//!! TODO: add some kind of oversampling to avoid aliasing
void spectrum_resample_input(
    float *target, unsigned int target_num_values, float target_lambda_min, float target_lambda_max,
    const float *source, const float *lambda_source, unsigned int source_num_values);

#if 0
// create blackbody spectrum according to Planck's law
void create_blackbody_spectrum(
    float *spectrum,
    unsigned int num_lambda,
    float lambda_min,
    float lambda_max,
    float temperature); // [K]
#endif

/// The MDL blackbody function implementation.
void mdl_blackbody(float sRGB[3], float kelvin);


/// known color spaces
enum Color_space_id {
    CS_XYZ,     // CIE XYZ
    CS_sRGB,    // (linear) sRGB, aka rec709 (HDTV)
    CS_ACES,    // Academy Color Encoding System (ACES2065-1, AP0 primaries)
    CS_ACEScg, // Academy Color Encoding System (ACEScg, AP1 primaries)
    CS_Rec2020  // U-HDTV
};

// conversion from and to CIE XYZ
const float *get_XYZ_to_cs(Color_space_id);
const float *get_cs_to_XYZ(Color_space_id);

void convert_XYZ_to_cs(float target[3], const float source[3], const Color_space_id target_id);
void convert_cs_to_XYZ(float target[3], const float source[3], const Color_space_id source_id);
float convert_cs_to_Y(const float source[3], const Color_space_id source_id);

// convert a reflectivity spectrum to a reflectivity color in a color space such that
// the result is identical for direct illumination (assuming the white point of the color space as
// illuminant)
// NOTE: result can be < 0.0 and > 1.0, treating that is up to the caller
void spectrum_to_cs_refl(
    float refl[3],
    const float *spectrum,
    unsigned int num_lambda,
    float lambda_min, float lambda_max,
    Color_space_id cs);

// re-construct a reflectivity spectrum from a color reflectivity (assuming the white point of the
// color space as illuminant)
// NOTE: result can be > 1.0 (if ignore_scale is set)
void cs_refl_to_spectrum(
    float values[SPECTRAL_XYZ_RES],
    const float color[3],
    Color_space_id cs,
    bool aggressive = false,    // use Smits' method to get maximum reflectivity in the sRGB gamut
                                // (sRGB-only, result lacks smoothness)
    bool ignore_scale = false); // don't do scaling such that the result is <= 1

// re-construct a emission spectrum from a color emission
void cs_emission_to_spectrum(
    float values[SPECTRAL_XYZ_RES],
    const float color[3],
    Color_space_id cs);


} // namespace spectral
} // namespace mdl
} // namespace mi

#endif // MDL_RUNTIME_SPECTRAL_I_SPECTRAL_H_


