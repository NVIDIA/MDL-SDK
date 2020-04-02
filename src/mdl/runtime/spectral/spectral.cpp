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

#include "pch.h"

#include "i_spectral.h"
#include "spectral_tables.h"
#include "chroma_grid.h"

#include "mdl/compiler/compilercore/compilercore_assert.h"

#include <cstring>
#include <cmath>
#include <algorithm>

namespace mi {
namespace mdl {
namespace spectral {

// get value from a spectrum using linear interpolation
float get_value_lerp(
     const float * const table_values,
     const unsigned int num_values,
     const float lambda_min,
     const float lambda_max,
     const float lambda)
{
    MDL_ASSERT(num_values > 1);

    const float f = (lambda - lambda_min) / (lambda_max - lambda_min) * (float)(num_values - 1);
    unsigned int b0 = (unsigned int)(std::max(floorf(f), 0.0f));
    if (b0 >= num_values)
        b0 = num_values - 1;
    const unsigned int b1 = (b0 == num_values - 1) ? b0 : b0 + 1;

    const float f1 = f - (float)b0;
    return table_values[b0] * (1.0f - f1) + table_values[b1] * f1;
}



void spectrum_to_XYZ(
    float XYZ[3], const float *spectrum, const unsigned int num_values, 
    const float lambda_min, const float lambda_max)
{
    XYZ[0] = 0.0f;
    XYZ[1] = 0.0f;
    XYZ[2] = 0.0f;

    for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
    {
        const float lambda = SPECTRAL_XYZ_LAMBDA_MIN + (float)i * SPECTRAL_XYZ_LAMBDA_STEP;

        const float val = get_value_lerp(spectrum, num_values, lambda_min, lambda_max, lambda);
	XYZ[0] += SPECTRAL_XYZ1931_X[i] * val;
	XYZ[1] += SPECTRAL_XYZ1931_Y[i] * val;
	XYZ[2] += SPECTRAL_XYZ1931_Z[i] * val;
    }

    const float scale = (float)(683.002) * SPECTRAL_XYZ_LAMBDA_STEP;
    XYZ[0] *= scale;
    XYZ[1] *= scale;
    XYZ[2] *= scale;
}

float spectrum_to_Y(
    const float *spectrum, const unsigned int num_values, 
    const float lambda_min, const float lambda_max)
{
    float sum = 0.0f;
    for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
    {
        const float lambda = SPECTRAL_XYZ_LAMBDA_MIN + (float)i * SPECTRAL_XYZ_LAMBDA_STEP;

        const float val = get_value_lerp(spectrum, num_values, lambda_min, lambda_max, lambda);
        sum += SPECTRAL_XYZ1931_Y[i] * val;
    }

    // conversion from Watt to lumen
    return sum * (float)(683.002) * SPECTRAL_XYZ_LAMBDA_STEP;
}

#if 0
void spectrum_scale_to_Y(
    float * const spectrum, const unsigned int num_values,
    const float lambda_min, const float lambda_max, const float Y)
{
    const float luminance = spectrum_to_Y(
	spectrum, num_values, lambda_min, lambda_max);
    
    const float scale = Y / luminance;
    for (unsigned int i = 0; i < num_values; ++i)
	spectrum[i] *= scale;
}

#endif

void spectrum_resample(
    float *const target, const unsigned int target_num_values, 
    const float target_lambda_min, const float target_lambda_max,
    const float *const source, unsigned int const source_num_values, 
    const float source_lambda_min, const float source_lambda_max)
{
    MDL_ASSERT(target_num_values > 1);
    const float step = (target_lambda_max - target_lambda_min) / (float)(target_num_values - 1);
    for (unsigned int i = 0; i < target_num_values; ++i)
    {
	const float lambda = target_lambda_min + (float)i * step;
	target[i] = 
	    get_value_lerp(source, source_num_values, source_lambda_min, source_lambda_max, lambda);
    }
}

// helper for function below
static float get_value_incremental(
    const float * const table_lambda,
    const float * const table_values,
    const unsigned int num_values,
    unsigned int * const search_pos,
    const float lambda)
{
    unsigned int p1 = *search_pos;
    while (p1 < num_values && lambda > table_lambda[p1]){
	++p1;
    }
    const unsigned int p0 = p1 > 0 ? p1 - 1 : p1;
    *search_pos = p0;
    if (p1 >= num_values)
	p1 = num_values - 1;
    if (p0 == p1)
	return table_values[p0];
    
    const float w0 = (lambda - table_lambda[p0]) / (table_lambda[p1] - table_lambda[p0]);
    return 
	table_values[p0] * w0 + table_values[p1] * (1.0f - w0);
}

void spectrum_resample_input(
    float * const target, const unsigned int target_num_values,
    const float target_lambda_min, const float target_lambda_max,
    const float * const source, const float * const lambda_source,
    const unsigned int source_num_values)
{
    MDL_ASSERT(target_num_values > 1);
    const float step = (target_lambda_max - target_lambda_min) / (float)(target_num_values - 1);
    unsigned int search_pos = 0;
    for (unsigned int i = 0; i < target_num_values; ++i)
    {
	const float lambda = target_lambda_min + (float)i * step;
	target[i] = 
	    get_value_incremental(lambda_source, source, source_num_values, &search_pos, lambda);
    }
}

//  blackbody emitter, compute intensity at specific wavelength (in nm)
//  and temperature (in Kelvin) using Planck's law 
static float blackbody(const float lambda, const float temperature)
{
    const float c = 2.9979e14f; // speed of light (um / s)
    const float h = 6.626e-22f; // Planck constant (scaled to um^2)
    const float k = 1.38e-11f;  // Boltzmann constant (scaled to um^2)

    // nm -> um
    const float x = lambda * 1e-3f;

    const float f = 2.0f * h * c * c / (x * x * x * x * x);

    return f / (expf(h * c / (x * k * temperature)) - 1.0f);
}

#if 0
void create_blackbody_spectrum(
    float * const spectrum,
    const unsigned int num_lambda,
    const float lambda_min,
    const float lambda_max,
    const float temperature)
{
    MDL_ASSERT(num_lambda > 1);
    const float step = (lambda_max - lambda_min) / (float)(num_lambda - 1);
    for (unsigned int i = 0; i < num_lambda; ++i)
    {
	const float lambda = lambda_min + step * (float)i;
	
	spectrum[i] = blackbody(lambda, temperature);
    }
}
#endif

static float const tf_xyz_to_srgb[] = {
     3.240600f, -1.537200f, -0.498600f,
    -0.968900f,  1.875800f,  0.041500f,
     0.055700f, -0.204000f,  1.057000f
};

static float const tf_xyz_to_rec2020[] = {
     1.7166511879712680f, -0.3556707837763924f, -0.2533662813736599f,
    -0.6666843518324890f,  1.6164812366349390f,  0.0157685458139111f,
     0.0176398574453108f, -0.0427706132578085f,  0.9421031212354738f
};

static float const tf_xyz_to_aces[] = {
     1.049811017497974f,  0.000000000000000f, -0.000097484540579f,
    -0.495903023077320f,  1.373313045815706f,  0.098240036057310f,
     0.000000000000000f,  0.000000000000000f,  0.991252018200499f
};

static float const tf_xyz_to_acescg[] = {
     1.64110464846739856f, -0.32481937949200446f, -0.23643640375142377f,
    -0.66364264630652392f,  1.61528239536072316f,  0.01675583735671893f,
     0.01172343360382976f, -0.00828552987826917f,  0.98852465086629981f
};

const float *get_XYZ_to_cs(const Color_space_id cs)
{
    switch (cs)
    {
        default:
            MDL_ASSERT(!"invalid color space id");
        case CS_XYZ:
            return NULL;
        case CS_sRGB:
            return tf_xyz_to_srgb;
        case CS_ACES:
            return tf_xyz_to_aces;
        case CS_ACEScg:
            return tf_xyz_to_acescg;
        case CS_Rec2020:
            return tf_xyz_to_rec2020;
    }
}

static float const tf_srgb_to_xyz[] = {
    0.4123955889674142f, 0.3575834307637148f, 0.1804926473817016f,
    0.2125862307855955f, 0.7151703037034108f, 0.0722004986433362f,
    0.0192972154917469f, 0.1191838645808485f, 0.9504971251315798f
};

static float const tf_aces_to_xyz[] = {
    0.952552395938186f, 0.000000000000000f,  0.000093678631660f,
    0.343966449765075f, 0.728166096613486f, -0.072132546378561f,
    0.000000000000000f, 0.000000000000000f,  1.008825184351586f
};

static float const tf_acescg_to_xyz[] = {
    0.66242137586444849f, 0.13400828779274149f, 0.15616717971318694f,
    0.27221523580404405f, 0.67410229616954809f, 0.05368246802640798f,
   -0.00557437342943430f, 0.00406085720584065f, 1.01020644376967783f
};

static float const tf_rec2020_to_xyz[] = {
    6.36958048301291e-01f, 1.44616903586208e-01f, 1.68880975164172e-01f,
    2.62700212011267e-01f, 6.77998071518871e-01f, 5.93017164698620e-02f,
    3.48808167679321e-17f, 2.80726930490874e-02f, 1.06098505771079e+00f
};


const float *get_cs_to_XYZ(const Color_space_id cs)
{
    switch (cs)
    {
        default:
            MDL_ASSERT(!"invalid color space id");
        case CS_XYZ:
            return NULL;
        case CS_sRGB:
            return tf_srgb_to_xyz;
        case CS_ACES:
            return tf_aces_to_xyz;
        case CS_ACEScg:
            return tf_acescg_to_xyz;
        case CS_Rec2020:
            return tf_rec2020_to_xyz;
    }
}


void convert_XYZ_to_cs(float target[3], const float source[3], const Color_space_id cs)
{
    const float *const m = get_XYZ_to_cs(cs);
    if (!cs)
    {
        target[0] = source[0];
        target[1] = source[1];
        target[2] = source[2];
        return;
    }

    target[0] = source[0] * m[0];
    target[1] = source[0] * m[3];
    target[2] = source[0] * m[6];
    for (unsigned int i = 1; i < 3; ++i) {
        target[0] += source[i] * m[i];
        target[1] += source[i] * m[3 + i];
        target[2] += source[i] * m[6 + i];
    }
}

void convert_cs_to_XYZ(float target[3], const float source[3], const Color_space_id cs)
{
    const float *const m = get_cs_to_XYZ(cs);
    if (!m)
    {
        target[0] = source[0];
        target[1] = source[1];
        target[2] = source[2];
        return;
    }

    target[0] = source[0] * m[0];
    target[1] = source[0] * m[3];
    target[2] = source[0] * m[6];
    for (unsigned int i = 1; i < 3; ++i) {
        target[0] += source[i] * m[i];
        target[1] += source[i] * m[3 + i];
        target[2] += source[i] * m[6 + i];
    }
}



/// The MDL blackbody function implementation.
void mdl_blackbody(float sRGB[3], float kelvin)
{
    const float threshold = 500.0f;
    if (kelvin < threshold)
        kelvin = threshold;

    float XYZ[3] = {0.0f, 0.0f, 0.0f};
    // code currently operates on full resolution of our tabulated color matching functions
    for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
    {
        const float lambda = SPECTRAL_XYZ_LAMBDA_MIN + (float)i * SPECTRAL_XYZ_LAMBDA_STEP;
        const float f = blackbody(lambda, kelvin);

        XYZ[0] += SPECTRAL_XYZ1931_X[i] * f;
        XYZ[1] += SPECTRAL_XYZ1931_Y[i] * f;
        XYZ[2] += SPECTRAL_XYZ1931_Z[i] * f;
    }

    XYZ[0] /= XYZ[1];
    XYZ[2] /= XYZ[1];
    XYZ[1] = 1.0f;

    convert_XYZ_to_cs(sRGB, XYZ, CS_sRGB);

    sRGB[0] = std::max(sRGB[0], 0.0f);
    sRGB[1] = std::max(sRGB[1], 0.0f);
    sRGB[2] = std::max(sRGB[2], 0.0f);
}


float convert_cs_to_Y(const float source[3], const Color_space_id cs)
{
    const float *const m = get_cs_to_XYZ(cs);
    if (!m)
        return source[1];

    return source[0] * m[3] + source[1] * m[4] + source[2] * m[5];
}

void spectrum_to_cs_refl(
    float refl[3],
    const float *const spectrum,
    const unsigned int num_lambda,
    const float lambda_min, const float lambda_max,
    const Color_space_id cs)
{
    // use white point of color space as illuminant spectrum
    //!! TODO: make illuminant configurable?
    const float *illuminant;
    switch (cs)
    {
        default:
            MDL_ASSERT(!"invalid color space id");
        case CS_XYZ:
            illuminant = NULL; // E
            break;
        case CS_sRGB:
        case CS_Rec2020:
            illuminant = D65;
            break;
        case CS_ACES:
        case CS_ACEScg:
            illuminant = D60;
            break;
    }

    // compute true spectral multiplication result converted to XYZ
    float XYZ_spectral[3] = {0.0f, 0.0f, 0.0f};
    // and compute illuminant in converted to XYZ
    float XYZ_illum[3] = {0.0f, 0.0f, 0.0f};
    for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
    {
        const float lambda = SPECTRAL_XYZ_LAMBDA_MIN + (float)i * SPECTRAL_XYZ_LAMBDA_STEP;


        const float illum = illuminant ? illuminant[i] : 1.0f;

        const float val = get_value_lerp(
                spectrum, num_lambda, lambda_min, lambda_max, lambda) * illum;

        XYZ_spectral[0] += SPECTRAL_XYZ1931_X[i] * val;
        XYZ_spectral[1] += SPECTRAL_XYZ1931_Y[i] * val;
        XYZ_spectral[2] += SPECTRAL_XYZ1931_Z[i] * val;

        XYZ_illum[0] += SPECTRAL_XYZ1931_X[i] * illum;
        XYZ_illum[1] += SPECTRAL_XYZ1931_Y[i] * illum;
        XYZ_illum[2] += SPECTRAL_XYZ1931_Z[i] * illum;
    }
    // note: can ignore actual scaling for both integrals (since it's identical)

    // convert both to the color space
    float cs_spectral[3];
    convert_XYZ_to_cs(cs_spectral, XYZ_spectral, cs);
    float cs_illum[3];
    convert_XYZ_to_cs(cs_illum, XYZ_illum, cs);

    // compute correspoding color space reflectivity (note: can be < 0.0 and > 1.0)
    refl[0] = cs_spectral[0] / cs_illum[0];
    refl[1] = cs_spectral[1] / cs_illum[1];
    refl[2] = cs_spectral[2] / cs_illum[2];
}

static void cs_refl_to_spectrum_smits(
    float values[SPECTRAL_XYZ_RES],
    const float color[3],
    const Color_space_id cs)
{
    MDL_ASSERT(cs == CS_sRGB); //!! implement a generic solution

    float col[3] = { color[0], color[1], color[2] };

    // for now simply Smits' method with sRGB tables
    const float c111 = std::min(col[0], std::min(col[1], col[2]));

    for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
        values[i] = c111;

    col[0] -= c111;
    col[1] -= c111;
    col[2] -= c111;

    if (col[0] > 0.0f && col[1] > 0.0f) {
        const float c110 = std::min(col[0], col[1]);
        for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
            values[i] += c110 * srgb_relative_s110[i];
        col[0] -= c110;
        col[1] -= c110;
    }
    else if (col[0] > 0.0f && col[2] > 0.0f) {
        const float c101 = std::min(col[0], col[2]);
        for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
            values[i] += c101 * srgb_relative_s101[i];
        col[0] -= c101;
        col[2] -= c101;
    }
    else {
        const float c011 = std::min(col[1], col[2]);
        for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
            values[i] += c011 * srgb_relative_s011[i];
        col[1] -= c011;
        col[2] -= c011;
    }

    if (col[0] > 0.0f) {
        for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
            values[i] += col[0] * srgb_relative_s100[i];
    }
    if (col[1]> 0.0f) {
        for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
            values[i] += col[1] * srgb_relative_s010[i];
    }
    if (col[2] > 0.0f) {
        for (unsigned int i = 0; i < SPECTRAL_XYZ_RES; ++i)
            values[i] += col[2] * srgb_relative_s001[i];
    }
}


void cs_refl_to_spectrum(
    float values[SPECTRAL_XYZ_RES],
    const float color[3],
    const Color_space_id cs,
    const bool aggressive,
    const bool ignore_scale)
{
    if (aggressive && (cs == CS_sRGB))
    {
        cs_refl_to_spectrum_smits(values, color, cs);
        return;
    }

    memset(values, 0, SPECTRAL_XYZ_RES * sizeof(float));

    const Chroma_grid_info *chroma_grid_info;
    const Chroma_cell *chroma_cells;
    const float *chroma_spectra;
    float illum_cs; // illuminant in color space (is 'white' in that color space, so only scalar)
    switch (cs)
    {
        default:
        case CS_XYZ:
            chroma_grid_info = &chroma_grid_info_e;
            chroma_cells = chroma_cells_e;
            chroma_spectra = chroma_spectra_e;
            illum_cs = (float)(1.0 / 3.0);
            break;
        case CS_ACES:
        case CS_ACEScg:
            chroma_grid_info = &chroma_grid_info_d60;
            chroma_cells = chroma_cells_d60;
            chroma_spectra = chroma_spectra_d60;
            illum_cs = 0.337670f;
            break;
        case CS_sRGB:
        case CS_Rec2020:
            chroma_grid_info = &chroma_grid_info_d65;
            chroma_cells = chroma_cells_d65;
            chroma_spectra = chroma_spectra_d65;
            illum_cs = 0.329000f;
            break;
    }

    const float val_cs[3] = {
        color[0] * illum_cs,
        color[1] * illum_cs,
        color[2] * illum_cs
    };

    float val_XYZ[3];
    convert_cs_to_XYZ(val_XYZ, val_cs, cs);

    const float sum = val_XYZ[0] + val_XYZ[1] + val_XYZ[2];
    const float x = val_XYZ[0] / sum;
    const float y = val_XYZ[1] / sum;

    unsigned int idx[4];
    float w[4];
    const unsigned int num = get_spectra(idx, w, x, y, chroma_grid_info, chroma_cells);
    if (num == 0)
        return;

    const float *illum = &chroma_spectra[chroma_grid_info->white_idx * (SPECTRAL_XYZ_RES + 1) + 1];

    float max_refl = 0.0f;
    for (unsigned int j = 0; j < num; ++j)
    {
        const float *s = &chroma_spectra[idx[j] * (SPECTRAL_XYZ_RES + 1) + 1];
        max_refl += w[j] * s[-1];

        for (unsigned int k = 0; k < SPECTRAL_XYZ_RES; ++k)
            values[k] += w[j] * s[k] / illum[k];
    }

    float scale = val_XYZ[1] / y;
    if (ignore_scale)
    {
        for (unsigned int j = 0; j < SPECTRAL_XYZ_RES; ++j)
            values[j] *= scale;
    }
    else
    {
        const float max_scale = max_refl > 0.0f ? 1.0f / max_refl : 0.0f;
        scale = std::min(scale, max_scale);

        for (unsigned int j = 0; j < SPECTRAL_XYZ_RES; ++j)
            values[j] = std::min(1.0f, values[j] * scale); //!! paranoia clamp to 1.0
    }
}


void cs_emission_to_spectrum(
    float values[SPECTRAL_XYZ_RES],
    const float color[3],
    const Color_space_id cs)
{
    memset(values, 0, SPECTRAL_XYZ_RES * sizeof(float));

    const Chroma_grid_info *chroma_grid_info;
    const Chroma_cell *chroma_cells;
    const float *chroma_spectra;
    switch (cs)
    {
        default:
        case CS_XYZ:
            chroma_grid_info = &chroma_grid_info_e;
            chroma_cells = chroma_cells_e;
            chroma_spectra = chroma_spectra_e;
            break;
        case CS_ACES:
        case CS_ACEScg:
            chroma_grid_info = &chroma_grid_info_d60;
            chroma_cells = chroma_cells_d60;
            chroma_spectra = chroma_spectra_d60;
            break;
        case CS_sRGB:
        case CS_Rec2020:
            chroma_grid_info = &chroma_grid_info_d65;
            chroma_cells = chroma_cells_d65;
            chroma_spectra = chroma_spectra_d65;
            break;
    }

    float val_XYZ[3];
    convert_cs_to_XYZ(val_XYZ, color, cs);

    const float sum = val_XYZ[0] + val_XYZ[1] + val_XYZ[2];
    const float x = val_XYZ[0] / sum;
    const float y = val_XYZ[1] / sum;

    unsigned int idx[4];
    float w[4];
    const unsigned int num = get_spectra(idx, w, x, y, chroma_grid_info, chroma_cells);
    if (num == 0)
        return;

    for (unsigned int j = 0; j < num; ++j)
    {
        const float *s = &chroma_spectra[idx[j] * (SPECTRAL_XYZ_RES + 1) + 1];

        for (unsigned int k = 0; k < SPECTRAL_XYZ_RES; ++k)
            values[k] += w[j] * s[k];
    }

    // do scaling since tables were constructed for simple matrix product "M_srgb * (x,y,z) * spectrum = color", not for integral spectral radiometric -> photometric
    const float scale = sum * (float)(1.0 / (683.002 * (SPECTRAL_XYZ_LAMBDA_STEP)));
    for (unsigned int j = 0; j < SPECTRAL_XYZ_RES; ++j)
        values[j] *= scale;
}


} // namespace mi
} // namespace mdl
} // namespace spectral
