/******************************************************************************
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "example_df_cuda.h"

#ifdef ENABLE_SPECTRAL
// Suppress the dummy spectral runtime stubs in texture_support_cuda.h. We provide real
// implementations further down that read the wavelengths from the spectral shading state
// and perform proper spectral upsampling.
#define TEX_SUPPORT_NO_DUMMY_SPECTRAL

// Pre-include the MDL target types so we can forward-declare the real spectral runtime
// functions before texture_support_cuda.h initializes the texture vtable that references them.
#include <mi/mdl/mdl_target_types.h>

extern "C" __device__ void rgb_to_spectral_ior(
    mi::mdl::tct_spectral_sample             *result,
    mi::mdl::Texture_handler_base const      *self_base,
    mi::mdl::Shading_state_material          *state,
    float const                               rgb[3]);
extern "C" __device__ void rgb_to_spectral_ior_deriv(
    mi::mdl::tct_spectral_sample                  *result,
    mi::mdl::Texture_handler_base const           *self_base,
    mi::mdl::Shading_state_material_with_derivs   *state,
    float const                                    rgb[3]);
extern "C" __device__ void rgb_to_spectral_reflectance(
    mi::mdl::tct_spectral_sample             *result,
    mi::mdl::Texture_handler_base const      *self_base,
    mi::mdl::Shading_state_material          *state,
    float const                               rgb[3]);
extern "C" __device__ void rgb_to_spectral_reflectance_deriv(
    mi::mdl::tct_spectral_sample                  *result,
    mi::mdl::Texture_handler_base const           *self_base,
    mi::mdl::Shading_state_material_with_derivs   *state,
    float const                                    rgb[3]);
extern "C" __device__ void rgb_to_spectral_luminance(
    mi::mdl::tct_spectral_sample             *result,
    mi::mdl::Texture_handler_base const      *self_base,
    mi::mdl::Shading_state_material          *state,
    float const                               rgb[3]);
extern "C" __device__ void rgb_to_spectral_luminance_deriv(
    mi::mdl::tct_spectral_sample                  *result,
    mi::mdl::Texture_handler_base const           *self_base,
    mi::mdl::Shading_state_material_with_derivs   *state,
    float const                                    rgb[3]);
extern "C" __device__ void rgb_to_spectral_volume_coefficient(
    mi::mdl::tct_spectral_sample             *result,
    mi::mdl::Texture_handler_base const      *self_base,
    mi::mdl::Shading_state_material          *state,
    float const                               rgb[3]);
extern "C" __device__ void rgb_to_spectral_volume_coefficient_deriv(
    mi::mdl::tct_spectral_sample                  *result,
    mi::mdl::Texture_handler_base const           *self_base,
    mi::mdl::Shading_state_material_with_derivs   *state,
    float const                                    rgb[3]);
extern "C" __device__ void get_wavelengths(
    mi::mdl::tct_spectral_sample             *result,
    mi::mdl::Texture_handler_base const      *self_base,
    mi::mdl::Shading_state_material          *state);
extern "C" __device__ void get_wavelengths_deriv(
    mi::mdl::tct_spectral_sample                  *result,
    mi::mdl::Texture_handler_base const           *self_base,
    mi::mdl::Shading_state_material_with_derivs   *state);
#endif

#include "texture_support_cuda.h"

// To reuse this sample code for the MDL SDK and MDL Core the corresponding namespaces are used.

// when this CUDA code is used in the context of an SDK example.
#if defined(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR)
    #define BSDF_USE_MATERIAL_IOR MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR
    using namespace mi::neuraylib;
// when this CUDA code is used in the context of an Core example.
#elif defined(MDL_CORE_BSDF_USE_MATERIAL_IOR)
    #define BSDF_USE_MATERIAL_IOR MDL_CORE_BSDF_USE_MATERIAL_IOR
    using namespace mi::mdl;
#endif

// If enabled, math::DX(state::texture_coordinates(0).xy) = float2(1, 0) and
// math::DY(state::texture_coordinates(0).xy) = float2(0, 1) will be used.
// #define USE_FAKE_DERIVATIVES

// Color mode for the BSDF/EDF data structures.
// The MDL backend's "libbsdf_enable_spectral" option controls which color mode the generated
// MDL functions expect, so this define MUST match how the PTX file was generated
// (see CMakeLists.txt and the host side).
#ifdef ENABLE_SPECTRAL
#define BSDF_TCCM TCCM_SPECTRAL_SAMPLING
typedef tct_spectral_sample Color_sample;
typedef tct_spectral_sample Pdf_sample;
#else
#define BSDF_TCCM TCCM_RGB
typedef float3 Color_sample;
typedef float  Pdf_sample;
#endif

#ifdef ENABLE_DERIVATIVES
typedef Material_expr_function_with_derivs  Mat_expr_func;
typedef Bsdf_init_function_with_derivs      Bsdf_init_func;
typedef Bsdf_sample_function_with_derivs    Bsdf_sample_func;
typedef Bsdf_evaluate_function_with_derivs  Bsdf_evaluate_func;
typedef Bsdf_pdf_function_with_derivs       Bsdf_pdf_func;
typedef Edf_init_function_with_derivs       Edf_init_func;
typedef Edf_sample_function_with_derivs     Edf_sample_func;
typedef Edf_evaluate_function_with_derivs   Edf_evaluate_func;
typedef Edf_pdf_function_with_derivs        Edf_pdf_func;
#ifdef ENABLE_SPECTRAL
typedef Shading_state_material_spectral_with_derivs Mdl_state;
#else
typedef Shading_state_material_with_derivs  Mdl_state;
#endif
typedef Texture_handler_deriv               Tex_handler;
#define TEX_VTABLE                          tex_deriv_vtable
#else
typedef Material_expr_function              Mat_expr_func;
typedef Bsdf_init_function                  Bsdf_init_func;
typedef Bsdf_sample_function                Bsdf_sample_func;
typedef Bsdf_evaluate_function              Bsdf_evaluate_func;
typedef Bsdf_pdf_function                   Bsdf_pdf_func;
typedef Edf_init_function                   Edf_init_func;
typedef Edf_sample_function                 Edf_sample_func;
typedef Edf_evaluate_function               Edf_evaluate_func;
typedef Edf_pdf_function                    Edf_pdf_func;
#ifdef ENABLE_SPECTRAL
typedef Shading_state_material_spectral     Mdl_state;
#else
typedef Shading_state_material              Mdl_state;
#endif
typedef Texture_handler                     Tex_handler;
#define TEX_VTABLE                          tex_vtable
#endif

// Custom structure representing the resources used by the generated code of a target code object.
// The layout has to match the host-side struct Target_code_data in example_cuda_shared.h.
struct Target_code_data
{
    size_t       num_textures;      // number of elements in the textures field
    Texture     *textures;          // a list of Texture objects, if used

    size_t       num_mbsdfs;        // number of elements in the mbsdfs field
    Mbsdf       *mbsdfs;            // a list of Mbsdf objects, if used

    char const  *ro_data_segment;   // the read-only data segment, if used
};


// all function types
union Mdl_function_ptr
{
    Mat_expr_func           *expression;
    Bsdf_init_func          *bsdf_init;
    Bsdf_sample_func        *bsdf_sample;
    Bsdf_evaluate_func      *bsdf_evaluate;
    Bsdf_pdf_func           *bsdf_pdf;
    Edf_init_func           *edf_init;
    Edf_sample_func         *edf_sample;
    Edf_evaluate_func       *edf_evaluate;
    Edf_pdf_func            *edf_pdf;
};


// function index offset depending on the target code
extern __constant__ unsigned int     mdl_target_code_offsets[];

// number of generated functions
extern __constant__ unsigned int     mdl_functions_count;

// the following arrays are indexed by an mdl_function_index
extern __constant__ Mdl_function_ptr mdl_functions[];
extern __constant__ unsigned int     mdl_arg_block_indices[];

// Identity matrix.
// The last row is always implied to be (0, 0, 0, 1).
__constant__ const float4 identity[3] = {
    {1.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 1.0f, 0.0f}
};


// the material provides pairs for each generated function to evaluate
// the functions and arg blocks array are indexed by:
// mdl_target_code_offsets[target_code_index] + function_index
typedef uint3 Mdl_function_index;
__device__ inline Mdl_function_index get_mdl_function_index(const uint2& index_pair)
{
    return make_uint3(
        index_pair.x,   // target_code_index
        index_pair.y,   // function_index inside target code
        mdl_target_code_offsets[index_pair.x] + index_pair.y); // global function index
}

// resource handler for accessing textures and other data
// depends on the target code (link unit)
struct Mdl_resource_handler
{
    __device__ Mdl_resource_handler()
    {
        m_tex_handler.vtable = &TEX_VTABLE;   // only required in 'vtable' mode, otherwise NULL
        data.shared_data = NULL;
        data.texture_handler = reinterpret_cast<Texture_handler_base *>(&m_tex_handler);
    }

    // reuse the handler with a different target code index
    __device__ inline void set_target_code_index(
        const Kernel_params& params, const Mdl_function_index& index)
    {
        m_tex_handler.num_textures = params.tc_data[index.x].num_textures;
        m_tex_handler.textures = params.tc_data[index.x].textures;
        m_tex_handler.num_mbsdfs = params.tc_data[index.x].num_mbsdfs;
        m_tex_handler.mbsdfs = params.tc_data[index.x].mbsdfs;
    }

    // a pointer to this data is passed to all generated functions
    Resource_data data;

private:
    Tex_handler m_tex_handler;
};


// checks if the indexed function can be evaluated or not
__device__ inline bool is_valid(const Mdl_function_index& index)
{
    return index.y != 0xFFFFFFFFu;
}

// get a pointer to the material parameters which is passed to all generated functions
__device__ inline const char* get_arg_block(
    const Kernel_params& params,
    const Mdl_function_index& index)
{
    return params.arg_block_list[mdl_arg_block_indices[index.z]];
}

// restores the normal since the BSDF init will change it
// sets the read-only data segment pointer for large arrays
__device__ inline void prepare_state(
    const Kernel_params& params,
    const Mdl_function_index& index,
    Mdl_state& state,
    const tct_float3& normal)
{
    state.ro_data_segment = params.tc_data[index.x].ro_data_segment;
    state.normal = normal;
}

// Expression functions
__device__ inline Mat_expr_func* as_expression(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 0].expression;
}

// BSDF functions
__device__ inline Bsdf_init_func* as_bsdf_init(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 0].bsdf_init;
}

__device__ inline Bsdf_sample_func* as_bsdf_sample(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 1].bsdf_sample;
}

__device__ inline Bsdf_evaluate_func* as_bsdf_evaluate(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 2].bsdf_evaluate;
}

__device__ inline Bsdf_pdf_func* as_bsdf_pdf(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 3].bsdf_pdf;
}

// EDF functions
__device__ inline Edf_init_func* as_edf_init(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 0].edf_init;
}

__device__ inline Edf_sample_func* as_edf_sample(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 1].edf_sample;
}

__device__ inline Edf_evaluate_func* as_edf_evaluate(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 2].edf_evaluate;
}

__device__ inline Edf_pdf_func* as_edf_pdf(const Mdl_function_index& index)
{
    return mdl_functions[index.z + 3].edf_pdf;
}


// 3d vector math utilities
__device__ inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ inline float3 operator*(const float3& a, const float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ inline float3 operator/(const float3& a, const float s)
{
    return make_float3(a.x / s, a.y / s, a.z / s);
}
__device__ inline void operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
__device__ inline void operator-=(float3& a, const float3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
__device__ inline void operator*=(float3& a, const float3& b)
{
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}
__device__ inline void operator*=(float3& a, const float& s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
__device__ inline float squared_length(const float3 &d)
{
    return d.x * d.x + d.y * d.y + d.z * d.z;
}
__device__ inline float3 normalize(const float3 &d)
{
    const float inv_len = 1.0f / sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    return make_float3(d.x * inv_len, d.y * inv_len, d.z * inv_len);
}
__device__ inline float dot(const float3 &u, const float3 &v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}
__device__ inline float3 cross(const float3 &u, const float3 &v)
{
    return make_float3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x);
}

//-------------------------------------------------------------------------------------------------
// Color_sample / Pdf_sample helpers
//
// In RGB mode Color_sample == float3 / Pdf_sample == float, in spectral mode they are
// tct_spectral_sample. The helpers below abstract over both so the path tracer can use the same
// code regardless of color mode.
//-------------------------------------------------------------------------------------------------

#ifdef ENABLE_SPECTRAL

__device__ inline tct_spectral_sample make_color_sample(float v)
{
    tct_spectral_sample s;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
        s.values[i] = v;
    return s;
}

__device__ inline tct_spectral_sample addcc(
    const tct_spectral_sample &a, const tct_spectral_sample &b)
{
    tct_spectral_sample r;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
        r.values[i] = a.values[i] + b.values[i];
    return r;
}

__device__ inline tct_spectral_sample mulcc(
    const tct_spectral_sample &a, const tct_spectral_sample &b)
{
    tct_spectral_sample r;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
        r.values[i] = a.values[i] * b.values[i];
    return r;
}

__device__ inline tct_spectral_sample mulcf(const tct_spectral_sample &a, float b)
{
    tct_spectral_sample r;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
        r.values[i] = a.values[i] * b;
    return r;
}

__device__ inline float get_main_pdf(const tct_spectral_sample &p)
{
    return p.values[0];
}

// D65 standard illuminant SPD, 360-830 nm in 5 nm steps (95 entries).
// Scaled so that the luminance integral (Y channel) equals 1.
__constant__ float s_cie_d65[95] = {
    6.462114e-06f, 6.839740e-06f, 7.217367e-06f, 7.070938e-06f,
    6.924510e-06f, 7.248223e-06f, 7.571951e-06f, 9.519149e-06f,
    1.146636e-05f, 1.207124e-05f, 1.267613e-05f, 1.281093e-05f,
    1.294573e-05f, 1.247813e-05f, 1.201053e-05f, 1.327021e-05f,
    1.452989e-05f, 1.537108e-05f, 1.621241e-05f, 1.626811e-05f,
    1.632381e-05f, 1.611929e-05f, 1.591492e-05f, 1.598850e-05f,
    1.606207e-05f, 1.556936e-05f, 1.507664e-05f, 1.511419e-05f,
    1.515188e-05f, 1.504436e-05f, 1.493684e-05f, 1.472817e-05f,
    1.451950e-05f, 1.472027e-05f, 1.492118e-05f, 1.469367e-05f,
    1.446616e-05f, 1.444122e-05f, 1.441642e-05f, 1.413611e-05f,
    1.385581e-05f, 1.360185e-05f, 1.334788e-05f, 1.331004e-05f,
    1.327220e-05f, 1.278016e-05f, 1.228811e-05f, 1.237960e-05f,
    1.247109e-05f, 1.244288e-05f, 1.241468e-05f, 1.228302e-05f,
    1.215136e-05f, 1.184583e-05f, 1.154031e-05f, 1.156876e-05f,
    1.159720e-05f, 1.134278e-05f, 1.108836e-05f, 1.110137e-05f,
    1.111438e-05f, 1.125732e-05f, 1.140026e-05f, 1.112358e-05f,
    1.084691e-05f, 1.025367e-05f, 9.660451e-06f, 9.791236e-06f,
    9.922021e-06f, 1.011183e-05f, 1.030166e-05f, 9.418694e-06f,
    8.535733e-06f, 9.109474e-06f, 9.683216e-06f, 1.004356e-05f,
    1.040391e-05f, 9.607591e-06f, 8.811283e-06f, 7.621443e-06f,
    6.431617e-06f, 7.844023e-06f, 9.256429e-06f, 9.019315e-06f,
    8.782200e-06f, 8.846020e-06f, 8.909840e-06f, 8.573684e-06f,
    8.237542e-06f, 7.718434e-06f, 7.199340e-06f, 7.579100e-06f,
    7.958860e-06f, 8.157816e-06f, 8.356785e-06f
};

__device__ inline float lookup_d65(float lambda)
{
    float f = (lambda - 360.0f) / (830.0f - 360.0f);
    if (f < 0.0f || f > 1.0f)
        return 0.0f;
    f *= float(95 - 1);
    int b0 = min(int(f), 95 - 1);
    int b1 = (b0 < (95 - 1)) ? (b0 + 1) : b0;
    float w1 = f - float(b0);
    return s_cie_d65[b0] * (1.0f - w1) + s_cie_d65[b1] * w1;
}

// RGB-to-spectral conversion.
// Uses Jendersie - "Fast Spectral Upsampling of Volume Attenuation Coefficients".
__device__ inline tct_spectral_sample rgb_to_spectral(
    const float3 &rgb, const tct_spectral_sample &lambdas, bool is_emission)
{
    tct_spectral_sample s;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        float lambda = lambdas.values[i];
        s.values[i] = (lambda < 485.0f) ? rgb.z : ((lambda < 595.9f) ? rgb.y : rgb.x);

        // for emission, apply spectral illuminant
        if (is_emission)
            s.values[i] *= lookup_d65(lambda);
    }
    return s;
}

// Convert a spectral sample to linear RGB.
__device__ inline float3 spectral_to_rgb(
    const tct_spectral_sample &values,
    const tct_spectral_sample &lambdas,
    bool is_reflectivity,
    const Kernel_params &params)
{
    float3 xyz = make_float3(0.0f, 0.0f, 0.0f);

    // weight by CIE XYZ color matching functions
    // (Wyman et al - "Simple Analytic Approximations to the CIE XYZ Color Matching Functions")
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        const float lambda = lambdas.values[i];
        if (lambda < 360.0f || lambda > 830.0f)
            continue;

        // for reflectivity values multiply by the spectral whitepoint of the RGB color space
        // (normalized to luminance 1)
        const float factor = is_reflectivity ? lookup_d65(lambda) : 1.0f;
        {
            const float p1 = (lambda - 442.0f) * ((lambda < 442.0f) ? 0.0624f : 0.0374f);
            const float p2 = (lambda - 599.8f) * ((lambda < 599.8f) ? 0.0264f : 0.0323f);
            const float p3 = (lambda - 501.1f) * ((lambda < 501.1f) ? 0.0490f : 0.0382f);
            xyz.x += (0.362f * expf(-0.5f * p1 * p1)
                    + 1.056f * expf(-0.5f * p2 * p2)
                    - 0.065f * expf(-0.5f * p3 * p3)) * values.values[i] * factor;
        }
        {
            const float p1 = (lambda - 568.8f) * ((lambda < 568.8f) ? 0.0213f : 0.0247f);
            const float p2 = (lambda - 530.9f) * ((lambda < 530.9f) ? 0.0613f : 0.0322f);
            xyz.y += (0.821f * expf(-0.5f * p1 * p1)
                    + 0.286f * expf(-0.5f * p2 * p2)) * values.values[i] * factor;
        }
        {
            const float p1 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
            const float p2 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);
            xyz.z += (1.217f * expf(-0.5f * p1 * p1)
                    + 0.681f * expf(-0.5f * p2 * p2)) * values.values[i] * factor;
        }
    }

    // apply scaling from radiometric to photometric units
    xyz *= 683.002f;

    // MDL_DF_SPECTRAL_SAMPLES samples uniformly on wavelength range
    if (params.spectral_max_wavelength != params.spectral_min_wavelength)
        xyz *= (params.spectral_max_wavelength - params.spectral_min_wavelength)
             / float(MDL_DF_SPECTRAL_SAMPLES);

    // convert to linear sRGB
    return make_float3(
        dot(xyz, make_float3( 3.240600f, -1.537200f, -0.498600f)),
        dot(xyz, make_float3(-0.968900f,  1.875800f,  0.041500f)),
        dot(xyz, make_float3( 0.055700f, -0.204000f,  1.057000f)));
}

//-------------------------------------------------------------------------------------------------
// Spectral runtime functions called by libbsdf for spectral upsampling of MDL color values.
//
// These override the default-zero stubs from texture_support_cuda.h with real implementations
// that read the wavelengths from the spectral shading state and perform proper spectral
// upsampling. Without these, every BSDF that uses a color material parameter would return zero
// in spectral mode, which causes the rendered image to be black.
//-------------------------------------------------------------------------------------------------

__device__ inline tct_spectral_sample state_spectral_wavelengths(
    const Shading_state_material *state)
{
    return static_cast<const Shading_state_material_spectral *>(state)->spectral_wavelengths;
}

__device__ inline tct_spectral_sample state_spectral_wavelengths(
    const Shading_state_material_with_derivs *state)
{
    return static_cast<const Shading_state_material_spectral_with_derivs *>(state)
        ->spectral_wavelengths;
}

// Spectral upsampling of an IOR value
// (piecewise-linear with point samples at 435, 546 and 700 nm for blue, green and red).
__device__ inline tct_spectral_sample rgb_to_spectral_ior_impl(
    const tct_spectral_sample &lambdas, const float rgb[3])
{
    tct_spectral_sample s;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        const float lambda = lambdas.values[i];
        if (lambda > 546.0f)
        {
            const float t = fminf((lambda - 546.0f) * (1.0f / (700.0f - 546.0f)), 1.0f);
            s.values[i] = t * rgb[0] + (1.0f - t) * rgb[1];
        }
        else
        {
            const float t = fmaxf((lambda - 435.0f) * (1.0f / (546.0f - 435.0f)), 0.0f);
            s.values[i] = t * rgb[1] + (1.0f - t) * rgb[2];
        }
    }
    return s;
}

// Spectral upsampling of a reflectance / volume coefficient value
// (Jendersie - "Fast Spectral Upsampling of Volume Attenuation Coefficients").
__device__ inline tct_spectral_sample rgb_to_spectral_jendersie(
    const tct_spectral_sample &lambdas, const float rgb[3], bool is_emission)
{
    tct_spectral_sample s;
    for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        const float lambda = lambdas.values[i];
        s.values[i] = (lambda < 485.0f) ? rgb[2] : ((lambda < 595.9f) ? rgb[1] : rgb[0]);
        if (is_emission)
            s.values[i] *= lookup_d65(lambda);
    }
    return s;
}

extern "C" __device__ void rgb_to_spectral_ior(
    tct_spectral_sample        *result,
    Texture_handler_base const *self_base,
    Shading_state_material     *state,
    float const                 rgb[3])
{
    *result = rgb_to_spectral_ior_impl(state_spectral_wavelengths(state), rgb);
}

extern "C" __device__ void rgb_to_spectral_ior_deriv(
    tct_spectral_sample                 *result,
    Texture_handler_base const          *self_base,
    Shading_state_material_with_derivs  *state,
    float const                          rgb[3])
{
    *result = rgb_to_spectral_ior_impl(state_spectral_wavelengths(state), rgb);
}

extern "C" __device__ void rgb_to_spectral_reflectance(
    tct_spectral_sample        *result,
    Texture_handler_base const *self_base,
    Shading_state_material     *state,
    float const                 rgb[3])
{
    *result = rgb_to_spectral_jendersie(
        state_spectral_wavelengths(state), rgb, /*is_emission=*/false);
}

extern "C" __device__ void rgb_to_spectral_reflectance_deriv(
    tct_spectral_sample                 *result,
    Texture_handler_base const          *self_base,
    Shading_state_material_with_derivs  *state,
    float const                          rgb[3])
{
    *result = rgb_to_spectral_jendersie(
        state_spectral_wavelengths(state), rgb, /*is_emission=*/false);
}

extern "C" __device__ void rgb_to_spectral_luminance(
    tct_spectral_sample        *result,
    Texture_handler_base const *self_base,
    Shading_state_material     *state,
    float const                 rgb[3])
{
    *result = rgb_to_spectral_jendersie(
        state_spectral_wavelengths(state), rgb, /*is_emission=*/true);
}

extern "C" __device__ void rgb_to_spectral_luminance_deriv(
    tct_spectral_sample                 *result,
    Texture_handler_base const          *self_base,
    Shading_state_material_with_derivs  *state,
    float const                          rgb[3])
{
    *result = rgb_to_spectral_jendersie(
        state_spectral_wavelengths(state), rgb, /*is_emission=*/true);
}

extern "C" __device__ void rgb_to_spectral_volume_coefficient(
    tct_spectral_sample        *result,
    Texture_handler_base const *self_base,
    Shading_state_material     *state,
    float const                 rgb[3])
{
    *result = rgb_to_spectral_jendersie(
        state_spectral_wavelengths(state), rgb, /*is_emission=*/false);
}

extern "C" __device__ void rgb_to_spectral_volume_coefficient_deriv(
    tct_spectral_sample                 *result,
    Texture_handler_base const          *self_base,
    Shading_state_material_with_derivs  *state,
    float const                          rgb[3])
{
    *result = rgb_to_spectral_jendersie(
        state_spectral_wavelengths(state), rgb, /*is_emission=*/false);
}

extern "C" __device__ void get_wavelengths(
    tct_spectral_sample        *result,
    Texture_handler_base const *self_base,
    Shading_state_material     *state)
{
    *result = state_spectral_wavelengths(state);
}

extern "C" __device__ void get_wavelengths_deriv(
    tct_spectral_sample                 *result,
    Texture_handler_base const          *self_base,
    Shading_state_material_with_derivs  *state)
{
    *result = state_spectral_wavelengths(state);
}

#else // !ENABLE_SPECTRAL

__device__ inline float3 make_color_sample(float v)
{
    return make_float3(v, v, v);
}

__device__ inline float3 addcc(const float3 &a, const float3 &b) { return a + b; }
__device__ inline float3 mulcc(const float3 &a, const float3 &b) { return a * b; }
__device__ inline float3 mulcf(const float3 &a, float b)         { return a * b; }
__device__ inline float   get_main_pdf(float p)                  { return p; }

#endif // ENABLE_SPECTRAL

// Random number generator based on the OptiX SDK
template<uint32_t N>
static __forceinline__ __device__ uint32_t tea(uint32_t v0, uint32_t v1)
{
    uint32_t s0 = 0;

    for (uint32_t n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Generate random uint32_t in [0, 2^24)
static __forceinline__ __device__ uint32_t lcg(uint32_t& prev)
{
    const uint32_t LCG_A = 1664525u;
    const uint32_t LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __forceinline__ __device__ float rnd(uint32_t& prev)
{
    return ((float)lcg(prev) / (float)0x01000000);
}

// direction to environment map texture coordinates
__device__ inline float2 environment_coords(const float3 &dir)
{
    const float u = atan2f(dir.z, dir.x) * (float)(0.5 / M_PI) + 0.5f;
    const float v = acosf(fmax(fminf(-dir.y, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
    return make_float2(u, v);
}

// importance sample the environment
__device__ inline float3 environment_sample(
    float3 &dir,
    float  &pdf,
    const  float3 &xi,
    const  Kernel_params &params)
{
    // importance sample an envmap pixel using an alias map
    const unsigned int size = params.env_size.x * params.env_size.y;
    const unsigned int idx = min((unsigned int)(xi.x * (float)size), size - 1);
    unsigned int env_idx;
    float xi_y = xi.y;
    if (xi_y < params.env_accel[idx].q) {
        env_idx = idx ;
        xi_y /= params.env_accel[idx].q;
    } else {
        env_idx = params.env_accel[idx].alias;
        xi_y = (xi_y - params.env_accel[idx].q) / (1.0f - params.env_accel[idx].q);
    }

    const unsigned int py = env_idx / params.env_size.x;
    const unsigned int px = env_idx % params.env_size.x;
    pdf = params.env_accel[env_idx].pdf;

    // uniformly sample spherical area of pixel
    const float u = (float)(px + xi_y) / (float)params.env_size.x;
    const float phi = u * (float)(2.0 * M_PI) - (float)M_PI;
    float sin_phi, cos_phi;
    sincosf(phi, &sin_phi, &cos_phi);
    const float step_theta = (float)M_PI / (float)params.env_size.y;
    const float theta0 = (float)(py) * step_theta;
    const float cos_theta = cosf(theta0) * (1.0f - xi.z) + cosf(theta0 + step_theta) * xi.z;
    const float theta = acosf(cos_theta);
    const float sin_theta = sinf(theta);
    dir = make_float3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

    // lookup filtered value
    const float v = theta * (float)(1.0 / M_PI);
    const float4 t = tex2D<float4>(params.env_tex, u, v);
    return make_float3(t.x, t.y, t.z) / pdf;
}

// evaluate the environment
__device__ inline float3 environment_eval(
    float &pdf,
    const float3 &dir,
    const Kernel_params &params)
{
    const float2 uv = environment_coords(dir);
    const unsigned int x =
        min((unsigned int)(uv.x * (float)params.env_size.x), params.env_size.x - 1);
    const unsigned int y =
        min((unsigned int)(uv.y * (float)params.env_size.y), params.env_size.y - 1);

    pdf = params.env_accel[y * params.env_size.x + x].pdf;
    const float4 t = tex2D<float4>(params.env_tex, uv.x, uv.y);
    return make_float3(t.x, t.y, t.z);
}


// Intersect a sphere with given radius located at the (0,0,0)
__device__ inline float intersect_sphere(
    const float3 &pos,
    const float3 &dir,
    const float radius)
{
    const float b = 2.0f * dot(dir, pos);
    const float c = dot(pos, pos) - radius * radius;

    float tmp = b * b - 4.0f * c;
    if (tmp < 0.0f)
        return -1.0f;

    tmp = sqrtf(tmp);
    const float t0 = (((b < 0.0f) ? -tmp : tmp) - b) * 0.5f;
    const float t1 = c / t0;

    const float m = fminf(t0, t1);
    return m > 0.0f ? m : fmaxf(t0, t1);
}

struct Ray_state {
    float3       contribution;        // accumulated radiance, always in linear sRGB
    Color_sample weight;              // path throughput; spectral or RGB depending on color mode
    float3 pos, pos_rx, pos_ry;
    float3 dir, dir_rx, dir_ry;
    bool inside;
    int intersection;
#ifdef ENABLE_SPECTRAL
    // wavelengths sampled for this path (in nm)
    tct_spectral_sample spectral_wavelengths;
    // running ratios pdf[i] / pdf[0] used for hero-wavelength MIS
    float spectral_pdf_ratios[MDL_DF_SPECTRAL_SAMPLES - 1];
#endif
};


#ifdef ENABLE_SPECTRAL
// Update spectral PDF ratios after a BSDF sample event and return the MIS weight scaling factor.
// Mirrors the implementation in df_native and df_vulkan.
__device__ inline float update_spectral_pdf_ratios(
    Ray_state &ray_state,
    const tct_spectral_sample &pdfs,
    bool specular,
    bool specular_dispersion)
{
    // The main wavelength has been used for sampling, so the MIS weight is
    // w = p[0] / sum(p) = 1 / (sum(p) / p[0])
    // for pdf p up to this point in the path.
    // Here we update the pdf ratios, compute the new weight and return the factor that
    // changes from the old to the new weight.

    if (specular && !specular_dispersion) // specular without dispersion: nothing to do
        return 1.0f;

    if (!specular_dispersion && pdfs.values[0] <= 0.0f) // really has zero probability
        return 0.0f;

    float inv_w_old = 1.0f;
    float inv_w_new = 1.0f;
    const float inv_p0 = specular_dispersion ? 0.0f : (1.0f / pdfs.values[0]);
    for (int i = 1; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
    {
        inv_w_old += ray_state.spectral_pdf_ratios[i - 1];
        ray_state.spectral_pdf_ratios[i - 1] *= pdfs.values[i] * inv_p0;
        inv_w_new += ray_state.spectral_pdf_ratios[i - 1];
    }

    return inv_w_old / inv_w_new;
}
#endif // ENABLE_SPECTRAL

__device__ inline bool trace_sphere(
    uint32_t &seed,
    Ray_state &ray_state,
    const Kernel_params &params)
{
    // intersect with geometry
    const float radius = 1.0f;
    const float t = intersect_sphere(ray_state.pos, ray_state.dir, radius);
    if (t < 0.0f) {
        if (ray_state.intersection == 0 && params.mdl_test_type != MDL_TEST_NO_ENV) {
            // primary ray miss, add environment contribution
            const float2 uv = environment_coords(ray_state.dir);
            const float4 texval = tex2D<float4>(params.env_tex, uv.x, uv.y);
            const float3 env_radiance = make_float3(texval.x, texval.y, texval.z);
#ifdef ENABLE_SPECTRAL
            // primary-ray miss: weight is still 1, convert env back to RGB via spectral upsample
            const Color_sample env_spec = rgb_to_spectral(
                env_radiance, ray_state.spectral_wavelengths, /*is_emission=*/true);
            ray_state.contribution += spectral_to_rgb(
                env_spec, ray_state.spectral_wavelengths, /*is_reflectivity=*/false, params);
#else
            ray_state.contribution += env_radiance;
#endif
        }
        return false;
    }

    // compute geometry state
    ray_state.pos += ray_state.dir * t;
    const float3 normal = normalize(ray_state.pos);

    const float phi = atan2f(normal.x, normal.z);
    const float theta = acosf(normal.y);

    const float3 uvw = make_float3(
        (phi * (float)(0.5 / M_PI) + 0.5f) * 2.0f,
        1.0f - theta * (float)(1.0 / M_PI),
        0.0f);

    // compute surface derivatives
    float sp, cp;
    sincosf(phi, &sp, &cp);
    const float st = sinf(theta);
    float3 tangent_u = make_float3(cp * st, 0.0f, -sp * st) * (float)M_PI * radius;
    float3 tangent_v = make_float3(sp * normal.y, -st, cp * normal.y) * (float)(-M_PI) * radius;

#ifdef ENABLE_DERIVATIVES
    tct_deriv_float3 pos = { ray_state.pos, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } };
    tct_deriv_float3 texture_coords[1] = {
        { uvw, { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } } };

    if (params.use_derivatives && ray_state.intersection == 0)
    {
#ifdef USE_FAKE_DERIVATIVES
        pos.dx = make_float3(1.0f, 0.0f, 0.0f);
        pos.dy = make_float3(0.0f, 1.0f, 0.0f);
        texture_coords[0].dx = make_float3(1.0f, 0.0f, 0.0f);
        texture_coords[0].dy = make_float3(0.0f, 1.0f, 0.0f);
#else
        // compute ray differential for one-pixel offset rays
        // ("Physically Based Rendering", 3rd edition, chapter 10.1.1)
        const float d = dot(normal, ray_state.pos);
        const float tx = (d - dot(normal, ray_state.pos_rx)) / dot(normal, ray_state.dir_rx);
        const float ty = (d - dot(normal, ray_state.pos_ry)) / dot(normal, ray_state.dir_ry);
        ray_state.pos_rx += ray_state.dir_rx * tx;
        ray_state.pos_ry += ray_state.dir_ry * ty;

        pos.dx = ray_state.pos_rx - ray_state.pos;
        pos.dy = ray_state.pos_ry - ray_state.pos;

        float4 A;
        float2 B_x, B_y;
        if (fabsf(normal.x) > fabsf(normal.y) && fabsf(normal.x) > fabsf(normal.z)) {
            B_x = make_float2(
                ray_state.pos_rx.y - ray_state.pos.y,
                ray_state.pos_rx.z - ray_state.pos.z);
            B_y = make_float2(
                ray_state.pos_ry.y - ray_state.pos.y,
                ray_state.pos_ry.z - ray_state.pos.z);
            A = make_float4(
                tangent_u.y, tangent_u.z, tangent_v.y, tangent_v.z);
        } else if (fabsf(normal.y) > fabsf(normal.z)) {
            B_x = make_float2(
                ray_state.pos_rx.x - ray_state.pos.x,
                ray_state.pos_rx.z - ray_state.pos.z);
            B_y = make_float2(
                ray_state.pos_ry.x - ray_state.pos.x,
                ray_state.pos_ry.z - ray_state.pos.z);
            A = make_float4(
                tangent_u.x, tangent_u.z, tangent_v.x, tangent_v.z);
        } else {
            B_x = make_float2(
                ray_state.pos_rx.x - ray_state.pos.x,
                ray_state.pos_rx.y - ray_state.pos.y);
            B_y = make_float2(
                ray_state.pos_ry.x - ray_state.pos.x,
                ray_state.pos_ry.y - ray_state.pos.y);
            A = make_float4(
                tangent_u.x, tangent_u.y, tangent_v.x, tangent_v.y);
        }

        const float det = A.x * A.w - A.y * A.z;
        if (fabsf(det) > 1e-10f) {
            const float inv_det = 1.0f / det;

            texture_coords[0].dx.x = inv_det * (A.w * B_x.x - A.z * B_x.y);
            texture_coords[0].dx.y = inv_det * (A.x * B_x.y - A.y * B_x.x);

            texture_coords[0].dy.x = inv_det * (A.w * B_y.x - A.z * B_y.y);
            texture_coords[0].dy.y = inv_det * (A.x * B_y.y - A.y * B_y.x);
        }
#endif
    }
#else
    tct_float3 texture_coords[1] = { uvw };
#endif
    tangent_u = normalize(tangent_u);
    tangent_v = normalize(tangent_v);

    float4 texture_results[16];

    // material of the current object
    Df_cuda_material material = params.material_buffer[params.current_material];

    // access textures and other resource data
    Mdl_resource_handler mdl_resources;

    // create state (field-by-field so both Shading_state_material and
    // Shading_state_material_spectral are handled uniformly)
    Mdl_state state;
    state.normal                = normal;
    state.geom_normal           = normal;
#ifdef ENABLE_DERIVATIVES
    state.position              = pos;
#else
    state.position              = ray_state.pos;
#endif
    state.animation_time        = 0.0f;
    state.text_coords           = texture_coords;
    state.tangent_u             = &tangent_u;
    state.tangent_v             = &tangent_v;
    state.text_results          = texture_results;
    state.ro_data_segment       = NULL;
    state.world_to_object       = identity;
    state.object_to_world       = identity;
    state.object_id             = 0;
    state.meters_per_scene_unit = 1.0f;
#ifdef ENABLE_SPECTRAL
    state.spectral_wavelengths  = ray_state.spectral_wavelengths;
#endif


    Mdl_function_index func_idx;

    // apply volume attenuation after first bounce
    // (assuming uniform absorption coefficient and ignoring scattering coefficient)
#ifndef ENABLE_SPECTRAL
    // TODO SPECTRAL: do proper spectral handling here
    if (ray_state.intersection > 0)
    {
        func_idx = get_mdl_function_index(material.volume_absorption);
        if (is_valid(func_idx)) {
            mdl_resources.set_target_code_index(params, func_idx);    // init resource handler
            const char* arg_block = get_arg_block(params, func_idx);  // get material parameters
            prepare_state(params, func_idx, state, normal); // init state

            float3 abs_coeff;
            as_expression(func_idx)(
                &abs_coeff, &state, &mdl_resources.data, arg_block);

            ray_state.weight.x *= abs_coeff.x > 0.0f ? expf(-abs_coeff.x * t) : 1.0f;
            ray_state.weight.y *= abs_coeff.y > 0.0f ? expf(-abs_coeff.y * t) : 1.0f;
            ray_state.weight.z *= abs_coeff.z > 0.0f ? expf(-abs_coeff.z * t) : 1.0f;
        }
    }
#endif // !ENABLE_SPECTRAL

    // add emission
    func_idx = get_mdl_function_index(material.edf);
    if (is_valid(func_idx))
    {
        // init for the use of the materials EDF
        mdl_resources.set_target_code_index(params, func_idx); // init resource handler
        const char* arg_block = get_arg_block(params, func_idx); // get material parameters
        prepare_state(params, func_idx, state, normal); // init state
        as_edf_init(func_idx)(&state, &mdl_resources.data, arg_block);

        // evaluate EDF
        Edf_evaluate_data<DF_HSM_NONE, (Target_code_color_mode)BSDF_TCCM> eval_data;
        eval_data.k1 = make_float3(-ray_state.dir.x, -ray_state.dir.y, -ray_state.dir.z);
        eval_data.edf = make_color_sample(0.0f);
        as_edf_evaluate(func_idx)(&eval_data, &state, &mdl_resources.data, arg_block);

        // evaluate intensity expression. With spectral rendering, the generated MDL function
        // returns a tct_spectral_sample, otherwise a float3.
        Color_sample emission_intensity = make_color_sample(0.0f);
        func_idx = get_mdl_function_index(material.emission_intensity);
        if (is_valid(func_idx))
        {
            // init for the use of the materials emission intensity
            mdl_resources.set_target_code_index(params, func_idx); // init resource handler
            arg_block = get_arg_block(params, func_idx); // get material parameters
            prepare_state(params, func_idx, state, normal); // init state

            as_expression(func_idx)(
                &emission_intensity, &state, &mdl_resources.data, arg_block);
        }

        // add emission: weight * intensity * edf_value
        const Color_sample emission_contrib =
            mulcc(ray_state.weight, mulcc(emission_intensity, eval_data.edf));
#ifdef ENABLE_SPECTRAL
        ray_state.contribution += spectral_to_rgb(
            emission_contrib, ray_state.spectral_wavelengths,
            /*is_reflectivity=*/false, params);
#else
        ray_state.contribution += emission_contrib;
#endif
    }


    func_idx = get_mdl_function_index(material.bsdf);
    if (is_valid(func_idx))
    {
        // init for the use of the materials BSDF
        mdl_resources.set_target_code_index(params, func_idx); // init resource handler
        const char* arg_block = get_arg_block(params, func_idx); // get material parameters
        prepare_state(params, func_idx, state, normal); // init state

        // initialize BSDF
        // Note, that this will change the state.normal (needs to be reset before using EDFs)
        as_bsdf_init(func_idx)(&state, &mdl_resources.data, arg_block);

        // reuse memory for function data
        union
        {
            Bsdf_sample_data<(Target_code_color_mode)BSDF_TCCM>                sample_data;
            Bsdf_evaluate_data<DF_HSM_NONE, (Target_code_color_mode)BSDF_TCCM> eval_data;
            Bsdf_pdf_data<(Target_code_color_mode)BSDF_TCCM>                   pdf_data;
        };

        // initialize shared fields
#ifdef ENABLE_SPECTRAL
        // In spectral mode all spectral samples must be set explicitly.
        if (ray_state.inside)
        {
            sample_data.ior1 = make_color_sample(BSDF_USE_MATERIAL_IOR);
            sample_data.ior2 = make_color_sample(1.0f);
        }
        else
        {
            sample_data.ior1 = make_color_sample(1.0f);
            sample_data.ior2 = make_color_sample(BSDF_USE_MATERIAL_IOR);
        }
#else
        if (ray_state.inside)
        {
            sample_data.ior1.x = BSDF_USE_MATERIAL_IOR;
            sample_data.ior2 = make_float3(1.0f, 1.0f, 1.0f);
        }
        else
        {
            sample_data.ior1 = make_float3(1.0f, 1.0f, 1.0f);
            sample_data.ior2.x = BSDF_USE_MATERIAL_IOR;
        }
#endif
        sample_data.k1 = make_float3(-ray_state.dir.x, -ray_state.dir.y, -ray_state.dir.z);

        // compute direct lighting for point light
        if (params.light_intensity.x > 0.0f ||
            params.light_intensity.y > 0.0f ||
            params.light_intensity.z > 0.0f)
        {
            float3 to_light = params.light_pos - ray_state.pos;
            const float check_sign = squared_length(params.light_pos) < 1.0f ? -1.0f : 1.0f;
            if (dot(to_light, normal) * check_sign > 0.0f)
            {

                const float inv_squared_dist = 1.0f / squared_length(to_light);
                const float3 f = params.light_intensity * inv_squared_dist * (float) (0.25 / M_PI);

                eval_data.k2 = to_light * sqrtf(inv_squared_dist);
                eval_data.bsdf_diffuse = make_color_sample(0.0f);
                eval_data.bsdf_glossy  = make_color_sample(0.0f);

                // evaluate the materials BSDF
                as_bsdf_evaluate(func_idx)(
                    &eval_data, &state, &mdl_resources.data, arg_block);

                // sample weight: weight * f, in active color mode. Point-light radiance is RGB
                // and treated as emission for the spectral upsample.
#ifdef ENABLE_SPECTRAL
                const Color_sample f_color = rgb_to_spectral(
                    f, ray_state.spectral_wavelengths, /*is_emission=*/true);
                const Color_sample w = mulcc(ray_state.weight, f_color);
#else
                const Color_sample w = ray_state.weight * f;
#endif
                const Color_sample contrib =
                    mulcc(addcc(eval_data.bsdf_diffuse, eval_data.bsdf_glossy), w);
#ifdef ENABLE_SPECTRAL
                ray_state.contribution += spectral_to_rgb(
                    contrib, ray_state.spectral_wavelengths,
                    /*is_reflectivity=*/false, params);
#else
                ray_state.contribution += contrib;
#endif
            }
        }

        // importance sample environment light
        if (params.mdl_test_type != MDL_TEST_SAMPLE && params.mdl_test_type != MDL_TEST_NO_ENV)
        {
            const float xi0 = rnd(seed);
            const float xi1 = rnd(seed);
            const float xi2 = rnd(seed);

            float3 light_dir;
            float pdf;
            const float3 f = environment_sample(light_dir, pdf, make_float3(xi0, xi1, xi2), params);

            const float cos_theta = dot(light_dir, normal);
            if (cos_theta > 0.0f && pdf > 0.0f)
            {
                eval_data.k2 = light_dir;
                eval_data.bsdf_diffuse = make_color_sample(0.0f);
                eval_data.bsdf_glossy  = make_color_sample(0.0f);

                // evaluate the materials BSDF
                as_bsdf_evaluate(func_idx)(
                    &eval_data, &state, &mdl_resources.data, arg_block);

                const float bsdf_main_pdf = get_main_pdf(eval_data.pdf);
                const float mis_weight =
                    (params.mdl_test_type == MDL_TEST_EVAL) ? 1.0f
                        : pdf / (pdf + bsdf_main_pdf);

                // sample weight: weight * f * mis_weight, in active color mode. Env radiance is
                // RGB and treated as emission for the spectral upsample.
#ifdef ENABLE_SPECTRAL
                const Color_sample f_color = rgb_to_spectral(
                    f, ray_state.spectral_wavelengths, /*is_emission=*/true);
                const Color_sample w = mulcf(mulcc(ray_state.weight, f_color), mis_weight);
#else
                const Color_sample w = ray_state.weight * f * mis_weight;
#endif
                const Color_sample contrib =
                    mulcc(addcc(eval_data.bsdf_diffuse, eval_data.bsdf_glossy), w);
#ifdef ENABLE_SPECTRAL
                ray_state.contribution += spectral_to_rgb(
                    contrib, ray_state.spectral_wavelengths,
                    /*is_reflectivity=*/false, params);
#else
                ray_state.contribution += contrib;
#endif
            }
        }

        // importance sample BSDF
        {
            sample_data.xi.x = rnd(seed);
            sample_data.xi.y = rnd(seed);
            sample_data.xi.z = rnd(seed);
            sample_data.xi.w = rnd(seed);


            // sample the materials BSDF
            as_bsdf_sample(func_idx)(&sample_data, &state, &mdl_resources.data, arg_block);


            if (sample_data.event_type == BSDF_EVENT_ABSORB)
                return false;

            ray_state.dir = sample_data.k2;

            const bool transmission = (sample_data.event_type & BSDF_EVENT_TRANSMISSION) != 0;
            if (transmission)
                ray_state.inside = !ray_state.inside;

            const bool is_specular = (sample_data.event_type & BSDF_EVENT_SPECULAR) != 0;

#ifdef ENABLE_SPECTRAL
            {
                // detect specular dispersion (specular transmission with wavelength-varying IOR)
                bool specular_dispersion = false;
                if (is_specular && transmission)
                {
                    for (int si = 1; si < MDL_DF_SPECTRAL_SAMPLES; ++si)
                    {
                        if (sample_data.ior1.values[si] != sample_data.ior1.values[0] ||
                            sample_data.ior2.values[si] != sample_data.ior2.values[0])
                        {
                            specular_dispersion = true;
                            break;
                        }
                    }
                }

                const float ratio_factor = update_spectral_pdf_ratios(
                    ray_state, sample_data.pdf, is_specular, specular_dispersion);
                ray_state.weight = mulcf(
                    mulcc(ray_state.weight, sample_data.bsdf_over_pdf),
                    ratio_factor);
            }
#else
            ray_state.weight *= sample_data.bsdf_over_pdf;
#endif


            if (ray_state.inside)
            {
                // avoid self-intersections
                ray_state.pos -= normal * 0.001f;

                return true; // continue bouncing in sphere
            }
            else if (params.mdl_test_type != MDL_TEST_NO_ENV &&
                params.mdl_test_type != MDL_TEST_EVAL)
            {
                // leaving sphere, add contribution from environment hit

                float pdf;
                const float3 f = environment_eval(pdf, sample_data.k2, params);

                float bsdf_pdf;
                if (params.mdl_test_type == MDL_TEST_MIS_PDF)
                {
                    const float3 k2 = sample_data.k2;
                    pdf_data.k2 = k2;

                    // get pdf corresponding to the materials BSDF
                    as_bsdf_pdf(func_idx)(&pdf_data, &state, &mdl_resources.data, arg_block);

                    bsdf_pdf = get_main_pdf(pdf_data.pdf);
                }
                else
                    bsdf_pdf = get_main_pdf(sample_data.pdf);

                if (is_specular || bsdf_pdf > 0.0f)
                {
                    const float mis_weight = is_specular ||
                        (params.mdl_test_type == MDL_TEST_SAMPLE) ? 1.0f :
                            bsdf_pdf / (pdf + bsdf_pdf);
#ifdef ENABLE_SPECTRAL
                    const Color_sample f_color = rgb_to_spectral(
                        f, ray_state.spectral_wavelengths, /*is_emission=*/true);
                    const Color_sample contrib = mulcf(
                        mulcc(ray_state.weight, f_color), mis_weight);
                    ray_state.contribution += spectral_to_rgb(
                        contrib, ray_state.spectral_wavelengths,
                        /*is_reflectivity=*/false, params);
#else
                    ray_state.contribution += ray_state.weight * f * mis_weight;
#endif
                }
            }
        }
    }

    return false;
}

__device__ inline float3 render_sphere(
    uint32_t &seed,
    const Kernel_params &params,
    const unsigned x,
    const unsigned y)
{
    const float inv_res_x = 1.0f / (float)params.resolution.x;
    const float inv_res_y = 1.0f / (float)params.resolution.y;

    const float dx = params.disable_aa ? 0.5f : rnd(seed);
    const float dy = params.disable_aa ? 0.5f : rnd(seed);

    const float2 screen_pos = make_float2(
        ((float)x + dx) * inv_res_x,
        ((float)y + dy) * inv_res_y);

    const float r    = (2.0f * screen_pos.x               - 1.0f);
    const float r_rx = (2.0f * (screen_pos.x + inv_res_x) - 1.0f);
    const float u    = (2.0f * screen_pos.y               - 1.0f);
    const float u_ry = (2.0f * (screen_pos.y + inv_res_y) - 1.0f);
    const float aspect = (float)params.resolution.y / (float)params.resolution.x;

    Ray_state ray_state;
    ray_state.contribution = make_float3(0.0f, 0.0f, 0.0f);
    ray_state.weight = make_color_sample(1.0f);
#ifdef ENABLE_SPECTRAL
    {
        // uniform stratified wavelength sampling across the configured range
        float xi = rnd(seed);
        for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES; ++i)
        {
            xi += 1.0f / float(MDL_DF_SPECTRAL_SAMPLES);
            if (xi > 1.0f) xi -= 1.0f;
            ray_state.spectral_wavelengths.values[i] =
                params.spectral_min_wavelength
                + xi * (params.spectral_max_wavelength - params.spectral_min_wavelength);
        }

        // initialize the spectral PDF ratios
        for (int i = 0; i < MDL_DF_SPECTRAL_SAMPLES - 1; ++i)
            ray_state.spectral_pdf_ratios[i] = 1.0f;
    }
#endif
    ray_state.pos = ray_state.pos_rx = ray_state.pos_ry = params.cam_pos;
    ray_state.dir = normalize(
        params.cam_dir * params.cam_focal + params.cam_right * r    + params.cam_up * aspect * u);
    ray_state.dir_rx = normalize(
        params.cam_dir * params.cam_focal + params.cam_right * r_rx + params.cam_up * aspect * u);
    ray_state.dir_ry = normalize(
        params.cam_dir * params.cam_focal + params.cam_right * r    + params.cam_up * aspect * u_ry);
    ray_state.inside = false;

    const unsigned int max_num_intersections = params.max_path_length - 1;
    for (ray_state.intersection = 0; ray_state.intersection < max_num_intersections;
            ++ray_state.intersection)
        if (!trace_sphere(seed, ray_state, params))
            break;

    return
        isfinite(ray_state.contribution.x) &&
        isfinite(ray_state.contribution.y) &&
        isfinite(ray_state.contribution.z) ? ray_state.contribution : make_float3(0.0f, 0.0f, 0.0f);
}


// exposure + simple Reinhard tonemapper + gamma
__device__ inline unsigned int display(float3 val, const float tonemap_scale)
{
    val *= tonemap_scale;
    const float burn_out = 0.1f;
    val.x *= (1.0f + val.x * burn_out) / (1.0f + val.x);
    val.y *= (1.0f + val.y * burn_out) / (1.0f + val.y);
    val.z *= (1.0f + val.z * burn_out) / (1.0f + val.z);
    const unsigned int r =
        (unsigned int)(255.0 * fminf(powf(fmaxf(val.x, 0.0f), (float)(1.0 / 2.2)), 1.0f));
    const unsigned int g =
        (unsigned int)(255.0 * fminf(powf(fmaxf(val.y, 0.0f), (float)(1.0 / 2.2)), 1.0f));
    const unsigned int b =
        (unsigned int)(255.0 * fminf(powf(fmaxf(val.z, 0.0f), (float)(1.0 / 2.2)), 1.0f));
    return 0xff000000 | (r << 16) | (g << 8) | b;
}

// CUDA kernel rendering simple geometry with IBL
extern "C" __global__ void render_sphere_kernel(
    const Kernel_params kernel_params)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kernel_params.resolution.x || y >= kernel_params.resolution.y)
        return;

    const unsigned int idx = y * kernel_params.resolution.x + x;
    uint32_t seed = tea<4>(idx, kernel_params.iteration_start);

    float3 value = make_float3(0.0f, 0.0f, 0.0f);
    for (unsigned int s = 0; s < kernel_params.iteration_num; ++s)
    {
        value += render_sphere(
            seed,
            kernel_params,
            x, y);
    }
    value *= 1.0f / (float)kernel_params.iteration_num;


    // accumulate
    if (kernel_params.iteration_start == 0)
        kernel_params.accum_buffer[idx] = value;
    else {
        kernel_params.accum_buffer[idx] = kernel_params.accum_buffer[idx] +
            (value - kernel_params.accum_buffer[idx]) *
                ((float)kernel_params.iteration_num /
                    (float)(kernel_params.iteration_start + kernel_params.iteration_num));
    }

    // update display buffer
    if (kernel_params.display_buffer)
        kernel_params.display_buffer[idx] =
            display(kernel_params.accum_buffer[idx], kernel_params.exposure_scale);
}
