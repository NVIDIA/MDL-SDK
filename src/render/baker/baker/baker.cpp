/******************************************************************************
 * Copyright (c) 2017-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"


// ENABLE_GPU_BAKING_OSS is mapped from the MDL_ENABLE_GPU_BAKER CMake option at the
// top of baker.h, which is included below before its first use.

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/itile.h>
#include <mi/neuraylib/vector_typedefs.h>

#include <base/system/main/access_module.h>
#include <base/system/main/module_registration.h>
#include <base/lib/log/i_log_logger.h>
#include <base/data/db/i_db_fragmented_job.h>
#include <base/data/db/i_db_access.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/image/image/i_image_utilities.h>
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/texture/i_texture.h>
#include <render/mdl/backends/backends_backends.h>
#include <base/hal/time/i_time.h>


#include "baker.h"


#ifdef ENABLE_GPU_BAKING_OSS
#include "baker_cuda_oss.h"
#ifdef ENABLE_GPU_BAKING_OSS
#  include <cuda.h>
#endif
#include <memory>
#endif

#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif

namespace MI {

namespace BAKER {

inline mi::Float32_3 from_polar(float theta, float phi) // polar coordinates
{
    const float cos_theta = -cosf(theta);
    const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    return mi::Float32_3(
        -sin_theta * cosf(phi),
        cos_theta,
        -sin_theta * sinf(phi));
}

inline float radinv2(unsigned int i)
{
    i = (i << 16) | (i >> 16);
    i = ((i & 0x00ff00ff) << 8) | ((i & 0xff00ff00) >> 8);
    i = ((i & 0x0f0f0f0f) << 4) | ((i & 0xf0f0f0f0) >> 4);
    i = ((i & 0x33333333) << 2) | ((i & 0xcccccccc) >> 2);
    i = ((i & 0x55555555) << 1) | ((i & 0xaaaaaaaa) >> 1);
    return (float)(i >> 8) * 0x1p-24f;
}

inline float fractf(const float x)
{
    return x - floorf(x);
}

// ----------------------------------------------------------------------------
// Baker_fragmented_job

class Baker_fragmented_job final : public DB::Fragmented_job
{
public:
    Baker_fragmented_job(
        const mi::neuraylib::ITarget_code* target_code,
        mi::neuraylib::ICanvas* texture,
        mi::Float32 min_u,
        mi::Float32 max_u,
        mi::Float32 min_v,
        mi::Float32 max_v,
        mi::Float32 animation_time,
        mi::Uint32 samples,
        mi::Uint32 state_flags,
        bool is_environment);

    virtual void execute_fragment(
        DB::Transaction* transaction,
        size_t           index,
        size_t           count,
        const mi::neuraylib::IJob_execution_context* context);

    bool successful() const { return m_failure == 0; }

    mi::Size get_fragment_count() const { return m_num_fragments; }

    // Determine if the baked texture is constant color.
    // Conditions are:
    // 1- All the "pixel equal flags" are true for all fragments
    // 2- All the first pixel are equal for all fragments
    // WARNING: Do not call this routine before the job is executed
    bool are_all_pixels_equal() const
    {
        mi::Size i;
        // Check if all pixels in all fragments are identical
        for (i = 0; i < m_num_fragments; i++)
        {
            if (m_all_pixels_equal[i] == false)
            {
                return false;
            }
        }
        // Check if all fragment reference pixels are identical
        for (i = 1/*skip first pixel*/; i < m_num_fragments; i++)
        {
            for (mi::Uint32 channel = 0; channel < 4/*mi::Float32_4*/; ++channel)
            {
                if (fabsf(m_first_pixel_color[0][channel] - m_first_pixel_color[i][channel]) > Baker_fragmented_job::m_epsilon)
                {
                    return false;
                }
            }
        }
        return true;
    }

protected:
    mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;
    mi::base::Handle<mi::neuraylib::ICanvas>            m_texture;

    mi::Uint32  m_tex_width;
    mi::Uint32  m_tex_height;
    MI::IMAGE::Pixel_type m_tex_pixel_type;
    mi::Float32 m_min_u;
    mi::Float32 m_max_u;
    mi::Float32 m_min_v;
    mi::Float32 m_max_v;
    mi::Float32 m_animation_time;
    mi::Uint32  m_num_samples;
    mi::Uint32  m_state_flags;
    bool        m_is_environment;
    mi::Float32 m_du;
    mi::Float32 m_dv;
    mi::Size    m_num_fragments;
    mi::Size    m_num_rows_per_frag;
    mi::Size    m_num_frags_with_extra_row;

    std::atomic_uint32_t m_failure;

    static constexpr mi::Uint32 MAX_FRAGMENT = 256;
    // Store a flag stating that all pixels are equal per fragment
    bool m_all_pixels_equal[MAX_FRAGMENT];
    // Store the first computed pixel per fragment
    mi::Float32_4 m_first_pixel_color[MAX_FRAGMENT];
    // Epsilon used to compare baked pixels; matches the GPU bakers'
    // constant-detection epsilon.
    static constexpr float m_epsilon = 1e-7f;
};

Baker_fragmented_job::Baker_fragmented_job(
    const mi::neuraylib::ITarget_code* target_code,
    mi::neuraylib::ICanvas* texture,
    const mi::Float32 min_u,
    const mi::Float32 max_u,
    const mi::Float32 min_v,
    const mi::Float32 max_v,
    const mi::Float32 animation_time,
    const mi::Uint32 samples,
    const mi::Uint32 state_flags,
    const bool is_environment)
    : m_target_code(target_code, mi::base::DUP_INTERFACE)
    , m_texture(texture, mi::base::DUP_INTERFACE)
    , m_min_u(min_u)
    , m_max_u(max_u)
    , m_min_v(min_v)
    , m_max_v(max_v)
    , m_animation_time(animation_time)
    , m_num_samples(samples)
    , m_state_flags(state_flags)
    , m_is_environment(is_environment)
    , m_failure(0)
{
    m_tex_width  = texture->get_resolution_x();
    m_tex_height = texture->get_resolution_y();
    m_tex_pixel_type = MI::IMAGE::convert_pixel_type_string_to_enum(texture->get_type());

    m_du = (mi::Float32)(1.0 / (mi::Float64)m_tex_width);
    m_dv = (mi::Float32)(1.0 / (mi::Float64)m_tex_height);

    m_num_fragments = std::min(m_tex_height, MAX_FRAGMENT);
    m_num_rows_per_frag = m_tex_height / m_num_fragments;
    m_num_frags_with_extra_row = m_tex_height - m_num_rows_per_frag * m_num_fragments;
}

static const mi::neuraylib::tct_float4_a16 s_unity[3] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f
};

static void prepare_cpu_state(
    mi::neuraylib::Shading_state_environment &state_env,
    mi::neuraylib::Shading_state_material &state,
    mi::Float32_3 *tex_coords,
    mi::Float32_3 *tangent_u,
    mi::Float32_3 *tangent_v,
    const mi::Float32 &animation_time,
    const unsigned int state_flags,
    const bool is_environment)
{
    if (is_environment)
        state_env.ro_data_segment = nullptr;
    else
    {
        if (state_flags & BAKER_STATE_POSITION_DIRECTION) {
            state.normal          = mi::Float32_3(0.0f, 0.0f, 0.0f);
            state.geom_normal     = mi::Float32_3(0.0f, 0.0f, 0.0f);
        } else {
            state.normal          = mi::Float32_3(0.0f, 0.0f, 1.0f);
            state.geom_normal     = mi::Float32_3(0.0f, 0.0f, 1.0f);
        }

        state.animation_time  = animation_time;
        state.ro_data_segment = nullptr;

        state.text_coords = tex_coords;
        state.tangent_u   = tangent_u;
        state.tangent_v   = tangent_v;

        for (uint32_t tex_index = 0; tex_index < BAKER_TEXTURE_SPACES; ++tex_index) {
            if (state_flags & BAKER_STATE_POSITION_DIRECTION) {
                tex_coords[tex_index] = mi::Float32_3(0.0f, 0.0f, 0.0f);
                tangent_u[tex_index] = mi::Float32_3(0.0f, 0.0f, 0.0f);
                tangent_v[tex_index] = mi::Float32_3(0.0f, 0.0f, 0.0f);
            }
            else {
                tangent_u[tex_index] = mi::Float32_3(1.0f, 0.0f, 0.0f);
                tangent_v[tex_index] = mi::Float32_3(0.0f, 1.0f, 0.0f);
            }
        }

        // text result are currently unused
        state.text_results = nullptr;

        // we have no uniform state here
        state.world_to_object       = &s_unity[0];
        state.object_to_world       = &s_unity[0];
        state.object_id             = 0;
        state.meters_per_scene_unit = 1.0f;
    }
}

void Baker_fragmented_job::execute_fragment(
    DB::Transaction* transaction,
    size_t           index,
    size_t           count,
    const mi::neuraylib::IJob_execution_context* context)
{
    const mi::Uint32 start_row = mi::Uint32(
        index * m_num_rows_per_frag +
        ((index < m_num_frags_with_extra_row) ? index : m_num_frags_with_extra_row));

    const mi::Uint32 end_row = mi::Uint32(
        start_row + m_num_rows_per_frag - 1 +
        ((index < m_num_frags_with_extra_row) ? 1 : 0));

    const mi::Uint32 start_col = 0;
    const mi::Uint32 end_col   = m_tex_width - 1;

    union {
        mi::neuraylib::Shading_state_environment state_env;
        mi::neuraylib::Shading_state_material state;
    };

    mi::Float32_3 tex_coords[BAKER_TEXTURE_SPACES];
    mi::Float32_3 tangent_u[BAKER_TEXTURE_SPACES];
    mi::Float32_3 tangent_v[BAKER_TEXTURE_SPACES];
    prepare_cpu_state(
        state_env, state, tex_coords, tangent_u, tangent_v, m_animation_time, m_state_flags, m_is_environment);

    mi::base::Handle<mi::neuraylib::ITile> tile(m_texture->get_tile());

    const float inv_spp = (float)(1.0 / (double)m_num_samples);
    const mi::Float32 range_v(m_max_v - m_min_v);
    const mi::Float32 range_u(m_max_u - m_min_u);
    for (mi::Uint32 i = start_row; i <= end_row; i++)
    {
        for (mi::Uint32 j = start_col; j <= end_col; j++)
        {
            mi::Float32_4 pixel(0.0f, 0.0f, 0.0f, 1.0f);
            mi::Float32_4 pixel_data(0.0f, 0.0f, 0.0f, 0.0f);

            for (mi::Uint32 k = 0; k < m_num_samples; k++) {

                const mi::Float32 y = ((((float)i + fractf(radinv2(k) + 0.5f)) * m_dv) * range_v) + m_min_v;
                const mi::Float32 x = ((((float)j + fractf((float)k * inv_spp + 0.5f)) * m_du) * range_u) + m_min_u;

                if (m_is_environment) {
                    const float phi = x * (float)(2.0 * M_PI);
                    const float theta = y * (float)(M_PI);
                    state_env.direction = from_polar(theta, phi);

                    if (m_target_code->execute_environment(
                            0, state_env, nullptr, (mi::Spectrum_struct*)&pixel.x) != 0) {
                        m_failure = 1;
                        return;
                    }
                } else {
                    if (m_state_flags & BAKER_STATE_POSITION_DIRECTION) {
                        const float phi = x * (float)(2.0 * M_PI);
                        const float theta = y * (float)(M_PI);
                        state.position = from_polar(theta, phi);
                    } else {
                        state.position = mi::Float32_3(x, y, 0.0f);
                        for (uint32_t tex_index = 0; tex_index < BAKER_TEXTURE_SPACES; ++tex_index) {
                            tex_coords[tex_index] = mi::Float32_3(x, y, 0.0f);
                        }
                    }

                    if (m_target_code->execute(
                            0, state, nullptr, nullptr, (void*)(&pixel.x)) != 0) {
                        m_failure = 1;
                        return;
                    }

                    // check for NaN
                    pixel.x = std::isnan(pixel.x) ? 0.0f : pixel.x;
                    pixel.y = std::isnan(pixel.y) ? 0.0f : pixel.y;
                    pixel.z = std::isnan(pixel.z) ? 0.0f : pixel.z;
                    pixel.w = std::isnan(pixel.w) ? 0.0f : pixel.w;

                    // For bool (Sint8) targets the MDL lambda writes a raw byte
                    // (0x00/0x01) into pixel.x.  Lift to 0.0f/1.0f before
                    // accumulation so float averaging and the > 0.5 majority-vote
                    // threshold below work correctly, matching the GPU path.
                    if (m_tex_pixel_type == IMAGE::PT_SINT8) {
                        unsigned char raw;
                        memcpy(&raw, &pixel.x, sizeof(raw));
                        pixel.x = (raw != 0) ? 1.0f : 0.0f;
                        pixel.y = pixel.z = pixel.w = 0.0f;
                    }
                }
                pixel_data += pixel;
            }
            pixel_data /= m_num_samples;

            bool apply_epsilon_threshold = true;
            if (m_tex_pixel_type == MI::IMAGE::Pixel_type::PT_SINT8) {
                // Majority-vote threshold: more than half the samples returned
                // true → pixel is true.  pixel_data.x is the average of the
                // per-sample 0.0f/1.0f values produced in the loop above.
                const float bv = (pixel_data.x > 0.5f) ? (1.0f / 255.0f) : 0.0f;
                pixel_data.x = pixel_data.y = pixel_data.z = bv;
                apply_epsilon_threshold = false;
            }
            tile->set_pixel(j, i, (mi::Float32*)&pixel_data.x);

            if (i == start_row && j == start_col) {
                // this is first pixel, store its color and initialize the constant flag
                m_first_pixel_color[index] = pixel_data;
                m_all_pixels_equal[index] = true;
            }
            else if (m_all_pixels_equal[index] == true) {
                // Until now, all pixels have constant color

                // Test the last computed pixel until the end of the fragment is reached
                // or a different pixel color is found
                for (mi::Uint32 channel = 0; channel < 4/*mi::Float32_4*/; ++channel) {
                    if (apply_epsilon_threshold) {
                        if (fabsf(pixel_data[channel] - m_first_pixel_color[index][channel]) > Baker_fragmented_job::m_epsilon) {
                            m_all_pixels_equal[index] = false;
                            break;
                        }
                    } else {
                        if (pixel_data[channel] != m_first_pixel_color[index][channel]) {
                            m_all_pixels_equal[index] = false;
                            break;
                        }
                    }
                }
            }
        }
    }
}


// Constructor.
Baker_code_impl::Baker_code_impl(
    mi::Uint32                         gpu_dev_id,
    const mi::neuraylib::ITarget_code *gpu_code,
    const mi::neuraylib::ITarget_code *cpu_code,
    const bool is_environment,
    bool is_uniform)
: m_gpu_dev_id(gpu_dev_id)
, m_gpu_code(gpu_code, mi::base::DUP_INTERFACE)
, m_cpu_code(cpu_code, mi::base::DUP_INTERFACE)
, m_is_environment(is_environment)
, m_is_uniform(is_uniform)
{

}

mi::Uint32 Baker_code_impl::get_used_gpu_device_id() const
{
    return m_gpu_dev_id;
}

const mi::neuraylib::ITarget_code* Baker_code_impl::get_gpu_target_code() const
{
    if (m_gpu_code) {
        m_gpu_code->retain();
        return m_gpu_code.get();
    }
    return nullptr;
}

const mi::neuraylib::ITarget_code* Baker_code_impl::get_cpu_target_code() const
{
    if (m_cpu_code) {
        m_cpu_code->retain();
        return m_cpu_code.get();
    }
    return nullptr;
}

void Baker_code_impl::gpu_failed() const
{
    m_gpu_code.reset();
}

Baker_module_impl::Baker_module_impl()
: m_mdlc_module()
, m_compiler()
, m_code_generator_jit()
#ifdef ENABLE_GPU_BAKING_OSS
, m_dev_ctx_cache_oss_lock()
, m_dev_ctx_cache_oss()
#endif
{
}

Baker_module_impl::~Baker_module_impl()
{
    // Out-of-line definition required for std::unique_ptr<Cuda_dynload> with
    // an only-forward-declared Cuda_dynload in baker.h.
}

bool Baker_module_impl::init()
{

    m_mdlc_module.set();
    m_compiler = m_mdlc_module->get_mdl();

    mi::base::Handle<mi::mdl::ICode_generator> generator(
        m_compiler->load_code_generator( "jit"));
    if( !generator)
        return false;
    m_code_generator_jit = generator->get_interface<mi::mdl::ICode_generator_jit>();

    return true;
}

void Baker_module_impl::exit()
{

#ifdef ENABLE_GPU_BAKING_OSS
    for (auto& kv : m_dev_ctx_cache_oss)
        cuCtxDestroy(kv.second);
    m_dev_ctx_cache_oss.clear();
#endif

    m_code_generator_jit = nullptr;
    m_compiler = nullptr;
    m_mdlc_module.reset();
}



#ifdef ENABLE_GPU_BAKING_OSS
bool Baker_module_impl::ensure_cuda_loaded_oss(int gpu_dev_id, unsigned& sm) const
{
    if (cuInit(0) != CUDA_SUCCESS) return false;

    int dev_count = 0;
    if (cuDeviceGetCount(&dev_count) != CUDA_SUCCESS || dev_count <= 0 ||
        gpu_dev_id < 0 || gpu_dev_id >= dev_count) {
        return false;
    }
    int dev = 0;
    if (cuDeviceGet(&dev, gpu_dev_id) != CUDA_SUCCESS) return false;
    int major = 0, minor = 0;
    if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
            != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
            != CUDA_SUCCESS) {
        return false;
    }
    sm = (unsigned)(major * 10 + minor);
    // The kernel is compiled for sm_50; require at least that.
    return sm >= 50;
}

CUcontext Baker_module_impl::get_dev_context_oss(int gpu_dev_id) const
{
    mi::base::Lock::Block block(&m_dev_ctx_cache_oss_lock);
    auto it = m_dev_ctx_cache_oss.find(gpu_dev_id);
    if (it != m_dev_ctx_cache_oss.end()) return it->second;

    int dev = 0;
    if (cuDeviceGet(&dev, gpu_dev_id) != CUDA_SUCCESS) return nullptr;

    CUcontext ctx = nullptr;
    if (cuCtxCreate(&ctx, /*flags=*/0, dev) != CUDA_SUCCESS || !ctx)
        return nullptr;

    // cuCtxCreate makes the new context current; pop so we don't leak state.
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);

    m_dev_ctx_cache_oss[gpu_dev_id] = ctx;
    return ctx;
}
#endif

const IBaker_code* Baker_module_impl::create_baker_code(
    DB::Transaction* transaction,
    const MDL::Mdl_compiled_material* compiled_material,
    const char* path,
    mi::neuraylib::Baker_resource resource,
    mi::Uint32 gpu_device_id,
    std::string& pixel_type,
    bool& is_uniform) const
{
    return create_baker_code_internal(
        transaction, compiled_material, nullptr, path,
        resource, gpu_device_id, pixel_type, is_uniform);
}

const IBaker_code* Baker_module_impl::create_environment_baker_code(
    DB::Transaction* transaction,
    const MDL::Mdl_function_call* environment_function,
    mi::neuraylib::Baker_resource resource,
    mi::Uint32 gpu_device_id,
    bool& is_uniform) const
{
    std::string pixel_type;
    return create_baker_code_internal(
        transaction, nullptr, environment_function, nullptr,
        resource, gpu_device_id, pixel_type, is_uniform);
}


const IBaker_code* Baker_module_impl::create_baker_code_internal(
    DB::Transaction* transaction,
    const MDL::Mdl_compiled_material* compiled_material,
    const MDL::Mdl_function_call* function_call,
    const char* path,
    mi::neuraylib::Baker_resource resource,
    mi::Uint32 gpu_device_id,
    std::string& pixel_type,
    bool& is_uniform) const
{
    TIME::Stopwatch mdl_time;
    mdl_time.start();

    if (compiled_material)
    {
        mi::base::Handle<const mi::mdl::IMaterial_instance> material(
            compiled_material->get_core_material_instance());

        const mi::mdl::DAG_node* node_result = nullptr;
        const mi::mdl::IValue* value_result = nullptr;
        std::string core_path = MDL::int_path_to_core_path(path);
        material->lookup_sub_expression(core_path.c_str(), node_result, value_result);
        if (!node_result && !value_result)
            return nullptr;

        const mi::mdl::IType* field_type
            = node_result ? node_result->get_type() : value_result->get_type();

        // convert MDL type to pixel type
        switch (field_type->get_kind()) {
            case mi::mdl::IType::TK_FLOAT:
                // we can bake to float
                pixel_type = "Float32";
                break;

            case mi::mdl::IType::TK_COLOR:
                // ... color
                pixel_type = "Rgb_fp";
                break;

            case mi::mdl::IType::TK_VECTOR:
            {
                const mi::mdl::IType_vector* field_type_vector
                    = mi::mdl::as<mi::mdl::IType_vector>(field_type);
                if (!field_type_vector) {
                    // should not happen
                    return 0;
                }
                const mi::mdl::IType* field_type_element = field_type_vector->get_element_type();
                if (field_type_element->get_kind() != mi::mdl::IType::TK_FLOAT) {
                    // unsupported vector type
                    return 0;
                }
                switch (field_type_vector->get_size()) {
                case 2:
                    pixel_type = "Float32<2>";
                    break;
                case 3:
                    pixel_type = "Float32<3>";
                    break;
                case 4:
                    pixel_type = "Float32<4>";
                    break;
                default:
                    // should not happen
                    return 0;
                }
            }
            break;

            case mi::mdl::IType::TK_BOOL:
                // ... or boolean
                pixel_type = "Sint8";
                break;

            default:
                // unsupported type
                return nullptr;
        }
    }

    bool use_cpu = false;
    bool use_gpu = false;
    unsigned sm = 0;

    switch (resource) {
    case mi::neuraylib::BAKE_ON_CPU:
        // should always work
        use_cpu = true;
        break;
    case mi::neuraylib::BAKE_ON_GPU:
        use_gpu = true;
        break;
    case mi::neuraylib::BAKE_ON_GPU_WITH_CPU_FALLBACK:
        use_gpu = use_cpu = true;
        break;
    }

#  if defined(ENABLE_GPU_BAKING_OSS)
    if (use_gpu) {
        unsigned probed_sm = 0;
        if (ensure_cuda_loaded_oss(gpu_device_id, probed_sm)) {
            sm = probed_sm;
        } else {
            if (!use_cpu) {
                log_error("CUDA driver/runtime not available; GPU baking failed. "
                          "Use Baker_resource::BAKE_ON_GPU_WITH_CPU_FALLBACK or "
                          "Baker_resource::BAKE_ON_CPU.");
            }
            use_gpu = false;
        }
    }
#  else  // !ENABLE_GPU_BAKING_OSS
    if (use_gpu) {
        if (use_cpu) {
            log_info("GPU baker not available in this build "
                     "(MDL_ENABLE_GPU_BAKER=OFF); falling back to CPU.");
        } else {
            log_error("GPU baker requested but not available: this MDL SDK was built "
                      "without CUDA support (MDL_ENABLE_GPU_BAKER=OFF). "
                      "Use Baker_resource::BAKE_ON_CPU or "
                      "Baker_resource::BAKE_ON_GPU_WITH_CPU_FALLBACK.");
        }
        use_gpu = false;
    }
#  endif

    if (!use_gpu && !use_cpu) {
        // no resource available
        return nullptr;
    }
    MDL::Execution_context context;

    // try GPU first
    mi::base::Handle<const mi::neuraylib::ITarget_code> gpu_code;

    if (use_gpu) {
        mi::base::Handle<mi::mdl::ICode_cache> code_cache(m_mdlc_module->get_code_cache());
        BACKENDS::Mdl_llvm_backend be_ptx(
            mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX,
            m_compiler.get(),
            m_code_generator_jit.get(),
            code_cache.get(),
            /*string_ids=*/true);

        be_ptx.set_option( "sm_version", std::to_string(sm).c_str());

        [[maybe_unused]] mi::Sint32 result = be_ptx.set_option( "num_texture_spaces", std::to_string(BAKER_TEXTURE_SPACES).c_str());
        ASSERT( M_BAKER, result == 0);

#ifdef ENABLE_GPU_BAKING_OSS
        // The open-source-release GPU baker pairs the generated PTX with the
        // baker's own device-side texture runtime in texture_support_cuda.h,
        // whose tex lookups are exposed through a vtable (tex_vtable) and take a
        // Texture_handler_base*. The generated PTX must therefore dispatch tex
        // lookups through that vtable. num_texture_results is 0 because the
        // baker passes no precomputed texture-results array in the shading state.
        be_ptx.set_option("tex_lookup_call_mode", "vtable");
        be_ptx.set_option("num_texture_results", "0");
#endif

        if (compiled_material)
            gpu_code = mi::base::make_handle(
                be_ptx.translate_material_expression(
                    transaction, compiled_material, path, "baker_lambda", &context));
        else {
            gpu_code = mi::base::make_handle(
                be_ptx.translate_environment(
                    transaction, function_call, "baker_lambda", &context));
        }

        ASSERT( M_BAKER, context.get_result() == 0 || context.get_result() == -2);

        if (!gpu_code && !use_cpu) {
            // we are enforced to use the GPU, but it failed
            return nullptr;
        }
    }

    mi::base::Handle<const mi::neuraylib::ITarget_code> cpu_code;
    if (use_cpu) {
        mi::base::Handle<mi::mdl::ICode_cache> code_cache(m_mdlc_module->get_code_cache());
        BACKENDS::Mdl_llvm_backend be_native(
            mi::neuraylib::IMdl_backend_api::MB_NATIVE,
            m_compiler.get(),
            m_code_generator_jit.get(),
            code_cache.get(),
            /*string_ids=*/true);

        [[maybe_unused]] mi::Sint32 result = be_native.set_option( "num_texture_spaces", std::to_string(BAKER_TEXTURE_SPACES).c_str());
        ASSERT( M_BAKER, result == 0);
        result = be_native.set_option("use_builtin_resource_handler", "on");
        ASSERT( M_BAKER, result == 0);

        result = context.set_option("fold_meters_per_scene_unit", true);
        ASSERT(M_BAKER, result == 0);

        if (compiled_material)
            cpu_code = mi::base::make_handle(
                be_native.translate_material_expression(
                    transaction, compiled_material, path, "baker_lambda", &context));
        else
            cpu_code = mi::base::make_handle(
                be_native.translate_environment(
                    transaction, function_call, "baker_lambda", &context));

        ASSERT(M_BAKER, context.get_result() == 0 || context.get_result() == -2);
        if (!cpu_code) {
            // compilation failed, CPU must succeed
            return nullptr;
        }
    }

    // Prefer the CPU code's reported state usage; fall back to the GPU code's
    // when only GPU code was generated. gpu_code is null in CPU-only builds.
    mi::neuraylib::ITarget_code::State_usage render_state_usage =
        cpu_code ? cpu_code->get_render_state_usage()
                 : (gpu_code ? gpu_code->get_render_state_usage()
                             : mi::neuraylib::ITarget_code::State_usage());

    // everything but state::texture_coordinate() and state::position() is constant for baking
    if (compiled_material)
        is_uniform = ((render_state_usage &
                       (mi::neuraylib::ITarget_code::SU_TEXTURE_COORDINATE |
                        mi::neuraylib::ITarget_code::SU_POSITION)) == 0);
    else
        is_uniform = ((render_state_usage &
                       mi::neuraylib::ITarget_code::SU_DIRECTION) == 0);

    mdl_time.stop();
    log_debug("MDL to target code time: %1.0f ms", mdl_time.elapsed() * 1000);

    return new Baker_code_impl(
        gpu_device_id,
        gpu_code.get(),
        cpu_code.get(),
        function_call != nullptr,
        is_uniform);
}


mi::Sint32 Baker_module_impl::bake_texture(
    DB::Transaction* transaction,
    const IBaker_code* baker_code,
    mi::neuraylib::ICanvas* texture,
    const mi::Uint32 samples,
    const mi::Uint32 state_flags) const
{
    return bake_texture(transaction, baker_code, texture, 0, 1, 0, 1, 0.0f, samples, state_flags);
}

mi::Sint32 Baker_module_impl::bake_texture(
    DB::Transaction* transaction,
    const IBaker_code* baker_code,
    mi::neuraylib::ICanvas* texture,
    mi::Float32 min_u,
    mi::Float32 max_u,
    mi::Float32 min_v,
    mi::Float32 max_v,
    mi::Float32 animation_time,
    const mi::Uint32 samples,
    mi::Uint32 state_flags) const
{
    Constant_result constant;
    bool is_constant = false;
    bool detect_constant_case = false;
    // Derive pixel_type from the canvas so the Sint8 → num_samples=1 guard
    // in bake_texture_internal fires correctly on this code path.  Without
    // this the check "(pixel_type && strcmp(pixel_type, "Sint8") == 0)" is
    // always false (pixel_type was NULL), so multi-sample baking with FTZ
    // flushes the denormal bool encoding to zero.
    const char* pixel_type = texture ? texture->get_type() : nullptr;
    return bake_texture_internal(
        transaction,
        baker_code,
        texture,
        constant,
        is_constant,
        min_u,
        max_u,
        min_v,
        max_v,
        animation_time,
        samples,
        pixel_type,
        state_flags,
        detect_constant_case);
}

mi::Sint32 Baker_module_impl::bake_texture_with_constant_detection(
    DB::Transaction         *transaction,
    const IBaker_code       *baker_code,
    mi::neuraylib::ICanvas  *texture,
    Constant_result         &constant,
    bool                    &is_constant,
    mi::Float32             min_u,
    mi::Float32             max_u,
    mi::Float32             min_v,
    mi::Float32             max_v,
    mi::Float32             animation_time,
    mi::Uint32              samples,
    const char              *pixel_type,
    mi::Uint32              state_flags) const
{
    bool detect_constant_case = true;
    return bake_texture_internal(
        transaction,
        baker_code,
        texture,
        constant,
        is_constant,
        min_u,
        max_u,
        min_v,
        max_v,
        animation_time,
        samples,
        pixel_type,
        state_flags,
        detect_constant_case);
}
bool set_constant_from_canvas(
    mi::neuraylib::ICanvas* texture,
    const std::string& pixel_type,
    BAKER::Baker_module::Constant_result& constant)
{
    mi::base::Handle<mi::neuraylib::ITile> tile(texture->get_tile());
    mi::math::Color color;
    tile->get_pixel(0, 0, &color.r);

    if (pixel_type == "Float32<2>")
    {
        constant.v.x = color.r;
        constant.v.y = color.g;
    }
    else if (pixel_type == "Rgb_fp" || pixel_type == "Float32<3>")
    {
        constant.v.x = color.r;
        constant.v.y = color.g;
        constant.v.z = color.b;
    }
    else if (pixel_type == "Float32<4>")
    {
        constant.v.x = color.r;
        constant.v.y = color.g;
        constant.v.z = color.b;
        constant.v.w = color.a;
    }
    else if (pixel_type == "Float32")
    {
        constant.f = color.r;
    }
    else if (pixel_type == "Sint8")
    {
        constant.b = color.r != 0.0f;
    }
    else
    {
        ASSERT(M_BAKER, false);
        return false;
    }
    return true;
}

mi::Sint32 Baker_module_impl::bake_texture_internal(
    DB::Transaction* transaction,
    const IBaker_code* baker_code,
    mi::neuraylib::ICanvas* texture,
    Constant_result & constant,
    bool & is_constant,
    mi::Float32 min_u,
    mi::Float32 max_u,
    mi::Float32 min_v,
    mi::Float32 max_v,
    mi::Float32 animation_time,
    mi::Uint32 samples,
    const char * pixel_type,
    mi::Uint32 state_flags,
    bool want_constant_detection) const
{
    // If uniform expression directly bake the expression as constant
    if (want_constant_detection) {
        if (baker_code->is_uniform()) {
            is_constant = true;
            return bake_constant(transaction, baker_code, constant, samples, pixel_type);
        }
    }

    is_constant = false;
    mi::base::Handle<const mi::neuraylib::ITarget_code> cpu_code(
        baker_code->get_cpu_target_code());


#ifdef ENABLE_GPU_BAKING_OSS
    {
        mi::base::Handle<const mi::neuraylib::ITarget_code> gpu_code_oss(
            baker_code->get_gpu_target_code());
        if (gpu_code_oss) {
            const bool is_env =
                static_cast<Baker_code_impl const *>(baker_code)->is_environment();
            log_info("Baking texture on GPU (device %u).",
                     baker_code->get_used_gpu_device_id());
            Baker_cuda_oss baker(
                this,
                baker_code->get_used_gpu_device_id(),
                gpu_code_oss.get(),
                texture,
                samples,
                min_u, max_u, min_v, max_v,
                animation_time,
                /*constant_target=*/ nullptr,
                /*constant_target_components=*/ 1,
                state_flags,
                is_env,
                want_constant_detection);

            if (baker.is_ready() && baker.bake_texture(transaction)) {
                if (want_constant_detection) {
                    is_constant = baker.are_all_pixels_equal();
                    if (is_constant) {
                        return set_constant_from_canvas(texture, pixel_type, constant) ? 0 : -1;
                    }
                }
                return 0;
            }
            if (cpu_code) {
                log_warning("Material expression execution failed on GPU, switching to CPU.");
                static_cast<Baker_code_impl const *>(baker_code)->gpu_failed();
            }
        }
    }
#endif

    if (cpu_code) {
        const bool is_env = static_cast<Baker_code_impl const *>(baker_code)->is_environment();
        log_info("Baking texture on CPU.");
        Baker_fragmented_job job(cpu_code.get(), texture, min_u, max_u, min_v, max_v, animation_time, samples, state_flags, is_env);
        transaction->execute_fragmented(&job, job.get_fragment_count());
        if (job.successful()) {
            if (want_constant_detection) {
                is_constant = job.are_all_pixels_equal();
                if (is_constant)
                {
                    return set_constant_from_canvas(texture, pixel_type, constant) ? 0 : -1;
                }
            }
            // success
            return 0;
        }
    }

    log_error("Material expression execution failed.");
    return -1;
}

mi::Sint32 Baker_module_impl::bake_constant(
    DB::Transaction   *transaction,
    const IBaker_code *baker_code,
    Constant_result   &constant,
    mi::Uint32        samples,
    const char        *pixel_type) const
{
    [[maybe_unused]] unsigned int n_comp = 0;
    if (strcmp(pixel_type, "Float32") == 0) {
        n_comp = 1;
    } else if (strcmp(pixel_type, "Float32<2>") == 0) {
        n_comp = 2;
    } else if (strcmp(pixel_type, "Float32<3>") == 0 || strcmp(pixel_type, "Rgb_fp") == 0) {
        n_comp = 3;
    } else if (strcmp(pixel_type, "Float32<4>") == 0) {
        n_comp = 4;
    } else if (strcmp(pixel_type, "Sint8") == 0) {
        n_comp = 0;
    } else {
        log_error("Material expression execution failed, unsupported constant type.");
        return -1;
    }

    mi::base::Handle<const mi::neuraylib::ITarget_code> gpu_code(
        baker_code->get_gpu_target_code());
    mi::base::Handle<const mi::neuraylib::ITarget_code> cpu_code(
        baker_code->get_cpu_target_code());

    const bool is_env = static_cast<Baker_code_impl const *>(baker_code)->is_environment();

    // Bake constant on the CPU if possible, this should be faster in most cases
    if (cpu_code) {
        union {
            mi::neuraylib::Shading_state_environment state_env;
            mi::neuraylib::Shading_state_material state;
        };

        mi::Float32_3 tex_coords[BAKER_TEXTURE_SPACES];
        mi::Float32_3 tangent_u[BAKER_TEXTURE_SPACES];
        mi::Float32_3 tangent_v[BAKER_TEXTURE_SPACES];
        prepare_cpu_state(
            state_env, state, tex_coords, tangent_u, tangent_v, /*animation_time*/0.0f, /*state_flags=*/0, is_env);

        if (is_env) {
            state_env.direction = mi::Float32_3(0.0f, 0.0f, 1.0f);
            if (cpu_code->execute_environment(
                    0, state_env, /*tex_handler=*/nullptr, &constant.s) == 0)
            {
                // success
                return 0;
            }
        } else {
            for (uint32_t tex_index = 0; tex_index < BAKER_TEXTURE_SPACES; ++tex_index) {
                tex_coords[tex_index] = state.position = mi::Float32_3(0.5f, 0.5f, 0);
            }
            if (cpu_code->execute(
                    0, state, /*tex_handler=*/nullptr, /*cap_args=*/nullptr, &constant.f) == 0)
            {
                // success
                return 0;
            }
        }
    }


#ifdef ENABLE_GPU_BAKING_OSS
    if (gpu_code) {
        Baker_cuda_oss baker(
            this,
            baker_code->get_used_gpu_device_id(),
            gpu_code.get(),
            /*texture=*/nullptr,
            samples,
            0, 1, 0, 1,
            0.0f,
            &constant.f,
            n_comp == 0 ? 1 : n_comp,
            /*state_flags=*/ 0,
            is_env,
            /*want_constant_detection*/false);

        if (baker.is_ready() && baker.bake_texture(transaction)) {
            return 0;
        }
        static_cast<Baker_code_impl const *>(baker_code)->gpu_failed();
    }
#endif

    log_error("Material expression execution failed.");
    return -1;
}


static SYSTEM::Module_registration<Baker_module_impl> s_module( SYSTEM::M_BAKER, "BAKER");

SYSTEM::Module_registration_entry* Baker_module::get_instance()
{
    return s_module.init_module( s_module.get_name());
}

} // namespace BAKER

} // namespace MI

