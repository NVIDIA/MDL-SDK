/******************************************************************************
 * Copyright (c) 2017-2024, NVIDIA CORPORATION. All rights reserved.
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


#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/core/ignore_unused.hpp>

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

MI_FORCE_INLINE float radinv2(const unsigned int i/*, const unsigned int scramble = 0*/);

// ----------------------------------------------------------------------------
// Baker_fragmented_job

class Baker_fragmented_job : public DB::Fragmented_job
{
public:
    Baker_fragmented_job(
        const mi::neuraylib::ITarget_code* target_code,
        mi::neuraylib::ICanvas* texture,
        mi::Float32 min_u,
        mi::Float32 max_u,
        mi::Float32 min_v,
        mi::Float32 max_v,
        mi::Uint32 samples,
        mi::Uint32 state_flags,
        bool is_environment);

    virtual void execute_fragment(
        DB::Transaction* transaction,
        size_t           index,
        size_t           count,
        const mi::neuraylib::IJob_execution_context* context);

    bool successful() const { return m_failure == 0; }

    mi::Size get_fragment_count() { return m_num_fragments; }

protected:
    mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;
    mi::base::Handle<mi::neuraylib::ICanvas>            m_texture;

    mi::Uint32  m_tex_width;
    mi::Uint32  m_tex_height;
    mi::Float32 m_min_u;
    mi::Float32 m_max_u;
    mi::Float32 m_min_v;
    mi::Float32 m_max_v;
    mi::Uint32  m_num_samples;
    mi::Uint32  m_state_flags;
    bool        m_is_environment;
    mi::Float32 m_du;
    mi::Float32 m_dv;
    mi::Size    m_num_fragments;
    mi::Size    m_num_rows_per_frag;
    mi::Size    m_num_frags_with_extra_row;

    std::atomic_uint32_t m_failure;
};

Baker_fragmented_job::Baker_fragmented_job(
    const mi::neuraylib::ITarget_code* target_code,
    mi::neuraylib::ICanvas* texture,
    const mi::Float32 min_u,
    const mi::Float32 max_u,
    const mi::Float32 min_v,
    const mi::Float32 max_v,
    const mi::Uint32 samples,
    const mi::Uint32 state_flags,
    const bool is_environment)
    : m_target_code(target_code, mi::base::DUP_INTERFACE)
    , m_texture(texture, mi::base::DUP_INTERFACE)
    , m_min_u(min_u)
    , m_max_u(max_u)
    , m_min_v(min_v)
    , m_max_v(max_v)
    , m_num_samples(samples)
    , m_state_flags(state_flags)
    , m_is_environment(is_environment)
    , m_failure(0)
{
    m_tex_width  = texture->get_resolution_x();
    m_tex_height = texture->get_resolution_y();

    m_du = (mi::Float32)(1.0 / (mi::Float64)m_tex_width);
    m_dv = (mi::Float32)(1.0 / (mi::Float64)m_tex_height);

    m_num_fragments = m_tex_height;
    m_num_rows_per_frag = m_tex_height / m_num_fragments;
    m_num_frags_with_extra_row = m_tex_height - m_num_rows_per_frag * m_num_fragments;
}

static mi::Float32_4_4 s_unity(1.0f);

static void prepare_cpu_state(
    mi::neuraylib::Shading_state_environment &state_env,
    mi::neuraylib::Shading_state_material &state,
    mi::Float32_3 &tex_coords,
    mi::Float32_3 &tangent_u,
    mi::Float32_3 &tangent_v,
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

        state.animation_time  = 0.0f;
        state.ro_data_segment = nullptr;

        state.text_coords = &tex_coords;
        state.tangent_u   = &tangent_u;
        state.tangent_v   = &tangent_v;

        if (state_flags & BAKER_STATE_POSITION_DIRECTION) {
            tex_coords = mi::Float32_3(0.0f, 0.0f, 0.0f);
            tangent_u = mi::Float32_3(0.0f, 0.0f, 0.0f);
            tangent_v = mi::Float32_3(0.0f, 0.0f, 0.0f);
        } else {
            tangent_u = mi::Float32_3(1.0f, 0.0f, 0.0f);
            tangent_v = mi::Float32_3(0.0f, 1.0f, 0.0f);
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

MI_FORCE_INLINE float fractf(const float x)
{
    return x - floorf(x);
}

void Baker_fragmented_job::execute_fragment(
    DB::Transaction* transaction,
    size_t           index,
    size_t           count,
    const mi::neuraylib::IJob_execution_context* context)
{
    mi::Uint32 start_row = mi::Uint32(
        index * m_num_rows_per_frag +
        ((index < m_num_frags_with_extra_row) ? index : m_num_frags_with_extra_row));

    mi::Uint32 end_row = mi::Uint32(
        start_row + m_num_rows_per_frag - 1 +
        ((index < m_num_frags_with_extra_row) ? 1 : 0));

    mi::Uint32 start_col = 0;
    mi::Uint32 end_col   = m_tex_width - 1;

    union {
        mi::neuraylib::Shading_state_environment state_env;
        mi::neuraylib::Shading_state_material state;
    };

    mi::Float32_3 tex_coords;
    mi::Float32_3 tangent_u;
    mi::Float32_3 tangent_v;
    prepare_cpu_state(
        state_env, state, tex_coords, tangent_u, tangent_v, m_state_flags, m_is_environment);

    mi::base::Handle<mi::neuraylib::ITile> tile(m_texture->get_tile());

    const float inv_spp = (float)(1.0 / (double)m_num_samples);
    const mi::Float32 range_v(m_max_v - m_min_v);
    const mi::Float32 range_u(m_max_u - m_min_u);
    for (mi::Uint32 i = start_row; i <= end_row; i++)
    {
        for (mi::Uint32 j = start_col; j <= end_col; j++)
        {
            mi::Float32_3 pixel(0.0f);
            mi::Float32_3 pixel_data = pixel;

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
                        tex_coords = mi::Float32_3(x, y, 0.0f);
                    }

                    if (m_target_code->execute(
                            0, state, nullptr, nullptr, (mi::Float32*)&pixel.x) != 0) {
                        m_failure = 1;
                        return;
                    }
                }
                pixel_data += pixel;
            }
            pixel_data /= m_num_samples;
            tile->set_pixel(j, i, (mi::Float32*)&pixel_data.x);
        }
    }
}


static size_t store_value(unsigned char *data, mi::base::Handle<MI::MDL::IValue const> iv)
{
    switch (iv->get_kind()) {
    case MI::MDL::IValue::VK_BOOL:
        {
            mi::base::Handle<MI::MDL::IValue_bool const> b(
                iv->get_interface<MI::MDL::IValue_bool>());
            *reinterpret_cast<bool *>(data) = b->get_value();
            return 1;
        }
    case MI::MDL::IValue::VK_INT:
        {
            mi::base::Handle<MI::MDL::IValue_int const> i(
                iv->get_interface<MI::MDL::IValue_int>());
            *reinterpret_cast<mi::Sint32 *>(data) = i->get_value();
            return 4;
        }
    case MI::MDL::IValue::VK_ENUM:
        {
            mi::base::Handle<MI::MDL::IValue_enum const> e(
                iv->get_interface<MI::MDL::IValue_enum>());
            *reinterpret_cast<mi::Sint32 *>(data) = e->get_value();
            return 4;
        }
    case MI::MDL::IValue::VK_FLOAT:
        {
            mi::base::Handle<MI::MDL::IValue_float const> f(
                iv->get_interface<MI::MDL::IValue_float>());
            *reinterpret_cast<mi::Float32 *>(data) = f->get_value();
            return 4;
        }
    case MI::MDL::IValue::VK_DOUBLE:
        {
            mi::base::Handle<MI::MDL::IValue_double const> d(
                iv->get_interface<MI::MDL::IValue_double>());
            *reinterpret_cast<mi::Float64 *>(data) = d->get_value();
            return 8;
        }
    case MI::MDL::IValue::VK_STRING:
        {
            ASSERT(M_BAKER, !"unsupported string value");
            return 8;
        }
    case MI::MDL::IValue::VK_VECTOR:
        {
            mi::base::Handle<MI::MDL::IValue_vector const> v(
                iv->get_interface<MI::MDL::IValue_vector>());

            size_t o = 0;
            for (size_t i = 0, n = v->get_size(); i < n; ++i) {
                mi::base::Handle<MI::MDL::IValue const> e(v->get_value(i));

                size_t no = store_value(data + o, e);
                o += no;
            }
            return o;
        }
    case MI::MDL::IValue::VK_MATRIX:
        {
            mi::base::Handle<MI::MDL::IValue_matrix const> m(
                iv->get_interface<MI::MDL::IValue_matrix>());

            size_t o = 0;
            for (size_t i = 0, n = m->get_size(); i < n; ++i) {
                mi::base::Handle<MI::MDL::IValue const> e(m->get_value(i));

                size_t no = store_value(data + o, e);
                o += no;
            }
            return o;
        }
    case MI::MDL::IValue::VK_COLOR:
        {
            mi::base::Handle<MI::MDL::IValue_color const> c(
                iv->get_interface<MI::MDL::IValue_color>());

            size_t o = 0;
            for (size_t i = 0, n = c->get_size(); i < n; ++i) {
                mi::base::Handle<MI::MDL::IValue const> e(c->get_value(i));

                size_t no = store_value(data + o, e);
                o += no;
            }
            return o;
        }
    case MI::MDL::IValue::VK_ARRAY:
        {
            mi::base::Handle<MI::MDL::IValue_array const> a(
                iv->get_interface<MI::MDL::IValue_array>());

            size_t o = 0;
            for (size_t i = 0, n = a->get_size(); i < n; ++i) {
                mi::base::Handle<MI::MDL::IValue const> e(a->get_value(i));

                size_t no = store_value(data + o, e);
                o += no;
            }
            return o;
        }
    case MI::MDL::IValue::VK_STRUCT:
        {
            ASSERT(M_BAKER, !"unsupported struct value");
            return 0;
        }
    case MI::MDL::IValue::VK_INVALID_DF:
    case MI::MDL::IValue::VK_TEXTURE:
    case MI::MDL::IValue::VK_LIGHT_PROFILE:
    case MI::MDL::IValue::VK_BSDF_MEASUREMENT:
    case MI::MDL::IValue::VK_FORCE_32_BIT:
        {
            ASSERT(M_BAKER, !"unexpected argument type");
            return 0;
        }
    }
    ASSERT(M_BAKER, !"unsupported value type");
    return 0;
}

// Constructor.
Baker_code_impl::Baker_code_impl(
    mi::Uint32                         gpu_dev_id,
    const mi::neuraylib::ITarget_code *gpu_code,
    const mi::neuraylib::ITarget_code *cpu_code,
    const bool is_environment)
: m_gpu_dev_id(gpu_dev_id)
, m_gpu_code(gpu_code, mi::base::DUP_INTERFACE)
, m_cpu_code(cpu_code, mi::base::DUP_INTERFACE)
, m_is_environment(is_environment)
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
    return NULL;
}

const mi::neuraylib::ITarget_code* Baker_code_impl::get_cpu_target_code() const
{
    if (m_cpu_code) {
        m_cpu_code->retain();
        return m_cpu_code.get();
    }
    return NULL;
}

void Baker_code_impl::gpu_failed() const
{
    m_gpu_code.reset();
}

Baker_module_impl::Baker_module_impl()
: m_mdlc_module()
, m_compiler()
, m_code_generator_jit()
{
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

    m_code_generator_jit = 0;
    m_compiler = 0;
    m_mdlc_module.reset();
}



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
        resource, gpu_device_id, pixel_type, is_uniform, false);
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
        resource, gpu_device_id, pixel_type, is_uniform, false);
}


const IBaker_code* Baker_module_impl::create_baker_code_internal(
    DB::Transaction* transaction,
    const MDL::Mdl_compiled_material* compiled_material,
    const MDL::Mdl_function_call* function_call,
    const char* path,
    mi::neuraylib::Baker_resource resource,
    mi::Uint32 gpu_device_id,
    std::string& pixel_type,
    bool& is_uniform,
    const bool use_custom_cpu_tex_runtime) const
{
    TIME::Stopwatch mdl_time;
    mdl_time.start();

    if (compiled_material)
    {
        mi::base::Handle<const MDL::IExpression> field(
            compiled_material->lookup_sub_expression( transaction, path));

        if( !field)
            return 0;

        mi::base::Handle<const MDL::IType> field_type( field->get_type());

        // convert MDL type to pixel type
        switch (field_type->get_kind()) {
            case MDL::IType::TK_FLOAT:
                // we can bake to float
                pixel_type = "Float32";
                break;

            case MDL::IType::TK_COLOR:
                // ... color
                pixel_type = "Rgb_fp";
                break;

            case MDL::IType::TK_VECTOR:
            {
                mi::base::Handle<const MDL::IType_vector> field_type_vector(
                    field_type->get_interface<MDL::IType_vector>());
                if (!field_type_vector) {
                    // should not happen
                    return 0;
                }
                if (field_type_vector->get_size() != 3) {
                    // unsupported vector type
                    return 0;
                }
                mi::base::Handle<const MDL::IType> field_type_element(
                    field_type_vector->get_element_type());
                if (field_type_element->get_kind() != MDL::IType::TK_FLOAT) {
                    // unsupported vector type
                    return 0;
                }
                // ... float3
                pixel_type = "Float32<3>";
            }
            break;

            case MDL::IType::TK_BOOL:
                // ... or boolean
                pixel_type = "Boolean";
                break;

            default:
                // unsupported type
                return 0;
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
    case mi::neuraylib::BAKER_RESOURCE_FORCE_32_BIT:
        // useless case to keep gcc happy
        break;
    }


    if (!use_gpu && !use_cpu) {
        // no resource available
        return 0;
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

        char sm_version[3];
        sm_version[0] = '0' + sm / 10;
        sm_version[1] = '0' + sm % 10;
        sm_version[2] = '\0';
        be_ptx.set_option( "sm_version", sm_version);

        mi::Sint32 result = be_ptx.set_option( "num_texture_spaces", "1");
        ASSERT( M_BAKER, result == 0);
        boost::ignore_unused( result);

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
            return 0;
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

        mi::Sint32 result = be_native.set_option( "num_texture_spaces", "1");
        ASSERT( M_BAKER, result == 0);
        if (use_custom_cpu_tex_runtime) {
            result = be_native.set_option("use_builtin_resource_handler", "off");
        } else {
            result = be_native.set_option("use_builtin_resource_handler", "on");
        }
        ASSERT( M_BAKER, result == 0);

        result = context.set_option("fold_meters_per_scene_unit", true);
        ASSERT(M_BAKER, result == 0);
        boost::ignore_unused(result);
        
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
            return 0;
        }
    }

    mi::neuraylib::ITarget_code::State_usage render_state_usage
        = cpu_code
            ? cpu_code->get_render_state_usage()
            : mi::neuraylib::ITarget_code::State_usage();

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
        function_call != nullptr);
}


mi::Sint32 Baker_module_impl::bake_texture(
    DB::Transaction* transaction,
    const IBaker_code* baker_code,
    mi::neuraylib::ICanvas* texture,
    const mi::Uint32 samples,
    const mi::Uint32 state_flags) const
{
    return bake_texture(transaction, baker_code, texture, 0, 1, 0, 1, samples, state_flags);
}
    
mi::Sint32 Baker_module_impl::bake_texture(
    DB::Transaction* transaction,
    const IBaker_code* baker_code,
    mi::neuraylib::ICanvas* texture,
    mi::Float32 min_u,
    mi::Float32 max_u,
    mi::Float32 min_v,
    mi::Float32 max_v,
    const mi::Uint32 samples,
    const mi::Uint32 state_flags) const
{
    mi::base::Handle<const mi::neuraylib::ITarget_code> cpu_code(
        baker_code->get_cpu_target_code());


    if (cpu_code) {
        const bool is_env = static_cast<Baker_code_impl const *>(baker_code)->is_environment();
        Baker_fragmented_job job(cpu_code.get(), texture, min_u, max_u, min_v, max_v, samples, state_flags, is_env);
        transaction->execute_fragmented(&job, texture->get_resolution_y());
        if (job.successful()) {
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

        mi::Float32_3 tex_coords;
        mi::Float32_3 tangent_u;
        mi::Float32_3 tangent_v;
        prepare_cpu_state(
            state_env, state, tex_coords, tangent_u, tangent_v, /*state_flags=*/0, is_env);

        if (is_env) {
            state_env.direction = mi::Float32_3(0.0f, 0.0f, 1.0f);
            if (cpu_code->execute_environment(
                    0, state_env, /*tex_handler=*/nullptr, &constant.s) == 0)
            {
                // success
                return 0;
            }
        } else {
            tex_coords = state.position = mi::Float32_3(0.5f, 0.5f, 0);
            if (cpu_code->execute(
                    0, state, /*tex_handler=*/nullptr, /*cap_args=*/nullptr, &constant.f) == 0)
            {
                // success
                return 0;
            }
        }
    }


    log_error("Material expression execution failed.");
    return -1;
}


static SYSTEM::Module_registration<Baker_module_impl> s_module( SYSTEM::M_BAKER, "BAKER");

SYSTEM::Module_registration_entry* Baker_module::get_instance()
{
    return s_module.init_module( s_module.get_name());
}


MI_FORCE_INLINE unsigned int __brev(unsigned int i)
{
    i =     (i << 16) | (i >> 16);
    i =    ((i & 0x00ff00ff) << 8) | ((i & 0xff00ff00) >> 8);
    i =    ((i & 0x0f0f0f0f) << 4) | ((i & 0xf0f0f0f0) >> 4);
    i =    ((i & 0x33333333) << 2) | ((i & 0xcccccccc) >> 2);
    return ((i & 0x55555555) << 1) | ((i & 0xaaaaaaaa) >> 1);
}

MI_FORCE_INLINE float radinv2(const unsigned int i/*, const unsigned int scramble = 0*/)
{
    return (float)((__brev(i) /*^ scramble*/)>>8) * 0x1p-24f;
}

} // namespace BAKER

} // namespace MI

