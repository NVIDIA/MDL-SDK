/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/optix7/optix7_mdl.cpp
//
// Simple renderer using compiled BSDFs with a material parameter editor GUI.

#define NOMINMAX
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#define _USE_MATH_DEFINES
#include <math.h>

#define OPENGL_INTEROP
#include <cuda.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cudaGL.h>
#include <cuda_runtime.h>
#include <vector_functions.h>


#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "optix7_mdl.h"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "optix7_mdl_helper.h"

#define GL_DISPLAY_CUDA
#include "utils/gl_display.h"

#define WINDOW_TITLE "MDL SDK OptiX 7 Example"


#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            exit_failure("CUDA error %d \"%s\" in file %s, line %u: \"%s\".\n", \
                err, cudaGetErrorString(err), __FILE__, __LINE__, #expr); \
        } \
    } while (false)
#endif

#define OPTIX_CHECK(expr) \
    do { \
        OptixResult err = (expr); \
        if (err != OPTIX_SUCCESS) { \
            exit_failure("OptiX error %d in file %s, line %u: \"%s\".\n", \
                err, __FILE__, __LINE__, #expr); \
        } \
    } while (false)

#define OPTIX_CHECK_LOG(expr) \
    do { \
        OptixResult err = (expr); \
        if (err != OPTIX_SUCCESS) { \
            exit_failure("OptiX error %d in file %s, line %u: \"%s\".\nLog:\n%s\n", \
                err, __FILE__, __LINE__, #expr, log); \
        } \
    } while (false)


//------------------------------------------------------------------------------
//
// Helper functions
//
//------------------------------------------------------------------------------

// Allocate memory on GPU and copy the given data to the allocated memory.
CUdeviceptr gpuMemDup(void const *data, size_t size)
{
    if (size == 0)
        return 0;

    CUdeviceptr device_ptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr), size));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(device_ptr),
        data,
        size,
        cudaMemcpyHostToDevice
    ));
    return device_ptr;
}

// Allocate memory on GPU and copy the given data to the allocated memory.
template <typename T>
CUdeviceptr gpuMemDup(T const &data)
{
    return gpuMemDup(&data, sizeof(T));
}

// Allocate memory on GPU and copy the given data to the allocated memory.
template <typename T, int N>
CUdeviceptr gpuMemDup(T const (&data)[N])
{
    return gpuMemDup(data, N * sizeof(T));
}

// Allocate memory on GPU and copy the given data to the allocated memory.
template<typename T>
CUdeviceptr gpuMemDup(std::vector<T> const &data)
{
    return gpuMemDup(data.data(), data.size() * sizeof(T));
}


//------------------------------------------------------------------------------
//
// Local types
//
//------------------------------------------------------------------------------

class Mesh;


template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template <>
struct SbtRecord<void>
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

typedef SbtRecord<void>         RayGenSbtRecord;
typedef SbtRecord<void>         MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;
typedef SbtRecord<void>         CallableSbtRecord;


struct Vertex
{
    float x, y, z, pad;
};


struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};


struct Instance
{
    Mesh *mesh;
    unsigned material_index;
    float transform[12];
};


struct MDLMaterial
{
    // Constructor, will take ownership of Material_info object.
    MDLMaterial(
        Material_info *mat_info = nullptr,
        OptixProgramGroup radiance_hit_group = nullptr,
        unsigned callable_base_index = 0)
    : mat_info(mat_info)
    , radiance_hit_group(radiance_hit_group)
    , callable_base_index(callable_base_index)
    , d_mdl_textures(0)
    , d_mdl_tex_handler(0)
    , d_scene_data_infos(0)
    {
    }

    // Move constructor.
    MDLMaterial(MDLMaterial &&other)
    : mat_info(other.mat_info)
    , radiance_hit_group(other.radiance_hit_group)
    , callable_base_index(other.callable_base_index)
    , mdl_textures(std::move(other.mdl_textures))
    , scene_data_infos(std::move(other.scene_data_infos))
    , d_mdl_textures(other.d_mdl_textures)
    , d_mdl_tex_handler(other.d_mdl_tex_handler)
    , d_scene_data_infos(other.d_scene_data_infos)
    {
        other.mat_info = nullptr;
        other.d_mdl_textures = 0;
        other.d_mdl_tex_handler = 0;
        other.d_scene_data_infos = 0;
    }

    ~MDLMaterial()
    {
        if (mat_info)
            delete mat_info;
        if (d_mdl_tex_handler)
            cudaFree(reinterpret_cast<void*>(d_mdl_tex_handler));
        if (d_mdl_textures)
            cudaFree(reinterpret_cast<void*>(d_mdl_textures));
        if (d_scene_data_infos)
            cudaFree(reinterpret_cast<void*>(d_scene_data_infos));
    }

    CUdeviceptr get_device_textures()
    {
        if (!d_mdl_textures && !mdl_textures.empty())
            d_mdl_textures = gpuMemDup(mdl_textures);
        return d_mdl_textures;
    }

    CUdeviceptr get_device_texture_handler()
    {
        if (!d_mdl_tex_handler) {
            Texture_handler tex_handler = {};
            tex_handler.num_textures = mdl_textures.size();
            tex_handler.textures = reinterpret_cast<Texture const *>(get_device_textures());
            d_mdl_tex_handler = gpuMemDup(tex_handler);
        }
        return d_mdl_tex_handler;
    }

    CUdeviceptr get_device_scene_data_infos()
    {
        if (!d_scene_data_infos && !scene_data_infos.empty())
            d_scene_data_infos = gpuMemDup(scene_data_infos);
        return d_scene_data_infos;
    }

    Material_info              *mat_info;
    OptixProgramGroup           radiance_hit_group;
    unsigned                    callable_base_index;
    std::vector<Texture>        mdl_textures;
    std::vector<SceneDataInfo>  scene_data_infos;

private:
    CUdeviceptr                 d_mdl_textures;
    CUdeviceptr                 d_mdl_tex_handler;
    CUdeviceptr                 d_scene_data_infos;
};


struct PathTracerState
{
    OptixDeviceContext             context = 0;

    Mesh                          *mesh = nullptr;
    std::vector<Instance>          instances;

    // Traversable handle for instance AS
    OptixTraversableHandle         ias_handle = 0;
    // Instance AS memory
    CUdeviceptr                    d_ias_output_buffer = 0;

    OptixModuleCompileOptions      module_compile_options = {};
    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              raygen_prog_group = 0;
    OptixProgramGroup              radiance_miss_group = 0;
    OptixProgramGroup              occlusion_miss_group = 0;
    std::vector<OptixProgramGroup> radiance_hit_groups;
    OptixProgramGroup              occlusion_hit_group = 0;
    std::vector<OptixProgramGroup> mdl_callable_groups;

    CUstream                       stream = 0;
    Params                         params;
    Params*                        d_params = nullptr;

    bool                           resize_dirty = false;
    bool                           camera_changed = true;

    float                          fovH = 96.0f;

    float                          cam_base_dist = 1.0f;
    float                          cam_zoom = 0.0f;
    double                         cam_theta = 0;
    double                         cam_phi = 0;

    // used to set the eye distance
    float                          scene_width = 0;

    // mouse state
    int32_t                        mouse_button = -1;
    double                         mouse_move_start_x = 0;
    double                         mouse_move_start_y = 0;

    OptixShaderBindingTable        sbt = {};

    Mdl_helper                    *mdl_helper = nullptr;
    std::vector<MDLMaterial>       mdl_materials;
    size_t                         cur_material_index = 0;
    std::map<std::string, std::pair<OptixProgramGroup, size_t>> optix_program_cache;
};


//--------------------------------------------------------------------------------------
// Environment light
//--------------------------------------------------------------------------------------

// Helper for initEnvironmentLight()
float build_alias_map(
    const float *data,
    const unsigned int size,
    EnvironmentAccel *accel)
{
    // create qs (normalized)
    float sum = 0.0f;
    for (unsigned int i = 0; i < size; ++i)
        sum += data[i];

    for (unsigned int i = 0; i < size; ++i)
        accel[i].q = (static_cast<float>(size) * data[i] / sum);

    // create partition table
    unsigned int *partition_table = static_cast<unsigned int *>(
        malloc(size * sizeof(unsigned int)));
    unsigned int s = 0u, large = size;
    for (unsigned int i = 0; i < size; ++i)
        partition_table[(accel[i].q < 1.0f) ? (s++) : (--large)] = accel[i].alias = i;

    // create alias map
    for (s = 0; s < large && large < size; ++s)
    {
        const unsigned int j = partition_table[s], k = partition_table[large];
        accel[j].alias = k;
        accel[k].q += accel[j].q - 1.0f;
        large = (accel[k].q < 1.0f) ? (large + 1u) : large;
    }

    free(partition_table);

    return sum;
}


bool initEnvironmentLight(PathTracerState& state, const std::string& env_path)
{
    mi::base::Handle<mi::neuraylib::ITransaction> transaction =
        state.mdl_helper->create_transaction();

    {
        // Load environment texture
        mi::base::Handle<mi::neuraylib::IImage> image(
            transaction->create<mi::neuraylib::IImage>("Image"));
        if (image->reset_file(env_path.c_str()) != 0)
        {
            std::cerr << "Failed to environment image: " << env_path << std::endl;
            return false;
        }

        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas());
        const mi::Uint32 rx = canvas->get_resolution_x();
        const mi::Uint32 ry = canvas->get_resolution_y();
        state.params.env_light.size.x = rx;
        state.params.env_light.size.y = ry;

        // Check, whether we need to convert the image
        char const *image_type = image->get_type();
        if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0) {
            canvas = state.mdl_helper->get_image_api()->convert(canvas.get(), "Color");
        }

        // Copy the image data to a CUDA array
        const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
        cudaArray_t env_tex_data;
        CUDA_CHECK(cudaMallocArray(&env_tex_data, &channel_desc, rx, ry));

        mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
        const float *pixels = static_cast<const float *>(tile->get_data());

        CUDA_CHECK(cudaMemcpy2DToArray(
            env_tex_data, 0, 0, pixels,
            rx * sizeof(float4), rx * sizeof(float4), ry, cudaMemcpyHostToDevice));

        // Create a CUDA texture
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = env_tex_data;

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeClamp; // don't sample beyond poles of env sphere
        tex_desc.addressMode[2] = cudaAddressModeWrap;
        tex_desc.filterMode     = cudaFilterModeLinear;
        tex_desc.readMode       = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(
            &state.params.env_light.texture, &res_desc, &tex_desc, nullptr));

        // Create importance sampling data
        EnvironmentAccel *env_accel_host = static_cast<EnvironmentAccel *>(
            malloc(rx * ry * sizeof(EnvironmentAccel)));
        float *importance_data = static_cast<float *>(malloc(rx * ry * sizeof(float)));
        float cos_theta0 = 1.0f;
        const float step_phi = float(2.0 * M_PI) / float(rx);
        const float step_theta = float(M_PI) / float(ry);
        for (unsigned int y = 0; y < ry; ++y)
        {
            const float theta1 = float(y + 1) * step_theta;
            const float cos_theta1 = std::cos(theta1);
            const float area = (cos_theta0 - cos_theta1) * step_phi;
            cos_theta0 = cos_theta1;

            for (unsigned int x = 0; x < rx; ++x) {
                const unsigned int idx = y * rx + x;
                const unsigned int idx4 = idx * 4;
                importance_data[idx] =
                    area * std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2]));
            }
        }
        const float inv_env_integral =
            1.0f / build_alias_map(importance_data, rx * ry, env_accel_host);
        free(importance_data);

        state.params.env_light.inv_env_integral = inv_env_integral;
        state.params.env_light.intensity = 1.0f;
        state.params.env_light.accel = reinterpret_cast<EnvironmentAccel*>(
            gpuMemDup(env_accel_host, rx * ry * sizeof(EnvironmentAccel)));
        free(env_accel_host);
    }

    transaction->commit();
    return true;
}


//--------------------------------------------------------------------------------------
// Mesh base class
//--------------------------------------------------------------------------------------

class Mesh
{
public:
    virtual ~Mesh()
    {
        cudaFree(reinterpret_cast<void*>(m_d_blas_output_buffer));
        cudaFree(reinterpret_cast<void*>(m_d_indices));
        cudaFree(reinterpret_cast<void*>(m_d_vertices));
    }

    virtual char const *getName() const = 0;

    CUdeviceptr getDeviceVertices() const { return m_d_vertices; }
    CUdeviceptr getDeviceIndices() const { return m_d_indices; }

    OptixTraversableHandle getBLASHandle() const { return m_blas_handle; }

protected:
    void buildBLAS(
        OptixDeviceContext context,
        size_t num_vertices,
        MeshVertex const *vertices,
        size_t num_index_triplets,
        short3 const *index_triplets)
    {
        // copy mesh data to device

        m_d_vertices = gpuMemDup(vertices, num_vertices * sizeof(*vertices));
        m_d_indices = gpuMemDup(index_triplets, num_index_triplets * sizeof(*index_triplets));

        uint32_t triangle_input_flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        input.triangleArray.vertexFormat         = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes  = sizeof(*vertices);
        input.triangleArray.numVertices          = uint32_t(num_vertices);
        input.triangleArray.vertexBuffers        = &m_d_vertices;

        input.triangleArray.indexFormat          = OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
        input.triangleArray.indexStrideInBytes   = sizeof(*index_triplets);
        input.triangleArray.numIndexTriplets     = uint32_t(num_index_triplets);
        input.triangleArray.indexBuffer          = m_d_indices;

        input.triangleArray.flags                = &triangle_input_flag;
        input.triangleArray.numSbtRecords        = 1;
        input.triangleArray.sbtIndexOffsetBuffer = 0;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes blas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            &accel_options,
            &input,
            1,  // num_build_inputs
            &blas_buffer_sizes
        ));

        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_temp_buffer),
            blas_buffer_sizes.tempSizeInBytes
        ));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&m_d_blas_output_buffer),
            blas_buffer_sizes.outputSizeInBytes
        ));

        OPTIX_CHECK(optixAccelBuild(
            context,
            0,                  // CUDA stream
            &accel_options,
            &input,
            1,                  // num build inputs
            d_temp_buffer,
            blas_buffer_sizes.tempSizeInBytes,
            m_d_blas_output_buffer,
            blas_buffer_sizes.outputSizeInBytes,
            &m_blas_handle,
            nullptr,            // emitted property list
            0                   // num emitted properties
        ));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    }

private:
    CUdeviceptr m_d_vertices;
    CUdeviceptr m_d_indices;
    CUdeviceptr m_d_blas_output_buffer;
    OptixTraversableHandle m_blas_handle;
};


//--------------------------------------------------------------------------------------
// A sphere mesh
//--------------------------------------------------------------------------------------
class Sphere : public Mesh
{
public:
    Sphere(OptixDeviceContext context, float radius, unsigned int slices, unsigned int stacks)
    {
        std::vector<short3> indices;
        std::vector<MeshVertex> vertices;

        const float step_phi = (float)(2.0 * M_PI / slices);
        const float step_theta = (float)(M_PI / stacks);

        // Calculate vertices
        vertices.reserve((stacks + 1) * (slices + 1));
        for (unsigned int i = 0; i <= stacks; ++i) {
            const float theta = step_theta * float(i);
            const float sin_t = sin(theta);
            const float cos_t = cos(theta);

            for (unsigned int j = 0; j <= slices; ++j) {
                const float phi = step_phi * float(j) + float(M_PI);
                const float sin_p = sin(phi);
                const float cos_p = cos(phi);

                const float3 p = make_float3(sin_p * sin_t, cos_t, cos_p * sin_t);

                float2 uv = make_float2(
                    (float)j / (float)slices * 2.0f,
                    1.0f - (float)i / (float)stacks
                );

                float3 tangent_u =
                    make_float3(cos_p * sin_t, 0.0f, -sin_p * sin_t) * float(M_PI);
                float3 tangent_v =
                    make_float3(sin_p * p.y, -sin_t, cos_p * p.y) * float(-M_PI);

                vertices.emplace_back(p * radius, p, tangent_u, tangent_v, uv);

                // Example scene data
                vertices.back().color = make_float3(sin_p, cos_p, sin(theta * 12));
                vertices.back().row_column = make_int2(i, j);
            }
        }

        // Calculate indices
        indices.reserve(stacks * slices * 2);
        for (unsigned int i = 0; i < stacks; ++i) {
            for (unsigned int j = 0; j < slices; ++j) {
                short p0 = short(i * (slices + 1) + j);
                short p1 = p0 + 1;
                short p2 = p0 + short(slices) + 1;
                short p3 = p2 + 1;

                indices.push_back(make_short3(p0, p1, p2));
                indices.push_back(make_short3(p1, p3, p2));
            }
        }

        buildBLAS(context, vertices.size(), vertices.data(), indices.size(), indices.data());
    }

    char const *getName() const override
    {
        return "Sphere";
    }
};


//--------------------------------------------------------------------------------------
// A cube mesh
//--------------------------------------------------------------------------------------
class Cube : public Mesh
{
public:
    Cube(OptixDeviceContext context, float radius)
    {
        MeshVertex vertices[] = {
            // Top
            MeshVertex(
                make_float3(-1.0f, 1.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f),
                make_float3(-1.0f, 0.0f,  0.0f), make_float3(0.0f, 0.0f, 1.0f),
                make_float2(1.0f, 0.0f)),
            MeshVertex(
                make_float3( 1.0f, 1.0f, -1.0f), make_float3(0.0f, 1.0f, 0.0f),
                make_float3(-1.0f, 0.0f,  0.0f), make_float3(0.0f, 0.0f, 1.0f),
                make_float2(0.0f, 0.0f)),
            MeshVertex(
                make_float3( 1.0f, 1.0f, 1.0f),  make_float3(0.0f, 1.0f, 0.0f),
                make_float3(-1.0f, 0.0f, 0.0f),  make_float3(0.0f, 0.0f, 1.0f),
                make_float2(0.0f, 1.0f)),
            MeshVertex(
                make_float3(-1.0f, 1.0f, 1.0f),  make_float3(0.0f, 1.0f, 0.0f),
                make_float3(-1.0f, 0.0f, 0.0f),  make_float3(0.0f, 0.0f, 1.0f),
                make_float2(1.0f, 1.0f)),

            // Bottom
            MeshVertex(
                make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, -1.0f, 0.0f),
                make_float3( 1.0f,  0.0f,  0.0f), make_float3(0.0f,  0.0f, 1.0f),
                make_float2(0.0f, 0.0f)),
            MeshVertex(
                make_float3(1.0f, -1.0f, -1.0f),  make_float3(0.0f, -1.0f, 0.0f),
                make_float3(1.0f,  0.0f,  0.0f),  make_float3(0.0f,  0.0f, 1.0f),
                make_float2(1.0f, 0.0f)),
            MeshVertex(
                make_float3(1.0f, -1.0f, 1.0f),   make_float3(0.0f, -1.0f, 0.0f),
                make_float3(1.0f,  0.0f, 0.0f),   make_float3(0.0f,  0.0f, 1.0f),
                make_float2(1.0f, 1.0f)),
            MeshVertex(
                make_float3(-1.0f, -1.0f, 1.0f),  make_float3(0.0f, -1.0f, 0.0f),
                make_float3( 1.0f,  0.0f, 0.0f),  make_float3(0.0f,  0.0f, 1.0f),
                make_float2(0.0f, 1.0f)),

            // Left
            MeshVertex(
                make_float3(-1.0f, -1.0f,  1.0f), make_float3(-1.0f,  0.0f, 0.0f),
                make_float3( 0.0f,  0.0f, -1.0f), make_float3( 0.0f, -1.0f, 0.0f),
                make_float2(0.0f, 1.0f)),
            MeshVertex(
                make_float3(-1.0f, -1.0f, -1.0f), make_float3(-1.0f,  0.0f, 0.0f),
                make_float3( 0.0f,  0.0f, -1.0f), make_float3( 0.0f, -1.0f, 0.0f),
                make_float2(1.0f, 1.0f)),
            MeshVertex(
                make_float3(-1.0f, 1.0f, -1.0f),  make_float3(-1.0f,  0.0f, 0.0f),
                make_float3( 0.0f, 0.0f, -1.0f),  make_float3( 0.0f, -1.0f, 0.0f),
                make_float2(1.0f, 0.0f)),
            MeshVertex(
                make_float3(-1.0f, 1.0f,  1.0f),  make_float3(-1.0f,  0.0f, 0.0f),
                make_float3( 0.0f, 0.0f, -1.0f),  make_float3( 0.0f, -1.0f, 0.0f),
                make_float2(0.0f, 0.0f)),

            // Right
            MeshVertex(
                make_float3(1.0f, -1.0f, 1.0f),  make_float3(1.0f,  0.0f, 0.0f),
                make_float3(0.0f,  0.0f, 1.0f),  make_float3(0.0f, -1.0f, 0.0f),
                make_float2(1.0f, 1.0f)),
            MeshVertex(
                make_float3(1.0f, -1.0f, -1.0f), make_float3(1.0f,  0.0f, 0.0f),
                make_float3(0.0f,  0.0f,  1.0f), make_float3(0.0f, -1.0f, 0.0f),
                make_float2(0.0f, 1.0f)),
            MeshVertex(
                make_float3(1.0f, 1.0f, -1.0f),  make_float3(1.0f,  0.0f, 0.0f),
                make_float3(0.0f, 0.0f,  1.0f),  make_float3(0.0f, -1.0f, 0.0f),
                make_float2(0.0f, 0.0f)),
            MeshVertex(
                make_float3(1.0f, 1.0f, 1.0f),   make_float3(1.0f,  0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f),   make_float3(0.0f, -1.0f, 0.0f),
                make_float2(1.0f, 0.0f)),

            // Front
            MeshVertex(
                make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f,  0.0f, -1.0f),
                make_float3(-1.0f,  0.0f,  0.0f), make_float3(0.0f, -1.0f,  0.0f),
                make_float2(0.0f, 1.0f)),
            MeshVertex(
                make_float3( 1.0f, -1.0f, -1.0f), make_float3(0.0f,  0.0f, -1.0f),
                make_float3(-1.0f,  0.0f,  0.0f), make_float3(0.0f, -1.0f,  0.0f),
                make_float2(1.0f, 1.0f)),
            MeshVertex(
                make_float3( 1.0f, 1.0f, -1.0f),  make_float3(0.0f,  0.0f, -1.0f),
                make_float3(-1.0f, 0.0f,  0.0f),  make_float3(0.0f, -1.0f,  0.0f),
                make_float2(1.0f, 0.0f)),
            MeshVertex(
                make_float3(-1.0f, 1.0f, -1.0f),  make_float3(0.0f,  0.0f, -1.0f),
                make_float3(-1.0f, 0.0f,  0.0f),  make_float3(0.0f, -1.0f,  0.0f),
                make_float2(0.0f, 0.0f)),

            // Back
            MeshVertex(
                make_float3(-1.0f, -1.0f, 1.0f), make_float3(0.0f,  0.0f, 1.0f),
                make_float3(-1.0f,  0.0f, 0.0f), make_float3(0.0f, -1.0f, 0.0f),
                make_float2(1.0f, 1.0f)),
            MeshVertex(
                make_float3( 1.0f, -1.0f, 1.0f), make_float3(0.0f,  0.0f, 1.0f),
                make_float3(-1.0f,  0.0f, 0.0f), make_float3(0.0f, -1.0f, 0.0f),
                make_float2(0.0f, 1.0f)),
            MeshVertex(
                make_float3( 1.0f, 1.0f, 1.0f),  make_float3(0.0f,  0.0f, 1.0f),
                make_float3(-1.0f, 0.0f, 0.0f),  make_float3(0.0f, -1.0f, 0.0f),
                make_float2(0.0f, 0.0f)),
            MeshVertex(
                make_float3(-1.0f,  1.0f, 1.0f), make_float3(0.0f,  0.0f, 1.0f),
                make_float3(-1.0f,  0.0f, 0.0f), make_float3(0.0f, -1.0f, 0.0f),
                make_float2(1.0f, 0.0f)),
        };

        size_t num_vertices = sizeof(vertices) / sizeof(*vertices);
        for (size_t i = 0; i < num_vertices; ++i) {
            vertices[i].position *= radius;
        }

        static const short3 index_triplets[] =
        {
            { 1, 3, 0 },
            { 1, 2, 3 },

            { 4, 6, 5 },
            { 4, 7, 6 },

            { 9, 11,  8 },
            { 9, 10, 11 },

            { 12, 14, 13 },
            { 12, 15, 14 },

            { 17, 19, 16 },
            { 17, 18, 19 },

            { 20, 22, 21 },
            { 20, 23, 22 }
        };
        size_t num_index_triplets = sizeof(index_triplets) / sizeof(*index_triplets);

        buildBLAS(context, num_vertices, vertices, num_index_triplets, index_triplets);
    }

    char const *getName() const override
    {
        return "Cube";
    }
};


//------------------------------------------------------------------------------
//
// Helper functions
//
//------------------------------------------------------------------------------

void optix_context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}


void createContext(PathTracerState& state, int cuda_device_id)
{
    // Ensure that we have a CUDA context
    CUDA_CHECK(cudaSetDevice(cuda_device_id));
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optix_context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    state.context = context;

    state.module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    state.module_compile_options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    state.module_compile_options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
#ifdef CONTRIB_IN_PAYLOAD
    state.pipeline_compile_options.numPayloadValues = 6;
#else
    state.pipeline_compile_options.numPayloadValues = 3;
#endif
    state.pipeline_compile_options.numAttributeValues = 2;
    state.pipeline_compile_options.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_NONE;
        //OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        //OPTIX_EXCEPTION_FLAG_DEBUG;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
}

OptixProgramGroup createRadianceClosestHitProgramGroup(
    PathTracerState& state,
    char const *module_code,
    size_t module_size)
{
    char   log[2048];
    size_t sizeof_log = sizeof(log);

    OptixModule mat_module = nullptr;
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &state.module_compile_options,
        &state.pipeline_compile_options,
        module_code,
        module_size,
        log,
        &sizeof_log,
        &mat_module
    ));

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = mat_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    sizeof_log = sizeof(log);
    OptixProgramGroup ch_hit_group = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hit_prog_group_desc,
        /*numProgramGroups=*/ 1,
        &program_group_options,
        log,
        &sizeof_log,
        &ch_hit_group
    ));

    return ch_hit_group;
}


void addMDLMaterial(
    PathTracerState& state,
    std::string const &material_name,
    bool class_compilation)
{
    mi::base::Handle<mi::neuraylib::ITransaction> transaction =
        state.mdl_helper->create_transaction();

#ifdef NO_DIRECT_CALL
    std::string bsdf_base_name   = "mdlcode";
    std::string thin_walled_name = "mdlcode_thin_walled";
#else
    // direct callable function names must start with OptiX semantic type
    // prefix "__direct_callable__" and need unique names
    std::string bsdf_base_name =
        "__direct_callable__mdlcode" + std::to_string(state.mdl_materials.size());
    std::string thin_walled_name =
        "__direct_callable__mdlcode_thin_walled" + std::to_string(state.mdl_materials.size());
#endif

    std::vector<mi::neuraylib::Target_function_description> descs;
    descs.push_back(mi::neuraylib::Target_function_description(
        "surface.scattering", bsdf_base_name.c_str()));
    descs.push_back(mi::neuraylib::Target_function_description(
        "thin_walled", thin_walled_name.c_str()));

    // Scope for compile result
    {
        Material_info *mat_info = nullptr;
        Mdl_helper::Compile_result compile_res(
            state.mdl_helper->compile_mdl_material(
                transaction.get(),
                material_name.c_str(),
                descs,
                class_compilation,
                &mat_info));

        if (!compile_res.target_code) {
            exit_failure(
                "Code generation failed: %s", state.mdl_helper->get_last_error_message().c_str());
        }

        OptixProgramGroup mat_hit_group;
        size_t callable_base_index;

        // Check whether we already created an OptiX program group for the same code
        // Note: the generated code may be the same, even with different material hashes,
        //       as we may not use the different part in this example (like volume coefficients)
        std::string code_str = std::string(
            compile_res.target_code->get_code(),
            compile_res.target_code->get_code_size());
        auto it = state.optix_program_cache.find(code_str);
        if (it != state.optix_program_cache.end()) {
            mat_hit_group = it->second.first;
            callable_base_index = it->second.second;
        } else {
#ifdef NO_DIRECT_CALL
            // In no-direct-call mode, we create one closest hit program per hash-unique material.
            mat_hit_group = createRadianceClosestHitProgramGroup(
                state,
                compile_res.target_code->get_code(),
                compile_res.target_code->get_code_size());

            state.radiance_hit_groups.push_back(mat_hit_group);

            callable_base_index = 0;  // not used in this mode
#else
            // In direct-call mode, create OptiX module from target code and create 5 callables,
            // 4 for the BSDF and 1 for thin_walled
            char   log[2048];
            size_t sizeof_log = sizeof(log);
            OptixModule module;
            OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
                state.context,
                &state.module_compile_options,
                &state.pipeline_compile_options,
                compile_res.target_code->get_code(),
                compile_res.target_code->get_code_size(),
                log,
                &sizeof_log,
                &module));

            // Create direct callable program group
            callable_base_index = state.mdl_callable_groups.size();
            state.mdl_callable_groups.resize(callable_base_index + 5);

            std::string init_name   = bsdf_base_name + "_init";
            std::string sample_name = bsdf_base_name + "_sample";
            std::string eval_name   = bsdf_base_name + "_evaluate";
            std::string pdf_name    = bsdf_base_name + "_pdf";

            OptixProgramGroupOptions callable_options = {};
            OptixProgramGroupDesc    callable_descs[5] = {};
            callable_descs[0].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            callable_descs[0].callables.moduleDC            = module;
            callable_descs[0].callables.entryFunctionNameDC = init_name.c_str();
            callable_descs[1].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            callable_descs[1].callables.moduleDC            = module;
            callable_descs[1].callables.entryFunctionNameDC = sample_name.c_str();
            callable_descs[2].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            callable_descs[2].callables.moduleDC            = module;
            callable_descs[2].callables.entryFunctionNameDC = eval_name.c_str();
            callable_descs[3].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            callable_descs[3].callables.moduleDC            = module;
            callable_descs[3].callables.entryFunctionNameDC = pdf_name.c_str();
            callable_descs[4].kind                          = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            callable_descs[4].callables.moduleDC            = module;
            callable_descs[4].callables.entryFunctionNameDC = thin_walled_name.c_str();
            sizeof_log = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(
                state.context,
                callable_descs,
                5,
                &callable_options,
                log,
                &sizeof_log,
                &state.mdl_callable_groups[callable_base_index]));

            // In direct-call mode, there will be only one closest hit program.
            // Create it, if it hasn't already been created
            if (state.radiance_hit_groups.empty()) {
                std::string ptx = mi::examples::io::read_text_file(
                    mi::examples::io::get_executable_folder() +
                    "/optix7_mdl_closest_hit_radiance.ptx");
                mat_hit_group = createRadianceClosestHitProgramGroup(
                    state,
                    ptx.c_str(),
                    ptx.size());

                state.radiance_hit_groups.push_back(mat_hit_group);
            } else {
                mat_hit_group = state.radiance_hit_groups.back();
            }
#endif

            state.optix_program_cache[code_str] =
                std::make_pair(mat_hit_group, callable_base_index);
        }

        state.mdl_materials.emplace_back(mat_info, mat_hit_group, unsigned(callable_base_index));

        MDLMaterial &new_mat = state.mdl_materials.back();

        // Prepare textures
        if (compile_res.textures.size() > 1) {
            for (mi::Size i = 1, n = compile_res.textures.size(); i < n; ++i) {
                state.mdl_helper->prepare_texture(
                    transaction.get(),
                    compile_res.textures[i].db_name.c_str(),
                    compile_res.textures[i].shape,
                    new_mat.mdl_textures);
            }
            new_mat.get_device_textures();
        }

        // TODO: Example does not support light profiles and BSDF measurements

        // Prepare scene data
        mi::Size num_scene_data = compile_res.compiled_material->get_referenced_scene_data_count();
        if (num_scene_data > 0) {
            // Collect scene data names
            std::set<std::string> scene_data_names;
            for (mi::Size i = 0; i < num_scene_data; ++i) {
                scene_data_names.insert(
                    compile_res.compiled_material->get_referenced_scene_data_name(i));
            }

            // Map from string IDs to scene data infos, skip the invalid string
            for (mi::Size i = 1, n = compile_res.target_code->get_string_constant_count();
                    i < n; ++i) {
                char const *name = compile_res.target_code->get_string_constant(i);
                if (name == nullptr)
                    continue;

                if (scene_data_names.count(name) != 0) {
                    if (i >= new_mat.scene_data_infos.size()) {
                        new_mat.scene_data_infos.resize(i + 1);
                    }
                    SceneDataInfo &info = new_mat.scene_data_infos[i];
                    if (strcmp(name, "vertex_color") == 0) {
                        info.data_kind = SceneDataInfo::DK_VERTEX_COLOR;
                        info.interpolation_mode = SceneDataInfo::IM_LINEAR;
                        info.is_uniform = false;
                    } else if (strcmp(name, "row_column") == 0) {
                        info.data_kind = SceneDataInfo::DK_ROW_COLUMN;
                        info.interpolation_mode = SceneDataInfo::IM_NEAREST;
                        info.is_uniform = false;
                    }
                }
            }
        }
    }

    transaction->commit();
}


void initMDL(
    PathTracerState& state,
    mi::examples::mdl::Configure_options const &configure_options,
    std::vector<std::string> const &material_names,
    bool class_compilation)
{
    state.mdl_helper = new Mdl_helper(
        configure_options,
        /*num_texture_spaces=*/  1,
        /*num_texture_results=*/ 16,
        /*enable_derivative=*/   false);

#ifdef NO_DIRECT_CALL
    // In no-direct-call mode, register closest hit shader code and texture runtime to be
    // linked and optimized with the generated code
    state.mdl_helper->set_renderer_module(
        mi::examples::io::get_executable_folder() + "/optix7_mdl_closest_hit_radiance.bc",
        /*visible_functions=*/ "__closesthit__radiance");
#endif

    for (std::string const &material_name : material_names)
        addMDLMaterial(state, material_name, class_compilation);
}


void buildMeshAccel(PathTracerState& state, std::string const &model)
{
    if (model == "sphere")
        state.mesh = new Sphere(state.context, 1.0f, 128, 128);
    else
        state.mesh = new Cube(state.context, 1.0f);
}


void buildScene(PathTracerState& state)
{
    // create a circle of spheres, one per provided material

    int n = int(state.mdl_materials.size());

    float tan_fovH_2 = tanf(float(state.fovH * M_PI / 180.f) / 2.f);

    float circumference, r;
    if (n == 1) {
        circumference = r = 0;
        state.scene_width = 2 * tan_fovH_2 * 3;  // gives an eye distance of 3
    } else {
        // half a sphere radius padding between spheres
        circumference = 2 * n * 1.5f;
        r = circumference / float(2 * M_PI);

        // plus 1.25 for the radius of a sphere and quarter a sphere margin
        state.scene_width = 2 * (r + 1.25f);
    }


    for (int i = 0; i < n; ++i) {
        float a = float(i * (2 * M_PI / n));
        float x = r * cosf(a);
        float y = r * sinf(a);

        state.instances.push_back(
            Instance{ state.mesh, i % state.mdl_materials.size(), {
                1, 0, 0, x,
                0, 1, 0, y,
                0, 0, 1, 0
            }});
    }

    // let camera look at whole sphere circle
    state.cam_base_dist = state.scene_width / 2 / tan_fovH_2;
    state.cam_theta = M_PI / 2;

    // TODO: the parallelogram light is black, so actually not used
    state.params.light.emission = make_float3(   0.0f,   0.0f,   0.0f);
    state.params.light.corner   = make_float3( -65.0f, 100.0f, -50.0f);
    state.params.light.v1       = make_float3(   0.0f,   0.0f, 100.0f);
    state.params.light.v2       = make_float3(-130.0f,   0.0f,   0.0f);
    state.params.light.normal   = normalize(cross(state.params.light.v1, state.params.light.v2));
}


void buildInstanceAccel(PathTracerState& state)
{
    std::vector<OptixInstance> optix_instances;
    optix_instances.reserve(state.instances.size());
    for (Instance &instance : state.instances) {
        OptixInstance optix_instance     = {};
        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = unsigned(optix_instances.size());
        optix_instance.sbtOffset         = instance.material_index * RAY_TYPE_COUNT;
        optix_instance.visibilityMask    = 255;
        optix_instance.traversableHandle = instance.mesh->getBLASHandle();
        memcpy(optix_instance.transform, instance.transform, sizeof(float) * 12);

        optix_instances.push_back(optix_instance);
    }

    CUdeviceptr d_instances = gpuMemDup(optix_instances);


    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = unsigned(optix_instances.size());


    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes tlas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &instance_input,
        1,                  // num build inputs
        &tlas_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        tlas_buffer_sizes.tempSizeInBytes
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.d_ias_output_buffer),
        tlas_buffer_sizes.outputSizeInBytes
    ));

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0,                  // CUDA stream
        &accel_options,
        &instance_input,
        1,                  // num build inputs
        d_temp_buffer,
        tlas_buffer_sizes.tempSizeInBytes,
        state.d_ias_output_buffer,
        tlas_buffer_sizes.outputSizeInBytes,
        &state.ias_handle,
        nullptr,            // emitted property list
        0                   // num emitted properties
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
}


void createModule(PathTracerState& state)
{
    std::string ptx = mi::examples::io::read_text_file(
        mi::examples::io::get_executable_folder() + "/optix7_mdl.ptx");

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &state.module_compile_options,
        &state.pipeline_compile_options,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &state.ptx_module
    ));
}


void createProgramGroups(PathTracerState& state)
{
    OptixProgramGroupOptions  program_group_options = {};

    char   log[2048];
    size_t sizeof_log = sizeof(log);

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &raygen_prog_group_desc,
            /*numProgramGroups=*/ 1,
            &program_group_options,
            log,
            &sizeof_log,
            &state.raygen_prog_group
        ));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            /*numProgramGroups=*/ 1,
            &program_group_options,
            log, &sizeof_log,
            &state.radiance_miss_group
        ));

        // NULL miss program for occlusion rays
        memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = nullptr;
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            /*numProgramGroups=*/ 1,
            &program_group_options,
            log, &sizeof_log,
            &state.occlusion_miss_group
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            /*numProgramGroups=*/ 1,
            &program_group_options,
            log, &sizeof_log,
            &state.occlusion_hit_group
        ));
    }
}


void createPipeline(PathTracerState& state)
{
    std::vector<OptixProgramGroup> program_groups;
    program_groups.push_back(state.raygen_prog_group);
    program_groups.push_back(state.radiance_miss_group);
    program_groups.push_back(state.occlusion_miss_group);
    for (OptixProgramGroup group : state.radiance_hit_groups)
        program_groups.push_back(group);
    program_groups.push_back(state.occlusion_hit_group);
    for (OptixProgramGroup group : state.mdl_callable_groups)
        program_groups.push_back(group);

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        unsigned(program_groups.size()),
        log,
        &sizeof_log,
        &state.pipeline
    ));

    // Calculate the stack sizes, so we can specify all parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup &pg : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes));
    }

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
#ifdef NO_DIRECT_CALL
    uint32_t max_dc_depth = 0;
#else
    uint32_t max_dc_depth = 1;
#endif
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 2;
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}


void createSBT(PathTracerState& state)
{
    // Create a SBT record for the ray generation program
    RayGenSbtRecord raygen_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &raygen_record));


    // Create SBT records for the miss program groups
    MissSbtRecord miss_records[RAY_TYPE_COUNT];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group, &miss_records[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_miss_group, &miss_records[1]));


    // Create SBT records for the hit groups
    std::vector<HitGroupSbtRecord> hitgroup_records;
    hitgroup_records.resize(RAY_TYPE_COUNT * state.mdl_materials.size());

    for (size_t i = 0, n = state.mdl_materials.size(); i < n; ++i) {
        MDLMaterial &mat = state.mdl_materials[i];

        {
            // SBT for radiance ray-type for ith material
            HitGroupSbtRecord &radiance_hit_group = hitgroup_records[i * RAY_TYPE_COUNT + 0];

            OPTIX_CHECK(optixSbtRecordPackHeader(mat.radiance_hit_group, &radiance_hit_group));
            radiance_hit_group.data.vertices =
                reinterpret_cast<MeshVertex*>(state.mesh->getDeviceVertices());
            radiance_hit_group.data.indices =
                reinterpret_cast<short3*>(state.mesh->getDeviceIndices());
            radiance_hit_group.data.texture_handler =
                reinterpret_cast<Texture_handler *>(mat.get_device_texture_handler());
            radiance_hit_group.data.mdl_callable_base_index = mat.callable_base_index;

            if (mat.mat_info->get_argument_block_size() > 0) {
                radiance_hit_group.data.arg_block = mat.mat_info->get_argument_block_data_gpu();
            }

            radiance_hit_group.data.scene_data_info =
                reinterpret_cast<SceneDataInfo *>(mat.get_device_scene_data_infos());
        }

        {
            // SBT for occlusion ray-type for ith material
            HitGroupSbtRecord &occlusion_hit_group = hitgroup_records[i * RAY_TYPE_COUNT + 1];

            OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_hit_group, &occlusion_hit_group));
        }
    }


    // Create SBT records for the direct callables (if used)
    std::vector<CallableSbtRecord> callable_records;
    callable_records.reserve(state.mdl_callable_groups.size());
    for (OptixProgramGroup &group : state.mdl_callable_groups) {
        callable_records.push_back(CallableSbtRecord());
        OPTIX_CHECK(optixSbtRecordPackHeader(group, &callable_records.back()));
    }


    // Fill the shader binding table
    state.sbt.raygenRecord                 = gpuMemDup(raygen_record);
    state.sbt.missRecordBase               = gpuMemDup(miss_records);
    state.sbt.missRecordStrideInBytes      = static_cast<uint32_t>(sizeof(MissSbtRecord));
    state.sbt.missRecordCount              = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase           = gpuMemDup(hitgroup_records);
    state.sbt.hitgroupRecordStrideInBytes  = static_cast<uint32_t>(sizeof(HitGroupSbtRecord));
    state.sbt.hitgroupRecordCount          = unsigned(RAY_TYPE_COUNT * state.mdl_materials.size());
    state.sbt.callablesRecordBase          = gpuMemDup(callable_records);
    state.sbt.callablesRecordStrideInBytes = sizeof(CallableSbtRecord);
    state.sbt.callablesRecordCount         = unsigned(callable_records.size());
}


void initLaunchParams(PathTracerState& state)
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float3)
    ));
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    state.params.subframe_index = 0u;

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

    state.params.handle = state.ias_handle;
}


void handleCameraUpdate(PathTracerState& state)
{
    if (!state.camera_changed)
        return;
    state.camera_changed = false;

    state.params.U = make_float3(
        float(cos(state.cam_phi)),
        0,
        float(-sin(state.cam_phi))
    );

    state.params.V = make_float3(
        float(-sin(state.cam_phi) * cos(state.cam_theta)),
        float(sin(state.cam_theta)),
        float(-cos(state.cam_phi) * cos(state.cam_theta))
    ) * float(state.params.height) / float(state.params.width);

    float3 cam_dir = make_float3(
        float(-sin(state.cam_phi) * sin(state.cam_theta)),
        float(-cos(state.cam_theta)),
        float(-cos(state.cam_phi) * sin(state.cam_theta))
    );

    state.params.W = cam_dir / tanf(state.fovH / 2 * float(2 * M_PI / 360));

    float dist = float(state.cam_base_dist * pow(0.95, double(state.cam_zoom)));
    state.params.eye = -dist * cam_dir;
}


void handleResize(mi::examples::mdl::GL_display *gl_display, PathTracerState& state)
{
    if (!state.resize_dirty)
        return;
    state.resize_dirty = false;

    if (gl_display)
        gl_display->resize(state.params.width, state.params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));

    if (state.params.width == 0 || state.params.height == 0)
        state.params.accum_buffer = 0;
    else
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&state.params.accum_buffer),
            state.params.width * state.params.height * sizeof(float3)
        ));
}


void updateState(mi::examples::mdl::GL_display *gl_display, PathTracerState& state)
{
    if (state.camera_changed || state.resize_dirty)
        state.params.subframe_index = 0;

    handleCameraUpdate(state);
    handleResize(gl_display, state);
}


void launchSubframe(mi::examples::mdl::GL_display *gl_display, PathTracerState& state)
{
    // Map display buffer, if present
    if (gl_display)
        state.params.frame_buffer = reinterpret_cast<uchar4*>(gl_display->map(state.stream));

    // Copy state parameters
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(state.d_params),
        &state.params, sizeof(Params),
        cudaMemcpyHostToDevice, state.stream
    ));

    // Launch
    OPTIX_CHECK(optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(Params),
        &state.sbt,
        state.params.width,   // launch width
        state.params.height,  // launch height
        1                     // launch depth
    ));

    // Unmap again, if necessary
    if (gl_display)
        gl_display->unmap(state.stream);

    // Synchronize to get proper frame times
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
}


void cleanupState(PathTracerState& state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    for (OptixProgramGroup &group : state.radiance_hit_groups) {
        OPTIX_CHECK(optixProgramGroupDestroy(group));
    }
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_miss_group));
    for (OptixProgramGroup &group : state.mdl_callable_groups) {
        OPTIX_CHECK(optixProgramGroupDestroy(group));
    }
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));


    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.callablesRecordBase)));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_ias_output_buffer)));

    cudaStreamDestroy(state.stream);

    delete state.mesh;

    // clear MDL materials which hold argument block objects before shutting down MDL SDK
    state.mdl_materials.clear();

    delete state.mdl_helper;
}


//------------------------------------------------------------------------------
//
// GUI
//
//------------------------------------------------------------------------------

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard || io.WantCaptureMouse)
        return;

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    PathTracerState* state = static_cast<PathTracerState*>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS) {
        state->mouse_button = button;
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            state->mouse_move_start_x = xpos;
            state->mouse_move_start_y = ypos;
        }
    } else {
        state->mouse_button = -1;
    }
}


void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard || io.WantCaptureMouse)
        return;

    PathTracerState* state = static_cast<PathTracerState*>(glfwGetWindowUserPointer(window));

    if (state->mouse_button == GLFW_MOUSE_BUTTON_LEFT) {
        state->cam_phi -= (xpos - state->mouse_move_start_x) * 0.001 * M_PI;
        state->cam_theta -= (ypos - state->mouse_move_start_y) * 0.001 * M_PI;
        state->cam_theta = std::max(state->cam_theta, 0.0);
        state->cam_theta = std::min(state->cam_theta, M_PI);

        state->mouse_move_start_x = xpos;
        state->mouse_move_start_y = ypos;

        state->camera_changed = true;
    }
}


void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    PathTracerState* state = static_cast<PathTracerState*>(glfwGetWindowUserPointer(window));
    state->params.width = res_x;
    state->params.height = res_y;
    state->camera_changed = true;
    state->resize_dirty = true;
}


void keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard || io.WantCaptureMouse)
        return;

    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
}


void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard || io.WantCaptureMouse)
        return;

    PathTracerState* state = static_cast<PathTracerState*>(glfwGetWindowUserPointer(window));

    if (yscroll > 0.0) {
        ++state->cam_zoom;
        state->camera_changed = true;
    } else if (yscroll < 0.0) {
        --state->cam_zoom;
        state->camera_changed = true;
    }
}


// Initialize OpenGL and create a window with an associated OpenGL context.
GLFWwindow *initGUI(PathTracerState& state)
{
    // Initialize GLFW
    check_success(glfwInit());
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create an OpenGL window and a context
    GLFWwindow *window = glfwCreateWindow(
        state.params.width, state.params.height, WINDOW_TITLE, nullptr, nullptr);
    if (!window) {
        exit_failure("Error creating OpenGL window!");
    }

    // Attach context to window
    glfwMakeContextCurrent(window);

    // Initialize GLEW to get OpenGL extensions
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        exit_failure("GLEW error: %s", glewGetErrorString(res));
    }

    // Disable VSync
    glfwSwapInterval(0);

    check_success(glGetError() == GL_NO_ERROR);

    // set callbacks
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetWindowSizeCallback(window, windowSizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetWindowUserPointer(window, &state);

    // ImGui initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
    io.IniFilename = nullptr;       // disable creating imgui.ini
    ImGui::StyleColorsDark();
    io.Fonts->AddFontDefault();
    ImGui::GetStyle().Alpha = 0.7f;

    return window;
}


void uninitGUI(GLFWwindow *window)
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}


void showMaterialGUI(PathTracerState &state)
{
    ImGui::SetNextWindowPos(ImVec2(10, 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin("Material parameters");
    ImGui::Text("CTRL + Click to manually enter numbers");

    std::string current_label = state.mdl_materials[state.cur_material_index].mat_info->name();

    if (ImGui::BeginCombo("material", current_label.c_str())) {
        // add selectable materials to the combo box
        for (size_t i = 0, n = state.mdl_materials.size(); i < n; ++i) {
            bool is_selected = i == state.cur_material_index;
            std::string label = state.mdl_materials[i].mat_info->name();
            if (ImGui::Selectable(label.c_str(), is_selected))
                state.cur_material_index = i;
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        };

        ImGui::EndCombo();
    }

    Material_info &mat_info = *state.mdl_materials[state.cur_material_index].mat_info;

    bool changed = false;
    const char *group_name = nullptr;
    int id = 0;
    for (std::list<Param_info>::iterator it = mat_info.params().begin(),
        end = mat_info.params().end(); it != end; ++it, ++id)
    {
        Param_info &param = *it;

        // Ensure unique ID even for parameters with same display names
        ImGui::PushID(id);

        // Group name changed? -> Start new group with new header
        if ((!param.group_name() != !group_name) ||
            (param.group_name() && (!group_name || strcmp(group_name, param.group_name()) != 0)))
        {
            ImGui::Separator();
            if (param.group_name() != nullptr)
                ImGui::Text("%s", param.group_name());
            group_name = param.group_name();
        }

        // Choose proper edit control depending on the parameter kind
        switch (param.kind()) {
        case Param_info::PK_FLOAT:
            changed |= ImGui::SliderFloat(
                param.display_name(),
                &param.data<float>(),
                param.range_min(),
                param.range_max());
            break;
        case Param_info::PK_FLOAT2:
            changed |= ImGui::SliderFloat2(
                param.display_name(),
                &param.data<float>(),
                param.range_min(),
                param.range_max());
            break;
        case Param_info::PK_FLOAT3:
            changed |= ImGui::SliderFloat3(
                param.display_name(),
                &param.data<float>(),
                param.range_min(),
                param.range_max());
            break;
        case Param_info::PK_COLOR:
            changed |= ImGui::ColorEdit3(
                param.display_name(),
                &param.data<float>());
            break;
        case Param_info::PK_BOOL:
            changed |= ImGui::Checkbox(
                param.display_name(),
                &param.data<bool>());
            break;
        case Param_info::PK_INT:
            changed |= ImGui::SliderInt(
                param.display_name(),
                &param.data<int>(),
                int(param.range_min()),
                int(param.range_max()));
            break;
        case Param_info::PK_ARRAY:
            {
                ImGui::Text("%s", param.display_name());
                ImGui::Indent(16.0f);
                char *ptr = &param.data<char>();
                for (mi::Size i = 0, n = param.array_size(); i < n; ++i) {
                    std::string idx_str = std::to_string(i);
                    switch (param.array_elem_kind()) {
                    case Param_info::PK_FLOAT:
                        changed |= ImGui::SliderFloat(
                            idx_str.c_str(),
                            reinterpret_cast<float *>(ptr),
                            param.range_min(),
                            param.range_max());
                        break;
                    case Param_info::PK_FLOAT2:
                        changed |= ImGui::SliderFloat2(
                            idx_str.c_str(),
                            reinterpret_cast<float *>(ptr),
                            param.range_min(),
                            param.range_max());
                        break;
                    case Param_info::PK_FLOAT3:
                        changed |= ImGui::SliderFloat3(
                            idx_str.c_str(),
                            reinterpret_cast<float *>(ptr),
                            param.range_min(),
                            param.range_max());
                        break;
                    case Param_info::PK_COLOR:
                        changed |= ImGui::ColorEdit3(
                            idx_str.c_str(),
                            reinterpret_cast<float *>(ptr));
                        break;
                    case Param_info::PK_BOOL:
                        changed |= ImGui::Checkbox(
                            param.display_name(),
                            reinterpret_cast<bool *>(ptr));
                        break;
                    case Param_info::PK_INT:
                        changed |= ImGui::SliderInt(
                            param.display_name(),
                            reinterpret_cast<int *>(ptr),
                            int(param.range_min()),
                            int(param.range_max()));
                        break;
                    default:
                        exit_failure("Array element type invalid or unhandled.");
                    }
                    ptr += param.array_pitch();
                }
                ImGui::Unindent(16.0f);
            }
            break;
        case Param_info::PK_ENUM:
            {
                int value = param.data<int>();
                std::string curr_value;

                const Enum_type_info *info = param.enum_info();
                for (size_t i = 0, n = info->values.size(); i < n; ++i) {
                    if (info->values[i].value == value) {
                        curr_value = info->values[i].name;
                        break;
                    }
                }

                if (ImGui::BeginCombo(param.display_name(), curr_value.c_str())) {
                    for (size_t i = 0, n = info->values.size(); i < n; ++i) {
                        const std::string &name = info->values[i].name;
                        bool is_selected = (curr_value == name);
                        if (ImGui::Selectable(
                            info->values[i].name.c_str(), is_selected)) {
                            param.data<int>() = info->values[i].value;
                            changed = true;
                        }
                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
            }
            break;
        case Param_info::PK_STRING:
        case Param_info::PK_TEXTURE:
        case Param_info::PK_LIGHT_PROFILE:
        case Param_info::PK_BSDF_MEASUREMENT:
            // Currently not supported by this example
            break;
        default:
            break;
        }

        ImGui::PopID();
    }

    ImGui::End();

    if (changed) {
        mat_info.update_arg_block_on_gpu(state.stream);
        state.params.subframe_index = 0;
    }
}


// Helper class for collecting and showing frame statistics.
class Frame_stats
{
public:
    // Constructor.
    //
    // \param update_min_interval  the minimum interval between statistic updates in seconds
    Frame_stats(double update_min_interval = 0.5)
        : m_update_min_interval(update_min_interval)
        , m_state_update_time(0.0)
        , m_render_time(0.0)
        , m_display_time(0.0)
        , m_last_update_frames(-1)
        , m_last_update_time(std::chrono::steady_clock::now())
    {
        m_stats_text[0] = 0;
    }

    void newFrame()
    {
        m_cur_state_start_time = std::chrono::steady_clock::now();
    }

    void endStateUpdate()
    {
        endState(m_state_update_time);
    }

    void endRender()
    {
        endState(m_render_time);
    }

    void endDisplay()
    {
        endState(m_display_time);
    }

    void showStats()
    {
        ++m_last_update_frames;
        if (m_last_update_frames <= 0)
            return;

        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin("##notitle", nullptr,
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoNav);

        if (m_cur_state_start_time - m_last_update_time > m_update_min_interval) {
            typedef std::chrono::duration<double, std::milli> durationMs;

            snprintf(m_stats_text, sizeof(m_stats_text),
                "%5.1f fps\n\n"
                "state update: %8.1f ms\n"
                "render:       %8.1f ms\n"
                "display:      %8.1f ms\n",
                m_last_update_frames / std::chrono::duration<double>(
                    m_cur_state_start_time - m_last_update_time).count(),
                (durationMs(m_state_update_time) / m_last_update_frames).count(),
                (durationMs(m_render_time) / m_last_update_frames).count(),
                (durationMs(m_display_time) / m_last_update_frames).count());

            m_last_update_time = m_cur_state_start_time;
            m_last_update_frames = 0;
            m_state_update_time = m_render_time = m_display_time =
                std::chrono::duration<double>::zero();
        }

        ImGui::TextUnformatted(m_stats_text);
        ImGui::End();
    }

private:
    void endState(std::chrono::duration<double> &state_time)
    {
        auto t = std::chrono::steady_clock::now();
        state_time += t - m_cur_state_start_time;
        m_cur_state_start_time = t;
    }

private:
    std::chrono::duration<double> m_update_min_interval;

    std::chrono::duration<double> m_state_update_time;
    std::chrono::duration<double> m_render_time;
    std::chrono::duration<double> m_display_time;

    char m_stats_text[128];
    int m_last_update_frames;
    std::chrono::steady_clock::time_point m_last_update_time;
    std::chrono::steady_clock::time_point m_cur_state_start_time;
};


//------------------------------------------------------------------------------
//
// Usage
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
    std::cerr
        << "Usage  : " << argv0 << " [options] [<material_name>]\n"
        << "Options:\n"
        << " --file | -f <filename> File for image output\n"
        << " --launch-samples | -s  Number of samples per pixel per launch (default: 16)\n"
        << " --mdl_path | -p <path> MDL search path, can occur multiple times.\n"
        << " --device <id>          run on CUDA device <id> (default: 0)\n"
        << " --hdr <filename>       HDR environment map (default: data/environment.hdr)\n"
        << " --res <res_x> <res_y>  Resolution (default: 1024 x 1024)\n"
        << " --fov <fov>            Horizontal field of view angle in degrees (default: 45.0)\n"
        << " --model <name>         Can be \"sphere\" or \"cube\" (default: sphere)\n"
        << " --nocc                 Disable class-compilation for MDL material\n"
        << " --help | -h            Print this usage message\n";
    exit(0);
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    PathTracerState state;
    state.params.width = 1024;
    state.params.height = 1024;
    state.params.samples_per_launch = 16;

    //
    // Parse command line options
    //
    mi::examples::mdl::Configure_options configure_options;
    std::string outfile;
    std::vector<std::string> material_names;
    std::string env_path;
    std::string model = "sphere";
    bool class_compilation = true;
    int cuda_device_id = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (!arg.empty() && arg[0] == '-') {
            if (arg == "--help" || arg == "-h") {
                printUsageAndExit(argv[0]);
            } else if (arg == "--file" || arg == "-f") {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                outfile = argv[++i];
            } else if (arg == "--launch-samples" || arg == "-s") {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                state.params.samples_per_launch = atoi(argv[++i]);
            } else if (arg == "--mdl_path" || arg == "-p") {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                configure_options.additional_mdl_paths.push_back(argv[++i]);
            } else if (arg == "--hdr") {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                env_path = argv[++i];
            } else if (arg == "--device") {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                cuda_device_id = atoi(argv[++i]);
            } else if (arg == "--res") {
                if (i >= argc - 2)
                    printUsageAndExit(argv[0]);
                state.params.width  = atoi(argv[++i]);
                state.params.height = atoi(argv[++i]);
            } else if (arg == "--fov") {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                state.fovH = float(atof(argv[++i]));
            } else if (arg == "--model") {
                if (i >= argc - 1)
                    printUsageAndExit(argv[0]);
                model = argv[++i];
                if (model != "sphere" && model != "cube")
                    printUsageAndExit(argv[0]);
            } else if (arg == "--nocc") {
                class_compilation = false;
            } else {
                std::cerr << "Unknown option '" << argv[i] << "'\n";
                printUsageAndExit(argv[0]);
            }
        } else {
            material_names.push_back(arg);
        }
    }

    // Use default material, if none was provided via command line
    if (material_names.empty())
        material_names.push_back("::nvidia::sdk_examples::tutorials::example_df");

    if (env_path.empty())
        env_path = mi::examples::mdl::get_examples_root() +
            "/mdl/nvidia/sdk_examples/resources/environment.hdr";

    try
    {
        //
        // Set up OptiX state
        //
        createContext(state, cuda_device_id);
        initMDL(state, configure_options, material_names, class_compilation);
        initEnvironmentLight(state, env_path);
        buildMeshAccel(state, model);
        buildScene(state);
        buildInstanceAccel(state);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);
        createSBT(state);
        initLaunchParams(state);


        if (outfile.empty()) {
            GLFWwindow* window = initGUI(state);

            //
            // Render loop
            //
            {
                mi::examples::mdl::GL_display gl_display(state.params.width, state.params.height);
                Frame_stats stats;

                do {
                    // start new frame, poll events and update state and material parameters
                    stats.newFrame();
                    glfwPollEvents();
                    updateState(&gl_display, state);

                    // don't render anything, if minimized
                    if (state.params.width == 0 || state.params.height == 0) {
                        // wait until something happens
                        glfwWaitEvents();
                        continue;
                    }

                    ImGui_ImplOpenGL3_NewFrame();
                    ImGui_ImplGlfw_NewFrame();
                    ImGui::NewFrame();
                    showMaterialGUI(state);

                    stats.endStateUpdate();

                    // render scene
                    launchSubframe(&gl_display, state);
                    stats.endRender();

                    // copy rendered image to frame buffer
                    gl_display.update_display();
                    stats.endDisplay();

                    // show frame statistics
                    stats.showStats();

                    // render ImGui windows to frame buffer
                    ImGui::Render();
                    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                    // swap frame buffers
                    glfwSwapBuffers(window);

                    ++state.params.subframe_index;
                } while (!glfwWindowShouldClose(window));

                CUDA_CHECK(cudaStreamSynchronize(state.stream));
            }

            uninitGUI(window);
        } else {
            updateState(nullptr, state);

            // render scene
            launchSubframe(nullptr, state);

            // copy rendered image to canvas
            mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                state.mdl_helper->get_image_api()->create_canvas(
                    "Rgb_fp", state.params.width, state.params.height));
            mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
            CUDA_CHECK(cudaMemcpy(
                tile->get_data(), state.params.accum_buffer,
                state.params.width * state.params.height * sizeof(float3),
                cudaMemcpyDeviceToHost));

            // export canvas to file
            state.mdl_helper->get_impexp_api()->export_canvas(outfile.c_str(), canvas.get());
        }

        cleanupState(state);
    }
    catch (std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
