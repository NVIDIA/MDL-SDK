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

// Code shared by CUDA MDL Core examples

#ifndef EXAMPLE_CUDA_SHARED_H
#define EXAMPLE_CUDA_SHARED_H

#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "example_shared_backends.h"

#include <cuda.h>
#ifdef OPENGL_INTEROP
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cudaGL.h>
#endif
#include <cuda_runtime.h>
#include <vector_functions.h>


// Structure representing an MDL texture, containing filtered and unfiltered CUDA texture
// objects and the size of the texture.
struct Texture
{
    Texture(
        cudaTextureObject_t  filtered_object,
        cudaTextureObject_t  unfiltered_object,
        uint3                size)
    : filtered_object(filtered_object)
    , unfiltered_object(unfiltered_object)
    , size(size)
    , inv_size(make_float3(1.0f / size.x, 1.0f / size.y, 1.0f / size.z))
    {}

    cudaTextureObject_t  filtered_object;    // uses filter mode cudaFilterModeLinear
    cudaTextureObject_t  unfiltered_object;  // uses filter mode cudaFilterModePoint
    uint3                size;               // size of the texture, needed for texel access
    float3               inv_size;           // the inverse values of the size of the texture
};

// Structure representing an MDL bsdf measurement.
// The fields must match the device-side counterpart in texture_support_cuda.h, so the data
// can be copied verbatim to the GPU.
struct Mbsdf
{
    Mbsdf()
    {
        for (unsigned i = 0; i < 2; ++i) {
            has_data[i] = 0u;
            eval_data[i] = 0;
            sample_data[i] = nullptr;
            albedo_data[i] = nullptr;
            max_albedo[i] = 0.0f;
            angular_resolution[i] = make_uint2(0u, 0u);
            inv_angular_resolution[i] = make_float2(0.0f, 0.0f);
            num_channels[i] = 0;
        }
    }

    void add(unsigned                  part_idx,
             const uint2&              angular_resolution_in,
             unsigned                  num_channels_in)
    {
        has_data[part_idx] = 1u;
        angular_resolution[part_idx] = angular_resolution_in;
        inv_angular_resolution[part_idx] = make_float2(
            1.0f / float(angular_resolution_in.x),
            1.0f / float(angular_resolution_in.y));
        num_channels[part_idx] = num_channels_in;
    }

    unsigned            has_data[2];            // true if there is a measurement for this part
    cudaTextureObject_t eval_data[2];           // uses filter mode cudaFilterModeLinear
    float               max_albedo[2];          // max albedo used to limit the multiplier
    float*              sample_data[2];         // CDFs for sampling a BSDF measurement
    float*              albedo_data[2];         // max albedo for each theta (isotropic)

    uint2           angular_resolution[2];      // size of the dataset, needed for texel access
    float2          inv_angular_resolution[2];  // the inverse values of the size of the dataset
    unsigned        num_channels[2];            // number of color channels (1 or 3)
};

// Structure representing the resources used by the generated code of a target code.
// The layout has to match the device-side struct Target_code_data in example_df_cuda.cu.
struct Target_code_data
{
    Target_code_data(
        size_t      num_textures,
        CUdeviceptr textures,
        size_t      num_mbsdfs,
        CUdeviceptr mbsdfs,
        CUdeviceptr ro_data_segment)
        : num_textures(num_textures)
        , textures(textures)
        , num_mbsdfs(num_mbsdfs)
        , mbsdfs(mbsdfs)
        , ro_data_segment(ro_data_segment)
    {}

    size_t      num_textures;      // number of elements in the textures field
    CUdeviceptr textures;          // a device pointer to a list of Texture objects, if used

    size_t      num_mbsdfs;        // number of elements in the mbsdfs field
    CUdeviceptr mbsdfs;            // a device pointer to a list of Mbsdf objects, if used

    CUdeviceptr ro_data_segment;   // a device pointer to the read-only data segment, if used
};


//------------------------------------------------------------------------------
//
// Helper functions
//
//------------------------------------------------------------------------------

// Return a textual representation of the given value.
template <typename T>
std::string to_string(T val)
{
    std::ostringstream stream;
    stream << val;
    return stream.str();
}


//------------------------------------------------------------------------------
//
// CUDA helper functions
//
//------------------------------------------------------------------------------

// Return a human-readable description for a CUDA driver or runtime error code.
inline const char* cuda_error_string(int err)
{
    const char* str = nullptr;
    if (cuGetErrorString(static_cast<CUresult>(err), &str) == CUDA_SUCCESS && str != nullptr) {
        return str;
    }
    return cudaGetErrorString(static_cast<cudaError_t>(err));
}

// Return a textual representation of a CUDA version number (e.g. 12060 -> "12.6").
inline std::string cuda_version_to_string(int version)
{
    int minor = version % 100;
    int major = version / 100;
    if (major % 10 == 0) {
        major /= 10;
    }
    if (minor % 10 == 0) {
        minor /= 10;
    }
    return to_string(major) + '.' + to_string(minor);
}

// Helper macro. Checks whether the expression is cudaSuccess and if not prints a message and
// resets the device and exits.
#define check_cuda_success(expr)                                                      \
    do {                                                                              \
        int err = (expr);                                                             \
        if (err != 0) {                                                               \
            fprintf(stderr, "CUDA error %d \"%s\" in file %s, line %u: \"%s\".\n",    \
                err, cuda_error_string(err), __FILE__, __LINE__, #expr);              \
            keep_console_open();                                                      \
            cudaDeviceReset();                                                        \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (false)

// Check that the installed CUDA driver supports the CUDA runtime version
// this example was compiled against.
inline void check_cuda_driver_version()
{
    int driver_version = 0;
    check_cuda_success(cuDriverGetVersion(&driver_version));
    if (driver_version < CUDART_VERSION) {
        fprintf(stderr,
            "CUDA driver version %s is too old; this example requires at least %s.\n"
            "Please update your NVIDIA display driver.\n",
            cuda_version_to_string(driver_version).c_str(),
            cuda_version_to_string(CUDART_VERSION).c_str());
        keep_console_open();
        exit(EXIT_FAILURE);
    }
}


// Initialize CUDA.
CUcontext init_cuda(
#ifdef OPENGL_INTEROP
    const bool opengl_interop
#endif
    )
{
    CUdevice cu_device;
    CUcontext cu_context;

    check_cuda_success(cuInit(0));
    check_cuda_driver_version();
#if defined(OPENGL_INTEROP) && !defined(__APPLE__)
    if (opengl_interop) {
        // Use first device used by OpenGL context
        unsigned int num_cu_devices;
        check_cuda_success(cuGLGetDevices(&num_cu_devices, &cu_device, 1, CU_GL_DEVICE_LIST_ALL));
    }
    else
#endif
        // Use first device
        check_cuda_success(cuDeviceGet(&cu_device, 0));

    check_cuda_success(cuCtxCreate(&cu_context, 0, cu_device));

    // For this example, increase printf CUDA buffer size to support a larger number
    // of MDL debug::print() calls per CUDA kernel launch
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 16 * 1024 * 1024);

    return cu_context;
}

// Uninitialize CUDA.
void uninit_cuda(CUcontext cuda_context)
{
    check_cuda_success(cuCtxDestroy(cuda_context));
}

// Allocate memory on GPU and copy the given data to the allocated memory.
CUdeviceptr gpu_mem_dup(void const *data, size_t size)
{
    CUdeviceptr device_ptr;
    check_cuda_success(cuMemAlloc(&device_ptr, size));
    check_cuda_success(cuMemcpyHtoD(device_ptr, data, size));
    return device_ptr;
}

// Allocate memory on GPU and copy the given data to the allocated memory.
template<typename T>
CUdeviceptr gpu_mem_dup(std::vector<T> const &data)
{
    return gpu_mem_dup(&data[0], data.size() * sizeof(T));
}


//------------------------------------------------------------------------------
//
// Material_gpu_context class
//
//------------------------------------------------------------------------------

// Helper class responsible for making textures and read-only data available to the GPU
// by generating and managing a list of Target_code_data objects.
class Material_gpu_context
{
public:
    // Constructor.
    Material_gpu_context(bool enable_derivatives)
        : m_enable_derivatives(enable_derivatives)
        , m_device_target_code_data_list(0)
        , m_device_target_argument_block_list(0)
    {
        // Use first entry as "not-used" block
        m_target_argument_block_list.push_back(0);
    }

    // Free all acquired resources.
    ~Material_gpu_context();

    // Prepare the needed data of the given target code.
    bool prepare_target_code_data(Target_code const *target_code);

    // Get a device pointer to the target code data list.
    CUdeviceptr get_device_target_code_data_list();

    // Get a device pointer to the target argument block list.
    CUdeviceptr get_device_target_argument_block_list();

    // Get a device pointer to the i'th target argument block.
    CUdeviceptr get_device_target_argument_block(size_t i)
    {
        // First entry is the "not-used" block, so start at index 1.
        if (i + 1 >= m_target_argument_block_list.size()) return 0;
        return m_target_argument_block_list[i + 1];
    }

    // Get the number of target argument blocks.
    size_t get_argument_block_count() const
    {
        return m_own_arg_blocks.size();
    }

    // Get the argument block of the i'th BSDF.
    // If the BSDF has no target argument block, size_t(~0) is returned.
    size_t get_bsdf_argument_block_index(size_t i) const
    {
        if (i >= m_bsdf_arg_block_indices.size()) return size_t(~0);
        return m_bsdf_arg_block_indices[i];
    }

    // Get a writable copy of the i'th target argument block.
    Argument_block *get_argument_block(size_t i)
    {
        if (i >= m_own_arg_blocks.size())
            return nullptr;
        return &m_own_arg_blocks[i];
    }

    // Get the layout of the i'th target argument block.
    mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> get_argument_block_layout(
        size_t i)
    {
        if (i >= m_arg_block_layouts.size())
            return mi::base::Handle<mi::mdl::IGenerated_code_value_layout const>();
        return m_arg_block_layouts[i];
    }

    // Update the i'th target argument block on the device with the data from the corresponding
    // block returned by get_argument_block().
    void update_device_argument_block(size_t i);
private:
    // Generate the next mip map level data, returns a pointer to a newly allocated buffer
    // and updates the width and height parameters with the new data.
    float4 *gen_mip_level(
        float4 *prev_data,
        unsigned &cur_width,
        unsigned &cur_height);

    // Prepare the texture identified by the texture_index for use by the texture access functions
    // on the GPU.
    bool prepare_texture(
        Target_code const       *code_ptx,
        size_t                texture_index,
        std::vector<Texture> &textures);

    // Prepare the BSDF measurement identified by mbsdf_index for use by the bsdf measurement
    // access functions on the GPU.
    bool prepare_mbsdf(
        Target_code const   *code_ptx,
        size_t               mbsdf_index,
        std::vector<Mbsdf>  &mbsdfs);

    // If true, mipmaps will be generated for all 2D textures.
    bool m_enable_derivatives;

    // The device pointer of the target code data list.
    CUdeviceptr m_device_target_code_data_list;

    // List of all target code data objects owned by this context.
    std::vector<Target_code_data> m_target_code_data_list;

    // The device pointer of the target argument block list.
    CUdeviceptr m_device_target_argument_block_list;

    // List of all target argument blocks owned by this context.
    std::vector<CUdeviceptr> m_target_argument_block_list;

    // List of all local, writable copies of the target argument blocks.
    std::vector<Argument_block> m_own_arg_blocks;

    // List of argument block indices per material BSDF.
    std::vector<size_t> m_bsdf_arg_block_indices;

    // List of all target argument block layouts.
    std::vector<mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> > m_arg_block_layouts;

    // List of all Texture objects owned by this context.
    std::vector<Texture> m_all_textures;

    // List of all CUDA arrays owned by this context.
    std::vector<cudaArray_t> m_all_texture_arrays;

    // List of all CUDA mipmapped arrays owned by this context.
    std::vector<cudaMipmappedArray_t> m_all_texture_mipmapped_arrays;

    // List of all Mbsdf objects owned by this context (used for CUDA resource cleanup).
    std::vector<Mbsdf> m_all_mbsdfs;
};

// Free all acquired resources.
Material_gpu_context::~Material_gpu_context()
{
    for (std::vector<cudaArray_t>::iterator it = m_all_texture_arrays.begin(),
            end = m_all_texture_arrays.end(); it != end; ++it) {
        check_cuda_success(cudaFreeArray(*it));
    }
    for (std::vector<cudaMipmappedArray_t>::iterator it = m_all_texture_mipmapped_arrays.begin(),
            end = m_all_texture_mipmapped_arrays.end(); it != end; ++it) {
        check_cuda_success(cudaFreeMipmappedArray(*it));
    }
    for (std::vector<Texture>::iterator it = m_all_textures.begin(),
            end = m_all_textures.end(); it != end; ++it) {
        check_cuda_success(cudaDestroyTextureObject(it->filtered_object));
        check_cuda_success(cudaDestroyTextureObject(it->unfiltered_object));
    }
    for (std::vector<Mbsdf>::iterator it = m_all_mbsdfs.begin(),
            end = m_all_mbsdfs.end(); it != end; ++it) {
        for (unsigned p = 0; p < 2; ++p) {
            if (it->has_data[p] != 0u) {
                check_cuda_success(cudaDestroyTextureObject(it->eval_data[p]));
                if (it->sample_data[p])
                    check_cuda_success(cuMemFree(
                        reinterpret_cast<CUdeviceptr>(it->sample_data[p])));
                if (it->albedo_data[p])
                    check_cuda_success(cuMemFree(
                        reinterpret_cast<CUdeviceptr>(it->albedo_data[p])));
            }
        }
    }
    for (std::vector<Target_code_data>::iterator it = m_target_code_data_list.begin(),
            end = m_target_code_data_list.end(); it != end; ++it) {
        if (it->textures)
            check_cuda_success(cuMemFree(it->textures));
        if (it->mbsdfs)
            check_cuda_success(cuMemFree(it->mbsdfs));
        if (it->ro_data_segment)
            check_cuda_success(cuMemFree(it->ro_data_segment));
    }
    for (std::vector<CUdeviceptr>::iterator it = m_target_argument_block_list.begin(),
            end = m_target_argument_block_list.end(); it != end; ++it) {
        if (*it != 0)
            check_cuda_success(cuMemFree(*it));
    }
    check_cuda_success(cuMemFree(m_device_target_code_data_list));
}

// Get a device pointer to the target code data list.
CUdeviceptr Material_gpu_context::get_device_target_code_data_list()
{
    if (!m_device_target_code_data_list)
        m_device_target_code_data_list = gpu_mem_dup(m_target_code_data_list);
    return m_device_target_code_data_list;
}

// Get a device pointer to the target argument block list.
CUdeviceptr Material_gpu_context::get_device_target_argument_block_list()
{
    if (!m_device_target_argument_block_list)
        m_device_target_argument_block_list = gpu_mem_dup(m_target_argument_block_list);
    return m_device_target_argument_block_list;
}

// Generate the next mip map level data, returns a pointer to a newly allocated buffer
// and updates the width and height parameters with the new data.
float4 *Material_gpu_context::gen_mip_level(
    float4 *prev_data,
    unsigned &cur_width,
    unsigned &cur_height)
{
    unsigned offsets_x[4] = { 0, 1, 0, 1 };
    unsigned offsets_y[4] = { 0, 0, 1, 1 };

    unsigned prev_width = cur_width, prev_height = cur_height;
    unsigned width  = std::max(prev_width / 2, 1u);
    unsigned height = std::max(prev_height / 2, 1u);

    float4 *data = static_cast<float4 *>(malloc(width * height * sizeof(float4)));

    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            // The current pixel (x,y) corresponds to the four pixels
            // [prev_x, prev_x+1] x [prev_y,prev_y+1] in the the previous layer.
            unsigned prev_x = 2 * x, prev_y = 2 * y;

            unsigned num_summands = 0;
            float4 sum = { 0, 0, 0, 0 };

            // Loop over the at most four pixels corresponding to pixel (x,y)
            for (unsigned i = 0; i < 4; ++i) {
                unsigned cur_prev_x = prev_x + offsets_x[i];
                unsigned cur_prev_y = prev_y + offsets_y[i];

                if (cur_prev_x >= prev_width || cur_prev_y >= prev_height)
                    continue;

                float4 *prev_ptr = prev_data + cur_prev_y * prev_width + cur_prev_x;
                sum.x += prev_ptr->x;
                sum.y += prev_ptr->y;
                sum.z += prev_ptr->z;
                sum.w += prev_ptr->w;
                ++num_summands;
            }
            float4 *data_ptr = data + y * width + x;
            data_ptr->x = sum.x / num_summands;
            data_ptr->y = sum.y / num_summands;
            data_ptr->z = sum.z / num_summands;
            data_ptr->w = sum.w / num_summands;
        }
    }

    // update width and height
    cur_width = width;
    cur_height = height;

    return data;
}

// Prepare the texture identified by the texture_index for use by the texture access functions
// on the GPU.
// Note: Currently no support for 3D textures, Udim textures, PTEX and cubemaps.
bool Material_gpu_context::prepare_texture(
    Target_code const       *code_ptx,
    size_t                texture_index,
    std::vector<Texture> &textures)
{
    Texture_data const *tex = code_ptx->get_texture(texture_index);
    if (!tex->is_valid()) {
        fprintf(stderr, "Error: Requested texture is invalid\n");
        return false;
    }

    unsigned width = tex->get_width(), height = tex->get_height();

    if (tex->get_shape() == mi::mdl::IType_texture::TS_BSDF_DATA) {
        unsigned char const *bsdf_data = tex->get_bsdf_data();
        if (bsdf_data == nullptr) {
            fprintf(stderr, "Error: bsdf data missing for requested texture\n");
            return false;
        }

        unsigned depth = tex->get_depth();

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));

        cudaArray_t device_tex_data;
        cudaExtent extent = make_cudaExtent(width, height, depth);
        check_cuda_success(cudaMalloc3DArray(&device_tex_data, &channel_desc, extent));

        cudaMemcpy3DParms copy_params = { 0 };
        copy_params.srcPtr = make_cudaPitchedPtr(
            const_cast<unsigned char *>(bsdf_data), width * sizeof(float), width, height);
        copy_params.dstArray = device_tex_data;
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyHostToDevice;
        check_cuda_success(cudaMemcpy3D(&copy_params));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_tex_data;

        m_all_texture_arrays.push_back(device_tex_data);

        // Create filtered texture object
        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0]   = cudaAddressModeClamp;
        tex_desc.addressMode[1]   = cudaAddressModeClamp;
        tex_desc.addressMode[2]   = cudaAddressModeClamp;
        tex_desc.filterMode       = cudaFilterModeLinear;
        tex_desc.readMode         = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;

        cudaTextureObject_t tex_obj = 0;
        check_cuda_success(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

        // Create unfiltered texture object and use a black border for access outside of the texture
        cudaTextureObject_t tex_obj_unfilt = 0;
        tex_desc.addressMode[0]   = cudaAddressModeBorder;
        tex_desc.addressMode[1]   = cudaAddressModeBorder;
        tex_desc.addressMode[2]   = cudaAddressModeBorder;
        tex_desc.filterMode       = cudaFilterModePoint;

        check_cuda_success(cudaCreateTextureObject(
            &tex_obj_unfilt, &res_desc, &tex_desc, nullptr));

        // Store texture infos in result vector
        textures.push_back(Texture(
            tex_obj,
            tex_obj_unfilt,
            make_uint3(tex->get_width(), tex->get_height(), tex->get_depth())));
        m_all_textures.push_back(textures.back());
        return true;
    }


    // For simplicity, the texture access functions are only implemented for float4 and gamma
    // is pre-applied here (all images are converted to linear space).

    std::vector<float> data(4 * width * height);
    std::shared_ptr<OIIO::ImageInput> image(tex->get_image());
    mi::Sint32 bytes_per_row = 4 * width * sizeof(float);
    bool success = image->read_image(
        /*subimage*/ 0,
        /*miplevel*/ 0,
        /*chbegin*/ 0,
        /*chend*/ 4,
        OIIO::TypeDesc::FLOAT,
        data.data() + (height-1) * 4 * width,
        /*xstride*/ 4 * sizeof(float),
        /*ystride*/ -bytes_per_row,
        /*zstride*/ OIIO::AutoStride);
    if (!success)
        return false;

    if (image->spec().nchannels <= 3)
        for (size_t i = 0, n = data.size(); i < n; i += 4)
            data[i+3] = 1.0f;

    // Convert to linear color space if necessary
    if (tex->get_gamma_mode() != mi::mdl::IValue_texture::gamma_linear) {
        for (size_t i = 0, n = data.size(); i < n; i += 4) {
            // Only adjust r, g and b, not alpha
            data[i  ] = std::pow(data[i  ], 2.2f);
            data[i+1] = std::pow(data[i+1], 2.2f);
            data[i+2] = std::pow(data[i+2], 2.2f);
        }
    }

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));

    // Copy image data to GPU array depending on texture shape (currently only 2D textures)
    if (tex->get_shape() != mi::mdl::IType_texture::TS_2D) {
        std::cerr << "Currently only 2D textures are supported!" << std::endl;
        return false;
    } else if (m_enable_derivatives) {
        // mipmapped textures use CUDA mipmapped arrays
        unsigned num_levels = 1 + unsigned(std::log2f(float(std::min(width, height))));
        cudaExtent extent = make_cudaExtent(width, height, 0);
        cudaMipmappedArray_t device_tex_miparray;
        check_cuda_success(cudaMallocMipmappedArray(
            &device_tex_miparray, &channel_desc, extent, num_levels));

        unsigned cur_width = width, cur_height = height;
        float4 *prev_data = nullptr;

        // create all mipmap levels and copy them to the CUDA arrays in the mipmapped array
        for (unsigned level = 0; level < num_levels; ++level) {
            float4 *cur_data;
            if (level == 0)
                cur_data = reinterpret_cast<float4 *>(data.data());
            else
                cur_data = gen_mip_level(prev_data, cur_width, cur_height);

            cudaArray_t device_cur_level_array;
            cudaGetMipmappedArrayLevel(&device_cur_level_array, device_tex_miparray, level);
            check_cuda_success(cudaMemcpy2DToArray(device_cur_level_array, 0, 0, cur_data,
                cur_width * sizeof(float4), cur_width * sizeof(float4), cur_height,
                cudaMemcpyHostToDevice));

            if (level >= 2)
                free(prev_data);
            prev_data = cur_data;
        }
        if (num_levels >= 2)
            free(prev_data);

        res_desc.resType = cudaResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = device_tex_miparray;

        m_all_texture_mipmapped_arrays.push_back(device_tex_miparray);
    } else {
        // 2D texture objects use CUDA arrays
        cudaArray_t device_tex_data;
        check_cuda_success(cudaMallocArray(&device_tex_data, &channel_desc, width, height));

        check_cuda_success(cudaMemcpy2DToArray(device_tex_data, 0, 0, data.data(),
            width * sizeof(float4), width * sizeof(float4), height,
            cudaMemcpyHostToDevice));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_tex_data;

        m_all_texture_arrays.push_back(device_tex_data);
    }

    // For cube maps we need clamped address mode to avoid artifacts in the corners
    cudaTextureAddressMode addr_mode =
        tex->get_shape() == mi::mdl::IType_texture::TS_CUBE
        ? cudaAddressModeClamp : cudaAddressModeWrap;

    // Create filtered texture object
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0]   = addr_mode;
    tex_desc.addressMode[1]   = addr_mode;
    tex_desc.addressMode[2]   = addr_mode;
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    if (res_desc.resType == cudaResourceTypeMipmappedArray) {
        tex_desc.mipmapFilterMode = cudaFilterModeLinear;
        tex_desc.maxAnisotropy = 16;
        tex_desc.minMipmapLevelClamp = 0.f;
        tex_desc.maxMipmapLevelClamp = 1000.f;  // default value in OpenGL
    }

    cudaTextureObject_t tex_obj = 0;
    check_cuda_success(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    // Create unfiltered texture object if necessary (cube textures have no texel functions)
    cudaTextureObject_t tex_obj_unfilt = 0;
    if (tex->get_shape() != mi::mdl::IType_texture::TS_CUBE) {
        // Use a black border for access outside of the texture
        tex_desc.addressMode[0]   = cudaAddressModeBorder;
        tex_desc.addressMode[1]   = cudaAddressModeBorder;
        tex_desc.addressMode[2]   = cudaAddressModeBorder;
        tex_desc.filterMode       = cudaFilterModePoint;

        check_cuda_success(cudaCreateTextureObject(
            &tex_obj_unfilt, &res_desc, &tex_desc, nullptr));
    }

    // Store texture infos in result vector
    textures.push_back(Texture(
        tex_obj,
        tex_obj_unfilt,
        make_uint3(tex->get_width(), tex->get_height(), tex->get_depth())));
    m_all_textures.push_back(textures.back());

    return true;
}

namespace
{
    // Build the CDFs / albedo / volume texture for one part (reflection or transmission)
    // of an MDL Core Bsdf_measurement_data and fill the corresponding fields of mbsdf_cuda.
    //
    // The math mirrors mdl_sdk/shared/example_cuda_shared.h::prepare_mbsdfs_part().
    bool prepare_core_mbsdf_part(
        unsigned                          part_idx,
        Mbsdf&                            mbsdf_cuda,
        const Bsdf_measurement_data&      bsdf_measurement,
        std::vector<cudaArray_t>&         tracked_arrays)
    {
        Bsdf_measurement_data::Part const part_enum = part_idx == 0
            ? Bsdf_measurement_data::PART_REFLECTION
            : Bsdf_measurement_data::PART_TRANSMISSION;

        if (!bsdf_measurement.has_part(part_enum))
            return true; // nothing to do

        Bsdf_measurement_data::Part_data const& dataset = bsdf_measurement.get_part(part_enum);

        // get dimensions
        uint2 res = make_uint2(dataset.resolution_theta, dataset.resolution_phi);
        unsigned num_channels =
            dataset.type == Bsdf_measurement_data::BDT_SCALAR ? 1u : 3u;
        mbsdf_cuda.add(part_idx, res, num_channels);

        // get data
        // layout: {1,3} * (index_theta_in * (res_phi * res_theta) +
        //                  index_theta_out * res_phi + index_phi)
        const float* src_data = dataset.data.data();

        // ----------------------------------------------------------------------------------------
        // prepare importance sampling data:
        // - for theta_in we will be able to perform a two stage CDF, first to select theta_out,
        //   and second to select phi_out
        // - maximum component is used to "probability" in case of colored measurements

        // CDF of the probability to select a certain theta_out for a given theta_in
        const unsigned int cdf_theta_size = res.x * res.x;

        // for each of theta_in x theta_out combination, a CDF of the probabilities to select a
        // a certain theta_out is stored
        const unsigned sample_data_size = cdf_theta_size + cdf_theta_size * res.y;
        std::vector<float> sample_data(sample_data_size);

        std::vector<float> albedo_data(res.x); // albedo for sampling refl. and transm.

        float* sample_data_theta = sample_data.data();                // first (theta) CDF
        float* sample_data_phi   = sample_data.data() + cdf_theta_size; // second (phi) CDFs

        const float s_theta = float(M_PI * 0.5) / float(res.x);  // step size
        const float s_phi   = float(M_PI)       / float(res.y);  // step size

        float max_albedo = 0.0f;
        for (unsigned int t_in = 0; t_in < res.x; ++t_in)
        {
            float sum_theta = 0.0f;
            float sintheta0_sqd = 0.0f;
            for (unsigned int t_out = 0; t_out < res.x; ++t_out)
            {
                const float sintheta1 = sinf(float(t_out + 1) * s_theta);
                const float sintheta1_sqd = sintheta1 * sintheta1;

                // BSDFs are symmetric: f(w_in, w_out) = f(w_out, w_in)
                // take the average of both measurements

                // area of two the surface elements (the ones we are averaging)
                const float mu = (sintheta1_sqd - sintheta0_sqd) * s_phi * 0.5f;
                sintheta0_sqd = sintheta1_sqd;

                // offset for both the thetas into the measurement data (select row in volume)
                const unsigned int offset_phi  = (t_in * res.x + t_out) * res.y;
                const unsigned int offset_phi2 = (t_out * res.x + t_in) * res.y;

                // build CDF for phi
                float sum_phi = 0.0f;
                for (unsigned int p_out = 0; p_out < res.y; ++p_out)
                {
                    const unsigned int idx  = offset_phi  + p_out;
                    const unsigned int idx2 = offset_phi2 + p_out;

                    float value = 0.0f;
                    if (num_channels == 3)
                    {
                        value = fmaxf(fmaxf(src_data[3 * idx  + 0], src_data[3 * idx  + 1]),
                                      fmaxf(src_data[3 * idx  + 2], 0.0f))
                              + fmaxf(fmaxf(src_data[3 * idx2 + 0], src_data[3 * idx2 + 1]),
                                      fmaxf(src_data[3 * idx2 + 2], 0.0f));
                    }
                    else /* num_channels == 1 */
                    {
                        value = fmaxf(src_data[idx], 0.0f) + fmaxf(src_data[idx2], 0.0f);
                    }

                    sum_phi += value * mu;
                    sample_data_phi[idx] = sum_phi;
                }

                // normalize CDF for phi
                for (unsigned int p_out = 0; p_out < res.y; ++p_out)
                {
                    const unsigned int idx = offset_phi + p_out;
                    sample_data_phi[idx] = (sum_phi > 0.0f) ? (sample_data_phi[idx] / sum_phi)
                                                           : 0.0f;
                }

                // build CDF for theta
                sum_theta += sum_phi;
                sample_data_theta[t_in * res.x + t_out] = sum_theta;
            }

            if (sum_theta > max_albedo)
                max_albedo = sum_theta;

            albedo_data[t_in] = sum_theta;

            // normalize CDF for theta
            for (unsigned int t_out = 0; t_out < res.x; ++t_out)
            {
                const unsigned int idx = t_in * res.x + t_out;
                sample_data_theta[idx] = (sum_theta > 0.0f) ? (sample_data_theta[idx] / sum_theta)
                                                            : 0.0f;
            }
        }

        // copy entire CDF data buffer to GPU
        CUdeviceptr sample_obj = 0;
        check_cuda_success(cuMemAlloc(&sample_obj, sample_data_size * sizeof(float)));
        check_cuda_success(cuMemcpyHtoD(
            sample_obj, sample_data.data(), sample_data_size * sizeof(float)));

        CUdeviceptr albedo_obj = 0;
        check_cuda_success(cuMemAlloc(&albedo_obj, res.x * sizeof(float)));
        check_cuda_success(cuMemcpyHtoD(albedo_obj, albedo_data.data(), res.x * sizeof(float)));

        mbsdf_cuda.sample_data[part_idx] = reinterpret_cast<float*>(sample_obj);
        mbsdf_cuda.albedo_data[part_idx] = reinterpret_cast<float*>(albedo_obj);
        mbsdf_cuda.max_albedo[part_idx]  = max_albedo;

        // ----------------------------------------------------------------------------------------
        // prepare evaluation data:
        // - simply store the measured data in a volume texture
        // - in case of color data, we store each sample in a vector4 to get texture support
        unsigned lookup_channels = (num_channels == 3) ? 4u : 1u;

        // make lookup data symmetric
        std::vector<float> lookup_data(size_t(lookup_channels) * res.y * res.x * res.x);
        for (unsigned int t_in = 0; t_in < res.x; ++t_in)
        {
            for (unsigned int t_out = 0; t_out < res.x; ++t_out)
            {
                const unsigned int offset_phi  = (t_in * res.x + t_out) * res.y;
                const unsigned int offset_phi2 = (t_out * res.x + t_in) * res.y;
                for (unsigned int p_out = 0; p_out < res.y; ++p_out)
                {
                    const unsigned int idx  = offset_phi  + p_out;
                    const unsigned int idx2 = offset_phi2 + p_out;

                    if (num_channels == 3)
                    {
                        lookup_data[4*idx+0] = (src_data[3*idx+0] + src_data[3*idx2+0]) * 0.5f;
                        lookup_data[4*idx+1] = (src_data[3*idx+1] + src_data[3*idx2+1]) * 0.5f;
                        lookup_data[4*idx+2] = (src_data[3*idx+2] + src_data[3*idx2+2]) * 0.5f;
                        lookup_data[4*idx+3] = 1.0f;
                    }
                    else
                    {
                        lookup_data[idx] = (src_data[idx] + src_data[idx2]) * 0.5f;
                    }
                }
            }
        }

        // Copy data to GPU array
        cudaArray_t device_mbsdf_data;
        cudaChannelFormatDesc channel_desc = (num_channels == 3)
            ? cudaCreateChannelDesc<float4>()    // float3 is not supported
            : cudaCreateChannelDesc<float>();

        // Allocate a 3D array on the GPU (phi_delta x theta_out x theta_in)
        cudaExtent extent = make_cudaExtent(res.y, res.x, res.x);
        check_cuda_success(cudaMalloc3DArray(&device_mbsdf_data, &channel_desc, extent, 0));

        // prepare and copy
        cudaMemcpy3DParms copy_params;
        memset(&copy_params, 0, sizeof(copy_params));
        copy_params.srcPtr = make_cudaPitchedPtr(
            (void*)(lookup_data.data()),                            // base pointer
            res.y * lookup_channels * sizeof(float),                // row pitch
            res.y,                                                  // width of slice
            res.x);                                                 // height of slice
        copy_params.dstArray = device_mbsdf_data;
        copy_params.extent   = extent;
        copy_params.kind     = cudaMemcpyHostToDevice;
        check_cuda_success(cudaMemcpy3D(&copy_params));

        tracked_arrays.push_back(device_mbsdf_data);

        cudaResourceDesc texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = device_mbsdf_data;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = 1;
        texDescr.filterMode       = cudaFilterModeLinear;
        texDescr.addressMode[0]   = cudaAddressModeClamp;
        texDescr.addressMode[1]   = cudaAddressModeClamp;
        texDescr.addressMode[2]   = cudaAddressModeClamp;
        texDescr.readMode         = cudaReadModeElementType;

        cudaTextureObject_t eval_tex_obj;
        check_cuda_success(cudaCreateTextureObject(&eval_tex_obj, &texRes, &texDescr, nullptr));
        mbsdf_cuda.eval_data[part_idx] = eval_tex_obj;

        return true;
    }
}

// Prepare the BSDF measurement identified by mbsdf_index for use by the BSDF measurement
// access functions on the GPU.
bool Material_gpu_context::prepare_mbsdf(
    Target_code const   *code_ptx,
    size_t               mbsdf_index,
    std::vector<Mbsdf>  &mbsdfs)
{
    Bsdf_measurement_data const *bm = code_ptx->get_bsdf_measurement(mbsdf_index);
    if (!bm) {
        fprintf(stderr, "Error: Requested BSDF measurement is missing\n");
        return false;
    }

    Mbsdf mbsdf_cuda;

    // skip invalid measurements (e.g. the slot 0 placeholder or files that could not be loaded)
    if (bm->is_valid()) {
        if (!prepare_core_mbsdf_part(0, mbsdf_cuda, *bm, m_all_texture_arrays))
            return false;
        if (!prepare_core_mbsdf_part(1, mbsdf_cuda, *bm, m_all_texture_arrays))
            return false;
    }

    mbsdfs.push_back(mbsdf_cuda);
    m_all_mbsdfs.push_back(mbsdfs.back());
    return true;
}

// Prepare the needed target code data of the given target code.
bool Material_gpu_context::prepare_target_code_data(Target_code const *target_code)
{
    // Target code data list may not have been retrieved already
    check_success(!m_device_target_code_data_list);

    // Handle the read-only data segments if necessary.
    // They are only created, if the "enable_ro_segment" backend option was set to "on".
    CUdeviceptr device_ro_data = 0;
    size_t ro_data_size = 0;
    if (char const *ro_data = target_code->get_ro_data_segment(ro_data_size)) {
        device_ro_data = gpu_mem_dup(ro_data, ro_data_size);
    }

    // Copy textures to GPU if the code has more than just the invalid texture
    CUdeviceptr device_textures = 0;
    size_t num_textures = target_code->get_texture_count();
    if (num_textures > 1) {
        std::vector<Texture> textures;

        // Loop over all textures skipping the first texture,
        // which is always the invalid texture
        for (size_t i = 1; i < num_textures; ++i) {
            if (!prepare_texture(target_code, i, textures))
                return false;
        }

        // Copy texture list to GPU
        device_textures = gpu_mem_dup(textures);
    }

    // Copy BSDF measurements to GPU if the code has more than just the invalid measurement
    CUdeviceptr device_mbsdfs = 0;
    size_t num_mbsdfs = target_code->get_bsdf_measurement_count();
    if (num_mbsdfs > 1) {
        std::vector<Mbsdf> mbsdfs;

        // Loop over all measurements skipping the first one,
        // which is always the invalid measurement
        for (size_t i = 1; i < num_mbsdfs; ++i) {
            if (!prepare_mbsdf(target_code, i, mbsdfs))
                return false;
        }

        // Copy mbsdf list to GPU
        device_mbsdfs = gpu_mem_dup(mbsdfs);
    }

    m_target_code_data_list.push_back(
        Target_code_data(num_textures, device_textures,
                         num_mbsdfs,   device_mbsdfs,
                         device_ro_data));

    for (size_t i = 0, num = target_code->get_argument_block_count(); i < num; ++i) {
        Argument_block const *arg_block = target_code->get_argument_block(i);
        CUdeviceptr dev_block = gpu_mem_dup(arg_block->get_data(), arg_block->get_size());
        m_target_argument_block_list.push_back(dev_block);
        m_own_arg_blocks.push_back(Argument_block(*arg_block));
        m_arg_block_layouts.push_back(
            mi::base::make_handle(target_code->get_argument_block_layout(i)));
    }

    for (size_t arg_block_index : target_code->get_argument_block_indices()) {
        m_bsdf_arg_block_indices.push_back(arg_block_index);
    }

    return true;
}

// Update the i'th target argument block on the device with the data from the corresponding
// block returned by get_argument_block().
void Material_gpu_context::update_device_argument_block(size_t i)
{
    CUdeviceptr device_ptr = get_device_target_argument_block(i);
    if (device_ptr == 0) return;

    Argument_block *arg_block = get_argument_block(i);
    check_cuda_success(cuMemcpyHtoD(
        device_ptr, arg_block->get_data(), arg_block->get_size()));
}

//------------------------------------------------------------------------------
//
// Material execution code
//
//------------------------------------------------------------------------------

// Helper function to create PTX source code for a non-empty 32-bit value array.
void print_array_u32(
    std::string &str, std::string const &name, unsigned count, std::string const &content)
{
    str += ".visible .const .align 4 .u32 " + name + "[";
    if (count == 0) {
        // PTX does not allow empty arrays, so use a dummy entry
        str += "1] = { 0 };\n";
    } else {
        str += to_string(count) + "] = { " + content + " };\n";
    }
}

// Helper function to create PTX source code for a non-empty function pointer array.
void print_array_func(
    std::string &str, std::string const &name, unsigned count, std::string const &content)
{
    str += ".visible .const .align 8 .u64 " + name + "[";
    if (count == 0) {
        // PTX does not allow empty arrays, so use a dummy entry
        str += "1] = { dummy_func };\n";
    } else {
        str += to_string(count) + "] = { " + content + " };\n";
    }
}

// Generate PTX array containing the references to all generated functions.
std::string generate_func_array_ptx(
    std::vector<std::unique_ptr<Target_code>> const &target_codes)
{
    // Create PTX header and mdl_expr_functions_count constant
    std::string src =
        ".version 4.0\n"
        ".target sm_20\n"
        ".address_size 64\n";

    // Workaround needed to let CUDA linker resolve the function pointers in the arrays.
    // Also used for "empty" function arrays.
    src += ".func dummy_func() { ret; }\n";

    std::string tc_offsets;
    std::string function_names;
    std::string tc_indices;
    std::string ab_indices;
    unsigned f_count = 0;

    // Iterate over all target codes
    for (size_t tc_index = 0, num = target_codes.size(); tc_index < num; ++tc_index)
    {
        Target_code const *target_code = target_codes[tc_index].get();

        // in case of multiple target codes, we need to address the functions by a pair of
        // target_code_index and function_index.
        // the elements in the resulting function array can then be index by offset + func_index.
        if (!tc_offsets.empty())
            tc_offsets += ", ";
        tc_offsets += to_string(f_count);

        // Collect all names and prototypes of callable functions within the current target code
        for (size_t func_index = 0, func_count = target_code->get_callable_function_count();
             func_index < func_count; ++func_index)
        {
            // add to function list
            if (!tc_indices.empty())
            {
                tc_indices += ", ";
                function_names += ", ";
                ab_indices += ", ";
            }

            // target code index in case of multiple link units
            tc_indices += to_string(tc_index);

            // name of the function
            function_names += target_code->get_callable_function(func_index);

            // Get argument block index and translate to 1 based list index (-> 0 = not-used)
            mi::Size ab_index = target_code->get_callable_function_argument_block_index(func_index);
            ab_indices += to_string(ab_index == mi::Size(~0) ? 0 : (ab_index + 1));
            f_count++;

            // Add prototype declaration
            src += target_code->get_callable_function_prototype(
                func_index, mi::mdl::IGenerated_code_executable::PL_PTX);
            src += '\n';
        }
    }

    // infos per target code (link unit)
    src += std::string(".visible .const .align 4 .u32 mdl_target_code_count = ")
        + to_string(target_codes.size()) + ";\n";
    print_array_u32(
        src, std::string("mdl_target_code_offsets"), unsigned(target_codes.size()), tc_offsets);

    // infos per function
    src += std::string(".visible .const .align 4 .u32 mdl_functions_count = ")
        + to_string(f_count) + ";\n";
    print_array_func(src, std::string("mdl_functions"), f_count, function_names);
    print_array_u32(src, std::string("mdl_arg_block_indices"), f_count, ab_indices);
    print_array_u32(src, std::string("mdl_target_code_indices"), f_count, tc_indices);

    return src;
}

// Build a linked CUDA kernel containing our kernel and all the generated code, making it
// available to the kernel via an added "mdl_expr_functions" array.
CUmodule build_linked_kernel(
    std::vector<std::unique_ptr<Target_code>> const &target_codes,
    const char *ptx_file,
    const char *kernel_function_name,
    CUfunction *out_kernel_function)
{
    // Generate PTX array containing the references to all generated functions.
    // The linker will resolve them to addresses.

    std::string ptx_func_array_src = generate_func_array_ptx(target_codes);
#ifdef DUMP_PTX
    std::cout << "Dumping CUDA PTX code for the \"mdl_expr_functions\" array:\n\n"
        << ptx_func_array_src << std::endl;
#endif

    // Link all generated code, our generated PTX array and our kernel together

    CUlinkState   cuda_link_state;
    CUmodule      cuda_module;
    void         *linked_cubin;
    size_t        linked_cubin_size;
    char          error_log[8192], info_log[8192];
    CUjit_option  options[4];
    void         *optionVals[4];

    // Setup the linker

    // Pass a buffer for info messages
    options[0] = CU_JIT_INFO_LOG_BUFFER;
    optionVals[0] = info_log;
    // Pass the size of the info buffer
    options[1] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    optionVals[1] = reinterpret_cast<void *>(uintptr_t(sizeof(info_log)));
    // Pass a buffer for error messages
    options[2] = CU_JIT_ERROR_LOG_BUFFER;
    optionVals[2] = error_log;
    // Pass the size of the error buffer
    options[3] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    optionVals[3] = reinterpret_cast<void *>(uintptr_t(sizeof(error_log)));

    check_cuda_success(cuLinkCreate(4, options, optionVals, &cuda_link_state));

    CUresult link_result = CUDA_SUCCESS;
    do {
        // Add all code generated by the MDL PTX backend
        for (size_t i = 0, num_target_codes = target_codes.size(); i < num_target_codes; ++i) {
            link_result = cuLinkAddData(
                cuda_link_state, CU_JIT_INPUT_PTX,
                const_cast<char *>(target_codes[i]->get_src_code()),
                target_codes[i]->get_src_code_size(),
                nullptr, 0, nullptr, nullptr);
            if (link_result != CUDA_SUCCESS) break;
        }
        if (link_result != CUDA_SUCCESS) break;

        // Add the "mdl_expr_functions" array PTX module
        link_result = cuLinkAddData(
            cuda_link_state, CU_JIT_INPUT_PTX,
            const_cast<char *>(ptx_func_array_src.c_str()),
            ptx_func_array_src.size(),
            nullptr, 0, nullptr, nullptr);
        if (link_result != CUDA_SUCCESS) break;

        // Add our kernel
        link_result = cuLinkAddFile(
            cuda_link_state, CU_JIT_INPUT_PTX,
            ptx_file, 0, nullptr, nullptr);
        if (link_result != CUDA_SUCCESS) break;

        // Link everything to a cubin
        link_result = cuLinkComplete(cuda_link_state, &linked_cubin, &linked_cubin_size);
    } while (false);
    if (link_result != CUDA_SUCCESS) {
        std::cerr << "PTX linker error:\n" << error_log << std::endl;
        check_cuda_success(link_result);
    }

    std::cout << "CUDA link completed. Linker output:\n" << info_log << std::endl;

    // Load the result and get the entrypoint of our kernel
    check_cuda_success(cuModuleLoadData(&cuda_module, linked_cubin));
    check_cuda_success(cuModuleGetFunction(
        out_kernel_function, cuda_module, kernel_function_name));

    int regs = 0;
    check_cuda_success(
        cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, *out_kernel_function));
    int lmem = 0;
    check_cuda_success(
        cuFuncGetAttribute(&lmem, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, *out_kernel_function));
    std::cout << "kernel uses " << regs << " registers and " << lmem << " lmem" << std::endl;

    // Cleanup
    check_cuda_success(cuLinkDestroy(cuda_link_state));

    return cuda_module;
}
#endif // EXAMPLE_CUDA_SHARED_H
