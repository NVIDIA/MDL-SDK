/******************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

// Code shared by CUDA MDL SDK examples

#ifndef EXAMPLE_CUDA_SHARED_H
#define EXAMPLE_CUDA_SHARED_H

#include <string>
#include <vector>
#include <sstream>
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#include "example_shared.h"
#include "compiled_material_traverser_base.h"

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
    explicit Texture(cudaTextureObject_t  filtered_object,
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
struct Mbsdf
{
    explicit Mbsdf()
    {
        for (unsigned i = 0; i < 2; ++i) {
            has_data[i] = 0u;
            eval_data[i] = 0;
            sample_data[i] = 0;
            albedo_data[i] = 0;
            this->max_albedo[i] = 0.0f;
            angular_resolution[i] = make_uint2(0u, 0u);
            inv_angular_resolution[i] = make_float2(0.0f, 0.0f);
            num_channels[i] = 0;
        }
    }

    void Add(mi::neuraylib::Mbsdf_part part,
             const uint2& angular_resolution, 
             unsigned num_channels)
    {
        unsigned part_idx = static_cast<unsigned>(part);

        this->has_data[part_idx] = 1u;
        this->angular_resolution[part_idx] = angular_resolution;
        this->inv_angular_resolution[part_idx] = make_float2(1.0f / float(angular_resolution.x),
                                                             1.0f / float(angular_resolution.y));
        this->num_channels[part_idx] = num_channels;
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


// Structure representing a Light Profile
struct Lightprofile
{
    explicit Lightprofile(
        uint2               angular_resolution = make_uint2(0, 0),
        float2              theta_phi_start = make_float2(0.0f, 0.0f),
        float2              theta_phi_delta = make_float2(0.0f, 0.0f),
        float               candela_multiplier = 0.0f,
        float               total_power = 0.0f,
        cudaTextureObject_t eval_data = 0,
        float               *cdf_data = nullptr)
    : angular_resolution(angular_resolution)
    , theta_phi_start(theta_phi_start)
    , theta_phi_delta(theta_phi_delta)
        , theta_phi_inv_delta(make_float2(0.0f, 0.0f))
    , candela_multiplier(candela_multiplier)
    , total_power(total_power)
    , eval_data(eval_data)
    , cdf_data(cdf_data)
    {
        theta_phi_inv_delta.x = theta_phi_delta.x ? (1.f / theta_phi_delta.x) : 0.f;
        theta_phi_inv_delta.y = theta_phi_delta.y ? (1.f / theta_phi_delta.y) : 0.f;
    }

    uint2           angular_resolution;     // angular resolution of the grid
    float2          theta_phi_start;        // start of the grid
    float2          theta_phi_delta;        // angular step size
    float2          theta_phi_inv_delta;    // inverse step size
    float           candela_multiplier;     // factor to rescale the normalized data
    float           total_power;

    cudaTextureObject_t eval_data;          // normalized data sampled on grid
    float*              cdf_data;           // CDFs for sampling a light profile
};

// Structure representing the resources used by the generated code of a target code.
struct Target_code_data
{
    Target_code_data(
        size_t num_textures,
        CUdeviceptr textures,
        size_t num_mbsdfs,
        CUdeviceptr mbsdfs,
        size_t num_lightprofiles,
        CUdeviceptr lightprofiles,
                     CUdeviceptr ro_data_segment)
        : num_textures(num_textures)
        , textures(textures)
        , num_mbsdfs(num_mbsdfs)
        , mbsdfs(mbsdfs)
        , num_lightprofiles(num_lightprofiles)
        , lightprofiles(lightprofiles)
        , ro_data_segment(ro_data_segment)
    {}

    size_t      num_textures;           // number of elements in the textures field
    CUdeviceptr textures;               // a device pointer to a list of Texture objects, if used

    size_t      num_mbsdfs;             // number of elements in the mbsdfs field
    CUdeviceptr mbsdfs;                 // a device pointer to a list of mbsdfs objects, if used

    size_t      num_lightprofiles;     // number of elements in the lightprofiles field
    CUdeviceptr lightprofiles;         // a device pointer to a list of mbsdfs objects, if used

    CUdeviceptr ro_data_segment;        // a device pointer to the read-only data segment, if used
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

// Collects the handles in a compiled material
class Handle_collector : public Compiled_material_traverser_base
{
public:
    // add all handle appearing in the provided material to the collectors handle list.
    explicit Handle_collector(
        mi::neuraylib::ITransaction* transaction,
        const mi::neuraylib::ICompiled_material* material)
    : Compiled_material_traverser_base()
    {
        traverse(material, transaction);
    }

    // get the collected handles.
    const std::vector<std::string>& get_handles() const { return m_handles; }

private:
    // Called when the traversal reaches a new element.
    void visit_begin(const mi::neuraylib::ICompiled_material* material,
                     const Compiled_material_traverser_base::Traversal_element& element,
                     void* context) override
    {
        // look for direct calls
        if (!element.expression ||
            element.expression->get_kind() != mi::neuraylib::IExpression::EK_DIRECT_CALL)
            return;

        // check if it is a distribution function
        auto transaction = static_cast<mi::neuraylib::ITransaction*>(context);

        const mi::base::Handle<const mi::neuraylib::IExpression_direct_call> expr_dcall(
            element.expression->get_interface<const mi::neuraylib::IExpression_direct_call
            >());
        const mi::base::Handle<const mi::neuraylib::IExpression_list> args(
            expr_dcall->get_arguments());
        const mi::base::Handle<const mi::neuraylib::IFunction_definition> func_def(
            transaction->access<mi::neuraylib::IFunction_definition>(
            expr_dcall->get_definition()));
        const mi::neuraylib::IFunction_definition::Semantics semantic = func_def->
            get_semantic();

        if (semantic < mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_FIRST
            || semantic > mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_LAST)
            return;

        // check if the last argument is a handle
        const mi::base::Handle<const mi::neuraylib::IExpression_list> arguments(
            expr_dcall->get_arguments());

        mi::Size arg_count = arguments->get_size();
        const char* name = arguments->get_name(arg_count - 1);
        if (strcmp(name, "handle") != 0)
            return;

        // get the handle value
        mi::base::Handle<const mi::neuraylib::IExpression> expr(
            arguments->get_expression(arg_count - 1));

        if (expr->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
            return; // is an error if 'handle' is a reserved parameter name

        const mi::base::Handle<const mi::neuraylib::IExpression_constant> expr_const(
            expr->get_interface<const mi::neuraylib::IExpression_constant>());

        const mi::base::Handle<const mi::neuraylib::IValue> value(expr_const->get_value());
        if (value->get_kind() != mi::neuraylib::IValue::VK_STRING)
            return;

        const mi::base::Handle<const mi::neuraylib::IValue_string> handle(
            value->get_interface<const mi::neuraylib::IValue_string>());

        std::string handle_value = handle->get_value() ? std::string(handle->get_value()) : "";

        if (std::find(m_handles.begin(), m_handles.end(), handle_value) == m_handles.end())
            m_handles.push_back(handle_value);
    }

    std::vector<std::string> m_handles;
};


//------------------------------------------------------------------------------
//
// CUDA helper functions
//
//------------------------------------------------------------------------------

// Helper macro. Checks whether the expression is cudaSuccess and if not prints a message and
// resets the device and exits.

#ifdef ENABLE_DEPRECATED_UTILIY_FUNCTIONS

#define check_cuda_success(expr) \
    do { \
        int err = (expr); \
        if (err != 0) { \
            fprintf(stderr, "CUDA error %d in file %s, line %u: \"%s\".\n", \
                err, __FILE__, __LINE__, #expr); \
            keep_console_open(); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while (false)

#else

#define check_cuda_success(expr) \
    do { \
        int err = (expr); \
        if (err != 0) { \
            cudaDeviceReset(); \
            exit_failure( "Error in file %s, line %u: \"%s\".\n", __FILE__, __LINE__, #expr); \
        } \
    } while (false)

#endif

// Initialize CUDA.
CUcontext init_cuda(
    int ordinal
#ifdef OPENGL_INTEROP
    , const bool opengl_interop
#endif
    )
{
    CUdevice cu_device;
    CUcontext cu_context;

    check_cuda_success(cuInit(0));
#if defined(OPENGL_INTEROP) && !defined(__APPLE__)
    if (opengl_interop) {
        // Use first device used by OpenGL context
        unsigned int num_cu_devices;
        check_cuda_success(cuGLGetDevices(&num_cu_devices, &cu_device, 1, CU_GL_DEVICE_LIST_ALL));
    }
    else
#endif
    {
        // Use given device
        check_cuda_success(cuDeviceGet(&cu_device, ordinal));
    }

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

/// Helper struct to delete CUDA (and related) resources.
template<typename T> struct Resource_deleter {
    /*compile error*/
};

template<> struct Resource_deleter<cudaArray_t> {
    void operator()(cudaArray_t res) { check_cuda_success(cudaFreeArray(res)); }
};

template<> struct Resource_deleter<cudaMipmappedArray_t> {
    void operator()(cudaMipmappedArray_t res) { check_cuda_success(cudaFreeMipmappedArray(res)); }
};

template<> struct Resource_deleter<Texture> {
    void operator()(Texture &res) {
        check_cuda_success(cudaDestroyTextureObject(res.filtered_object));
        check_cuda_success(cudaDestroyTextureObject(res.unfiltered_object));
    }
};

template<> struct Resource_deleter<Mbsdf> {
    void operator()(Mbsdf &res) {
        for (size_t i = 0; i < 2; ++i) {
            if (res.has_data[i] != 0u) {
                check_cuda_success(cudaDestroyTextureObject(res.eval_data[i]));
                check_cuda_success(cuMemFree(reinterpret_cast<CUdeviceptr>(res.sample_data[i])));
                check_cuda_success(cuMemFree(reinterpret_cast<CUdeviceptr>(res.albedo_data[i])));
            }
        }
    }
};

template<> struct Resource_deleter<Lightprofile> {
    void operator()(Lightprofile res) {
        if (res.cdf_data)
            check_cuda_success(cuMemFree((CUdeviceptr)res.cdf_data));
    }
};

template<> struct Resource_deleter<Target_code_data> {
    void operator()(Target_code_data &res) {
        if (res.textures)
            check_cuda_success(cuMemFree(res.textures));
        if (res.ro_data_segment)
            check_cuda_success(cuMemFree(res.ro_data_segment));
    }
};

template<> struct Resource_deleter<CUdeviceptr> {
    void operator()(CUdeviceptr res) {
        if (res != 0)
            check_cuda_success(cuMemFree(res));
    }
};

/// Holds one resource, not copyable.
template<typename T, typename D = Resource_deleter<T> >
struct Resource_handle {
    Resource_handle(T res) : m_res(res) {}

    ~Resource_handle() {
        D deleter;
        deleter(m_res);
    }

    T &get() { return m_res; }

    T const &get() const { return m_res; }

    void set(T res) { m_res = res; }

private:
    // No copy possible.
    Resource_handle(Resource_handle const &);
    Resource_handle &operator=(Resource_handle const &);

private:
    T m_res;
};

/// Hold one container of resources, not copyable.
template<typename T, typename C = std::vector<T>, typename D = Resource_deleter<T> >
struct Resource_container {
    Resource_container() : m_cont() {}

    ~Resource_container() {
        D deleter;
        typedef typename C::iterator I;
        for (I it(m_cont.begin()), end(m_cont.end()); it != end; ++it) {
            T &r = *it;
            deleter(r);
        }
    }

    C &operator*() { return m_cont; }

    C const &operator*() const { return m_cont; }

    C *operator->() { return &m_cont; }

    C const *operator->() const { return &m_cont; }

private:
    // No copy possible.
    Resource_container(Resource_container const &);
    Resource_container &operator=(Resource_container const &);

private:
    C m_cont;
};

// Allocate memory on GPU and copy the given data to the allocated memory.
CUdeviceptr gpu_mem_dup(void const *data, size_t size)
{
    CUdeviceptr device_ptr;
    check_cuda_success(cuMemAlloc(&device_ptr, size));
    check_cuda_success(cuMemcpyHtoD(device_ptr, data, size));
    return device_ptr;
}

// Allocate memory on GPU and copy the given data to the allocated memory.
template <typename T>
CUdeviceptr gpu_mem_dup(Resource_handle<T> const *data, size_t size)
{
    return gpu_mem_dup((void *)data->get(), size);
}

// Allocate memory on GPU and copy the given data to the allocated memory.
template<typename T>
CUdeviceptr gpu_mem_dup(std::vector<T> const &data)
{
    return gpu_mem_dup(&data[0], data.size() * sizeof(T));
}

// Allocate memory on GPU and copy the given data to the allocated memory.
template<typename T, typename C>
CUdeviceptr gpu_mem_dup(Resource_container<T,C> const &cont)
{
    return gpu_mem_dup(*cont);
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
    Material_gpu_context(bool enable_derivatives)
        : m_enable_derivatives(enable_derivatives)
        , m_device_target_code_data_list(0)
        , m_device_target_argument_block_list(0)
    {
        // Use first entry as "not-used" block
        m_target_argument_block_list->push_back(0);
    }

    // Prepare the needed data of the given target code.
    bool prepare_target_code_data(
        mi::neuraylib::ITransaction          *transaction,
        mi::neuraylib::IImage_api            *image_api,
        mi::neuraylib::ITarget_code const    *target_code,
        std::vector<size_t> const            &arg_block_indices);

    // Get a device pointer to the target code data list.
    CUdeviceptr get_device_target_code_data_list();

    // Get a device pointer to the target argument block list.
    CUdeviceptr get_device_target_argument_block_list();

    // Get a device pointer to the i'th target argument block.
    CUdeviceptr get_device_target_argument_block(size_t i)
    {
        // First entry is the "not-used" block, so start at index 1.
        if (i + 1 >= m_target_argument_block_list->size())
            return 0;
        return (*m_target_argument_block_list)[i + 1];
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
    mi::base::Handle<mi::neuraylib::ITarget_argument_block> get_argument_block(size_t i)
    {
        if (i >= m_own_arg_blocks.size())
            return mi::base::Handle<mi::neuraylib::ITarget_argument_block>();
        return m_own_arg_blocks[i];
    }

    // Get the layout of the i'th target argument block.
    mi::base::Handle<mi::neuraylib::ITarget_value_layout const> get_argument_block_layout(size_t i)
    {
        if (i >= m_arg_block_layouts.size())
            return mi::base::Handle<mi::neuraylib::ITarget_value_layout const>();
        return m_arg_block_layouts[i];
    }

    // Update the i'th target argument block on the device with the data from the corresponding
    // block returned by get_argument_block().
    void update_device_argument_block(size_t i);
private:
    // Copy the image data of a canvas to a CUDA array.
    void copy_canvas_to_cuda_array(cudaArray_t device_array, mi::neuraylib::ICanvas const *canvas);

    // Prepare the texture identified by the texture_index for use by the texture access functions
    // on the GPU.
    bool prepare_texture(
        mi::neuraylib::ITransaction       *transaction,
        mi::neuraylib::IImage_api         *image_api,
        mi::neuraylib::ITarget_code const *code_ptx,
        mi::Size                           texture_index,
        std::vector<Texture>              &textures);

    // Prepare the mbsdf identified by the mbsdf_index for use by the bsdf measurement access 
    // functions on the GPU.
    bool prepare_mbsdf(
        mi::neuraylib::ITransaction       *transaction,
        mi::neuraylib::ITarget_code const *code_ptx,
        mi::Size                           mbsdf_index,
        std::vector<Mbsdf>                &mbsdfs);

    // Prepare the mbsdf identified by the mbsdf_index for use by the bsdf measurement access 
    // functions on the GPU.
    bool prepare_lightprofile(
        mi::neuraylib::ITransaction       *transaction,
        mi::neuraylib::ITarget_code const *code_ptx,
        mi::Size                           lightprofile_index,
        std::vector<Lightprofile>        &lightprofiles);

    // If true, mipmaps will be generated for all 2D textures.
    bool m_enable_derivatives;

    // The device pointer of the target code data list.
    Resource_handle<CUdeviceptr> m_device_target_code_data_list;

    // List of all target code data objects owned by this context.
    Resource_container<Target_code_data> m_target_code_data_list;

    // The device pointer of the target argument block list.
    Resource_handle<CUdeviceptr> m_device_target_argument_block_list;

    // List of all target argument blocks owned by this context.
    Resource_container<CUdeviceptr> m_target_argument_block_list;

    // List of all local, writable copies of the target argument blocks.
    std::vector<mi::base::Handle<mi::neuraylib::ITarget_argument_block> > m_own_arg_blocks;

    // List of argument block indices per material BSDF.
    std::vector<size_t> m_bsdf_arg_block_indices;

    // List of all target argument block layouts.
    std::vector<mi::base::Handle<mi::neuraylib::ITarget_value_layout const> > m_arg_block_layouts;

    // List of all Texture objects owned by this context.
    Resource_container<Texture> m_all_textures;

    // List of all MBSDFs objects owned by this context.
    Resource_container<Mbsdf> m_all_mbsdfs;

    // List of all Light profiles objects owned by this context.
    Resource_container<Lightprofile> m_all_lightprofiles;

    // List of all CUDA arrays owned by this context.
    Resource_container<cudaArray_t> m_all_texture_arrays;

    // List of all CUDA mipmapped arrays owned by this context.
    Resource_container<cudaMipmappedArray_t> m_all_texture_mipmapped_arrays;
};

// Get a device pointer to the target code data list.
CUdeviceptr Material_gpu_context::get_device_target_code_data_list()
{
    if (!m_device_target_code_data_list.get())
        m_device_target_code_data_list.set(gpu_mem_dup(m_target_code_data_list));
    return m_device_target_code_data_list.get();
}

// Get a device pointer to the target argument block list.
CUdeviceptr Material_gpu_context::get_device_target_argument_block_list()
{
    if (!m_device_target_argument_block_list.get())
        m_device_target_argument_block_list.set(gpu_mem_dup(m_target_argument_block_list));
    return m_device_target_argument_block_list.get();
}

// Copy the image data of a canvas to a CUDA array.
void Material_gpu_context::copy_canvas_to_cuda_array(
    cudaArray_t device_array,
    mi::neuraylib::ICanvas const *canvas)
{
    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
    mi::Float32 const *data = static_cast<mi::Float32 const *>(tile->get_data());
    check_cuda_success(cudaMemcpy2DToArray(
        device_array, 0, 0, data,
        canvas->get_resolution_x() * sizeof(float) * 4,
        canvas->get_resolution_x() * sizeof(float) * 4,
        canvas->get_resolution_y(),
        cudaMemcpyHostToDevice));
}

// Prepare the texture identified by the texture_index for use by the texture access functions
// on the GPU.
bool Material_gpu_context::prepare_texture(
    mi::neuraylib::ITransaction       *transaction,
    mi::neuraylib::IImage_api         *image_api,
    mi::neuraylib::ITarget_code const *code_ptx,
    mi::Size                           texture_index,
    std::vector<Texture>              &textures)
{
    // Get access to the texture data by the texture database name from the target code.
    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        transaction->access<mi::neuraylib::ITexture>(code_ptx->get_texture(texture_index)));
    mi::base::Handle<const mi::neuraylib::IImage> image(
        transaction->access<mi::neuraylib::IImage>(texture->get_image()));
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas());
    mi::Uint32 tex_width = canvas->get_resolution_x();
    mi::Uint32 tex_height = canvas->get_resolution_y();
    mi::Uint32 tex_layers = canvas->get_layers_size();
    char const *image_type = image->get_type();

    if (image->is_uvtile()) {
        std::cerr << "The example does not support uvtile textures!" << std::endl;
        return false;
    }

    if (canvas->get_tiles_size_x() != 1 || canvas->get_tiles_size_y() != 1) {
        std::cerr << "The example does not support tiled images!" << std::endl;
        return false;
    }

    // For simplicity, the texture access functions are only implemented for float4 and gamma
    // is pre-applied here (all images are converted to linear space).

    // Convert to linear color space if necessary
    if (texture->get_effective_gamma() != 1.0f) {
        // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
        mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
            image_api->convert(canvas.get(), "Color"));
        gamma_canvas->set_gamma(texture->get_effective_gamma());
        image_api->adjust_gamma(gamma_canvas.get(), 1.0f);
        canvas = gamma_canvas;
    } else if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0) {
        // Convert to expected format
        canvas = image_api->convert(canvas.get(), "Color");
    }

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));

    // Copy image data to GPU array depending on texture shape
    mi::neuraylib::ITarget_code::Texture_shape texture_shape =
        code_ptx->get_texture_shape(texture_index);
    if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube ||
        texture_shape == mi::neuraylib::ITarget_code::Texture_shape_3d ||
        texture_shape == mi::neuraylib::ITarget_code::Texture_shape_bsdf_data) {
        // Cubemap and 3D texture objects require 3D CUDA arrays

        if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube &&
            tex_layers != 6) {
            std::cerr << "Invalid number of layers (" << tex_layers
                << "), cubemaps must have 6 layers!" << std::endl;
            return false;
        }

        // Allocate a 3D array on the GPU
        cudaExtent extent = make_cudaExtent(tex_width, tex_height, tex_layers);
        cudaArray_t device_tex_array;
        check_cuda_success(cudaMalloc3DArray(
            &device_tex_array, &channel_desc, extent,
            texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube ?
            cudaArrayCubemap : 0));

        // Prepare the memcpy parameter structure
        cudaMemcpy3DParms copy_params;
        memset(&copy_params, 0, sizeof(copy_params));
        copy_params.dstArray = device_tex_array;
        copy_params.extent = make_cudaExtent(tex_width, tex_height, 1);
        copy_params.kind = cudaMemcpyHostToDevice;

        // Copy the image data of all layers (the layers are not consecutive in memory)
        for (mi::Uint32 layer = 0; layer < tex_layers; ++layer) {
            mi::base::Handle<const mi::neuraylib::ITile> tile(
                canvas->get_tile(0, 0, layer));
            float const *data = static_cast<float const *>(tile->get_data());

            copy_params.srcPtr = make_cudaPitchedPtr(
                const_cast<float *>(data), tex_width * sizeof(float) * 4,
                tex_width, tex_height);
            copy_params.dstPos = make_cudaPos(0, 0, layer);

            check_cuda_success(cudaMemcpy3D(&copy_params));
        }

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_tex_array;

        m_all_texture_arrays->push_back(device_tex_array);
    } else if (m_enable_derivatives) {
        // mipmapped textures use CUDA mipmapped arrays
        mi::Uint32 num_levels = image->get_levels();
        cudaExtent extent = make_cudaExtent(tex_width, tex_height, 0);
        cudaMipmappedArray_t device_tex_miparray;
        check_cuda_success(cudaMallocMipmappedArray(
            &device_tex_miparray, &channel_desc, extent, num_levels));

        // create all mipmap levels and copy them to the CUDA arrays in the mipmapped array
        mi::base::Handle<mi::IArray> mipmaps(image_api->create_mipmaps(canvas.get(), 1.0f));

        for (mi::Uint32 level = 0; level < num_levels; ++level) {
            mi::base::Handle<mi::neuraylib::ICanvas const> level_canvas;
            if (level == 0)
                level_canvas = canvas;
            else {
                mi::base::Handle<mi::IPointer> mipmap_ptr(mipmaps->get_element<mi::IPointer>(level - 1));
                level_canvas = mipmap_ptr->get_pointer<mi::neuraylib::ICanvas>();
            }
            cudaArray_t device_level_array;
            cudaGetMipmappedArrayLevel(&device_level_array, device_tex_miparray, level);
            copy_canvas_to_cuda_array(device_level_array, level_canvas.get());
        }

        res_desc.resType = cudaResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = device_tex_miparray;

        m_all_texture_mipmapped_arrays->push_back(device_tex_miparray);
    } else {
        // 2D texture objects use CUDA arrays
        cudaArray_t device_tex_array;
        check_cuda_success(cudaMallocArray(
            &device_tex_array, &channel_desc, tex_width, tex_height));

        copy_canvas_to_cuda_array(device_tex_array, canvas.get());

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = device_tex_array;

        m_all_texture_arrays->push_back(device_tex_array);
    }

    // For cube maps we need clamped address mode to avoid artifacts in the corners
    cudaTextureAddressMode addr_mode =
        texture_shape == mi::neuraylib::ITarget_code::Texture_shape_cube
        ? cudaAddressModeClamp
        : cudaAddressModeWrap;

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
    if (texture_shape != mi::neuraylib::ITarget_code::Texture_shape_cube) {
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
        make_uint3(tex_width, tex_height, tex_layers)));
    m_all_textures->push_back(textures.back());

    return true;
}

namespace 
{
    bool prepare_mbsdfs_part(mi::neuraylib::Mbsdf_part part, Mbsdf& mbsdf_cuda_representation, 
                             const mi::neuraylib::IBsdf_measurement* bsdf_measurement)
    {
        mi::base::Handle<const mi::neuraylib::Bsdf_isotropic_data> dataset;
        switch (part)
        {
            case mi::neuraylib::MBSDF_DATA_REFLECTION:
                dataset = bsdf_measurement->get_reflection<mi::neuraylib::Bsdf_isotropic_data>();
                break;
            case mi::neuraylib::MBSDF_DATA_TRANSMISSION:
                dataset = bsdf_measurement->get_transmission<mi::neuraylib::Bsdf_isotropic_data>();
                break;
        }

        // no data, fine
        if (!dataset)
            return true;

        // get dimensions
        uint2 res;
        res.x = dataset->get_resolution_theta();
        res.y = dataset->get_resolution_phi();
        unsigned num_channels = dataset->get_type() == mi::neuraylib::BSDF_SCALAR ? 1 : 3;
        mbsdf_cuda_representation.Add(part, res, num_channels);


        // get data
        mi::base::Handle<const mi::neuraylib::IBsdf_buffer> buffer(dataset->get_bsdf_buffer());
        // {1,3} * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)

        const mi::Float32* src_data = buffer->get_data();

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
        float* sample_data = new float[sample_data_size];

        float* albedo_data = new float[res.x]; // albedo for sampling reflection and transmission

        float* sample_data_theta = sample_data;                // begin of the first (theta) CDF
        float* sample_data_phi = sample_data + cdf_theta_size; // begin of the second (phi) CDFs

        const float s_theta = (float) (M_PI * 0.5) / float(res.x);  // step size
        const float s_phi = (float) (M_PI) / float(res.y);          // step size

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

                // offset for both the thetas into the measurement data (select row in the volume) 
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
                        value = fmax(fmaxf(src_data[3 * idx  + 0], src_data[3 * idx  + 1]),
                                     fmaxf(src_data[3 * idx  + 2], 0.0f))
                              + fmax(fmaxf(src_data[3 * idx2 + 0], src_data[3 * idx2 + 1]),
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
                    sample_data_phi[idx] = sample_data_phi[idx] / sum_phi;
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
                sample_data_theta[idx] = sample_data_theta[idx] / sum_theta;
            }
        }

        // copy entire CDF data buffer to GPU
        CUdeviceptr sample_obj = 0;
        check_cuda_success(cuMemAlloc(&sample_obj, sample_data_size * sizeof(float)));
        check_cuda_success(cuMemcpyHtoD(sample_obj, sample_data, sample_data_size * sizeof(float)));
        delete[] sample_data;

        CUdeviceptr albedo_obj = 0;
        check_cuda_success(cuMemAlloc(&albedo_obj, res.x * sizeof(float)));
        check_cuda_success(cuMemcpyHtoD(albedo_obj, albedo_data, res.x * sizeof(float)));
        delete[] albedo_data;


        mbsdf_cuda_representation.sample_data[part] = reinterpret_cast<float*>(sample_obj);
        mbsdf_cuda_representation.albedo_data[part] = reinterpret_cast<float*>(albedo_obj);
        mbsdf_cuda_representation.max_albedo[part] = max_albedo;

        // ----------------------------------------------------------------------------------------
        // prepare evaluation data:
        // - simply store the measured data in a volume texture
        // - in case of color data, we store each sample in a vector4 to get texture support
        unsigned lookup_channels = (num_channels == 3) ? 4 : 1;

        // make lookup data symmetric
        float* lookup_data = new float[lookup_channels * res.y * res.x * res.x];
        for (unsigned int t_in = 0; t_in < res.x; ++t_in)
        {
            for (unsigned int t_out = 0; t_out < res.x; ++t_out)
            {
                const unsigned int offset_phi = (t_in * res.x + t_out) * res.y;
                const unsigned int offset_phi2 = (t_out * res.x + t_in) * res.y;
                for (unsigned int p_out = 0; p_out < res.y; ++p_out)
                {
                    const unsigned int idx = offset_phi + p_out;
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
        cudaChannelFormatDesc channel_desc = (num_channels == 3
            ? cudaCreateChannelDesc<float4>() // float3 is not supported
            : cudaCreateChannelDesc<float>());

        // Allocate a 3D array on the GPU (phi_delta x theta_out x theta_in)
        cudaExtent extent = make_cudaExtent(res.y, res.x, res.x); 
        check_cuda_success(cudaMalloc3DArray(&device_mbsdf_data, &channel_desc, extent, 0));

        // prepare and copy
        cudaMemcpy3DParms copy_params;
        memset(&copy_params, 0, sizeof(copy_params));
        copy_params.srcPtr = make_cudaPitchedPtr(
            (void*)(lookup_data),                                   // base pointer
            res.y * lookup_channels * sizeof(float),                // row pitch
            res.y,                                                  // width of slice
            res.x);                                                 // height of slice
        copy_params.dstArray = device_mbsdf_data;
        copy_params.extent = extent;
        copy_params.kind = cudaMemcpyHostToDevice;
        check_cuda_success(cudaMemcpy3D(&copy_params));
        delete[] lookup_data;

        cudaResourceDesc    texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));
        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = device_mbsdf_data;

        cudaTextureDesc     texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));
        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeClamp;   
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        texDescr.readMode = cudaReadModeElementType;

        cudaTextureObject_t eval_tex_obj;
        check_cuda_success(cudaCreateTextureObject(&eval_tex_obj, &texRes, &texDescr, nullptr));
        mbsdf_cuda_representation.eval_data[part] = eval_tex_obj;

        return true;
    }
}

bool Material_gpu_context::prepare_mbsdf(
    mi::neuraylib::ITransaction       *transaction,
    mi::neuraylib::ITarget_code const *code_ptx,
    mi::Size                           mbsdf_index,
    std::vector<Mbsdf>                &mbsdfs)
{
    // Get access to the texture data by the texture database name from the target code.
    mi::base::Handle<const mi::neuraylib::IBsdf_measurement> mbsdf(
        transaction->access<mi::neuraylib::IBsdf_measurement>(
        code_ptx->get_bsdf_measurement(mbsdf_index)));

    Mbsdf mbsdf_cuda;

    // handle reflection and transmission
    if (!prepare_mbsdfs_part(mi::neuraylib::MBSDF_DATA_REFLECTION, mbsdf_cuda, mbsdf.get()))
        return false;
    if (!prepare_mbsdfs_part(mi::neuraylib::MBSDF_DATA_TRANSMISSION, mbsdf_cuda, mbsdf.get()))
        return false;

    mbsdfs.push_back(mbsdf_cuda);
    m_all_mbsdfs->push_back(mbsdfs.back());
    return true;
}

bool Material_gpu_context::prepare_lightprofile(
    mi::neuraylib::ITransaction       *transaction,
    mi::neuraylib::ITarget_code const *code_ptx,
    mi::Size                           lightprofile_index,
    std::vector<Lightprofile>         &lightprofiles)
{

    // Get access to the texture data by the texture database name from the target code.
    mi::base::Handle<const mi::neuraylib::ILightprofile> lprof_nr(
        transaction->access<mi::neuraylib::ILightprofile>(
        code_ptx->get_light_profile(lightprofile_index)));

    uint2 res = make_uint2(lprof_nr->get_resolution_theta(), lprof_nr->get_resolution_phi());
    float2 start = make_float2(lprof_nr->get_theta(0), lprof_nr->get_phi(0));
    float2 delta = make_float2(lprof_nr->get_theta(1) - start.x, lprof_nr->get_phi(1) - start.y);

    // phi-mayor: [res.x x res.y]
    const float* data = lprof_nr->get_data();

    // -------------------------------------------------------------------------------------------- 
    // compute total power
    // compute inverse CDF data for sampling
    // sampling will work on cells rather than grid nodes (used for evaluation)

    // first (res.x-1) for the cdf for sampling theta
    // rest (rex.x-1) * (res.y-1) for the individual cdfs for sampling phi (after theta)
    size_t cdf_data_size = (res.x - 1) + (res.x - 1) * (res.y - 1);
    float* cdf_data = new float[cdf_data_size];

    float debug_total_erea = 0.0f;
    float sum_theta = 0.0f;
    float total_power = 0.0f;
    float cos_theta0 = cosf(start.x);
    for (unsigned int t = 0; t < res.x - 1; ++t)
    {
        const float cos_theta1 = cosf(start.x + float(t + 1) * delta.x);

        // area of the patch (grid cell)
        // \mu = int_{theta0}^{theta1} sin{theta} \delta theta
        const float mu = cos_theta0 - cos_theta1;
        cos_theta0 = cos_theta1;

        // build CDF for phi
        float* cdf_data_phi = cdf_data + (res.x - 1) + t * (res.y - 1);
        float sum_phi = 0.0f;
        for (unsigned int p = 0; p < res.y - 1; ++p)
        {
            // the probability to select a patch corresponds to the value times area
            // the value of a cell is the average of the corners
            // omit the *1/4 as we normalize in the end
            float value = data[p * res.x + t]
                + data[p * res.x + t + 1]
                + data[(p + 1) * res.x + t]
                + data[(p + 1) * res.x + t + 1];

            sum_phi += value * mu;
            cdf_data_phi[p] = sum_phi;
            debug_total_erea += mu;
        }

        // normalize CDF for phi
        for (unsigned int p = 0; p < res.y - 2; ++p)
            cdf_data_phi[p] = sum_phi ? (cdf_data_phi[p] / sum_phi) : 0.0f;

        cdf_data_phi[res.y - 2] = 1.0f;

        // build CDF for theta
        sum_theta += sum_phi;
        cdf_data[t] = sum_theta;
    }
    total_power = sum_theta * 0.25f * delta.y;

    // normalize CDF for theta
    for (unsigned int t = 0; t < res.x - 2; ++t)
        cdf_data[t] = sum_theta ? (cdf_data[t] / sum_theta) : cdf_data[t];

    cdf_data[res.x - 2] = 1.0f;

    // copy entire CDF data buffer to GPU
    CUdeviceptr cdf_data_obj = 0;
    check_cuda_success(cuMemAlloc(&cdf_data_obj, cdf_data_size * sizeof(float)));
    check_cuda_success(cuMemcpyHtoD(cdf_data_obj, cdf_data, cdf_data_size * sizeof(float)));
    delete[] cdf_data;

    // -------------------------------------------------------------------------------------------- 
    // prepare evaluation data
    //  - use a 2d texture that allows bilinear interpolation
    // Copy data to GPU array
    cudaArray_t device_lightprofile_data;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

    // 2D texture objects use CUDA arrays
    check_cuda_success(cudaMallocArray(&device_lightprofile_data, &channel_desc, res.x, res.y));
    check_cuda_success(cudaMemcpy2DToArray(
        device_lightprofile_data, 0, 0, data,
        res.x * sizeof(float), res.x * sizeof(float), res.y, cudaMemcpyHostToDevice));

    // Create filtered texture object
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = device_lightprofile_data;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeClamp;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.borderColor[1] = 1.0f;
    tex_desc.borderColor[2] = 1.0f;
    tex_desc.borderColor[3] = 1.0f;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    cudaTextureObject_t tex_obj = 0;
    check_cuda_success(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    double multiplier = lprof_nr->get_candela_multiplier();
    Lightprofile lprof(
        res,
        start,
        delta,
        float(multiplier),
        float(total_power * multiplier),
        tex_obj,
        reinterpret_cast<float*>(cdf_data_obj));

    lightprofiles.push_back(lprof);
    m_all_lightprofiles->push_back(lightprofiles.back());
    return true;
}

// Prepare the needed target code data of the given target code.
bool Material_gpu_context::prepare_target_code_data(
    mi::neuraylib::ITransaction          *transaction,
    mi::neuraylib::IImage_api            *image_api,
    mi::neuraylib::ITarget_code const    *target_code,
    std::vector<size_t> const            &arg_block_indices)
{
    // Target code data list may not have been retrieved already
    check_success(m_device_target_code_data_list.get() == 0);

    // Handle the read-only data segments if necessary.
    // They are only created, if the "enable_ro_segment" backend option was set to "on".
    CUdeviceptr device_ro_data = 0;
    if (target_code->get_ro_data_segment_count() > 0) {
        device_ro_data = gpu_mem_dup(
            target_code->get_ro_data_segment_data(0),
            target_code->get_ro_data_segment_size(0));
    }

    // Copy textures to GPU if the code has more than just the invalid texture
    CUdeviceptr device_textures = 0;
    mi::Size num_textures = target_code->get_texture_count();
    if (num_textures > 1) {
        std::vector<Texture> textures;

        // Loop over all textures skipping the first texture,
        // which is always the invalid texture
        for (mi::Size i = 1; i < num_textures; ++i) {
            if (!prepare_texture(
                    transaction, image_api, target_code, i, textures))
                return false;
        }

        // Copy texture list to GPU
        device_textures = gpu_mem_dup(textures);
    }

    // Copy MBSDFs to GPU if the code has more than just the invalid mbsdf
    CUdeviceptr device_mbsdfs = 0;
    mi::Size num_mbsdfs = target_code->get_bsdf_measurement_count();
    if (num_mbsdfs > 1) {
        std::vector<Mbsdf> mbsdfs;

        // Loop over all mbsdfs skipping the first mbsdf,
        // which is always the invalid mbsdf
        for (mi::Size i = 1; i < num_mbsdfs; ++i) {
            if (!prepare_mbsdf(
                transaction, target_code, i, mbsdfs))
                return false;
        }

        // Copy mbsdf list to GPU
        device_mbsdfs = gpu_mem_dup(mbsdfs);
    }

    // Copy light profiles to GPU if the code has more than just the invalid light profile
    CUdeviceptr device_lightprofiles = 0;
    mi::Size num_lightprofiles = target_code->get_light_profile_count();
    if (num_lightprofiles > 1) {
        std::vector<Lightprofile> lightprofiles;

        // Loop over all profiles skipping the first profile,
        // which is always the invalid profile
        for (mi::Size i = 1; i < num_lightprofiles; ++i) {
            if (!prepare_lightprofile(
                transaction, target_code, i, lightprofiles))
                return false;
        }

        // Copy light profile list to GPU
        device_lightprofiles = gpu_mem_dup(lightprofiles);
    }

    (*m_target_code_data_list).push_back(
        Target_code_data(num_textures, device_textures,
                         num_mbsdfs, device_mbsdfs,
                         num_lightprofiles, device_lightprofiles,
                         device_ro_data));

    for (mi::Size i = 0, num = target_code->get_argument_block_count(); i < num; ++i) {
        mi::base::Handle<mi::neuraylib::ITarget_argument_block const> arg_block(
            target_code->get_argument_block(i));
        CUdeviceptr dev_block = gpu_mem_dup(arg_block->get_data(), arg_block->get_size());
        m_target_argument_block_list->push_back(dev_block);
        m_own_arg_blocks.push_back(mi::base::make_handle(arg_block->clone()));
        m_arg_block_layouts.push_back(
            mi::base::make_handle(target_code->get_argument_block_layout(i)));
    }

    for (size_t arg_block_index : arg_block_indices) {
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

    mi::base::Handle<mi::neuraylib::ITarget_argument_block> arg_block(get_argument_block(i));
    check_cuda_success(cuMemcpyHtoD(
        device_ptr, arg_block->get_data(), arg_block->get_size()));
}


//------------------------------------------------------------------------------
//
// MDL material compilation code
//
//------------------------------------------------------------------------------

class Material_compiler {
public:
    // Constructor.
    Material_compiler(
        mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
        mi::neuraylib::IMdl_backend_api* mdl_backend_api,
        mi::neuraylib::IMdl_factory* mdl_factory,
        mi::neuraylib::ITransaction* transaction,
        unsigned num_texture_results,
        bool enable_derivatives,
        bool fold_ternary_on_df,
        bool enable_auxiliary,
        const std::string& df_handle_mode);

    // Loads an MDL module and returns the module DB.
    std::string load_module(const std::string& mdl_module_name);

    // Add a subexpression of a given material to the link unit.
    // path is the path of the sub-expression.
    // fname is the function name in the generated code.
    // If class_compilation is true, the material will use class compilation.
    bool add_material_subexpr(
        const std::string& qualified_module_name,
        const std::string& material_simple_name,
        const char* path,
        const char* fname,
        bool class_compilation=false);

    // Add a distribution function of a given material to the link unit.
    // path is the path of the sub-expression.
    // fname is the function name in the generated code.
    // If class_compilation is true, the material will use class compilation.
    bool add_material_df(
        const std::string& qualified_module_name,
        const std::string& material_simple_name,
        const char* path,
        const char* base_fname,
        bool class_compilation=false);

    // Add (multiple) MDL distribution function and expressions of a material to this link unit.
    // For each distribution function it results in four functions, suffixed with \c "_init",
    // \c "_sample", \c "_evaluate", and \c "_pdf". Functions can be selected by providing a
    // a list of \c Target_function_descriptions. Each of them needs to define the \c path, the root
    // of the expression that should be translated. After calling this function, each element of
    // the list will contain information for later usage in the application,
    // e.g., the \c argument_block_index and the \c function_index.
    bool add_material(
        const std::string& qualified_module_name,
        const std::string& material_simple_name,
        mi::neuraylib::Target_function_description* function_descriptions,
        mi::Size description_count,
        bool class_compilation);

    // Generates CUDA PTX target code for the current link unit.
    mi::base::Handle<const mi::neuraylib::ITarget_code> generate_cuda_ptx();

    typedef std::vector<mi::base::Handle<mi::neuraylib::IMaterial_definition const> >
        Material_definition_list;

    // Get the list of used material definitions.
    // There will be one entry per add_* call.
    Material_definition_list const &get_material_defs()
    {
        return m_material_defs;
    }

    typedef std::vector<mi::base::Handle<mi::neuraylib::ICompiled_material const> >
        Compiled_material_list;

    // Get the list of compiled materials.
    // There will be one entry per add_* call.
    Compiled_material_list const &get_compiled_materials()
    {
        return m_compiled_materials;
    }

    /// Get the list of argument block indices per material.
    std::vector<size_t> const &get_argument_block_indices() const {
        return m_arg_block_indexes;
    }

    /// Get the set of handles present in added materials.
    /// Only available after calling 'add_material' at least once.
    const std::vector<std::string>& get_handles() const {
        return m_handles;
    }

private:
    // Creates an instance of the given material.
    mi::neuraylib::IMaterial_instance* create_material_instance(
        const std::string& qualified_module_name,
        const std::string& material_simple_name);

    // Compiles the given material instance in the given compilation modes.
    mi::neuraylib::ICompiled_material* compile_material_instance(
        mi::neuraylib::IMaterial_instance* material_instance,
        bool class_compilation);

private:
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_mdl_impexp_api;
    mi::base::Handle<mi::neuraylib::IMdl_backend>    m_be_cuda_ptx;
    mi::base::Handle<mi::neuraylib::IMdl_factory>    m_mdl_factory;
    mi::base::Handle<mi::neuraylib::ITransaction>    m_transaction;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_context;
    mi::base::Handle<mi::neuraylib::ILink_unit>             m_link_unit;

    Material_definition_list  m_material_defs;
    Compiled_material_list    m_compiled_materials;
    std::vector<size_t>       m_arg_block_indexes;
    std::vector<std::string>  m_handles;
};

// Constructor.
Material_compiler::Material_compiler(
        mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
        mi::neuraylib::IMdl_backend_api* mdl_backend_api,
        mi::neuraylib::IMdl_factory* mdl_factory,
        mi::neuraylib::ITransaction* transaction,
        unsigned num_texture_results,
        bool enable_derivatives,
        bool fold_ternary_on_df,
        bool enable_auxiliary,
        const std::string& df_handle_mode)
    : m_mdl_impexp_api(mdl_impexp_api, mi::base::DUP_INTERFACE)
    , m_be_cuda_ptx(mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX))
    , m_mdl_factory(mdl_factory, mi::base::DUP_INTERFACE)
    , m_transaction(transaction, mi::base::DUP_INTERFACE)
    , m_context(mdl_factory->create_execution_context())
    , m_link_unit()
{
    check_success(m_be_cuda_ptx->set_option("num_texture_spaces", "1") == 0);

    // Option "enable_ro_segment": Default is disabled.
    // If you have a lot of big arrays, enabling this might speed up compilation.
    // check_success(m_be_cuda_ptx->set_option("enable_ro_segment", "on") == 0);

    if (enable_derivatives) {
        // Option "texture_runtime_with_derivs": Default is disabled.
        // We enable it to get coordinates with derivatives for texture lookup functions.
        check_success(m_be_cuda_ptx->set_option("texture_runtime_with_derivs", "on") == 0);
    }

    // Option "tex_lookup_call_mode": Default mode is vtable mode.
    // You can switch to the slower vtable mode by commenting out the next line.
    check_success(m_be_cuda_ptx->set_option("tex_lookup_call_mode", "direct_call") == 0);

    // Option "num_texture_results": Default is 0.
    // Set the size of a renderer provided array for texture results in the MDL SDK state in number
    // of float4 elements processed by the init() function.
    check_success(m_be_cuda_ptx->set_option(
        "num_texture_results",
        to_string(num_texture_results).c_str()) == 0);

    if (enable_auxiliary) {
        // Option "enable_auxiliary": Default is disabled.
        // We enable it to create an additional 'auxiliary' function that can be called on each
        // distribution function to fill an albedo and normal buffer e.g. for denoising.
        check_success(m_be_cuda_ptx->set_option("enable_auxiliary", "on") == 0);
    }


    // Option "df_handle_slot_mode": Default is "none".
    // When using light path expressions, individual parts of the distribution functions can be
    // selected using "handles". The contribution of each of those parts has to be evaluated during
    // rendering. This option controls how many parts are evaluated with each call into the
    // generated "evaluate" and "auxiliary" functions and how the data is passed.
    // The CUDA backend supports pointers, which means an externally managed buffer of arbitrary 
    // size is used to transport the contributions of each part.
    check_success(m_be_cuda_ptx->set_option("df_handle_slot_mode", df_handle_mode.c_str()) == 0);

    // Option "scene_data_names": Default is "".
    // Uncomment the line below to enable calling the scene data runtime functions
    // for any scene data names or specify a comma-separated list of names for which
    // you may provide scene data. The example runtime functions always return the
    // default values, which is the same as not supporting any scene data.
    //     m_be_cuda_ptx->set_option("scene_data_names", "*");

    // force experimental to true for now
    m_context->set_option("experimental", true);

    m_context->set_option("fold_ternary_on_df", fold_ternary_on_df);

    // After we set the options, we can create the link unit
    m_link_unit = mi::base::make_handle(m_be_cuda_ptx->create_link_unit(transaction, m_context.get()));
}

std::string Material_compiler::load_module(const std::string& mdl_module_name)
{
    // load module
    m_mdl_impexp_api->load_module(m_transaction.get(), mdl_module_name.c_str(), m_context.get());
    if (!print_messages(m_context.get()))
        exit_failure("Failed to load module: %s", mdl_module_name.c_str());

    // get and return the DB name
    mi::base::Handle<const mi::IString> db_module_name(
        m_mdl_factory->get_db_module_name(mdl_module_name.c_str()));
    return db_module_name->get_c_str();
}

// Creates an instance of the given material.
mi::neuraylib::IMaterial_instance* Material_compiler::create_material_instance(
    const std::string& qualified_module_name,
    const std::string& material_simple_name)
{
    // Load mdl module.
    m_mdl_impexp_api->load_module(
        m_transaction.get(), qualified_module_name.c_str(), m_context.get());
    if (!print_messages(m_context.get())) {
        // module has errors
        return nullptr;
    }

    // get db name
    mi::base::Handle<const mi::IString> module_db_name(
        m_mdl_factory->get_db_module_name(qualified_module_name.c_str()));

    std::string material_db_name =
        std::string(module_db_name->get_c_str()) + "::" + material_simple_name;

    // Create a material instance from the material definition
    // with the default arguments.
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        m_transaction->access<mi::neuraylib::IMaterial_definition>(
            material_db_name.c_str()));
    if (!material_definition) {
        // material with given name does not exists
        print_message(
            mi::base::details::MESSAGE_SEVERITY_ERROR,
            mi::neuraylib::IMessage::MSG_COMPILER_DAG,
            (
                "Material '" +
                material_simple_name +
                "' does not exists in '" +
                qualified_module_name + "'").c_str());
        return nullptr;
    }

    m_material_defs.push_back(material_definition);

    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
        material_definition->create_material_instance(0, &result));
    check_success(result == 0);

    material_instance->retain();
    return material_instance.get();
}

// Compiles the given material instance in the given compilation modes.
mi::neuraylib::ICompiled_material *Material_compiler::compile_material_instance(
    mi::neuraylib::IMaterial_instance* material_instance,
    bool class_compilation)
{
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance->create_compiled_material(flags, m_context.get()));
    check_success(print_messages(m_context.get()));

    m_compiled_materials.push_back(compiled_material);

    compiled_material->retain();
    return compiled_material.get();
}

// Generates CUDA PTX target code for the current link unit.
mi::base::Handle<const mi::neuraylib::ITarget_code> Material_compiler::generate_cuda_ptx()
{
    mi::base::Handle<const mi::neuraylib::ITarget_code> code_cuda_ptx(
        m_be_cuda_ptx->translate_link_unit(m_link_unit.get(), m_context.get()));
    check_success(print_messages(m_context.get()));
    check_success(code_cuda_ptx);

#ifdef DUMP_PTX
    std::cout << "Dumping CUDA PTX code:\n\n"
        << code_cuda_ptx->get_code() << std::endl;
#endif

    return code_cuda_ptx;
}

// Add a subexpression of a given material to the link unit.
// path is the path of the sub-expression.
// fname is the function name in the generated code.
bool Material_compiler::add_material_subexpr(
    const std::string& qualified_module_name,
    const std::string& material_simple_name,
    const char* path,
    const char* fname,
    bool class_compilation)
{
    mi::neuraylib::Target_function_description desc;
    desc.path = path;
    desc.base_fname = fname;
    add_material(qualified_module_name, material_simple_name, &desc, 1, class_compilation);
    return desc.return_code == 0;
}

// Add a distribution function of a given material to the link unit.
// path is the path of the sub-expression.
// fname is the function name in the generated code.
bool Material_compiler::add_material_df(
    const std::string& qualified_module_name,
    const std::string& material_simple_name,
    const char* path,
    const char* base_fname,
    bool class_compilation)
{
    mi::neuraylib::Target_function_description desc;
    desc.path = path;
    desc.base_fname = base_fname;
    add_material(qualified_module_name, material_simple_name, &desc, 1, class_compilation);
    return desc.return_code == 0;
}

// Add (multiple) MDL distribution function and expressions of a material to this link unit.
// For each distribution function it results in four functions, suffixed with \c "_init",
// \c "_sample", \c "_evaluate", and \c "_pdf". Functions can be selected by providing a
// a list of \c Target_function_description. Each of them needs to define the \c path, the root
// of the expression that should be translated. After calling this function, each element of
// the list will contain information for later usage in the application,
// e.g., the \c argument_block_index and the \c function_index.
bool Material_compiler::add_material(
    const std::string& qualified_module_name,
    const std::string& material_simple_name,
    mi::neuraylib::Target_function_description* function_descriptions,
    mi::Size description_count,
    bool class_compilation)
{
    if (description_count == 0)
        return false;

    // Load the given module and create a material instance
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
        create_material_instance(qualified_module_name, material_simple_name));
    if (!material_instance)
        return false;

    // Compile the material instance in instance compilation mode
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        compile_material_instance(material_instance.get(), class_compilation));

    m_link_unit->add_material(
        compiled_material.get(), function_descriptions, description_count,
        m_context.get());

    // Note: the same argument_block_index is filled into all function descriptions of a
    //       material, if any function uses it
    m_arg_block_indexes.push_back(function_descriptions[0].argument_block_index);

    return print_messages(m_context.get());
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
    const std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code> > &target_codes)
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
        mi::base::Handle<const mi::neuraylib::ITarget_code> const &target_code = 
            target_codes[tc_index];

        // in case of multiple target codes, we need to address the functions by a pair of 
        // target_code_index and function_index.
        // the elements in the resulting function array can then be index by offset + func_index.
        if(!tc_offsets.empty())
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
                func_index, mi::neuraylib::ITarget_code::SL_PTX);
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
    std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code> > const &target_codes,
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
                const_cast<char *>(target_codes[i]->get_code()),
                target_codes[i]->get_code_size(),
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

    std::cout << "CUDA link completed." << std::endl;
    if (info_log[0])
        std::cout << "Linker output:\n" << info_log << std::endl;

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
    std::cout << "Kernel uses " << regs << " registers and " << lmem << " lmem and has a size of "
        << linked_cubin_size << " bytes." << std::endl;

    // Cleanup
    check_cuda_success(cuLinkDestroy(cuda_link_state));

    return cuda_module;
}

#endif // EXAMPLE_CUDA_SHARED_H


