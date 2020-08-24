/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "texture.h"
#include "base_application.h"
#include "buffer.h"
#include "command_queue.h"
#include "descriptor_heap.h"
#include "mdl_material.h"
#include "mdl_sdk.h"

namespace mi { namespace examples { namespace mdl_d3d12
{

Texture* Texture::create_texture_2d(
    Base_application* app,
    GPU_access gpu_access,
    size_t width,
    size_t height,
    DXGI_FORMAT format,
    const std::string& debug_name)
{
    return new Texture(app, gpu_access, Texture_dimension::Texture_2D,
        width, height, 1, format, debug_name);
}

// ------------------------------------------------------------------------------------------------

Texture* Texture::create_texture_3d(
    Base_application* app,
    GPU_access gpu_access,
    size_t width,
    size_t height,
    size_t depth,
    DXGI_FORMAT format,
    const std::string& debug_name)
{
    return new Texture(app, gpu_access, Texture_dimension::Texture_3D,
        width, height, depth, format, debug_name);
}

// ------------------------------------------------------------------------------------------------

Texture::Texture(
    Base_application* app,
    GPU_access gpu_access,
    Texture_dimension dimension,
    size_t width,
    size_t height,
    size_t depth,
    DXGI_FORMAT format,
    const std::string& debug_name)

    : m_app(app)
    , m_debug_name(debug_name)
    , m_dimension(dimension)
    , m_gpu_access(gpu_access)
    , m_width(width)
    , m_height(height)
    , m_depth(depth)
    , m_format(format)
    , m_pixel_stride_in_byte(0)
    , m_resource(nullptr)
    , m_resource_upload(nullptr)
    , m_resource_download(nullptr)
    , m_latest_scheduled_state(D3D12_RESOURCE_STATE_COMMON)
    , m_opt_swap_chain(nullptr)
{
    create();
}

// ------------------------------------------------------------------------------------------------

Texture::Texture(
    Base_application* app,
    IDXGISwapChain1* swap_chain,
    size_t swap_chain_buffer_index,
    const std::string& debug_name)

    : m_app(app)
    , m_debug_name(debug_name)
    , m_gpu_access(GPU_access::render_target)
    , m_dimension(Texture_dimension::Texture_2D)
    , m_width(0)
    , m_height(0)
    , m_depth(1)
    , m_format(DXGI_FORMAT_R8G8B8A8_UNORM)
    , m_pixel_stride_in_byte(0)
    , m_resource(nullptr)
    , m_resource_upload(nullptr)
    , m_resource_download(nullptr)
    , m_latest_scheduled_state(D3D12_RESOURCE_STATE_COMMON)
    , m_opt_swap_chain(swap_chain)
    , m_opt_swap_chain_buffer_index(swap_chain_buffer_index)
{
    DXGI_SWAP_CHAIN_DESC1 swap_desc;
    throw_on_failure(m_opt_swap_chain->GetDesc1(&swap_desc),
        "Failed to get Swap Chain description.", SRC);

    m_width = swap_desc.Width;
    m_height = swap_desc.Height;
    m_format = swap_desc.Format;

    create();
}

// ------------------------------------------------------------------------------------------------

bool Texture::get_srv_description(
    D3D12_SHADER_RESOURCE_VIEW_DESC& out_desc,
    Texture_dimension dimension) const
{
    if (!mi::examples::enums::has_flag(m_gpu_access, GPU_access::shader_resource))
    {
        log_error("Texture has not shader resource access: " + m_debug_name, SRC);
        return false;
    }

    out_desc = {};
    out_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    out_desc.Format = m_format;

    if (dimension == Texture_dimension::Undefined)
        dimension = m_dimension;

    switch (dimension)
    {
        case Texture_dimension::Texture_2D:
            out_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            break;
        case Texture_dimension::Texture_3D:
            out_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            break;
        default:
            log_error("Texture has no valid dimension: " + m_debug_name, SRC);
            return false;
    }
    out_desc.Texture2D.MipLevels = 1;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Texture::get_uav_description(D3D12_UNORDERED_ACCESS_VIEW_DESC& out_desc) const
{
    if (!mi::examples::enums::has_flag(m_gpu_access, GPU_access::unorder_access))
    {
        log_error("Texture has not unordered access: " + m_debug_name, SRC);
        return false;
    }

    out_desc = {};
    switch (m_dimension)
    {
        case Texture_dimension::Texture_2D:
            out_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            break;
        case Texture_dimension::Texture_3D:
            out_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            break;
        default:
            log_error("Texture has no valid dimension: " + m_debug_name, SRC);
            return false;
    }
    out_desc.Format = m_format;
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Texture::create()
{
    // handle swap chain case
    if (m_opt_swap_chain)
    {
        if (log_on_failure(m_opt_swap_chain->GetBuffer(static_cast<uint32_t>(
            m_opt_swap_chain_buffer_index), IID_PPV_ARGS(&m_resource)),
            "Failed to get resource from Swap Chain." , SRC))
            return false;
    }
    else
    {
        // non swap chain textures
        CD3DX12_RESOURCE_DESC resource_desc;
        switch (m_dimension)
        {
            case Texture_dimension::Texture_2D:
                resource_desc = CD3DX12_RESOURCE_DESC::Tex2D(
                    m_format,
                    static_cast<uint64_t>(m_width),
                    static_cast<uint32_t>(m_height),
                    static_cast<uint16_t>(1),
                    static_cast<uint16_t>(1), // mip levels (only one)
                    static_cast<uint16_t>(1), // sample count
                    static_cast<uint32_t>(0), // sample quality
                    D3D12_RESOURCE_FLAG_NONE);
                break;
            case Texture_dimension::Texture_3D:
                resource_desc = CD3DX12_RESOURCE_DESC::Tex3D(
                    m_format,
                    static_cast<uint64_t>(m_width),
                    static_cast<uint32_t>(m_height),
                    static_cast<uint16_t>(m_depth),
                    static_cast<uint16_t>(1), // mip levels (only one)
                    D3D12_RESOURCE_FLAG_NONE);
                break;
            default:
                log_error("Texture has no valid dimension: " + m_debug_name, SRC);
                return false;
        }

        bool init_with_clear_value = true;

        if (mi::examples::enums::has_flag(m_gpu_access, GPU_access::shader_resource))
        {
            m_latest_scheduled_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            init_with_clear_value = false;
        }

        if (mi::examples::enums::has_flag(m_gpu_access, GPU_access::depth_stencil_target))
        {
            resource_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
            m_latest_scheduled_state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        }

        if (mi::examples::enums::has_flag(m_gpu_access, GPU_access::render_target))
        {
            resource_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            m_latest_scheduled_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
        }

        if (mi::examples::enums::has_flag(m_gpu_access, GPU_access::unorder_access))
        {
            resource_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            m_latest_scheduled_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            init_with_clear_value = false;
        }

        if (log_on_failure(m_app->get_device()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &resource_desc,
            static_cast<D3D12_RESOURCE_STATES>(m_latest_scheduled_state),
            nullptr,
            IID_PPV_ARGS(&m_resource)), "Failed to create texture resource.", SRC))
            return false;
        set_debug_name(m_resource.Get(), m_debug_name);
    }

    // get and cache pixel stride
    size_t row_data_size;
    D3D12_RESOURCE_DESC desc = m_resource->GetDesc();
    m_app->get_device()->GetCopyableFootprints(
        &desc, 0, 1, 0, nullptr, nullptr, &row_data_size, nullptr);
    m_pixel_stride_in_byte = row_data_size / m_width;
    return true;
}

// ------------------------------------------------------------------------------------------------

size_t Texture::get_gpu_row_pitch() const
{
    return round_to_power_of_two(
        m_width * m_pixel_stride_in_byte, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
}

// ------------------------------------------------------------------------------------------------

bool Texture::upload(D3DCommandList* command_list, const uint8_t* data, size_t data_row_pitch)
{
    if (!m_resource) {
        log_error("Resource is not valid: " + m_debug_name, SRC);
        return false;
    }

    // Note, this is not able to handle mip maps
    // it will also create a new buffer every time,
    // some logic is required to reuse existing buffers if the size matches

    size_t buffer_size = GetRequiredIntermediateSize(m_resource.Get(), 0, 1);
    const uint8_t* buffer = data;

    // need to enforce alignment?
    if(data_row_pitch == -1)
        data_row_pitch = m_width * m_pixel_stride_in_byte;

    size_t gpu_row_pitch = get_gpu_row_pitch();

    if (gpu_row_pitch != data_row_pitch)
    {
        buffer = new uint8_t[buffer_size];
        memset((void*) buffer, 0, buffer_size);

        for (size_t r = 0; r < m_height; ++r)
            memcpy((void*) (buffer + r * gpu_row_pitch),
                    (void*) (data + r * data_row_pitch),
                    data_row_pitch);
    }

    // create a resource that allows to upload data
    if (!m_resource_upload)
    {
        if (log_on_failure(m_app->get_device()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(buffer_size),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&m_resource_upload)),
            "Failed to create upload texture resource.", SRC))
            return false;

        set_debug_name(m_resource_upload.Get(), m_debug_name + "_Upload");
    }

    D3D12_SUBRESOURCE_DATA subresource_data = {};
    subresource_data.pData = buffer;
    subresource_data.RowPitch = gpu_row_pitch;
    subresource_data.SlicePitch = gpu_row_pitch * m_height;

    D3D12_RESOURCE_STATES saved = m_latest_scheduled_state;
    transition_to(command_list, D3D12_RESOURCE_STATE_COPY_DEST);
    auto res = UpdateSubresources(
        command_list, m_resource.Get(), m_resource_upload.Get(), 0, 0, 1, &subresource_data);
    transition_to(command_list, saved);

    if (buffer != data)
        delete[] buffer;

    if (res != buffer_size) {
        log_error("Failed to upload texture data to GPU: " + m_debug_name, SRC);
        return false;
    }

    return true;
}

// ------------------------------------------------------------------------------------------------

bool Texture::download(void* data)
{
    // copy texture buffer to the staging resources
    D3D12_RESOURCE_DESC desc = m_resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
    m_app->get_device()->GetCopyableFootprints(
        &desc, 0, 1, 0, &layout, nullptr, nullptr, nullptr);

    size_t buffer_size = layout.Footprint.RowPitch * layout.Footprint.Height;

    // create a resource that allows to download data
    if (!m_resource_download)
    {
        if (log_on_failure(m_app->get_device()->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(buffer_size),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&m_resource_download)),
            "Failed to create download texture resource.", SRC))
            return false;

        set_debug_name(m_resource_download.Get(), m_debug_name + "_Download");
    }

    auto command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_COPY);
    auto command_list = command_queue->get_command_list();

    D3D12_TEXTURE_COPY_LOCATION dstLocation = {
        m_resource_download.Get(),
        D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        { layout }
    };

    D3D12_TEXTURE_COPY_LOCATION srcLocation = {
        m_resource.Get(),
        D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        {}
    };

    D3D12_BOX box;
    box.left = 0;
    box.top = 0;
    box.right = static_cast<UINT>(m_width);
    box.bottom = static_cast<UINT>(m_height);
    box.front = 0;
    box.back = static_cast<UINT>(m_depth);

    // copy texture region to download buffer
    command_list->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, &box);

    // execute copy and wait till finished
    command_queue->execute_command_list(command_list);
    command_queue->flush();

    uint8_t* mapped_buffer;
    D3D12_RANGE range = {0, buffer_size};
    if (log_on_failure(m_resource_download->Map(0, &range, (void**) &mapped_buffer),
        "Failed to map download buffer: " + m_debug_name, SRC))
        return false;

    // copy line by line to flip the image
    uint8_t* destination = (uint8_t*) data;

    for (size_t y = 0; y < m_height; ++y)
        memcpy(destination + m_pixel_stride_in_byte * m_width * y,
                mapped_buffer + layout.Footprint.RowPitch * (m_height - y - 1),
                m_pixel_stride_in_byte * m_width);

    m_resource_download->Unmap(0, nullptr);
    return true;
}

// ------------------------------------------------------------------------------------------------

void Texture::transition_to(D3DCommandList* command_list, D3D12_RESOURCE_STATES state)
{
    if (m_latest_scheduled_state == state)
        return;

    command_list->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        m_resource.Get(), m_latest_scheduled_state, state));

    m_latest_scheduled_state = state;
}

// ------------------------------------------------------------------------------------------------

bool Texture::resize(size_t width, size_t height)
{
    return resize(width, height, m_depth);
}

// ------------------------------------------------------------------------------------------------

bool Texture::resize(size_t width, size_t height, size_t depth)
{
    if (m_dimension != Texture_dimension::Texture_3D && depth != 1)
    {
        log_error("Setting 'depth' of non-3D textures is invalid: " + m_debug_name, SRC);
        return false;
    }

    if (m_width == width && m_height == height && m_depth == depth)
        return true;

    m_resource.Reset();
    m_width = width;
    m_height = height;
    m_depth = depth;

    if (m_resource_upload)
        m_resource_upload.Reset();
    m_resource_upload = nullptr;

    if (m_resource_download)
        m_resource_download.Reset();
    m_resource_download = nullptr;

    return create();
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Environment::Environment(Base_application * app, const std::string& file_path)
    : m_app(app)
    , m_debug_name(file_path + "_Environment")
    , m_texture(nullptr)
    , m_sampling_buffer(nullptr)
    , m_integral(0.0f)
    , m_first_resource_heap_handle()
{
    if (!create(file_path))
    {
        if (m_texture) delete m_texture;
        if (m_sampling_buffer) delete m_sampling_buffer;
        m_resource_descriptor_table.clear();
    }
}

// ------------------------------------------------------------------------------------------------

Environment::~Environment()
{
    if (m_texture) delete m_texture;
    if (m_sampling_buffer) delete m_sampling_buffer;
    m_app->get_resource_descriptor_heap()->free_views(m_first_resource_heap_handle);
}

// ------------------------------------------------------------------------------------------------

void Environment::transition_to(D3DCommandList* command_list, D3D12_RESOURCE_STATES state)
{
    if(!m_texture)
    {
        log_error("Environment object is invalid: " + m_debug_name, SRC);
        return;
    }

    m_texture->transition_to(command_list, state);
}

// ------------------------------------------------------------------------------------------------

namespace
{

// Helper for create_environment()
float build_alias_map(
    const float *data,
    const unsigned int size,
    std::vector<Environment::Sample_data>& sampling_data)
{
    // create qs (normalized)
    float sum = 0.0f;
    for (unsigned int i = 0; i < size; ++i)
        sum += data[i];

    for (unsigned int i = 0; i < size; ++i)
        sampling_data[i].q = (static_cast<float>(size) * data[i] / sum);

    // create partition table
    unsigned int *partition_table = static_cast<unsigned int*>(
        malloc(size * sizeof(unsigned int)));
    unsigned int s = 0u, large = size;
    for (unsigned int i = 0; i < size; ++i)
        partition_table[(sampling_data[i].q < 1.0f)
            ? (s++)
            : (--large)] = sampling_data[i].alias = i;

    // create alias map
    for (s = 0; s < large && large < size; ++s)
    {
        const unsigned int j = partition_table[s], k = partition_table[large];
        sampling_data[j].alias = k;
        sampling_data[k].q += sampling_data[j].q - 1.0f;
        large = (sampling_data[k].q < 1.0f) ? (large + 1u) : large;
    }

    free(partition_table);

    return sum;
}

} // anonymous

// ------------------------------------------------------------------------------------------------

bool Environment::create(const std::string& file_path)
{
    {
        // Load environment texture
        mi::base::Handle<mi::neuraylib::IImage>image(
            m_app->get_mdl_sdk().get_transaction().create<mi::neuraylib::IImage>("Image"));

        if (image->reset_file(file_path.c_str()) != 0)
        {
            log_error("Failed to load image for: " + m_debug_name, SRC);
            return false;
        }

        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas());
        const size_t rx = canvas->get_resolution_x();
        const size_t ry = canvas->get_resolution_y();

        // Check, whether we need to convert the image
        char const *image_type = image->get_type();
        if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0)
            canvas = m_app->get_mdl_sdk().get_image_api().convert(canvas.get(), "Color");

        mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
        const float *pixels = static_cast<const float *>(tile->get_data());

        // create a command list for uploading data to the GPU
        Command_queue* command_queue = m_app->get_command_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
        D3DCommandList* command_list = command_queue->get_command_list();

        // create the d3d texture
        m_texture = Texture::create_texture_2d(
            m_app, GPU_access::shader_resource, rx, ry, DXGI_FORMAT_R32G32B32A32_FLOAT,
            file_path + "_Texture");

        // create sampling data
        m_sampling_buffer = new Structured_buffer<Environment::Sample_data>(
            m_app, rx * ry, file_path + "_SamplingBuffer");

        std::vector<Environment::Sample_data> sampling_data(rx * ry);

        float *importance_data = static_cast<float *>(malloc(rx * ry * sizeof(float)));
        float cos_theta0 = 1.0f;
        const float step_phi = (2.0f * mdl_d3d12::PI) / float(rx);
        const float step_theta = (mdl_d3d12::PI) / float(ry);
        for (unsigned int y = 0; y < ry; ++y)
        {
            const float theta1 = float(y + 1) * step_theta;
            const float cos_theta1 = std::cos(theta1);
            const float area = (cos_theta0 - cos_theta1) * step_phi;
            cos_theta0 = cos_theta1;

            for (unsigned int x = 0; x < rx; ++x)
            {
                const unsigned int idx = static_cast<const unsigned int>(y * rx + x);
                const unsigned int idx4 = idx * 4;
                importance_data[idx] =
                    area * std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2]));
            }
        }

        m_integral = build_alias_map(importance_data, unsigned int(rx * ry), sampling_data);
        free(importance_data);

        // copy data to the GPU
        if (!m_texture->upload(command_list, (const uint8_t*) pixels)) return false;

        m_sampling_buffer->set_data(sampling_data);
        if (!m_sampling_buffer->upload(command_list)) return false;

        command_queue->execute_command_list(command_list);
    }

    // reserve continuous part of the descriptor
    auto resource_heap = m_app->get_resource_descriptor_heap();
    m_first_resource_heap_handle = resource_heap->reserve_views(2);
    if (!m_first_resource_heap_handle.is_valid())
        return false;

    // create a resource views for the env map
    if (!resource_heap->create_shader_resource_view(
        m_texture, Texture_dimension::Texture_2D, m_first_resource_heap_handle))
            return false;

    // create another view for the sampling buffer
    Descriptor_heap_handle second = m_first_resource_heap_handle.create_offset(1);
    if (!second.is_valid() ||
        !resource_heap->create_shader_resource_view(m_sampling_buffer, second))
            return false;

    // bind read-only data segment to shader register(t0, space1)
    m_resource_descriptor_table.register_srv(0, 1, m_first_resource_heap_handle);
    // note, passing the handle directly assumed a global root signature starting at the
    // beginning of the heap! TODO, make this more flexible

    // bind read-only data segment to shader register(t1, space1)
    m_resource_descriptor_table.register_srv(1, 1, second);

    return true;
}

}}} // mi::examples::mdl_d3d12
