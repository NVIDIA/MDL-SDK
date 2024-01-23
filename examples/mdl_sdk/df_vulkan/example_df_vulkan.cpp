/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/df_vulkan/example_df_vulkan.cpp
 //
 // Simple Vulkan renderer using compiled BSDFs with a material parameter editor GUI.

#include "example_shared.h"
#include "example_vulkan_shared.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <numeric>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cassert>

// Enable this to dump the generated GLSL code to stdout.
//#define DUMP_GLSL

static const VkFormat g_accumulation_texture_format = VK_FORMAT_R32G32B32A32_SFLOAT;

// Local group size for the path tracing compute shader
static const uint32_t g_local_size_x = 16;
static const uint32_t g_local_size_y = 8;

// Descriptor set bindings. Used as a define in the shaders.
static const uint32_t g_binding_beauty_buffer = 0;
static const uint32_t g_binding_aux_albedo_buffer = 1;
static const uint32_t g_binding_aux_normal_buffer = 2;
static const uint32_t g_binding_render_params = 3;
static const uint32_t g_binding_environment_map = 4;
static const uint32_t g_binding_environment_sampling_data = 5;
static const uint32_t g_binding_material_textures_indices = 6;
static const uint32_t g_binding_material_textures_2d = 7;
static const uint32_t g_binding_material_textures_3d = 8;
static const uint32_t g_binding_ro_data_buffer = 9;

static const uint32_t g_set_ro_data_buffer = 0;
static const uint32_t g_set_material_textures = 0;

// Command line options structure.
struct Options
{
    bool no_window = false;
    std::string output_file = "output.exr";
    uint32_t res_x = 1024;
    uint32_t res_y = 1024;
    uint32_t num_images = 3;
    uint32_t samples_per_pixel = 4096;
    uint32_t samples_per_iteration = 8;
    uint32_t max_path_length = 4;
    float cam_fov = 96.0f;
    mi::Float32_3 cam_pos = { 0.0f, 0.0f, 3.0f };
    mi::Float32_3 light_pos = { 10.0f, 0.0f, 5.0f };
    mi::Float32_3 light_intensity = { 0.0f, 0.0f, 0.0f };
    std::string hdr_file = "nvidia/sdk_examples/resources/environment.hdr";
    float hdr_intensity = 1.0f;
    bool use_class_compilation = false;
    std::string material_name = "::nvidia::sdk_examples::tutorials::example_df";
    bool enable_validation_layers = false;
};

struct Vulkan_texture
{
    VkImage image = nullptr;
    VkImageView image_view = nullptr;
    VkDeviceMemory device_memory = nullptr;

    void destroy(VkDevice device)
    {
        vkDestroyImageView(device, image_view, nullptr);
        vkDestroyImage(device, image, nullptr);
        vkFreeMemory(device, device_memory, nullptr);
    }
};

struct Vulkan_buffer
{
    VkBuffer buffer = nullptr;
    VkDeviceMemory device_memory = nullptr;

    void destroy(VkDevice device)
    {
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, device_memory, nullptr);
    }
};


//------------------------------------------------------------------------------
// MDL-Vulkan resource interop
//------------------------------------------------------------------------------

// Creates the storage buffer for the material's read-only data.
Vulkan_buffer create_ro_data_buffer(
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkQueue queue,
    VkCommandPool command_pool,
    const mi::neuraylib::ITarget_code* target_code)
{
    Vulkan_buffer ro_data_buffer;

    mi::Size num_segments = target_code->get_ro_data_segment_count();
    if (num_segments == 0)
        return ro_data_buffer;

    if (num_segments > 1)
    {
        std::cerr << "Multiple data segments (SSBOs) are defined for read-only data."
            << " This should not be the case if a storage buffer is used.\n";
        terminate();
    }

    { // Create the storage buffer in device local memory (VRAM)
        VkBufferCreateInfo buffer_create_info = {};
        buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_create_info.size = target_code->get_ro_data_segment_size(0);
        buffer_create_info.usage
            = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VK_CHECK(vkCreateBuffer(
            device, &buffer_create_info, nullptr, &ro_data_buffer.buffer));

        // Allocate device memory for the buffer.
        ro_data_buffer.device_memory = mi::examples::vk::allocate_and_bind_buffer_memory(
            device, physical_device, ro_data_buffer.buffer,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    {
        mi::examples::vk::Staging_buffer staging_buffer(device, physical_device,
            target_code->get_ro_data_segment_size(0), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        // Memcpy the read-only data into the staging buffer
        void* mapped_data = staging_buffer.map_memory();
        std::memcpy(mapped_data, target_code->get_ro_data_segment_data(0),
            target_code->get_ro_data_segment_size(0));
        staging_buffer.unmap_memory();

        // Upload the read-only data from the staging buffer into the storage buffer
        mi::examples::vk::Temporary_command_buffer command_buffer(device, command_pool);
        command_buffer.begin();

        VkBufferCopy copy_region = {};
        copy_region.size = target_code->get_ro_data_segment_size(0);

        vkCmdCopyBuffer(command_buffer.get(),
            staging_buffer.get(), ro_data_buffer.buffer, 1, &copy_region);

        command_buffer.end_and_submit(queue);
    }

    return ro_data_buffer;
}

// Creates the image and image view for the given texture index.
Vulkan_texture create_material_texture(
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkQueue queue,
    VkCommandPool command_pool,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IImage_api* image_api,
    const mi::neuraylib::ITarget_code* target_code,
    mi::Size texture_index)
{
    // Get access to the texture data by the texture database name from the target code.
    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        transaction->access<mi::neuraylib::ITexture>(target_code->get_texture(texture_index)));
    mi::base::Handle<const mi::neuraylib::IImage> image(
        transaction->access<mi::neuraylib::IImage>(texture->get_image())); 
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
    mi::Uint32 tex_width = canvas->get_resolution_x();
    mi::Uint32 tex_height = canvas->get_resolution_y();
    mi::Uint32 tex_layers = canvas->get_layers_size();
    char const* image_type = image->get_type(0, 0);

    if (image->is_uvtile() || image->is_animated())
    {
        std::cerr << "The example does not support uvtile and/or animated textures!" << std::endl;
        terminate();
    }

    // For simplicity, the texture access functions are only implemented for float4 and gamma
    // is pre-applied here (all images are converted to linear space).

    // Convert to linear color space if necessary
    if (texture->get_effective_gamma(0, 0) != 1.0f)
    {
        // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
        mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
            image_api->convert(canvas.get(), "Color"));
        gamma_canvas->set_gamma(texture->get_effective_gamma(0, 0));
        image_api->adjust_gamma(gamma_canvas.get(), 1.0f);
        canvas = gamma_canvas;
    }
    else if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0)
    {
        // Convert to expected format
        canvas = image_api->convert(canvas.get(), "Color");
    }

    // Create the Vulkan image
    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // This example supports only 2D and 3D textures (no PTEX or cube)
    mi::neuraylib::ITarget_code::Texture_shape texture_shape
        = target_code->get_texture_shape(texture_index);

    if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_2d)
    {
        image_create_info.imageType = VK_IMAGE_TYPE_2D;
        image_create_info.extent.width = tex_width;
        image_create_info.extent.height = tex_height;
        image_create_info.extent.depth = 1;
        image_create_info.arrayLayers = 1;
        image_create_info.mipLevels = 1;
    }
    else if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_3d
        || texture_shape == mi::neuraylib::ITarget_code::Texture_shape_bsdf_data)
    {
        image_create_info.imageType = VK_IMAGE_TYPE_3D;
        image_create_info.extent.width = tex_width;
        image_create_info.extent.height = tex_height;
        image_create_info.extent.depth = tex_layers;
        image_create_info.arrayLayers = 1;
        image_create_info.mipLevels = 1;
    }
    else
    {
        std::cerr << "Unsupported texture shape!" << std::endl;
        terminate();
    }

    Vulkan_texture material_texture;

    VK_CHECK(vkCreateImage(device, &image_create_info, nullptr,
        &material_texture.image));

    // Allocate device memory for the texture.
    material_texture.device_memory = mi::examples::vk::allocate_and_bind_image_memory(
        device, physical_device, material_texture.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    {
        size_t layer_size = tex_width * tex_height * sizeof(float) * 4; // RGBA32F
        size_t staging_buffer_size = layer_size * tex_layers;
        mi::examples::vk::Staging_buffer staging_buffer(device, physical_device,
            staging_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        // Memcpy the read-only data into the staging buffer
        uint8_t* mapped_data = static_cast<uint8_t*>(staging_buffer.map_memory());
        for (mi::Uint32 layer = 0; layer < tex_layers; layer++)
        {
            mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(layer));
            std::memcpy(mapped_data, tile->get_data(), layer_size);
            mapped_data += layer_size;
        }
        staging_buffer.unmap_memory();

        // Upload the read-only data from the staging buffer into the storage buffer
        mi::examples::vk::Temporary_command_buffer command_buffer(device, command_pool);
        command_buffer.begin();

        {
            VkImageMemoryBarrier image_memory_barrier = {};
            image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            image_memory_barrier.srcAccessMask = 0;
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_memory_barrier.image = material_texture.image;
            image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_memory_barrier.subresourceRange.baseMipLevel = 0;
            image_memory_barrier.subresourceRange.levelCount = 1;
            image_memory_barrier.subresourceRange.baseArrayLayer = 0;
            image_memory_barrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(command_buffer.get(),
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &image_memory_barrier);
        }

        VkBufferImageCopy copy_region = {};
        copy_region.bufferOffset = 0;
        copy_region.bufferRowLength = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.mipLevel = 0;
        copy_region.imageSubresource.baseArrayLayer = 0;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageOffset = { 0, 0, 0 };
        copy_region.imageExtent = { tex_width, tex_height, tex_layers };

        vkCmdCopyBufferToImage(command_buffer.get(), staging_buffer.get(),
            material_texture.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        {
            VkImageMemoryBarrier image_memory_barrier = {};
            image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_memory_barrier.image = material_texture.image;
            image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_memory_barrier.subresourceRange.baseMipLevel = 0;
            image_memory_barrier.subresourceRange.levelCount = 1;
            image_memory_barrier.subresourceRange.baseArrayLayer = 0;
            image_memory_barrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(command_buffer.get(),
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &image_memory_barrier);
        }

        command_buffer.end_and_submit(queue);
    }

    // Create the image view
    VkImageViewCreateInfo image_view_create_info = {};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.image = material_texture.image;
    image_view_create_info.viewType = (image_create_info.imageType == VK_IMAGE_TYPE_2D)
        ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D;
    image_view_create_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = 0;
    image_view_create_info.subresourceRange.levelCount = 1;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(
        device, &image_view_create_info, nullptr, &material_texture.image_view));

    return material_texture;
}


//------------------------------------------------------------------------------
// Application and rendering logic
//------------------------------------------------------------------------------
class Df_vulkan_app : public mi::examples::vk::Vulkan_example_app
{
public:
    Df_vulkan_app(
        mi::base::Handle<mi::neuraylib::ITransaction> transaction,
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api,
        mi::base::Handle<mi::neuraylib::IImage_api> image_api,
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code,
        const Options& options)
    : Vulkan_example_app(mdl_impexp_api.get(), image_api.get())
    , m_transaction(transaction)
    , m_target_code(target_code)
    , m_options(options)
    {
    }

    virtual void init_resources() override;
    virtual void cleanup_resources() override;

    // All framebuffer size dependent resources need to be recreated
    // when the swapchain is recreated due to not being optimal anymore
    // or because the window was resized.
    virtual void recreate_size_dependent_resources() override;

    // Updates the application logic. This is called right before the
    // next frame is rendered.
    virtual void update(float elapsed_seconds, uint32_t image_index) override;

    // Populates the current frame's command buffer. The base application's
    // render pass has already been started at this point.
    virtual void render(VkCommandBuffer command_buffer, uint32_t image_index) override;

    // Window event handlers.
    virtual void key_callback(int key, int action) override;
    virtual void mouse_button_callback(int button, int action) override;
    virtual void mouse_scroll_callback(float offset_x, float offset_y) override;
    virtual void mouse_move_callback(float pos_x, float pos_y) override;
    virtual void resized_callback(uint32_t width, uint32_t height) override;

private:
    struct Camera_state
    {
        float base_distance;
        float theta;
        float phi;
        float zoom;
    };

    struct Render_params
    {
        alignas(16) mi::Float32_3 cam_pos;
        alignas(16) mi::Float32_3 cam_dir;
        alignas(16) mi::Float32_3 cam_right;
        alignas(16) mi::Float32_3 cam_up;
        float cam_focal;
        alignas(16) mi::Float32_3 point_light_pos;
        alignas(16) mi::Float32_3 point_light_color;
        float point_light_intensity;
        float environment_intensity_factor;
        float environment_inv_integral;
        uint32_t max_path_length;
        uint32_t samples_per_iteration;
        uint32_t progressive_iteration;
    };

private:
    void update_camera_render_params(const Camera_state& cam_state);

    void create_material_textures_index_buffer(const std::vector<uint32_t>& indices);

    void create_accumulation_images();

    VkShaderModule create_path_trace_shader_module();

    // Creates the descriptors set layout which is used to create the
    // pipeline layout. Here the number of material resources is declared.
    void create_descriptor_set_layouts();

    // Create the pipeline layout and state for rendering a fullscreen triangle.
    void create_pipeline_layouts();
    void create_pipelines();

    void create_render_params_buffers();

    void create_environment_map();

    // Creates the descriptor pool and set that hold enough space for all
    // material resources, and are used during rendering to access the
    // the resources.
    void create_descriptor_pool_and_sets();

    void update_accumulation_image_descriptors();

private:
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<const mi::neuraylib::ITarget_code> m_target_code;
    Options m_options;

    Vulkan_texture m_beauty_texture;
    Vulkan_texture m_auxiliary_albedo_texture;
    Vulkan_texture m_auxiliary_normal_texture;
    VkSampler m_linear_sampler = nullptr;

    VkRenderPass m_path_trace_render_pass = nullptr;
    VkPipelineLayout m_path_trace_pipeline_layout = nullptr;
    VkPipelineLayout m_display_pipeline_layout = nullptr;
    VkPipeline m_path_trace_pipeline = nullptr;
    VkPipeline m_display_pipeline = nullptr;

    VkDescriptorSetLayout m_path_trace_descriptor_set_layout = nullptr;
    VkDescriptorSetLayout m_display_descriptor_set_layout = nullptr;
    VkDescriptorPool m_descriptor_pool = nullptr;
    std::vector<VkDescriptorSet> m_path_trace_descriptor_sets;
    VkDescriptorSet m_display_descriptor_set;
    std::vector<Vulkan_buffer> m_render_params_buffers;
    std::vector<void*> m_render_params_buffer_data_ptrs;

    Vulkan_texture m_environment_map;
    Vulkan_buffer m_environment_sampling_data_buffer;
    VkSampler m_environment_sampler;

    // Material resources
    Vulkan_buffer m_ro_data_buffer;
    Vulkan_buffer m_material_textures_index_buffer;
    std::vector<Vulkan_texture> m_material_textures_2d;
    std::vector<Vulkan_texture> m_material_textures_3d;

    Render_params m_render_params;
    bool m_camera_moved = true; // Force a clear in first frame
    uint32_t m_display_buffer_index = 0; // Which buffer to display

    // Camera movement
    Camera_state m_camera_state;
    mi::Float32_2 m_mouse_start;
    bool m_camera_moving = false;
};

void Df_vulkan_app::init_resources()
{
    glslang::InitializeProcess();

    m_linear_sampler = mi::examples::vk::create_linear_sampler(m_device);
    
    // Create the render resources for the material
    m_ro_data_buffer = create_ro_data_buffer(m_device, m_physical_device,
        m_graphics_queue, m_command_pool, m_target_code.get());

    // Record the indices of each texture in their respective array
    // e.g. the indices of 2D textures in the m_material_textures_2d array
    std::vector<uint32_t> material_textures_indices;

    // Create the textures for the material
    if (m_target_code->get_texture_count() > 0)
    {
        // The first texture (index = 0) is always the invalid texture in MDL
        material_textures_indices.reserve(m_target_code->get_texture_count() - 1);

        for (mi::Size i = 1; i < m_target_code->get_texture_count(); i++)
        {
            Vulkan_texture texture = create_material_texture(
                m_device, m_physical_device, m_graphics_queue, m_command_pool,
                m_transaction.get(), m_image_api.get(), m_target_code.get(), i);

            switch (m_target_code->get_texture_shape(i))
            {
            case mi::neuraylib::ITarget_code::Texture_shape_2d:
                material_textures_indices.push_back(static_cast<uint32_t>(m_material_textures_2d.size()));
                m_material_textures_2d.push_back(texture);
                break;

            case mi::neuraylib::ITarget_code::Texture_shape_3d:
            case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data:
                material_textures_indices.push_back(static_cast<uint32_t>(m_material_textures_3d.size()));
                m_material_textures_3d.push_back(texture);
                break;

            default:
                std::cerr << "Unsupported texture shape!" << std::endl;
                terminate();
                break;
            }
        }
    }

    create_material_textures_index_buffer(material_textures_indices);

    create_descriptor_set_layouts();
    create_pipeline_layouts();
    create_accumulation_images();
    create_pipelines();
    create_render_params_buffers();
    create_environment_map();
    create_descriptor_pool_and_sets();

    // Initialize render parameters
    m_render_params.progressive_iteration = 0;
    m_render_params.max_path_length = m_options.max_path_length;
    m_render_params.samples_per_iteration = m_options.samples_per_iteration;

    m_render_params.point_light_pos = m_options.light_pos;
    m_render_params.point_light_intensity
        = std::max(std::max(m_options.light_intensity.x, m_options.light_intensity.y), m_options.light_intensity.z);
    m_render_params.point_light_color = m_render_params.point_light_intensity > 0.0f
        ? m_options.light_intensity / m_render_params.point_light_intensity
        : mi::Float32_3(0.0f, 0.0f, 0.0f);
    m_render_params.environment_intensity_factor = m_options.hdr_intensity;

    const float fov = m_options.cam_fov;
    const float to_radians = static_cast<float>(M_PI / 180.0);
    m_render_params.cam_focal = 1.0f / mi::math::tan(fov / 2.0f * to_radians);

    // Setup camera
    const mi::Float32_3 camera_pos = m_options.cam_pos;
    mi::Float32_3 inv_dir = camera_pos / mi::math::length(camera_pos);
    m_camera_state.base_distance = mi::math::length(camera_pos);
    m_camera_state.phi = mi::math::atan2(inv_dir.x, inv_dir.z);
    m_camera_state.theta = mi::math::acos(inv_dir.y);
    m_camera_state.zoom = 0;

    update_camera_render_params(m_camera_state);
}

void Df_vulkan_app::cleanup_resources()
{
    // In headless mode we output the accumulation buffers to files
    if (m_options.no_window)
    {
        std::string filename_base = m_options.output_file;
        std::string filename_ext;

        size_t dot_pos = m_options.output_file.rfind('.');
        if (dot_pos != std::string::npos) {
            filename_base = m_options.output_file.substr(0, dot_pos);
            filename_ext = m_options.output_file.substr(dot_pos);
        }

        VkImage output_images[] = {
            m_beauty_texture.image,
            m_auxiliary_albedo_texture.image,
            m_auxiliary_normal_texture.image
        };
        std::string output_filenames[] = {
            filename_base + filename_ext,
            filename_base + "_albedo" + filename_ext,
            filename_base + "_normal" + filename_ext
        };

        for (uint32_t i = 0; i < 3; i++)
        {
            uint32_t bpp = mi::examples::vk::get_image_format_bpp(g_accumulation_texture_format);
            std::vector<uint8_t> pixels = mi::examples::vk::copy_image_to_buffer(
                m_device, m_physical_device, m_command_pool, m_graphics_queue,
                output_images[i], m_image_width, m_image_height, bpp,
                VK_IMAGE_LAYOUT_GENERAL, false);

            mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                m_image_api->create_canvas("Color", m_image_width, m_image_height));
            mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile());
            std::memcpy(tile->get_data(), pixels.data(), pixels.size());
            canvas = m_image_api->convert(canvas.get(), "Rgb_fp");
            m_mdl_impexp_api->export_canvas(output_filenames[i].c_str(), canvas.get());
        }
    }

    // Cleanup resources
    m_material_textures_index_buffer.destroy(m_device);
    for (Vulkan_texture& texture : m_material_textures_3d)
        texture.destroy(m_device);
    for (Vulkan_texture& texture : m_material_textures_2d)
        texture.destroy(m_device);
    m_ro_data_buffer.destroy(m_device);

    m_environment_sampling_data_buffer.destroy(m_device);
    m_environment_map.destroy(m_device);

    for (Vulkan_buffer& buffer : m_render_params_buffers)
    {
        vkUnmapMemory(m_device, buffer.device_memory);
        buffer.destroy(m_device);
    }

    vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_display_descriptor_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_path_trace_descriptor_set_layout, nullptr);
    vkDestroyPipelineLayout(m_device, m_display_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(m_device, m_path_trace_pipeline_layout, nullptr);
    vkDestroyPipeline(m_device, m_path_trace_pipeline, nullptr);
    vkDestroyPipeline(m_device, m_display_pipeline, nullptr);
    vkDestroyRenderPass(m_device, m_path_trace_render_pass, nullptr);
    vkDestroySampler(m_device, m_linear_sampler, nullptr);

    m_auxiliary_normal_texture.destroy(m_device);
    m_auxiliary_albedo_texture.destroy(m_device);
    m_beauty_texture.destroy(m_device);
    vkDestroySampler(m_device, m_environment_sampler, nullptr);

    glslang::FinalizeProcess();
}

// All framebuffer size dependent resources need to be recreated
// when the swapchain is recreated due to not being optimal anymore
// or because the window was resized.
void Df_vulkan_app::recreate_size_dependent_resources()
{
    vkDestroyPipeline(m_device, m_path_trace_pipeline, nullptr);
    vkDestroyPipeline(m_device, m_display_pipeline, nullptr);
    vkDestroyRenderPass(m_device, m_path_trace_render_pass, nullptr);

    m_auxiliary_normal_texture.destroy(m_device);
    m_auxiliary_albedo_texture.destroy(m_device);
    m_beauty_texture.destroy(m_device);

    create_accumulation_images();
    create_pipelines();

    update_accumulation_image_descriptors();
}

// Updates the application logic. This is called right before the
// next frame is rendered.
void Df_vulkan_app::update(float elapsed_seconds, uint32_t image_index)
{
    if (m_camera_moved)
        m_render_params.progressive_iteration = 0;

    // Update current frame's render params uniform buffer
    std::memcpy(m_render_params_buffer_data_ptrs[image_index],
        &m_render_params, sizeof(Render_params));

    m_render_params.progressive_iteration += m_options.samples_per_iteration;

    if (!m_options.no_window)
    {
        std::string window_title = "MDL SDK DF Vulkan Example | Press keys 1 - 3 for output buffers | Iteration: ";
        window_title += std::to_string(m_render_params.progressive_iteration);
        glfwSetWindowTitle(m_window, window_title.c_str());
    }
}

// Populates the current frame's command buffer. The base application's
// render pass has already been started at this point.
void Df_vulkan_app::render(VkCommandBuffer command_buffer, uint32_t image_index)
{
    const VkImage accum_images[] = {
        m_beauty_texture.image,
        m_auxiliary_albedo_texture.image,
        m_auxiliary_normal_texture.image
    };

    // Path trace compute pass
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_path_trace_pipeline);

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_path_trace_pipeline_layout, 0, 1, &m_path_trace_descriptor_sets[image_index],
        0, nullptr);

    if (m_camera_moved)
    {
        m_camera_moved = false;

        for (VkImage image : accum_images)
        {
            VkClearColorValue clear_color = { 0.0f, 0.0f, 0.0f, 0.0f };

            VkImageSubresourceRange clear_range;
            clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            clear_range.baseMipLevel = 0;
            clear_range.levelCount = 1;
            clear_range.baseArrayLayer = 0;
            clear_range.layerCount = 1;

            vkCmdClearColorImage(command_buffer,
                image, VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &clear_range);
        }
    }

    uint32_t group_count_x = (m_image_width + g_local_size_x - 1) / g_local_size_x;
    uint32_t group_count_y = (m_image_width + g_local_size_y - 1) / g_local_size_y;
    vkCmdDispatch(command_buffer, group_count_x, group_count_y, 1);

    for (VkImage image : accum_images)
    {
        VkImageMemoryBarrier image_memory_barrier = {};
        image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        image_memory_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barrier.image = image;
        image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_memory_barrier.subresourceRange.baseMipLevel = 0;
        image_memory_barrier.subresourceRange.levelCount = 1;
        image_memory_barrier.subresourceRange.baseArrayLayer = 0;
        image_memory_barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &image_memory_barrier);
    }

    // Display render pass
    VkRenderPassBeginInfo render_pass_begin_info = {};
    render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_begin_info.renderPass = m_main_render_pass;
    render_pass_begin_info.framebuffer = m_framebuffers[image_index];
    render_pass_begin_info.renderArea = { {0, 0}, {m_image_width, m_image_height} };

    VkClearValue clear_values[2];
    clear_values[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
    clear_values[1].depthStencil = { 1.0f, 0 };
    render_pass_begin_info.clearValueCount = std::size(clear_values);
    render_pass_begin_info.pClearValues = clear_values;

    vkCmdBeginRenderPass(
        command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(
        command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_display_pipeline);

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_display_pipeline_layout, 0, 1, &m_display_descriptor_set, 0, nullptr);

    vkCmdPushConstants(command_buffer, m_display_pipeline_layout,
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(uint32_t), &m_display_buffer_index);

    vkCmdDraw(command_buffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(command_buffer);
}

// Handles keyboard input from the window.
void Df_vulkan_app::key_callback(int key, int action)
{
    // Handle only key press events
    if (action != GLFW_PRESS)
        return;

    if (key == GLFW_KEY_ENTER)
        request_screenshot();

    if (key == GLFW_KEY_SPACE)
        m_camera_moved = true;

    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_3)
        m_display_buffer_index = key - GLFW_KEY_1;
}

void Df_vulkan_app::mouse_button_callback(int button, int action)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        m_camera_moving = (action == GLFW_PRESS);

        double mouse_x, mouse_y;
        glfwGetCursorPos(m_window, &mouse_x, &mouse_y);
        m_mouse_start.x = static_cast<float>(mouse_x);
        m_mouse_start.y = static_cast<float>(mouse_y);
    }
}

void Df_vulkan_app::mouse_scroll_callback(float offset_x, float offset_y)
{
    if (offset_y < 0.0f)
        m_camera_state.zoom -= 1.0f;
    else if (offset_y > 0.0f)
        m_camera_state.zoom += 1.0f;

    update_camera_render_params(m_camera_state);
    m_camera_moved = true;
}

void Df_vulkan_app::mouse_move_callback(float pos_x, float pos_y)
{
    if (m_camera_moving)
    {
        float dx = pos_x - m_mouse_start.x;
        float dy = pos_y - m_mouse_start.y;
        m_mouse_start.x = pos_x;
        m_mouse_start.y = pos_y;

        m_camera_state.phi -= static_cast<float>(dx * 0.001f * M_PI);
        m_camera_state.theta -= static_cast<float>(dy * 0.001f * M_PI);
        m_camera_state.theta = mi::math::max(
            m_camera_state.theta, static_cast<float>(0.0f * M_PI));
        m_camera_state.theta = mi::math::min(
            m_camera_state.theta, static_cast<float>(1.0f * M_PI));

        update_camera_render_params(m_camera_state);
        m_camera_moved = true;
    }
}

// Gets called when the window is resized.
void Df_vulkan_app::resized_callback(uint32_t width, uint32_t height)
{
    m_camera_moved = true;
}

void Df_vulkan_app::update_camera_render_params(const Camera_state& cam_state)
{
    m_render_params.cam_dir.x = -mi::math::sin(cam_state.phi) * mi::math::sin(cam_state.theta);
    m_render_params.cam_dir.y = -mi::math::cos(cam_state.theta);
    m_render_params.cam_dir.z = -mi::math::cos(cam_state.phi) * mi::math::sin(cam_state.theta);

    m_render_params.cam_right.x = mi::math::cos(cam_state.phi);
    m_render_params.cam_right.y = 0.0f;
    m_render_params.cam_right.z = -mi::math::sin(cam_state.phi);

    m_render_params.cam_up.x = -mi::math::sin(cam_state.phi) * mi::math::cos(cam_state.theta);
    m_render_params.cam_up.y = mi::math::sin(cam_state.theta);
    m_render_params.cam_up.z = -mi::math::cos(cam_state.phi) * mi::math::cos(cam_state.theta);

    const float dist = cam_state.base_distance * mi::math::pow(0.95f, cam_state.zoom);
    m_render_params.cam_pos.x = -m_render_params.cam_dir.x * dist;
    m_render_params.cam_pos.y = -m_render_params.cam_dir.y * dist;
    m_render_params.cam_pos.z = -m_render_params.cam_dir.z * dist;
}

void Df_vulkan_app::create_material_textures_index_buffer(const std::vector<uint32_t>& indices)
{
    if (indices.empty())
        return;

    // The uniform buffer has std140 layout which means each array entry must be the size of a vec4 (16 byte)
    std::vector<uint32_t> buffer_data(indices.size() * 4);
    for (size_t i = 0; i < indices.size(); i++)
        buffer_data[i * 4] = indices[i];

    const size_t num_buffer_data_bytes = buffer_data.size() * sizeof(uint32_t);

    { // Create the uniform buffer in device local memory (VRAM)
        VkBufferCreateInfo buffer_create_info = {};
        buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_create_info.size = num_buffer_data_bytes;
        buffer_create_info.usage
            = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VK_CHECK(vkCreateBuffer(
            m_device, &buffer_create_info, nullptr, &m_material_textures_index_buffer.buffer));

        // Allocate device memory for the buffer.
        m_material_textures_index_buffer.device_memory = mi::examples::vk::allocate_and_bind_buffer_memory(
            m_device, m_physical_device, m_material_textures_index_buffer.buffer,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    {
        mi::examples::vk::Staging_buffer staging_buffer(m_device, m_physical_device,
            num_buffer_data_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        // Memcpy the read-only data into the staging buffer
        void* mapped_data = staging_buffer.map_memory();
        std::memcpy(mapped_data, buffer_data.data(), num_buffer_data_bytes);
        staging_buffer.unmap_memory();

        // Upload the read-only data from the staging buffer into the storage buffer
        mi::examples::vk::Temporary_command_buffer command_buffer(m_device, m_command_pool);
        command_buffer.begin();

        VkBufferCopy copy_region = {};
        copy_region.size = num_buffer_data_bytes;

        vkCmdCopyBuffer(command_buffer.get(),
            staging_buffer.get(), m_material_textures_index_buffer.buffer, 1, &copy_region);

        command_buffer.end_and_submit(m_graphics_queue);
    }
}

void Df_vulkan_app::create_accumulation_images()
{
    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = g_accumulation_texture_format;
    image_create_info.extent.width = m_image_width;
    image_create_info.extent.height = m_image_height;
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = VK_IMAGE_USAGE_STORAGE_BIT
        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(m_device, &image_create_info, nullptr, &m_beauty_texture.image));
    VK_CHECK(vkCreateImage(m_device, &image_create_info, nullptr, &m_auxiliary_albedo_texture.image));
    VK_CHECK(vkCreateImage(m_device, &image_create_info, nullptr, &m_auxiliary_normal_texture.image));

    m_beauty_texture.device_memory = mi::examples::vk::allocate_and_bind_image_memory(
        m_device, m_physical_device, m_beauty_texture.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_auxiliary_albedo_texture.device_memory = mi::examples::vk::allocate_and_bind_image_memory(
        m_device, m_physical_device, m_auxiliary_albedo_texture.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_auxiliary_normal_texture.device_memory = mi::examples::vk::allocate_and_bind_image_memory(
        m_device, m_physical_device, m_auxiliary_normal_texture.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    { // Transition image layout
        mi::examples::vk::Temporary_command_buffer command_buffer(m_device, m_command_pool);
        command_buffer.begin();

        VkImageMemoryBarrier image_memory_barrier = {};
        image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        image_memory_barrier.srcAccessMask = 0;
        image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_memory_barrier.subresourceRange.baseMipLevel = 0;
        image_memory_barrier.subresourceRange.levelCount = 1;
        image_memory_barrier.subresourceRange.baseArrayLayer = 0;
        image_memory_barrier.subresourceRange.layerCount = 1;

        const VkImage images_to_transition[] = {
            m_beauty_texture.image,
            m_auxiliary_albedo_texture.image,
            m_auxiliary_normal_texture.image
        };

        for (VkImage image : images_to_transition)
        {
            image_memory_barrier.image = image;

            vkCmdPipelineBarrier(command_buffer.get(),
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &image_memory_barrier);
        }

        command_buffer.end_and_submit(m_graphics_queue);
    }

    VkImageViewCreateInfo image_view_create_info = {};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = g_accumulation_texture_format;
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = 0;
    image_view_create_info.subresourceRange.levelCount = 1;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount = 1;
    
    image_view_create_info.image = m_beauty_texture.image;
    VK_CHECK(vkCreateImageView(
        m_device, &image_view_create_info, nullptr, &m_beauty_texture.image_view));

    image_view_create_info.image = m_auxiliary_albedo_texture.image;
    VK_CHECK(vkCreateImageView(
        m_device, &image_view_create_info, nullptr, &m_auxiliary_albedo_texture.image_view));

    image_view_create_info.image = m_auxiliary_normal_texture.image;
    VK_CHECK(vkCreateImageView(
        m_device, &image_view_create_info, nullptr, &m_auxiliary_normal_texture.image_view));
}

VkShaderModule Df_vulkan_app::create_path_trace_shader_module()
{
    std::string df_glsl_source = m_target_code->get_code();

    std::string path_trace_shader_source = mi::examples::io::read_text_file(
        mi::examples::io::get_executable_folder() + "/" + "path_trace.comp");

    std::vector<std::string> defines;
    defines.push_back("LOCAL_SIZE_X=" + std::to_string(g_local_size_x));
    defines.push_back("LOCAL_SIZE_Y=" + std::to_string(g_local_size_y));

    defines.push_back("BINDING_RENDER_PARAMS=" + std::to_string(g_binding_render_params));
    defines.push_back("BINDING_ENV_MAP=" + std::to_string(g_binding_environment_map));
    defines.push_back("BINDING_ENV_MAP_SAMPLING_DATA=" + std::to_string(g_binding_environment_sampling_data));
    defines.push_back("BINDING_BEAUTY_BUFFER=" + std::to_string(g_binding_beauty_buffer));
    defines.push_back("BINDING_AUX_ALBEDO_BUFFER=" + std::to_string(g_binding_aux_albedo_buffer));
    defines.push_back("BINDING_AUX_NORMAL_BUFFER=" + std::to_string(g_binding_aux_normal_buffer));

    defines.push_back("NUM_MATERIAL_TEXTURES_2D=" + std::to_string(m_material_textures_2d.size()));
    defines.push_back("NUM_MATERIAL_TEXTURES_3D=" + std::to_string(m_material_textures_3d.size()));
    defines.push_back("SET_MATERIAL_TEXTURES_INDICES=" + std::to_string(g_set_material_textures));
    defines.push_back("SET_MATERIAL_TEXTURES_2D=" + std::to_string(g_set_material_textures));
    defines.push_back("SET_MATERIAL_TEXTURES_3D=" + std::to_string(g_set_material_textures));
    defines.push_back("BINDING_MATERIAL_TEXTURES_INDICES=" + std::to_string(g_binding_material_textures_indices));
    defines.push_back("BINDING_MATERIAL_TEXTURES_2D=" + std::to_string(g_binding_material_textures_2d));
    defines.push_back("BINDING_MATERIAL_TEXTURES_3D=" + std::to_string(g_binding_material_textures_3d));

    // Check if functions for backface were generated
    for (mi::Size i = 0; i < m_target_code->get_callable_function_count(); i++)
    {
        const char* fname = m_target_code->get_callable_function(i);

        if (std::strcmp(fname, "mdl_backface_bsdf_sample") == 0)
            defines.push_back("BACKFACE_BSDF");
        else if (std::strcmp(fname, "mdl_backface_edf_sample") == 0)
            defines.push_back("BACKFACE_EDF");
        else if (std::strcmp(fname, "mdl_backface_emission_intensity") == 0)
            defines.push_back("BACKFACE_EMISSION_INTENSITY");
    }

    return mi::examples::vk::create_shader_module_from_sources(m_device,
        { df_glsl_source, path_trace_shader_source }, EShLangCompute, defines);
}

// Creates the descriptors set layout which is used to create the
// pipeline layout. Here the number of material resources is declared.
void Df_vulkan_app::create_descriptor_set_layouts()
{
    {
        VkDescriptorSetLayoutBinding render_params_layout_binding = {};
        render_params_layout_binding.binding = g_binding_render_params;
        render_params_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        render_params_layout_binding.descriptorCount = 1;
        render_params_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding env_map_layout_binding = {};
        env_map_layout_binding.binding = g_binding_environment_map;
        env_map_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        env_map_layout_binding.descriptorCount = 1;
        env_map_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding env_sampling_data_layout_binding = {};
        env_sampling_data_layout_binding.binding = g_binding_environment_sampling_data;
        env_sampling_data_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        env_sampling_data_layout_binding.descriptorCount = 1;
        env_sampling_data_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding textures_indices_layout_binding = {};
        textures_indices_layout_binding.binding = g_binding_material_textures_indices;
        textures_indices_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        textures_indices_layout_binding.descriptorCount = 1;
        textures_indices_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding textures_2d_layout_binding = {};
        textures_2d_layout_binding.binding = g_binding_material_textures_2d;
        textures_2d_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textures_2d_layout_binding.descriptorCount
            = static_cast<uint32_t>(m_material_textures_2d.size());
        textures_2d_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding textures_3d_layout_binding = {};
        textures_3d_layout_binding.binding = g_binding_material_textures_3d;
        textures_3d_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        textures_3d_layout_binding.descriptorCount
            = static_cast<uint32_t>(m_material_textures_3d.size());
        textures_3d_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding beauty_buffer_layout_binding = {};
        beauty_buffer_layout_binding.binding = g_binding_beauty_buffer;
        beauty_buffer_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        beauty_buffer_layout_binding.descriptorCount = 1;
        beauty_buffer_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding aux_albedo_buffer_layout_binding
            = beauty_buffer_layout_binding;
        aux_albedo_buffer_layout_binding.binding = g_binding_aux_albedo_buffer;

        VkDescriptorSetLayoutBinding aux_albedo_normal_layout_binding
            = beauty_buffer_layout_binding;
        aux_albedo_normal_layout_binding.binding = g_binding_aux_normal_buffer;

        VkDescriptorSetLayoutBinding ro_data_buffer_layout_binding = {};
        ro_data_buffer_layout_binding.binding = g_binding_ro_data_buffer;
        ro_data_buffer_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ro_data_buffer_layout_binding.descriptorCount = 1;
        ro_data_buffer_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const VkDescriptorSetLayoutBinding bindings[] = {
            render_params_layout_binding,
            env_map_layout_binding,
            env_sampling_data_layout_binding,
            textures_indices_layout_binding,
            textures_2d_layout_binding,
            textures_3d_layout_binding,
            beauty_buffer_layout_binding,
            aux_albedo_buffer_layout_binding,
            aux_albedo_normal_layout_binding,
            ro_data_buffer_layout_binding
        };

        VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
        descriptor_set_layout_create_info.sType
            = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_set_layout_create_info.bindingCount = std::size(bindings);
        descriptor_set_layout_create_info.pBindings = bindings;

        VK_CHECK(vkCreateDescriptorSetLayout(
            m_device, &descriptor_set_layout_create_info, nullptr, &m_path_trace_descriptor_set_layout));
    }

    {
        VkDescriptorSetLayoutBinding layout_bindings[3];
        for (uint32_t i = 0; i < 3; i++)
        {
            layout_bindings[i].binding = i;
            layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            layout_bindings[i].descriptorCount = 1;
            layout_bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            layout_bindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
        descriptor_set_layout_create_info.sType
            = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_set_layout_create_info.bindingCount = std::size(layout_bindings);
        descriptor_set_layout_create_info.pBindings = layout_bindings;

        VK_CHECK(vkCreateDescriptorSetLayout(
            m_device, &descriptor_set_layout_create_info, nullptr, &m_display_descriptor_set_layout));
    }
}

void Df_vulkan_app::create_pipeline_layouts()
{
    { // Path trace compute pipeline layout
        VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
        pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_create_info.setLayoutCount = 1;
        pipeline_layout_create_info.pSetLayouts = &m_path_trace_descriptor_set_layout;
        pipeline_layout_create_info.pushConstantRangeCount = 0;
        pipeline_layout_create_info.pPushConstantRanges = nullptr;

        VK_CHECK(vkCreatePipelineLayout(
            m_device, &pipeline_layout_create_info, nullptr, &m_path_trace_pipeline_layout));
    }

    { // Display graphics pipeline layout
        VkPushConstantRange push_constant_range;
        push_constant_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
        pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_create_info.setLayoutCount = 1;
        pipeline_layout_create_info.pSetLayouts = &m_display_descriptor_set_layout;
        pipeline_layout_create_info.pushConstantRangeCount = 1;
        pipeline_layout_create_info.pPushConstantRanges = &push_constant_range;

        VK_CHECK(vkCreatePipelineLayout(
            m_device, &pipeline_layout_create_info, nullptr, &m_display_pipeline_layout));
    }
}

void Df_vulkan_app::create_pipelines()
{
    { // Create path trace compute pipeline
        VkShaderModule path_trace_compute_shader = create_path_trace_shader_module();

        VkPipelineShaderStageCreateInfo compute_shader_stage = {};
        compute_shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compute_shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        compute_shader_stage.module = path_trace_compute_shader;
        compute_shader_stage.pName = "main";

        VkComputePipelineCreateInfo pipeline_create_info = {};
        pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_create_info.stage = compute_shader_stage;
        pipeline_create_info.layout = m_path_trace_pipeline_layout;

        vkCreateComputePipelines(
            m_device, nullptr, 1, &pipeline_create_info, nullptr, &m_path_trace_pipeline);

        vkDestroyShaderModule(m_device, path_trace_compute_shader, nullptr);
    }

    { // Create display graphics pipeline
        VkShaderModule fullscreen_triangle_vertex_shader
            = mi::examples::vk::create_shader_module_from_file(
                m_device, "display.vert", EShLangVertex);
        VkShaderModule display_fragment_shader
            = mi::examples::vk::create_shader_module_from_file(
                m_device, "display.frag", EShLangFragment);

        VkPipelineVertexInputStateCreateInfo vertex_input_state = {};
        vertex_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {};
        input_assembly_state.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly_state.primitiveRestartEnable = false;

        VkViewport viewport = { 0.0f, 0.0f, (float)m_image_width, (float)m_image_height, 0.0f, 1.0f };
        VkRect2D scissor_rect = { {0, 0}, {m_image_width, m_image_height} };

        VkPipelineViewportStateCreateInfo viewport_state = {};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor_rect;

        VkPipelineRasterizationStateCreateInfo rasterization_state = {};
        rasterization_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization_state.depthClampEnable = false;
        rasterization_state.rasterizerDiscardEnable = false;
        rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization_state.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterization_state.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterization_state.depthBiasEnable = false;
        rasterization_state.lineWidth = 1.0f;

        VkPipelineDepthStencilStateCreateInfo depth_stencil_state = {};
        depth_stencil_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_stencil_state.depthTestEnable = false;
        depth_stencil_state.depthWriteEnable = false;
        depth_stencil_state.depthBoundsTestEnable = false;
        depth_stencil_state.stencilTestEnable = false;

        VkPipelineMultisampleStateCreateInfo multisample_state = {};
        multisample_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineShaderStageCreateInfo vertex_shader_stage_info = {};
        vertex_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertex_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertex_shader_stage_info.module = fullscreen_triangle_vertex_shader;
        vertex_shader_stage_info.pName = "main";

        VkPipelineShaderStageCreateInfo fragment_shader_stage_info = {};
        fragment_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragment_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragment_shader_stage_info.module = display_fragment_shader;
        fragment_shader_stage_info.pName = "main";

        const VkPipelineShaderStageCreateInfo shader_stages[] = {
            vertex_shader_stage_info,
            fragment_shader_stage_info
        };

        VkPipelineColorBlendAttachmentState color_blend_attachment = {};
        color_blend_attachment.blendEnable = false;
        color_blend_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo color_blend_state = {};
        color_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend_state.logicOpEnable = false;
        color_blend_state.attachmentCount = 1;
        color_blend_state.pAttachments = &color_blend_attachment;

        VkGraphicsPipelineCreateInfo pipeline_create_info = {};
        pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_create_info.stageCount = std::size(shader_stages);
        pipeline_create_info.pStages = shader_stages;
        pipeline_create_info.pVertexInputState = &vertex_input_state;
        pipeline_create_info.pInputAssemblyState = &input_assembly_state;
        pipeline_create_info.pViewportState = &viewport_state;
        pipeline_create_info.pRasterizationState = &rasterization_state;
        pipeline_create_info.pDepthStencilState = &depth_stencil_state;
        pipeline_create_info.pMultisampleState = &multisample_state;
        pipeline_create_info.pColorBlendState = &color_blend_state;
        pipeline_create_info.layout = m_display_pipeline_layout;
        pipeline_create_info.renderPass = m_main_render_pass;
        pipeline_create_info.subpass = 0;

        VK_CHECK(vkCreateGraphicsPipelines(
            m_device, nullptr, 1, &pipeline_create_info, nullptr, &m_display_pipeline));

        vkDestroyShaderModule(m_device, fullscreen_triangle_vertex_shader, nullptr);
        vkDestroyShaderModule(m_device, display_fragment_shader, nullptr);
    }
}

void Df_vulkan_app::create_render_params_buffers()
{
    m_render_params_buffers.resize(m_image_count);
    m_render_params_buffer_data_ptrs.resize(m_image_count);

    for (uint32_t i = 0; i < m_image_count; i++)
    {
        VkBufferCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = sizeof(Render_params);
        create_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK(vkCreateBuffer(
            m_device, &create_info, nullptr, &m_render_params_buffers[i].buffer));

        m_render_params_buffers[i].device_memory = mi::examples::vk::allocate_and_bind_buffer_memory(
            m_device, m_physical_device, m_render_params_buffers[i].buffer,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VK_CHECK(vkMapMemory(m_device, m_render_params_buffers[i].device_memory,
            0, sizeof(Render_params), 0, &m_render_params_buffer_data_ptrs[i]));
    }
}

void Df_vulkan_app::create_environment_map()
{
    // Load environment texture
    mi::base::Handle<mi::neuraylib::IImage> image(
        m_transaction->create<mi::neuraylib::IImage>("Image"));
    check_success(image->reset_file(m_options.hdr_file.c_str()) == 0);

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
    const mi::Uint32 res_x = canvas->get_resolution_x();
    const mi::Uint32 res_y = canvas->get_resolution_y();

    // Check, whether we need to convert the image
    char const* image_type = image->get_type(0, 0);
    if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0)
        canvas = m_image_api->convert(canvas.get(), "Color");

    // Create the Vulkan texture
    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_create_info.extent = { res_x, res_y, 1 };
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(m_device, &image_create_info, nullptr, &m_environment_map.image));

    m_environment_map.device_memory = mi::examples::vk::allocate_and_bind_image_memory(
        m_device, m_physical_device, m_environment_map.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkImageViewCreateInfo image_view_create_info = {};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.image = m_environment_map.image;
    image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    image_view_create_info.format = image_create_info.format;
    image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_view_create_info.subresourceRange.baseMipLevel = 0;
    image_view_create_info.subresourceRange.levelCount = 1;
    image_view_create_info.subresourceRange.baseArrayLayer = 0;
    image_view_create_info.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(
        m_device, &image_view_create_info, nullptr, &m_environment_map.image_view));

    { // Upload image data to the GPU
        size_t staging_buffer_size = res_x * res_y * sizeof(float) * 4; // RGBA32F
        mi::examples::vk::Staging_buffer staging_buffer(m_device, m_physical_device,
            staging_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        // Memcpy the read-only data into the staging buffer
        mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile());
        void* mapped_data = staging_buffer.map_memory();
        std::memcpy(mapped_data, tile->get_data(), staging_buffer_size);
        staging_buffer.unmap_memory();

        // Upload the read-only data from the staging buffer into the storage buffer
        mi::examples::vk::Temporary_command_buffer command_buffer(m_device, m_command_pool);
        command_buffer.begin();

        {
            VkImageMemoryBarrier image_memory_barrier = {};
            image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            image_memory_barrier.srcAccessMask = 0;
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_memory_barrier.image = m_environment_map.image;
            image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_memory_barrier.subresourceRange.baseMipLevel = 0;
            image_memory_barrier.subresourceRange.levelCount = 1;
            image_memory_barrier.subresourceRange.baseArrayLayer = 0;
            image_memory_barrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(command_buffer.get(),
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &image_memory_barrier);
        }

        VkBufferImageCopy copy_region = {};
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageExtent = { res_x, res_y, 1 };

        vkCmdCopyBufferToImage(
            command_buffer.get(), staging_buffer.get(), m_environment_map.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        {
            VkImageMemoryBarrier image_memory_barrier = {};
            image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_memory_barrier.image = m_environment_map.image;
            image_memory_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image_memory_barrier.subresourceRange.baseMipLevel = 0;
            image_memory_barrier.subresourceRange.levelCount = 1;
            image_memory_barrier.subresourceRange.baseArrayLayer = 0;
            image_memory_barrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(command_buffer.get(),
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &image_memory_barrier);
        }

        command_buffer.end_and_submit(m_graphics_queue);
    }

    // Create alias map
    struct Env_accel {
        uint32_t alias;
        float q;
    };

    auto build_alias_map = [](
        const std::vector<float>& data,
        std::vector<Env_accel>& accel) -> float
    {
        // Create qs (normalized)
        float sum = std::accumulate(data.begin(), data.end(), 0.0f);
        uint32_t size = static_cast<uint32_t>(data.size());

        for (uint32_t i = 0; i < size; i++)
            accel[i].q = static_cast<float>(size) * data[i] / sum;

        // Create partition table
        std::vector<uint32_t> partition_table(size);
        uint32_t s = 0;
        uint32_t large = size;
        for (uint32_t i = 0; i < size; i++)
            partition_table[(accel[i].q < 1.0f) ? (s++) : (--large)] = accel[i].alias = i;

        // Create alias map
        for (s = 0; s < large && large < size; ++s)
        {
            uint32_t j = partition_table[s];
            uint32_t k = partition_table[large];
            accel[j].alias = k;
            accel[k].q += accel[j].q - 1.0f;
            large = (accel[k].q < 1.0f) ? (large + 1) : large;
        }

        return sum;
    };

    // Create importance sampling data
    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile());
    const float* pixels = static_cast<const float*>(tile->get_data());

    std::vector<Env_accel> env_accel_data(res_x * res_y);
    std::vector<float> importance_data(res_x * res_y);
    float cos_theta0 = 1.0f;
    const float step_phi = static_cast<float>(2.0 * M_PI / res_x);
    const float step_theta = static_cast<float>(M_PI / res_y);
    for (uint32_t y = 0; y < res_y; y++)
    {
        const float theta1 = static_cast<float>(y + 1) * step_theta;
        const float cos_theta1 = std::cos(theta1);
        const float area = (cos_theta0 - cos_theta1) * step_phi;
        cos_theta0 = cos_theta1;

        for (uint32_t x = 0; x < res_x; x++) {
            const uint32_t idx = y * res_x + x;
            const uint32_t idx4 = idx * 4;
            const float max_channel
                = std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2]));
            importance_data[idx] = area * max_channel;
        }
    }
    float integral = build_alias_map(importance_data, env_accel_data);
    m_render_params.environment_inv_integral = 1.0f / integral;

    // Create Vulkan buffer for importance sampling data
    VkBufferCreateInfo buffer_create_info = {};
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.size = env_accel_data.size() * sizeof(Env_accel);
    buffer_create_info.usage
        = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VK_CHECK(vkCreateBuffer(
        m_device, &buffer_create_info, nullptr, &m_environment_sampling_data_buffer.buffer));

    m_environment_sampling_data_buffer.device_memory
        = mi::examples::vk::allocate_and_bind_buffer_memory(
            m_device, m_physical_device, m_environment_sampling_data_buffer.buffer,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    {
        mi::examples::vk::Staging_buffer staging_buffer(m_device, m_physical_device,
            buffer_create_info.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        // Memcpy the read-only data into the staging buffer
        void* mapped_data = staging_buffer.map_memory();
        std::memcpy(mapped_data, env_accel_data.data(), env_accel_data.size() * sizeof(Env_accel));
        staging_buffer.unmap_memory();

        // Upload the read-only data from the staging buffer into the storage buffer
        mi::examples::vk::Temporary_command_buffer command_buffer(m_device, m_command_pool);
        command_buffer.begin();

        VkBufferCopy copy_region = {};
        copy_region.size = buffer_create_info.size;

        vkCmdCopyBuffer(command_buffer.get(),
            staging_buffer.get(), m_environment_sampling_data_buffer.buffer, 1, &copy_region);

        command_buffer.end_and_submit(m_graphics_queue);
    }

    // Create sampler
    VkSamplerCreateInfo sampler_create_info = {};
    sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_create_info.magFilter = VK_FILTER_LINEAR;
    sampler_create_info.minFilter = VK_FILTER_LINEAR;
    sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_create_info.unnormalizedCoordinates = false;

    VK_CHECK(vkCreateSampler(m_device, &sampler_create_info, nullptr, &m_environment_sampler));
}

// Creates the descriptor pool and set that hold enough space for all
// material resources, and are used during rendering to access the
// the resources.
void Df_vulkan_app::create_descriptor_pool_and_sets()
{
    // Reserve enough space. This is way too much, but sizing them perfectly
    // would make the code less readable.
    VkDescriptorPoolSize uniform_buffer_pool_size;
    uniform_buffer_pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniform_buffer_pool_size.descriptorCount = 100;

    VkDescriptorPoolSize texture_pool_size;
    texture_pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texture_pool_size.descriptorCount = 100;

    VkDescriptorPoolSize storage_buffer_pool_size;
    storage_buffer_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    storage_buffer_pool_size.descriptorCount = 100;

    VkDescriptorPoolSize storage_image_pool_size;
    storage_image_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    storage_image_pool_size.descriptorCount = 100;

    const VkDescriptorPoolSize pool_sizes[] = {
        uniform_buffer_pool_size,
        texture_pool_size,
        storage_buffer_pool_size,
        storage_image_pool_size
    };

    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.maxSets = m_image_count + 1; // img_cnt for path_trace + 1 set for display
    descriptor_pool_create_info.poolSizeCount = std::size(pool_sizes);
    descriptor_pool_create_info.pPoolSizes = pool_sizes;

    VK_CHECK(vkCreateDescriptorPool(
        m_device, &descriptor_pool_create_info, nullptr, &m_descriptor_pool));

    // Allocate descriptor set
    {
        std::vector<VkDescriptorSetLayout> set_layouts(
            m_image_count, m_path_trace_descriptor_set_layout);

        VkDescriptorSetAllocateInfo descriptor_set_alloc_info = {};
        descriptor_set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptor_set_alloc_info.descriptorPool = m_descriptor_pool;
        descriptor_set_alloc_info.descriptorSetCount = m_image_count;
        descriptor_set_alloc_info.pSetLayouts = set_layouts.data();

        m_path_trace_descriptor_sets.resize(m_image_count);
        VK_CHECK(vkAllocateDescriptorSets(
            m_device, &descriptor_set_alloc_info, m_path_trace_descriptor_sets.data()));
    }

    {
        VkDescriptorSetAllocateInfo descriptor_set_alloc_info = {};
        descriptor_set_alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptor_set_alloc_info.descriptorPool = m_descriptor_pool;
        descriptor_set_alloc_info.descriptorSetCount = 1;
        descriptor_set_alloc_info.pSetLayouts = &m_display_descriptor_set_layout;

        VK_CHECK(vkAllocateDescriptorSets(
            m_device, &descriptor_set_alloc_info, &m_display_descriptor_set));
    }

    // Populate descriptor sets
    std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
    std::vector<VkDescriptorImageInfo> descriptor_image_infos;
    std::vector<VkWriteDescriptorSet> descriptor_writes;

    // Reserve enough space. This is way too much, but sizing them perfectly
    // would make the code less readable.
    descriptor_buffer_infos.reserve(100);
    descriptor_image_infos.reserve(100);

    for (uint32_t i = 0; i < m_image_count; i++)
    {
        VkWriteDescriptorSet descriptor_write = {};
        descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_write.dstSet = m_path_trace_descriptor_sets[i];

        { // Render params buffer
            VkDescriptorBufferInfo descriptor_buffer_info = {};
            descriptor_buffer_info.buffer = m_render_params_buffers[i].buffer;
            descriptor_buffer_info.range = sizeof(Render_params);
            descriptor_buffer_infos.push_back(descriptor_buffer_info);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
            descriptor_write.dstBinding = g_binding_render_params;
            descriptor_write.descriptorCount = 1;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_write.pBufferInfo = &descriptor_buffer_infos.back();
            descriptor_writes.push_back(descriptor_write);
        }

        { // Environment map
            VkDescriptorImageInfo descriptor_image_info = {};
            descriptor_image_info.sampler = m_environment_sampler;
            descriptor_image_info.imageView = m_environment_map.image_view;
            descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            descriptor_image_infos.push_back(descriptor_image_info);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
            descriptor_write.dstBinding = g_binding_environment_map;
            descriptor_write.descriptorCount = 1;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptor_write.pImageInfo = &descriptor_image_infos.back();
            descriptor_writes.push_back(descriptor_write);
        }

        { // Environment map sampling data
            VkDescriptorBufferInfo descriptor_buffer_info = {};
            descriptor_buffer_info.buffer = m_environment_sampling_data_buffer.buffer;
            descriptor_buffer_info.range = VK_WHOLE_SIZE;
            descriptor_buffer_infos.push_back(descriptor_buffer_info);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
            descriptor_write.dstBinding = g_binding_environment_sampling_data;
            descriptor_write.descriptorCount = 1;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptor_write.pBufferInfo = &descriptor_buffer_infos.back();
            descriptor_writes.push_back(descriptor_write);
        }

        // Material textures index buffer
        if (m_material_textures_index_buffer.buffer)
        {
            VkDescriptorBufferInfo descriptor_buffer_info = {};
            descriptor_buffer_info.buffer = m_material_textures_index_buffer.buffer;
            descriptor_buffer_info.range = VK_WHOLE_SIZE;
            descriptor_buffer_infos.push_back(descriptor_buffer_info);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
            descriptor_write.dstBinding = g_binding_material_textures_indices;
            descriptor_write.descriptorCount = 1;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptor_write.pBufferInfo = &descriptor_buffer_infos.back();
            descriptor_writes.push_back(descriptor_write);
        }

        // Material textures
        const std::vector<Vulkan_texture>* material_textures_arrays[] = {
            &m_material_textures_2d,
            &m_material_textures_3d
        };
        const uint32_t material_textures_bindings[] = {
            g_binding_material_textures_2d,
            g_binding_material_textures_3d
        };
        for (size_t dim = 0; dim < 2; dim++)
        {
            const auto& textures = *material_textures_arrays[dim];
            const uint32_t binding = material_textures_bindings[dim];

            for (size_t tex = 0; tex < textures.size(); tex++)
            {
                VkDescriptorImageInfo descriptor_image_info = {};
                descriptor_image_info.sampler = m_linear_sampler;
                descriptor_image_info.imageView = textures[tex].image_view;
                descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                descriptor_image_infos.push_back(descriptor_image_info);

                VkWriteDescriptorSet descriptor_write = {};
                descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
                descriptor_write.dstBinding = binding;
                descriptor_write.dstArrayElement = tex;
                descriptor_write.descriptorCount = 1;
                descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptor_write.pImageInfo = &descriptor_image_infos.back();
                descriptor_writes.push_back(descriptor_write);
            }
        }

        // Read-only data buffer
        if (m_ro_data_buffer.buffer)
        {
            VkDescriptorBufferInfo descriptor_buffer_info = {};
            descriptor_buffer_info.buffer = m_ro_data_buffer.buffer;
            descriptor_buffer_info.range = VK_WHOLE_SIZE;
            descriptor_buffer_infos.push_back(descriptor_buffer_info);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
            descriptor_write.dstBinding = g_binding_ro_data_buffer;
            descriptor_write.descriptorCount = 1;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptor_write.pBufferInfo = &descriptor_buffer_infos.back();
            descriptor_writes.push_back(descriptor_write);
        }
    }

    vkUpdateDescriptorSets(
        m_device, static_cast<uint32_t>(descriptor_writes.size()),
        descriptor_writes.data(), 0, nullptr);

    update_accumulation_image_descriptors();
}

void Df_vulkan_app::update_accumulation_image_descriptors()
{
    std::vector<VkWriteDescriptorSet> descriptor_writes;

    std::vector<VkDescriptorImageInfo> descriptor_image_infos;
    descriptor_image_infos.reserve(m_image_count * 3 + 3);

    for (uint32_t i = 0; i < m_image_count; i++)
    {
        VkDescriptorImageInfo descriptor_image_info = {};
        descriptor_image_info.imageView = m_beauty_texture.image_view;
        descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        descriptor_image_infos.push_back(descriptor_image_info);

        VkWriteDescriptorSet descriptor_write = {};
        descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
        descriptor_write.dstBinding = g_binding_beauty_buffer;
        descriptor_write.descriptorCount = 1;
        descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptor_write.pImageInfo = &descriptor_image_infos.back();
        descriptor_writes.push_back(descriptor_write);

        descriptor_image_info.imageView = m_auxiliary_albedo_texture.image_view;
        descriptor_image_infos.push_back(descriptor_image_info);

        descriptor_write.dstBinding = g_binding_aux_albedo_buffer;
        descriptor_write.pImageInfo = &descriptor_image_infos.back();
        descriptor_writes.push_back(descriptor_write);

        descriptor_image_info.imageView = m_auxiliary_normal_texture.image_view;
        descriptor_image_infos.push_back(descriptor_image_info);

        descriptor_write.dstBinding = g_binding_aux_normal_buffer;
        descriptor_write.pImageInfo = &descriptor_image_infos.back();
        descriptor_writes.push_back(descriptor_write);
    }

    VkDescriptorImageInfo descriptor_info = {};
    descriptor_info.imageView = m_beauty_texture.image_view;
    descriptor_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    descriptor_image_infos.push_back(descriptor_info);

    VkWriteDescriptorSet descriptor_write = {};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = m_display_descriptor_set;
    descriptor_write.dstBinding = 0;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorCount = 1;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptor_write.pImageInfo = &descriptor_image_infos.back();
    descriptor_writes.push_back(descriptor_write);

    descriptor_info.imageView = m_auxiliary_albedo_texture.image_view;
    descriptor_image_infos.push_back(descriptor_info);

    descriptor_write.dstBinding = 1;
    descriptor_write.pImageInfo = &descriptor_image_infos.back();
    descriptor_writes.push_back(descriptor_write);

    descriptor_info.imageView = m_auxiliary_normal_texture.image_view;
    descriptor_image_infos.push_back(descriptor_info);

    descriptor_write.dstBinding = 2;
    descriptor_write.pImageInfo = &descriptor_image_infos.back();
    descriptor_writes.push_back(descriptor_write);

    vkUpdateDescriptorSets(
        m_device, static_cast<uint32_t>(descriptor_writes.size()),
        descriptor_writes.data(), 0, nullptr);
}


//------------------------------------------------------------------------------
// MDL material compilation helpers
//------------------------------------------------------------------------------
mi::neuraylib::IFunction_call* create_material_instance(
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_execution_context* context,
    const std::string& material_name)
{
    // Split material name into module and simple material name
    std::string module_name, material_simple_name;
    mi::examples::mdl::parse_cmd_argument_material_name(
        material_name, module_name, material_simple_name);

    // Load module
    mdl_impexp_api->load_module(transaction, module_name.c_str(), context);
    if (!print_messages(context))
        exit_failure("Loading module '%s' failed.", module_name.c_str());

    // Get the database name for the module we loaded and check if
    // the module exists in the database.
    mi::base::Handle<const mi::IString> module_db_name(
        mdl_factory->get_db_definition_name(module_name.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
    if (!module)
        exit_failure("Failed to access the loaded module.");

    // To access the material in the database we need to know the exact material
    // signature, so we append the arguments to the full name (with module).
    std::string material_db_name
        = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;
    material_db_name = mi::examples::mdl::add_missing_material_signature(
        module.get(), material_db_name.c_str());

    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
    if (!material_definition)
        exit_failure("Failed to access material definition '%s'.", material_db_name.c_str());

    // Create material instance
    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(
        material_definition->create_function_call(nullptr, &result));
    if (result != 0)
        exit_failure("Failed to instantiate material '%s'.", material_db_name.c_str());

    material_instance->retain();
    return material_instance.get();
}

mi::neuraylib::ICompiled_material* compile_material_instance(
    mi::neuraylib::IFunction_call* material_instance,
    mi::neuraylib::IMdl_execution_context* context,
    bool class_compilation)
{
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance2(
        material_instance->get_interface<mi::neuraylib::IMaterial_instance>());

    mi::Uint32 compile_flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance2->create_compiled_material(compile_flags, context));
    check_success(print_messages(context));

    compiled_material->retain();
    return compiled_material.get();
}

const mi::neuraylib::ITarget_code* generate_glsl_code(
    mi::neuraylib::ICompiled_material* compiled_material,
    mi::neuraylib::IMdl_backend_api* mdl_backend_api,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_execution_context* context)
{
    // Add compiled material to link unit
    mi::base::Handle<mi::neuraylib::IMdl_backend> be_glsl(
        mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_GLSL));

    check_success(be_glsl->set_option("glsl_version", "450") == 0);
    check_success(be_glsl->set_option("glsl_place_uniforms_into_ssbo", "on") == 0);
    check_success(be_glsl->set_option("glsl_max_const_data", "0") == 0);
    check_success(be_glsl->set_option("glsl_uniform_ssbo_binding",
        std::to_string(g_binding_ro_data_buffer).c_str()) == 0);
    check_success(be_glsl->set_option("glsl_uniform_ssbo_set",
        std::to_string(g_set_ro_data_buffer).c_str()) == 0);
    check_success(be_glsl->set_option("num_texture_spaces", "1") == 0);
    check_success(be_glsl->set_option("enable_auxiliary", "on") == 0);
    check_success(be_glsl->set_option("df_handle_slot_mode", "none") == 0);
    
    mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(
        be_glsl->create_link_unit(transaction, context));

    // Specify which functions to generate
    std::vector<mi::neuraylib::Target_function_description> function_descs;
    function_descs.emplace_back("thin_walled", "mdl_thin_walled");
    function_descs.emplace_back("surface.scattering", "mdl_bsdf");
    function_descs.emplace_back("surface.emission.emission", "mdl_edf");
    function_descs.emplace_back("surface.emission.intensity", "mdl_emission_intensity");
    function_descs.emplace_back("volume.absorption_coefficient", "mdl_absorption_coefficient");
    function_descs.emplace_back("geometry.cutout_opacity", "mdl_cutout_opacity");

    // Try to determine if the material is thin walled so we can check
    // if backface functions need to be generated.
    bool is_thin_walled_function = true;
    bool thin_walled_value = false;
    mi::base::Handle<const mi::neuraylib::IExpression> thin_walled_expr(
        compiled_material->lookup_sub_expression("thin_walled"));
    if (thin_walled_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
        mi::base::Handle<const mi::neuraylib::IExpression_constant> thin_walled_const(
            thin_walled_expr->get_interface<const mi::neuraylib::IExpression_constant>());
        mi::base::Handle<const mi::neuraylib::IValue_bool> thin_walled_bool(
            thin_walled_const->get_value<mi::neuraylib::IValue_bool>());

        is_thin_walled_function = false;
        thin_walled_value = thin_walled_bool->get_value();
    }

    // Back faces could be different for thin walled materials
    bool need_backface_bsdf = false;
    bool need_backface_edf = false;
    bool need_backface_emission_intensity = false;

    if (is_thin_walled_function || thin_walled_value)
    {
        // First, backfacs DFs are only considered for thin_walled materials

        // Second, we only need to generate new code if surface and backface are different
        need_backface_bsdf =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_SCATTERING)
            != compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_SCATTERING);
        need_backface_edf =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_EDF_EMISSION)
            != compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_EDF_EMISSION);
        need_backface_emission_intensity =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_INTENSITY)
            != compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_INTENSITY);

        // Third, either the bsdf or the edf need to be non-default (black)
        mi::base::Handle<const mi::neuraylib::IExpression> scattering_expr(
            compiled_material->lookup_sub_expression("backface.scattering"));
        mi::base::Handle<const mi::neuraylib::IExpression> emission_expr(
            compiled_material->lookup_sub_expression("backface.emission.emission"));

        if (scattering_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT
            && emission_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
        {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> scattering_expr_constant(
                scattering_expr->get_interface<mi::neuraylib::IExpression_constant>());
            mi::base::Handle<const mi::neuraylib::IValue> scattering_value(
                scattering_expr_constant->get_value());

            mi::base::Handle<const mi::neuraylib::IExpression_constant> emission_expr_constant(
                emission_expr->get_interface<mi::neuraylib::IExpression_constant>());
            mi::base::Handle<const mi::neuraylib::IValue> emission_value(
                emission_expr_constant->get_value());

            if (scattering_value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF
                && emission_value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF)
            {
                need_backface_bsdf = false;
                need_backface_edf = false;
                need_backface_emission_intensity = false;
            }
        }
    }

    if (need_backface_bsdf)
        function_descs.emplace_back("backface.scattering", "mdl_backface_bsdf");

    if (need_backface_edf)
        function_descs.emplace_back("backface.emission.emission", "mdl_backface_edf");

    if (need_backface_emission_intensity)
        function_descs.emplace_back("backface.emission.intensity", "mdl_backface_emission_intensity");

    link_unit->add_material(
        compiled_material, function_descs.data(), function_descs.size(), context);
    check_success(print_messages(context));

    // Generate GLSL code
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
        be_glsl->translate_link_unit(link_unit.get(), context));
    check_success(print_messages(context));
    check_success(target_code);

    target_code->retain();
    return target_code.get();
}


//------------------------------------------------------------------------------
// Command line helpers
//------------------------------------------------------------------------------
void print_usage(char const* prog_name)
{
    std::cout
        << "Usage: " << prog_name << " [options] [<material_name|full_mdle_path>]\n"
        << "Options:\n"
        << "  -h|--help                 print this text and exit\n"
        << "  -v|--version              print the MDL SDK version string and exit\n"
        << "  --nowin                   don't show interactive display\n"
        << "  --res <res_x> <res_y>     resolution (default: 1024x768)\n"
        << "  --numimg <n>              swapchain image count (default: 3)\n"
        << "  -o|--output <outputfile>  image file to write result in nowin mode (default: output.exr)\n"
        << "  --spp <num>               samples per pixel, only used for --nowin (default: 4096)\n"
        << "  --spi <num>               samples per render loop iteration (default: 8)\n"
        << "  --max_path_length <num>   maximum path length (default: 4)\n"
        << "  -f|--fov <fov>            the camera field of view in degrees (default: 96.0)\n"
        << "  -p|--pos <x> <y> <z>      set the camera position (default: 0 0 3).\n"
        << "                            The camera will always look towards (0, 0, 0)\n"
        << "  -l|--light <x> <y> <z>    adds an omnidirectional light with the given position"
        << "             <r> <g> <b>    and intensity\n"
        << "  --hdr <path>              hdr image file used for the environment map\n"
        << "                            (default: nvidia/sdk_examples/resources/environment.hdr)\n"
        << "  --hdr_intensity <value>   intensity of the environment map (default: 1.0)\n"
        << "  --cc                      the material will be compiled using class compilation\n"
        << "  --mdl_path <path>         additional MDL search path, can occur multiple times\n"
        << "  --vkdebug                 enable the Vulkan validation layers"
        << std::endl;

    exit(EXIT_FAILURE);
}

void parse_command_line(int argc, char* argv[], Options& options,
    bool& print_version_and_exit, mi::examples::mdl::Configure_options& mdl_configure_options)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg[0] == '-')
        {
            if (arg == "-v" || arg == "--version")
                print_version_and_exit = true;
            else if (arg == "--nowin")
                options.no_window = true;
            else if (arg == "--res" && i < argc - 2)
            {
                options.res_x = std::max(atoi(argv[++i]), 1);
                options.res_y = std::max(atoi(argv[++i]), 1);
            }
            else if (arg == "--numimg" && i < argc - 1)
                options.num_images = std::max(atoi(argv[++i]), 2);
            else if (arg == "-o" && i < argc - 1)
                options.output_file = argv[++i];
            else if (arg == "--spp" && i < argc - 1)
                options.samples_per_pixel = std::atoi(argv[++i]);
            else if (arg == "--spi" && i < argc - 1)
                options.samples_per_iteration = std::atoi(argv[++i]);
            else if (arg == "--max_path_length" && i < argc - 1)
                options.max_path_length = std::atoi(argv[++i]);
            else if ((arg == "-f" || arg == "--fov") && i < argc - 1)
                options.cam_fov = static_cast<float>(std::atof(argv[++i]));
            else if ((arg == "-p" || arg == "--pos") && i < argc - 3)
            {
                options.cam_pos.x = static_cast<float>(std::atof(argv[++i]));
                options.cam_pos.y = static_cast<float>(std::atof(argv[++i]));
                options.cam_pos.z = static_cast<float>(std::atof(argv[++i]));
            }
            else if ((arg == "-l" || arg == "--light") && i < argc - 6)
            {
                options.light_pos.x = static_cast<float>(std::atof(argv[++i]));
                options.light_pos.y = static_cast<float>(std::atof(argv[++i]));
                options.light_pos.z = static_cast<float>(std::atof(argv[++i]));
                options.light_intensity.x = static_cast<float>(std::atof(argv[++i]));
                options.light_intensity.y = static_cast<float>(std::atof(argv[++i]));
                options.light_intensity.z = static_cast<float>(std::atof(argv[++i]));
            }
            else if (arg == "--hdr" && i < argc - 1)
                options.hdr_file = argv[++i];
            else if (arg == "--hdr_intensity" && i < argc - 1)
                options.hdr_intensity = static_cast<float>(std::atof(argv[++i]));
            else if (arg == "--mdl_path" && i < argc - 1)
                mdl_configure_options.additional_mdl_paths.push_back(argv[++i]);
            else if (arg == "--cc")
                options.use_class_compilation = true;
            else if (arg == "--vkdebug")
                options.enable_validation_layers = true;
            else
            {
                if (arg != "-h" && arg != "--help")
                    std::cout << "Unknown option: \"" << arg << "\"" << std::endl;

                print_usage(argv[0]);
            }
        }
        else
            options.material_name = arg;
    }
}


//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int MAIN_UTF8(int argc, char* argv[])
{
    Options options;
    bool print_version_and_exit = false;
    mi::examples::mdl::Configure_options configure_options;
    parse_command_line(argc, argv, options, print_version_and_exit, configure_options);

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(
        mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Handle the --version flag
    if (print_version_and_exit)
    {
        // Print library version information
        mi::base::Handle<const mi::neuraylib::IVersion> version(
            neuray->get_api_component<const mi::neuraylib::IVersion>());
        std::cout << version->get_string() << "\n";

        // Free the handles and unload the MDL SDK
        version = nullptr;
        neuray = nullptr;
        if (!mi::examples::mdl::unload())
            exit_failure("Failed to unload the SDK.");

        exit_success();
    }

    // Configure the MDL SDK
    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 result = neuray->start();
    if (result != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", result);

    {
        // Create a transaction
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        // Access needed API components
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>());

        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
            neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

        mi::base::Handle<mi::neuraylib::IImage_api> image_api(
            neuray->get_api_component<mi::neuraylib::IImage_api>());

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());
        
        {
            // Load and compile material, and generate GLSL code
            mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(
                create_material_instance(mdl_impexp_api.get(), mdl_factory.get(),
                    transaction.get(), context.get(), options.material_name));

            mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
                compile_material_instance(material_instance.get(), context.get(),
                    options.use_class_compilation));

            mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
                generate_glsl_code(compiled_material.get(), mdl_backend_api.get(),
                    transaction.get(), context.get()));

        #ifdef DUMP_GLSL
            std::cout << "Dumping GLSL target code:\n\n" << target_code->get_code() << "\n\n";
        #endif

            // Start application
            mi::examples::vk::Vulkan_example_app::Config app_config;
            app_config.window_title = "MDL SDK DF Vulkan Example";
            app_config.image_width = options.res_x;
            app_config.image_height = options.res_y;
            app_config.image_count = options.num_images;
            app_config.headless = options.no_window;
            app_config.iteration_count = options.samples_per_pixel / options.samples_per_iteration;
            app_config.enable_validation_layers = options.enable_validation_layers;

            Df_vulkan_app app(transaction, mdl_impexp_api, image_api, target_code, options);
            app.run(app_config);
        }

        transaction->commit();
    }

    // Shut down the MDL SDK
    if (neuray->shutdown() != 0)
        exit_failure("Failed to shutdown the SDK.");

    // Unload the MDL SDK
    neuray = nullptr;
    if (!mi::examples::mdl::unload())
        exit_failure("Failed to unload the SDK.");

    exit_success();
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
