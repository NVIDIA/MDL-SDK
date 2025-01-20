/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_core/df_vulkan/example_df_vulkan.cpp
 //
 // Simple Vulkan renderer using compiled BSDFs with a material parameter editor GUI.

#include "example_shared_backends.h"
#include "example_vulkan_shared.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <numeric>
#define _USE_MATH_DEFINES
#include <math.h>

static const VkFormat g_accumulation_texture_format = VK_FORMAT_R32G32B32A32_SFLOAT;

// Local group size for the path tracing compute shader
static const uint32_t g_local_size_x = 16;
static const uint32_t g_local_size_y = 8;

// Descriptor set bindings. Used as a define in the shaders.
static const uint32_t g_binding_beauty_buffer = 0;
static const uint32_t g_binding_aux_albedo_diffuse_buffer = 1;
static const uint32_t g_binding_aux_albedo_glossy_buffer = 2;
static const uint32_t g_binding_aux_normal_buffer = 3;
static const uint32_t g_binding_aux_roughness_buffer = 4;
static const uint32_t g_binding_render_params = 5;
static const uint32_t g_binding_environment_map = 6;
static const uint32_t g_binding_environment_sampling_data = 7;
static const uint32_t g_binding_material_textures_2d = 8;
static const uint32_t g_binding_material_textures_3d = 9;
static const uint32_t g_binding_ro_data_buffer = 10;
static const uint32_t g_binding_argument_block_buffer = 11;

static const uint32_t g_set_ro_data_buffer = 0;
static const uint32_t g_set_argument_block_buffer = 0;
static const uint32_t g_set_material_textures = 0;

// Command line options structure.
struct Options
{
    bool no_window = false;
    std::string output_file = "output.exr";
    uint32_t res_x = 1024;
    uint32_t res_y = 768;
    uint32_t num_images = 3;
    int32_t device_index = -1;
    uint32_t samples_per_pixel = 4096;
    uint32_t samples_per_iteration = 8;
    uint32_t max_path_length = 4;
    float cam_fov = 96.0f;
    Float32_3 cam_pos = { 0.0f, 0.0f, 3.0f };
    Float32_3 light_pos = { 10.0f, 0.0f, 5.0f };
    Float32_3 light_intensity = { 1.0f, 0.95f, 0.9f };
    bool light_enabled = false;
    std::string hdr_file = "nvidia/sdk_examples/resources/environment.hdr";
    float hdr_intensity = 1.0f;
    bool use_class_compilation = true;
    uint32_t tex_results_cache_size = 16;
    bool enable_ro_segment = false;
    bool disable_ssbo = false;
    uint32_t max_const_data = 1024;
    std::string material_name = "::nvidia::sdk_examples::tutorials::example_df";
    bool enable_validation_layers = false;
    bool enable_shader_optimization = true;
    bool dump_glsl = false;
    bool hide_gui = false;
    bool enable_bsdf_flags = false;
    mi::mdl::Df_flags allowed_scatter_mode = mi::mdl::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
    std::vector<std::string> additional_mdl_paths;
};

Vulkan_buffer create_storage_buffer(
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkQueue queue,
    VkCommandPool command_pool,
    const void* buffer_data,
    size_t buffer_size)
{
    Vulkan_buffer storage_buffer;

    { // Create the storage buffer in device local memory (VRAM)
        VkBufferCreateInfo buffer_create_info = {};
        buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_create_info.size = buffer_size;
        buffer_create_info.usage
            = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VK_CHECK(vkCreateBuffer(
            device, &buffer_create_info, nullptr, &storage_buffer.buffer));

        // Allocate device memory for the buffer.
        storage_buffer.device_memory = allocate_and_bind_buffer_memory(
            device, physical_device, storage_buffer.buffer,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    {
        Staging_buffer staging_buffer(
            device, physical_device, buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        // Memcpy the data into the staging buffer
        void* mapped_data = staging_buffer.map_memory();
        std::memcpy(mapped_data, buffer_data, buffer_size);
        staging_buffer.unmap_memory();

        // Upload the data from the staging buffer into the storage buffer
        Temporary_command_buffer command_buffer(device, command_pool);
        command_buffer.begin();

        VkBufferCopy copy_region = {};
        copy_region.size = buffer_size;

        vkCmdCopyBuffer(command_buffer.get(),
            staging_buffer.get(), storage_buffer.buffer, 1, &copy_region);

        command_buffer.end_and_submit(queue);
    }

    return storage_buffer;
}


//------------------------------------------------------------------------------
// MDL-Vulkan resource interop
//------------------------------------------------------------------------------

// Creates the image and image view for the given texture index.
Vulkan_texture create_material_texture(
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkQueue queue,
    VkCommandPool command_pool,
    const Target_code* target_code,
    size_t texture_index)
{
    const Texture_data* texture_data = target_code->get_texture(texture_index);
    if (!texture_data->is_valid())
    {
        std::cerr << "Error: Requested texture is invalid\n";
        exit_failure();
    }

    mi::Uint32 tex_width = texture_data->get_width();
    mi::Uint32 tex_height = texture_data->get_height();
    mi::Uint32 tex_layers = texture_data->get_depth();

    // Create the Vulkan image
    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // This example supports only 2D and 3D textures (no PTEX or cube)
    mi::mdl::IType_texture::Shape texture_shape = texture_data->get_shape();
    
    if (texture_shape == mi::mdl::IType_texture::TS_2D)
    {
        image_create_info.imageType = VK_IMAGE_TYPE_2D;
        image_create_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        image_create_info.extent.width = tex_width;
        image_create_info.extent.height = tex_height;
        image_create_info.extent.depth = 1;
        image_create_info.arrayLayers = 1;
        image_create_info.mipLevels = 1;
    }
    else if (texture_shape == mi::mdl::IType_texture::TS_3D
        || texture_shape == mi::mdl::IType_texture::TS_BSDF_DATA)
    {
        image_create_info.imageType = VK_IMAGE_TYPE_3D;
        if (texture_shape == mi::mdl::IType_texture::TS_BSDF_DATA)
            image_create_info.format = VK_FORMAT_R32_SFLOAT;
        else
            image_create_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        image_create_info.extent.width = tex_width;
        image_create_info.extent.height = tex_height;
        image_create_info.extent.depth = tex_layers;
        image_create_info.arrayLayers = 1;
        image_create_info.mipLevels = 1;
    }
    else
    {
        std::cerr << "Error: Unsupported texture shape!\n";
        exit_failure();
    }

    Vulkan_texture material_texture;

    VK_CHECK(vkCreateImage(device, &image_create_info, nullptr,
        &material_texture.image));

    // Allocate device memory for the texture.
    material_texture.device_memory = allocate_and_bind_image_memory(
        device, physical_device, material_texture.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    uint32_t bpp = get_image_format_bpp(image_create_info.format);
    size_t layer_size = tex_width * tex_height * bpp;
    size_t staging_buffer_size = layer_size * tex_layers;
    Staging_buffer staging_buffer(device, physical_device,
        staging_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    if (texture_shape == mi::mdl::IType_texture::TS_BSDF_DATA)
    {
        const unsigned char* bsdf_data = texture_data->get_bsdf_data();
        if (!bsdf_data)
        {
            std::cerr << "Error: bsdf data missing for requested texture\n";
            exit_failure();
        }

        size_t bsdf_data_size = tex_width * tex_height * tex_layers * sizeof(float);
        uint8_t* mapped_data = static_cast<uint8_t*>(staging_buffer.map_memory());
        std::memcpy(mapped_data, bsdf_data, bsdf_data_size);
        staging_buffer.unmap_memory();
    }
    else
    {
        // For simplicity, the texture access functions are only implemented for float4 and gamma
        // is pre-applied here (all images are converted to linear space).

        std::vector<float> pixel_data(4 * tex_width * tex_height);
        std::shared_ptr<OIIO::ImageInput> image(texture_data->get_image());
        mi::Sint32 bytes_per_row = 4 * tex_width * sizeof(float);
        bool success = image->read_image(
            /*subimage*/ 0,
            /*miplevel*/ 0,
            /*chbegin*/ 0,
            /*chend*/ 4,
            OIIO::TypeDesc::FLOAT,
            pixel_data.data() + (tex_height - 1) * 4 * tex_width,
            /*xstride*/ 4 * sizeof(float),
            /*ystride*/ -bytes_per_row);
        if (!success)
        {
            std::cerr << "Error: Failed to read image data for requested texture\n";
            exit_failure();
        }

        if (image->spec().nchannels <= 3)
            for (size_t i = 0, n = pixel_data.size(); i < n; i += 4)
                pixel_data[i + 3] = 1.0f;

        // Convert to linear color space if necessary
        if (texture_data->get_gamma_mode() != mi::mdl::IValue_texture::gamma_linear) {
            for (size_t i = 0, n = pixel_data.size(); i < n; i += 4) {
                // Only adjust r, g and b, not alpha
                pixel_data[i + 0] = std::pow(pixel_data[i + 0], 2.2f);
                pixel_data[i + 1] = std::pow(pixel_data[i + 1], 2.2f);
                pixel_data[i + 2] = std::pow(pixel_data[i + 2], 2.2f);
            }
        }

        uint8_t* mapped_data = static_cast<uint8_t*>(staging_buffer.map_memory());
        std::memcpy(mapped_data, pixel_data.data(), staging_buffer_size);
        staging_buffer.unmap_memory();
    }

    {
        // Upload the read-only data from the staging buffer into the storage buffer
        Temporary_command_buffer command_buffer(device, command_pool);
        command_buffer.begin();

        transitionImageLayout(command_buffer.get(),
            /*image=*/           material_texture.image,
            /*src_access_mask=*/ 0,
            /*dst_access_mask=*/ VK_ACCESS_TRANSFER_WRITE_BIT,
            /*old_layout=*/      VK_IMAGE_LAYOUT_UNDEFINED,
            /*new_layout=*/      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            /*src_stage_mask=*/  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            /*dst_stage_mask=*/  VK_PIPELINE_STAGE_TRANSFER_BIT);

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

        transitionImageLayout(command_buffer.get(),
            /*image=*/           material_texture.image,
            /*src_access_mask=*/ VK_ACCESS_TRANSFER_WRITE_BIT,
            /*dst_access_mask=*/ VK_ACCESS_SHADER_READ_BIT,
            /*old_layout=*/      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            /*new_layout=*/      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            /*src_stage_mask=*/  VK_PIPELINE_STAGE_TRANSFER_BIT,
            /*dst_stage_mask=*/  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        command_buffer.end_and_submit(queue);
    }

    // Create the image view
    VkImageViewCreateInfo image_view_create_info = {};
    image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    image_view_create_info.image = material_texture.image;
    image_view_create_info.viewType = (image_create_info.imageType == VK_IMAGE_TYPE_2D)
        ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D;
    image_view_create_info.format = image_create_info.format;
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
class Df_vulkan_app : public Vulkan_base_app
{
public:
    Df_vulkan_app(
        Target_code* target_code,
        const Material_instance& material_instance,
        size_t argument_block_index,
        const mi::mdl::IMDL* mdl_compiler,
        const Options& options)
    : m_target_code(target_code)
    , m_material_instance(&material_instance)
    , m_argument_block_index(argument_block_index)
    , m_mdl_compiler(mdl_compiler)
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
    virtual void update(float elapsed_seconds, uint32_t frame_index) override;

    // Populates the current frame's command buffer. The base application's
    // render pass has already been started at this point.
    virtual void render(VkCommandBuffer command_buffer, uint32_t frame_index, uint32_t image_index) override;

    // Window event handlers.
    virtual void key_callback(int key, int action, int mods) override;
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
        alignas(16) Float32_3 cam_pos;
        alignas(16) Float32_3 cam_dir;
        alignas(16) Float32_3 cam_right;
        alignas(16) Float32_3 cam_up;
        float cam_focal;
        alignas(16) Float32_3 point_light_pos;
        alignas(16) Float32_3 point_light_color;
        float point_light_intensity;
        float environment_intensity_factor;
        float environment_inv_integral;
        uint32_t max_path_length;
        uint32_t samples_per_iteration;
        uint32_t progressive_iteration;
        uint32_t bsdf_data_flags;
    };

private:
    void do_settings_and_stats_gui();

    void update_camera_render_params(const Camera_state& cam_state);

    void create_accumulation_images();

    // Creates the rendering shader module. The generated GLSL target code,
    // GLSL MDL renderer runtime implementation, and renderer code are compiled
    // and linked together here.
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

    void create_query_pool();

    void update_accumulation_image_descriptors();

    void write_accum_images_to_files();

private:
    Target_code* m_target_code;
    const Material_instance* m_material_instance;
    size_t m_argument_block_index;
    const mi::mdl::IMDL* m_mdl_compiler;
    Options m_options;

    enum AccumImage
    {
        ACCUM_IMAGE_BEAUTY = 0,
        ACCUM_IMAGE_AUX_ALBEDO_DIFFUSE,
        ACCUM_IMAGE_AUX_ALBEDO_GLOSSY,
        ACCUM_IMAGE_AUX_NORMAL,
        ACCUM_IMAGE_AUX_ROUGHNESS,
        ACCUM_IMAGE_COUNT
    };
    Vulkan_texture m_accum_images[ACCUM_IMAGE_COUNT];
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
    VkQueryPool m_query_pool;

    Vulkan_texture m_environment_map;
    Vulkan_buffer m_environment_sampling_data_buffer;
    VkSampler m_environment_sampler;

    // Material resources
    Vulkan_buffer m_ro_data_buffer;
    Vulkan_buffer m_argument_block_buffer;
    std::vector<Staging_buffer> m_argument_block_staging_buffers;
    std::vector<Vulkan_texture> m_material_textures_2d;
    std::vector<Vulkan_texture> m_material_textures_3d;
    std::unique_ptr<Argument_block> m_argument_block;

    Render_params m_render_params;
    bool m_camera_moved = true; // Force a clear in first frame
    bool m_material_changed = false; // If the argument buffer needs to be updated
    uint32_t m_display_buffer_index = 0; // Which buffer to display
    bool m_show_gui = true;
    bool m_first_stats_update = true;
    double m_last_stats_update;
    float m_render_time = 0.0f;
    bool m_vsync_enabled = true;
    
    // Camera movement
    Camera_state m_camera_state;
    Float32_2 m_mouse_start;
    bool m_camera_moving = false;
};

void Df_vulkan_app::init_resources()
{
    glslang::InitializeProcess();

    m_linear_sampler = create_linear_sampler(m_device);
    
    // Create the render resources for the material
    //
    // Create the storage buffer for the material's read-only data
    size_t ro_data_size;
    const char* ro_data = m_target_code->get_ro_data_segment(ro_data_size);
    if (ro_data)
    {
        m_ro_data_buffer = create_storage_buffer(
            m_device, m_physical_device, m_graphics_queue, m_command_pool, ro_data, ro_data_size);
    }

    // Create the storage buffers for the material's argument block
    bool is_class_compiled = (m_argument_block_index != ~0);
    if (is_class_compiled)
    {
        size_t num_argument_blocks = m_target_code->get_argument_block_count();
        if (num_argument_blocks > 0 && m_argument_block_index < num_argument_blocks)
        {
            // We create our own copy of the argument data block, so we can modify the material parameters
            const Argument_block* readonly_argument_block = m_target_code->get_argument_block(m_argument_block_index);
            m_argument_block = std::make_unique<Argument_block>(*readonly_argument_block);

            // We create a device-local storage buffer for the argument block for simplicity since we
            // we don't expect to change the material parameters every frame.
            // This choice is not meant to be a recommendation. The appropriate memory properties and
            // buffer update strategies depend on the application.
            m_argument_block_buffer = create_storage_buffer(
                m_device, m_physical_device, m_graphics_queue, m_command_pool, m_argument_block->get_data(), m_argument_block->get_size());

            // Need the staging buffer so we can update the argument block storage buffers later
            m_argument_block_staging_buffers.reserve(m_image_count);
            for (uint32_t i = 0; i < m_image_count; i++)
            {
                m_argument_block_staging_buffers.emplace_back(
                    m_device, m_physical_device, m_argument_block->get_size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            }
        }
    }

    // Record the indices of each texture in their respective array
    // e.g. the indices of 2D textures in the m_material_textures_2d array
    std::vector<uint32_t> material_textures_indices;

    // Create the textures for the material
    if (m_target_code->get_texture_count() > 0)
    {
        // The first texture (index = 0) is always the invalid texture in MDL
        material_textures_indices.reserve(m_target_code->get_texture_count() - 1);

        for (size_t i = 1; i < m_target_code->get_texture_count(); i++)
        {
            Vulkan_texture texture = create_material_texture(
                m_device, m_physical_device, m_graphics_queue, m_command_pool, m_target_code, i);

            switch (m_target_code->get_texture(i)->get_shape())
            {
            case mi::mdl::IType_texture::TS_2D:
                material_textures_indices.push_back(static_cast<uint32_t>(m_material_textures_2d.size()));
                m_material_textures_2d.push_back(texture);
                break;
            case mi::mdl::IType_texture::TS_3D:
            case mi::mdl::IType_texture::TS_BSDF_DATA:
                material_textures_indices.push_back(static_cast<uint32_t>(m_material_textures_3d.size()));
                m_material_textures_3d.push_back(texture);
                break;
            case mi::mdl::IType_texture::TS_CUBE:
            case mi::mdl::IType_texture::TS_PTEX:
                std::cerr << "Unsupported texture shape!" << std::endl;
                exit_failure();
                break;
            }
        }
    }

    create_descriptor_set_layouts();
    create_pipeline_layouts();
    create_accumulation_images();
    create_pipelines();
    create_render_params_buffers();
    create_environment_map();
    create_descriptor_pool_and_sets();
    create_query_pool();

    // Initialize render parameters
    m_render_params.progressive_iteration = 0;
    m_render_params.max_path_length = m_options.max_path_length;
    m_render_params.samples_per_iteration = m_options.samples_per_iteration;
    m_render_params.bsdf_data_flags = m_options.allowed_scatter_mode;

    m_render_params.point_light_pos = m_options.light_pos;
    m_render_params.point_light_intensity = m_options.light_enabled
        ? std::max(std::max(m_options.light_intensity.x, m_options.light_intensity.y), m_options.light_intensity.z)
        : 0.0f;
    m_render_params.point_light_color = m_render_params.point_light_intensity > 0.0f
        ? m_options.light_intensity / m_render_params.point_light_intensity
        : m_options.light_intensity;
    m_render_params.environment_intensity_factor = m_options.hdr_intensity;

    const float fov = m_options.cam_fov;
    const float to_radians = static_cast<float>(M_PI / 180.0);
    m_render_params.cam_focal = 1.0f / mi::math::tan(fov / 2.0f * to_radians);

    // Setup camera
    const Float32_3 camera_pos = m_options.cam_pos;
    Float32_3 inv_dir = camera_pos / mi::math::length(camera_pos);
    m_camera_state.base_distance = mi::math::length(camera_pos);
    m_camera_state.phi = mi::math::atan2(inv_dir.x, inv_dir.z);
    m_camera_state.theta = mi::math::acos(inv_dir.y);
    m_camera_state.zoom = 0;

    update_camera_render_params(m_camera_state);

    // Init gui
    if (!m_options.no_window)
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGui_ImplVulkan_InitInfo imgui_init_info = {};
        imgui_init_info.Instance = m_instance;
        imgui_init_info.PhysicalDevice = m_physical_device;
        imgui_init_info.Device = m_device;
        imgui_init_info.QueueFamily = m_graphics_queue_family_index;
        imgui_init_info.Queue = m_graphics_queue;
        imgui_init_info.DescriptorPool = m_descriptor_pool;
        imgui_init_info.RenderPass = m_main_render_pass;
        imgui_init_info.MinImageCount = m_image_count;
        imgui_init_info.ImageCount = m_image_count;

        ImGui_ImplGlfw_InitForVulkan(m_window, false);
        ImGui_ImplVulkan_Init(&imgui_init_info);
        ImGui::GetIO().IniFilename = nullptr; // disable creating imgui.ini
        ImGui::StyleColorsDark();
        ImGui::GetStyle().Alpha = 0.7f;
        ImGui::GetStyle().WindowBorderSize = 0;

        glfwSetCharCallback(m_window, ImGui_ImplGlfw_CharCallback);

        m_show_gui = !m_options.hide_gui;
    }
}

void Df_vulkan_app::cleanup_resources()
{
    // In headless mode we output the accumulation buffers to files
    if (m_options.no_window)
        write_accum_images_to_files();

    // Cleanup imgui
    if (!m_options.no_window)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    // Cleanup resources
    for (Vulkan_texture& texture : m_material_textures_3d)
        texture.destroy(m_device);
    for (Vulkan_texture& texture : m_material_textures_2d)
        texture.destroy(m_device);
    m_ro_data_buffer.destroy(m_device);
    m_argument_block_buffer.destroy(m_device);
    m_argument_block_staging_buffers.clear();

    m_environment_sampling_data_buffer.destroy(m_device);
    m_environment_map.destroy(m_device);

    for (Vulkan_buffer& buffer : m_render_params_buffers)
        buffer.destroy(m_device);

    vkDestroyQueryPool(m_device, m_query_pool, nullptr);
    vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_display_descriptor_set_layout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_path_trace_descriptor_set_layout, nullptr);
    vkDestroyPipelineLayout(m_device, m_display_pipeline_layout, nullptr);
    vkDestroyPipelineLayout(m_device, m_path_trace_pipeline_layout, nullptr);
    vkDestroyPipeline(m_device, m_path_trace_pipeline, nullptr);
    vkDestroyPipeline(m_device, m_display_pipeline, nullptr);
    vkDestroyRenderPass(m_device, m_path_trace_render_pass, nullptr);
    vkDestroySampler(m_device, m_linear_sampler, nullptr);

    for (Vulkan_texture& texture : m_accum_images)
        texture.destroy(m_device);
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

    for (Vulkan_texture& texture : m_accum_images)
        texture.destroy(m_device);

    create_accumulation_images();
    create_pipelines();

    update_accumulation_image_descriptors();
}

// Updates the application logic. This is called right before the
// next frame is rendered.
void Df_vulkan_app::update(float elapsed_seconds, uint32_t frame_index)
{
    uint64_t timestamps[2];
    VkResult result = vkGetQueryPoolResults(m_device, m_query_pool, frame_index * 2, 2,
        sizeof(uint64_t) * 2, timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (result == VK_SUCCESS)
    {
        auto time_now = glfwGetTime();
        if (time_now - m_last_stats_update > 0.5 || m_first_stats_update)
        {
            VkPhysicalDeviceProperties device_properties;
            vkGetPhysicalDeviceProperties(m_physical_device, &device_properties);

            m_render_time = (float)((timestamps[1] - timestamps[0])
                * (double)device_properties.limits.timestampPeriod * 1e-6);

            m_first_stats_update = false;
            m_last_stats_update = time_now;
        }
    }

    if (!m_options.no_window && m_show_gui)
        do_settings_and_stats_gui();

    if (m_camera_moved)
        m_render_params.progressive_iteration = 0;

    // Update current frame's render params uniform buffer
    std::memcpy(m_render_params_buffers[frame_index].mapped_data,
        &m_render_params, sizeof(Render_params));

    m_render_params.progressive_iteration += m_options.samples_per_iteration;
}

// Populates the current frame's command buffer.
void Df_vulkan_app::render(VkCommandBuffer command_buffer, uint32_t frame_index, uint32_t image_index)
{
    // Update the storage buffer for the material's argument block if any argument value changed
    if (m_material_changed)
    {
        m_material_changed = false;

        void* mapped_data = m_argument_block_staging_buffers[frame_index].map_memory();
        std::memcpy(mapped_data, m_argument_block->get_data(), m_argument_block->get_size());
        m_argument_block_staging_buffers[frame_index].unmap_memory();

        VkBufferCopy copy_region = {};
        copy_region.size = m_argument_block->get_size();

        vkCmdCopyBuffer(command_buffer,
            m_argument_block_staging_buffers[frame_index].get(), m_argument_block_buffer.buffer, 1, &copy_region);
    }

    // Path trace compute pass
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_path_trace_pipeline);

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_path_trace_pipeline_layout, 0, 1, &m_path_trace_descriptor_sets[frame_index],
        0, nullptr);

    if (m_camera_moved)
    {
        m_camera_moved = false;

        for (const Vulkan_texture& accum_image : m_accum_images)
        {
            VkClearColorValue clear_color = { 0.0f, 0.0f, 0.0f, 0.0f };

            VkImageSubresourceRange clear_range;
            clear_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            clear_range.baseMipLevel = 0;
            clear_range.levelCount = 1;
            clear_range.baseArrayLayer = 0;
            clear_range.layerCount = 1;

            vkCmdClearColorImage(command_buffer,
                accum_image.image, VK_IMAGE_LAYOUT_GENERAL, &clear_color, 1, &clear_range);
        }
    }

    vkCmdResetQueryPool(command_buffer, m_query_pool, frame_index * 2, 2);
    vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_query_pool, frame_index * 2);

    uint32_t group_count_x = (m_image_width + g_local_size_x - 1) / g_local_size_x;
    uint32_t group_count_y = (m_image_width + g_local_size_y - 1) / g_local_size_y;
    vkCmdDispatch(command_buffer, group_count_x, group_count_y, 1);

    vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_query_pool, frame_index * 2 + 1);

    for (const Vulkan_texture& accum_image : m_accum_images)
    {
        transitionImageLayout(command_buffer,
            /*image=*/           accum_image.image,
            /*src_access_mask=*/ VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            /*dst_access_mask=*/ VK_ACCESS_SHADER_READ_BIT,
            /*old_layout=*/      VK_IMAGE_LAYOUT_GENERAL,
            /*new_layout=*/      VK_IMAGE_LAYOUT_GENERAL,
            /*src_stage_mask=*/  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            /*dst_stage_mask=*/  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
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

    if (!m_options.no_window && m_show_gui)
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);
    vkCmdEndRenderPass(command_buffer);
}

// Handles keyboard input from the window.
void Df_vulkan_app::key_callback(int key, int action, int mods)
{
    if (action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard)
    {
        if (key == GLFW_KEY_ENTER)
            request_screenshot();

        if (key == GLFW_KEY_SPACE)
            m_show_gui = !m_show_gui;

        if (key == GLFW_KEY_R)
            m_camera_moved = true;

        if (key >= GLFW_KEY_1 && key <= GLFW_KEY_6)
            m_display_buffer_index = key - GLFW_KEY_1;
    }
    
    ImGui_ImplGlfw_KeyCallback(m_window, key, glfwGetKeyScancode(key), action, mods);
}

void Df_vulkan_app::mouse_button_callback(int button, int action)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && !ImGui::GetIO().WantCaptureMouse)
    {
        m_camera_moving = (action == GLFW_PRESS);

        double mouse_x, mouse_y;
        glfwGetCursorPos(m_window, &mouse_x, &mouse_y);
        m_mouse_start.x = static_cast<float>(mouse_x);
        m_mouse_start.y = static_cast<float>(mouse_y);
    }

    ImGui_ImplGlfw_MouseButtonCallback(m_window, button, action, 0);
}

void Df_vulkan_app::mouse_scroll_callback(float offset_x, float offset_y)
{
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        if (offset_y < 0.0f)
            m_camera_state.zoom -= 1.0f;
        else if (offset_y > 0.0f)
            m_camera_state.zoom += 1.0f;

        update_camera_render_params(m_camera_state);
        m_camera_moved = true;
    }

    ImGui_ImplGlfw_ScrollCallback(m_window, offset_x, offset_y);
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

// Type trait to get the value type for a given type.
template<typename T> struct Value_trait { /* error */ };
template<> struct Value_trait<float> { using IVALUE_TYPE = mi::mdl::IValue_float; };
template<> struct Value_trait<const char*> { using IVALUE_TYPE = mi::mdl::IValue_string; };

template<typename T>
bool get_annotation_argument_value(mi::mdl::DAG_call const* anno, int index, T& res)
{
    const mi::mdl::DAG_constant* dag_const =
        mi::mdl::as<mi::mdl::DAG_constant>(anno->get_argument(index));
    if (!dag_const)
        return false;

    using IValue_type = typename Value_trait<T>::IVALUE_TYPE;

    const IValue_type* val = mi::mdl::as<IValue_type>(dag_const->get_value());
    if (!val)
        return false;

    res = val->get_value();
    return true;
}

void Df_vulkan_app::do_settings_and_stats_gui()
{
    bool changed = false;

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Stats overlay
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoDecoration
        | ImGuiWindowFlags_AlwaysAutoResize
        | ImGuiWindowFlags_NoSavedSettings
        | ImGuiWindowFlags_NoFocusOnAppearing
        | ImGuiWindowFlags_NoNav
        | ImGuiWindowFlags_NoMove;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 window_pos(viewport->WorkPos.x + viewport->WorkSize.x - 10.0f, viewport->WorkPos.y + 10.0f);
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, ImVec2(1, 0));
    ImGui::SetNextWindowSize(ImVec2(230, 0));
    ImGui::SetNextWindowBgAlpha(0.4f);

    ImGui::Begin("stats overlay", nullptr, window_flags);
    ImGui::Text("%s", ("progressive iteration: " + std::to_string(m_render_params.progressive_iteration)).c_str());
    ImGui::Separator();
    ImGui::Text("%s", ("render time: " + std::to_string(m_render_time) + " ms").c_str());
    ImGui::End();

    // Settings window
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, m_options.use_class_compilation ? 550.0f : 275.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(0.4f);
    ImGui::Begin("Settings");
    ImGui::PushItemWidth(-200);

    ImGui::Text("Display options:");
    ImGui::Separator();
    if (ImGui::Checkbox("Enable VSync", &m_vsync_enabled))
    {
        set_vsync_enabled(m_vsync_enabled);
        changed = true;
    }
    {
        const char* options[] = { "Beauty", "Albedo", "Albedo (Diffuse)", "Albedo (Glossy)", "Normal", "Roughness"};
        changed |= ImGui::Combo("Buffer", (int*)&m_display_buffer_index, options, IM_ARRAYSIZE(options));
    }
    if (m_options.enable_bsdf_flags)
    {
        const char* options[] = { "None", "Reflect only", "Transmit only", "Reflect+Transmit" };
        changed |= ImGui::Combo("BSDF flags", (int*)&m_render_params.bsdf_data_flags, options, IM_ARRAYSIZE(options));
    }
    ImGui::Spacing();

    ImGui::Text("Light parameters:");
    ImGui::Separator();
    changed |= ImGui::ColorEdit3("Point Light Color", m_render_params.point_light_color.begin());
    changed |= ImGui::DragFloat("Point Light Intensity", &m_render_params.point_light_intensity, 10.0f, 0.0f, std::numeric_limits<float>::max());
    changed |= ImGui::DragFloat("Environment Intensity", &m_render_params.environment_intensity_factor, 0.01f, 0.0f, std::numeric_limits<float>::max());
    ImGui::Spacing();

    ImGui::Text("Material parameters:");
    ImGui::Separator();
    if (m_options.use_class_compilation)
    {
        bool material_changed = false;

        mi::base::Handle<const mi::mdl::IGenerated_code_value_layout> layout(
            m_target_code->get_argument_block_layout(m_argument_block_index));

        auto mat_instance = m_material_instance->get_material_instance();
        for (size_t i = 0; i < mat_instance->get_parameter_count(); i++)
        {
            const char* parameter_name = mat_instance->get_parameter_name(i);
            const mi::mdl::IValue* arg = mat_instance->get_parameter_default(i);

            const char* display_name = parameter_name;
            float range_min = -std::numeric_limits<float>::max();
            float range_max = std::numeric_limits<float>::max();
            
            size_t dag_param_index = m_material_instance->get_dag_parameter_index(parameter_name);
            if (dag_param_index != ~0)
            {
                bool has_soft_range = false;
                for (size_t k = 0; k < m_material_instance->get_dag_parameter_annotation_count(dag_param_index); k++)
                {
                    const mi::mdl::DAG_call* anno = mi::mdl::as<mi::mdl::DAG_call>(
                        m_material_instance->get_dag_parameter_annotation(dag_param_index, k));
                    if (!anno)
                        continue;
                    switch (anno->get_semantic())
                    {
                    case mi::mdl::IDefinition::DS_SOFT_RANGE_ANNOTATION:
                        has_soft_range = true;
                        get_annotation_argument_value(anno, 0, range_min);
                        get_annotation_argument_value(anno, 1, range_max);
                        break;
                    case mi::mdl::IDefinition::DS_HARD_RANGE_ANNOTATION:
                        if (!has_soft_range)
                        {
                            get_annotation_argument_value(anno, 0, range_min);
                            get_annotation_argument_value(anno, 1, range_max);
                        }
                        break;
                    case mi::mdl::IDefinition::DS_DISPLAY_NAME_ANNOTATION:
                        get_annotation_argument_value(anno, 0, display_name);
                        break;
                    default:
                        break;
                    }
                }
            }

            mi::mdl::IGenerated_code_value_layout::State state = layout->get_nested_state(i);
            mi::mdl::IValue::Kind argument_kind;
            size_t argument_size;
            size_t data_offset = layout->get_layout(argument_kind, argument_size, state);
            char* data_ptr = m_argument_block->get_data() + data_offset;

            switch (argument_kind)
            {
            case mi::mdl::IValue::VK_FLOAT:
                material_changed |= ImGui::DragFloat(display_name, reinterpret_cast<float*>(data_ptr), 0.01f, range_min, range_max);
                break;
            case mi::mdl::IValue::VK_INT:
                material_changed |= ImGui::DragInt(display_name, reinterpret_cast<int*>(data_ptr), 0.25f, (int)range_min, (int)range_max);
                break;
            case mi::mdl::IValue::VK_BOOL:
                material_changed |= ImGui::Checkbox(display_name, reinterpret_cast<bool*>(data_ptr));
                break;
            case mi::mdl::IValue::VK_VECTOR:
            {
                const mi::mdl::IValue_vector* value = mi::mdl::as<mi::mdl::IValue_vector>(arg);
                const mi::mdl::IType_vector* vector_type = value->get_type();
                if (vector_type->get_size() == 2)
                    material_changed |= ImGui::DragFloat2(display_name, reinterpret_cast<float*>(data_ptr), 0.01f, range_min, range_max);
                else if (vector_type->get_size() == 3)
                    material_changed |= ImGui::DragFloat3(display_name, reinterpret_cast<float*>(data_ptr), 0.01f, range_min, range_max);
                else if (vector_type->get_size() == 4)
                    material_changed |= ImGui::DragFloat4(display_name, reinterpret_cast<float*>(data_ptr), 0.01f, range_min, range_max);
                break;
            }
            case mi::mdl::IValue::VK_RGB_COLOR:
                material_changed |= ImGui::ColorEdit3(display_name, reinterpret_cast<float*>(data_ptr));
                break;
            case mi::mdl::IValue::VK_ENUM:
            {
                const mi::mdl::IValue_enum* value = mi::mdl::as<mi::mdl::IValue_enum>(arg);
                const mi::mdl::IType_enum* enum_type = value->get_type();

                int* current_value = reinterpret_cast<int*>(data_ptr);
                int current_index = 0;

                std::vector<const char*> names;
                names.reserve(enum_type->get_value_count());
                for (size_t index = 0; index < enum_type->get_value_count(); index++)
                {
                    const mi::mdl::IType_enum::Value* enum_value = enum_type->get_value(index);
                    names.push_back(enum_value->get_symbol()->get_name());
                    if (*current_value == enum_value->get_code())
                        current_index = static_cast<int>(index);
                }

                material_changed |= ImGui::Combo(display_name, &current_index, names.data(), names.size());
                *current_value = enum_type->get_value(current_index)->get_code();
                break;
            }
            case mi::mdl::IValue::VK_STRING:
            {
                String_constant_table& constant_table = m_target_code->get_string_constant_table();

                std::vector<char> buffer;
                size_t max_len = constant_table.get_max_length();
                max_len = max_len > 63 ? max_len + 1 : 64;
                buffer.resize(max_len);

                int* value = reinterpret_cast<int*>(data_ptr);
                const char* str = constant_table.get_string(*value);
                strcpy(buffer.data(), str ? str : "");

                if (ImGui::InputText(display_name, buffer.data(), buffer.size(), ImGuiInputTextFlags_EnterReturnsTrue))
                {
                    *value = constant_table.get_id_for_string(buffer.data());
                    material_changed = true;
                }
                break;
            }
            case mi::mdl::IValue::VK_TEXTURE:
            {
                std::vector<const char*> urls;
                urls.reserve(m_target_code->get_texture_count());
                urls.push_back("<unset>");
                for (size_t index = 1; index < m_target_code->get_texture_count(); index++)
                    urls.push_back(m_target_code->get_texture((int)index)->get_url());

                material_changed |= ImGui::Combo(display_name, reinterpret_cast<int*>(data_ptr), urls.data(), urls.size());
                break;
            }
            default:
                break;
            }
        }

        changed |= material_changed;
        m_material_changed = material_changed;
    }
    else
        ImGui::Text("Parameter editing requires class compilation.");

    if (changed)
        m_camera_moved = true;

    ImGui::End();
    ImGui::Render();
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

    for (Vulkan_texture& accum_image : m_accum_images)
    {
        VK_CHECK(vkCreateImage(m_device, &image_create_info, nullptr, &accum_image.image));

        accum_image.device_memory = allocate_and_bind_image_memory(
            m_device, m_physical_device, accum_image.image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    { // Transition image layout
        Temporary_command_buffer command_buffer(m_device, m_command_pool);
        command_buffer.begin();

        for (Vulkan_texture& accum_image : m_accum_images)
        {
            transitionImageLayout(command_buffer.get(),
                /*image=*/           accum_image.image,
                /*src_access_mask=*/ 0,
                /*dst_access_mask=*/ VK_ACCESS_SHADER_READ_BIT,
                /*old_layout=*/      VK_IMAGE_LAYOUT_UNDEFINED,
                /*new_layout=*/      VK_IMAGE_LAYOUT_GENERAL,
                /*src_stage_mask=*/  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                /*dst_stage_mask=*/  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
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

    for (Vulkan_texture& accum_image : m_accum_images)
    {
        image_view_create_info.image = accum_image.image;

        VK_CHECK(vkCreateImageView(
            m_device, &image_view_create_info, nullptr, &accum_image.image_view));
    }
}

VkShaderModule Df_vulkan_app::create_path_trace_shader_module()
{
    std::string df_glsl_source = m_target_code->get_src_code();
    std::string path_trace_shader_source = read_text_file(get_executable_folder() + "/" + "path_trace.comp");

    std::vector<std::string> defines;
    defines.push_back("LOCAL_SIZE_X=" + std::to_string(g_local_size_x));
    defines.push_back("LOCAL_SIZE_Y=" + std::to_string(g_local_size_y));

    defines.push_back("BINDING_RENDER_PARAMS=" + std::to_string(g_binding_render_params));
    defines.push_back("BINDING_ENV_MAP=" + std::to_string(g_binding_environment_map));
    defines.push_back("BINDING_ENV_MAP_SAMPLING_DATA=" + std::to_string(g_binding_environment_sampling_data));
    defines.push_back("BINDING_BEAUTY_BUFFER=" + std::to_string(g_binding_beauty_buffer));
    defines.push_back("BINDING_AUX_ALBEDO_DIFFUSE_BUFFER=" + std::to_string(g_binding_aux_albedo_diffuse_buffer));
    defines.push_back("BINDING_AUX_ALBEDO_GLOSSY_BUFFER=" + std::to_string(g_binding_aux_albedo_glossy_buffer));
    defines.push_back("BINDING_AUX_NORMAL_BUFFER=" + std::to_string(g_binding_aux_normal_buffer));
    defines.push_back("BINDING_AUX_ROUGHNESS_BUFFER=" + std::to_string(g_binding_aux_roughness_buffer));

    defines.push_back("MDL_SET_MATERIAL_TEXTURES_2D=" + std::to_string(g_set_material_textures));
    defines.push_back("MDL_SET_MATERIAL_TEXTURES_3D=" + std::to_string(g_set_material_textures));
    defines.push_back("MDL_SET_MATERIAL_ARGUMENT_BLOCK=" + std::to_string(g_set_argument_block_buffer));
    defines.push_back("MDL_SET_MATERIAL_RO_DATA_SEGMENT=" + std::to_string(g_set_ro_data_buffer));
    defines.push_back("MDL_BINDING_MATERIAL_TEXTURES_2D=" + std::to_string(g_binding_material_textures_2d));
    defines.push_back("MDL_BINDING_MATERIAL_TEXTURES_3D=" + std::to_string(g_binding_material_textures_3d));
    defines.push_back("MDL_BINDING_MATERIAL_ARGUMENT_BLOCK=" + std::to_string(g_binding_argument_block_buffer));
    defines.push_back("MDL_BINDING_MATERIAL_RO_DATA_SEGMENT=" + std::to_string(g_binding_ro_data_buffer));

    if (m_options.tex_results_cache_size > 0)
        defines.push_back("NUM_TEX_RESULTS=" + std::to_string(m_options.tex_results_cache_size));

    if (m_options.enable_ro_segment)
        defines.push_back("USE_RO_DATA_SEGMENT");

    // Check if functions for backface were generated
    for (size_t i = 0; i < m_target_code->get_callable_function_count(); i++)
    {
        const char* fname = m_target_code->get_callable_function(i);

        if (std::strcmp(fname, "mdl_backface_bsdf_sample") == 0)
            defines.push_back("MDL_HAS_BACKFACE_BSDF");
        else if (std::strcmp(fname, "mdl_backface_edf_sample") == 0)
            defines.push_back("MDL_HAS_BACKFACE_EDF");
        else if (std::strcmp(fname, "mdl_backface_emission_intensity") == 0)
            defines.push_back("MDL_HAS_BACKFACE_EMISSION_INTENSITY");
    }

    auto t0 = std::chrono::steady_clock::now();
    VkShaderModule shader_module = create_shader_module_from_sources(
        m_device, { df_glsl_source, path_trace_shader_source }, EShLangCompute,
        defines, m_options.enable_shader_optimization);
    auto t1 = std::chrono::steady_clock::now();
    if (!m_path_trace_pipeline) // Print only the first time
        std::cout << "Compile GLSL to SPIR-V: " << std::chrono::duration<float>(t1 - t0).count() << "s\n";

    return shader_module;
}

// Creates the descriptors set layout which is used to create the
// pipeline layout. Here the number of material resources is declared.
void Df_vulkan_app::create_descriptor_set_layouts()
{
    {
        auto make_binding = [](uint32_t binding, VkDescriptorType type, uint32_t count)
        {
            VkDescriptorSetLayoutBinding layout_binding = {};
            layout_binding.binding = binding;
            layout_binding.descriptorType = type;
            layout_binding.descriptorCount = count;
            layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            return layout_binding;
        };

        // We reserve enough space for the maximum amount of textures we expect (arbitrary number here)
        // in combination with descriptor indexing. The partially bound flag is used because all textures
        // in MDL share the same index range, but in GLSL we must use seperate arrays. This leads to "holes"
        // in the GLSL texture arrays since for each index only one of the texture arrays is populated.
        // We can also leave the descriptor sets empty with the partially bound flag, in case no
        // textures of the corresponding shapes is used.
        // See Df_vulkan_app::create_descriptor_pool_and_sets() for how the descriptor sets are populated.
        const uint32_t max_num_textures = 100;

        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(make_binding(g_binding_render_params, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1));
        bindings.push_back(make_binding(g_binding_environment_map, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1));
        bindings.push_back(make_binding(g_binding_environment_sampling_data, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1));
        bindings.push_back(make_binding(g_binding_material_textures_2d, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_num_textures));
        bindings.push_back(make_binding(g_binding_material_textures_3d, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_num_textures));
        bindings.push_back(make_binding(g_binding_beauty_buffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1));
        bindings.push_back(make_binding(g_binding_aux_albedo_diffuse_buffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1));
        bindings.push_back(make_binding(g_binding_aux_albedo_glossy_buffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1));
        bindings.push_back(make_binding(g_binding_aux_normal_buffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1));
        bindings.push_back(make_binding(g_binding_aux_roughness_buffer, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1));
        bindings.push_back(make_binding(g_binding_ro_data_buffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1));
        bindings.push_back(make_binding(g_binding_argument_block_buffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1));

        VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
        descriptor_set_layout_create_info.sType
            = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_set_layout_create_info.bindingCount = static_cast<uint32_t>(bindings.size());
        descriptor_set_layout_create_info.pBindings = bindings.data();

        std::vector<VkDescriptorBindingFlags> binding_flags(bindings.size(), 0);
        binding_flags[3] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT; // g_binding_material_textures_2d
        binding_flags[4] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT; // g_binding_material_textures_3d

        VkDescriptorSetLayoutBindingFlagsCreateInfoEXT descriptor_set_layout_binding_flags = {};
        descriptor_set_layout_binding_flags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
        descriptor_set_layout_binding_flags.bindingCount = static_cast<uint32_t>(bindings.size());
        descriptor_set_layout_binding_flags.pBindingFlags = binding_flags.data();
        descriptor_set_layout_create_info.pNext = &descriptor_set_layout_binding_flags;

        VK_CHECK(vkCreateDescriptorSetLayout(
            m_device, &descriptor_set_layout_create_info, nullptr, &m_path_trace_descriptor_set_layout));
    }

    {
        VkDescriptorSetLayoutBinding layout_bindings[5];
        for (uint32_t i = 0; i < 5; i++)
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
        VkShaderModule fullscreen_triangle_vertex_shader = create_shader_module_from_file(
            m_device, "display.vert", EShLangVertex, {}, m_options.enable_shader_optimization);
        VkShaderModule display_fragment_shader = create_shader_module_from_file(
            m_device, "display.frag", EShLangFragment, {}, m_options.enable_shader_optimization);

        m_display_pipeline = create_fullscreen_triangle_graphics_pipeline(
            m_device, m_display_pipeline_layout, fullscreen_triangle_vertex_shader,
            display_fragment_shader, m_main_render_pass, 0, m_image_width, m_image_height, false);

        vkDestroyShaderModule(m_device, fullscreen_triangle_vertex_shader, nullptr);
        vkDestroyShaderModule(m_device, display_fragment_shader, nullptr);
    }
}

void Df_vulkan_app::create_render_params_buffers()
{
    m_render_params_buffers.resize(m_image_count);

    for (uint32_t i = 0; i < m_image_count; i++)
    {
        VkBufferCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        create_info.size = sizeof(Render_params);
        create_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK(vkCreateBuffer(
            m_device, &create_info, nullptr, &m_render_params_buffers[i].buffer));

        m_render_params_buffers[i].device_memory = allocate_and_bind_buffer_memory(
            m_device, m_physical_device, m_render_params_buffers[i].buffer,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VK_CHECK(vkMapMemory(m_device, m_render_params_buffers[i].device_memory,
            0, sizeof(Render_params), 0, &m_render_params_buffers[i].mapped_data));
    }
}

void Df_vulkan_app::create_environment_map()
{
    Texture_data env_map_data(
        m_options.hdr_file.c_str(), m_mdl_compiler->create_entity_resolver(nullptr));
    if (!env_map_data.is_valid())
    {
        std::cerr << "Error: Requested environment map texture data is invalid\n";
        exit_failure();
    }

    auto image = env_map_data.get_image();

    const OIIO::ImageSpec& spec = image->spec();
    const mi::Uint32 res_x = spec.width;
    const mi::Uint32 res_y = spec.height;

    std::vector<float> pixel_data(4 * res_x * res_y);
    mi::Sint32 bytes_per_row = 4 * res_x * sizeof(float);
    bool success = image->read_image(
        /*subimage*/ 0,
        /*miplevel*/ 0,
        /*chbegin*/ 0,
        /*chend*/ 4,
        OIIO::TypeDesc::FLOAT,
        pixel_data.data() + (res_y - 1) * 4 * res_x,
        /*xstride*/ 4 * sizeof(float),
        /*ystride*/ -bytes_per_row);
    if (!success)
    {
        std::cerr << "Error: Failed to read image data for environment map\n";
        exit_failure();
    }

    if (spec.nchannels <= 3)
        for (size_t i = 0, n = pixel_data.size(); i < n; i += 4)
            pixel_data[i + 3] = 1.0f;

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

    m_environment_map.device_memory = allocate_and_bind_image_memory(
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
        Staging_buffer staging_buffer(m_device, m_physical_device,
            staging_buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        void* mapped_data = staging_buffer.map_memory();
        std::memcpy(mapped_data, pixel_data.data(), staging_buffer_size);
        staging_buffer.unmap_memory();

        // Upload the read-only data from the staging buffer into the storage buffer
        Temporary_command_buffer command_buffer(m_device, m_command_pool);
        command_buffer.begin();

        transitionImageLayout(command_buffer.get(),
            /*image=*/           m_environment_map.image,
            /*src_access_mask=*/ 0,
            /*dst_access_mask=*/ VK_ACCESS_TRANSFER_WRITE_BIT,
            /*old_layout=*/      VK_IMAGE_LAYOUT_UNDEFINED,
            /*new_layout=*/      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            /*src_stage_mask=*/  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            /*dst_stage_mask=*/  VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy copy_region = {};
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageExtent = { res_x, res_y, 1 };

        vkCmdCopyBufferToImage(
            command_buffer.get(), staging_buffer.get(), m_environment_map.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

        transitionImageLayout(command_buffer.get(),
            /*image=*/           m_environment_map.image,
            /*src_access_mask=*/ VK_ACCESS_TRANSFER_WRITE_BIT,
            /*dst_access_mask=*/ VK_ACCESS_SHADER_READ_BIT,
            /*old_layout=*/      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            /*new_layout=*/      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            /*src_stage_mask=*/  VK_PIPELINE_STAGE_TRANSFER_BIT,
            /*dst_stage_mask=*/  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

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
    const float* pixels = pixel_data.data();

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
    m_environment_sampling_data_buffer = create_storage_buffer(m_device, m_physical_device,
        m_graphics_queue, m_command_pool, env_accel_data.data(), env_accel_data.size() * sizeof(Env_accel));

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
    const VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100 }
    };

    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.maxSets = m_image_count + 2; // img_cnt for path_trace + 1 set for display and imgui each
    descriptor_pool_create_info.poolSizeCount = std::size(pool_sizes);
    descriptor_pool_create_info.pPoolSizes = pool_sizes;
    descriptor_pool_create_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; // required for imgui

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
    descriptor_image_infos.reserve(1000);

    for (uint32_t i = 0; i < m_image_count; i++)
    {
        VkWriteDescriptorSet descriptor_write = {};
        descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_write.dstSet = m_path_trace_descriptor_sets[i];

        { // Render params buffer
            VkDescriptorBufferInfo descriptor_buffer_info = {};
            descriptor_buffer_info.buffer = m_render_params_buffers[i].buffer;
            descriptor_buffer_info.range = VK_WHOLE_SIZE;
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

        // Material textures
        // 
        // We rely on the partially bound bit when creating the descriptor set layout,
        // so we can leave holes in the descriptor sets (or leave them empty).
        // For each MDL texture index only one of the GLSL texture arrays is populated.
        size_t texture_2d_index = 0;
        size_t texture_3d_index = 0;

        for (size_t tex = 1; tex < m_target_code->get_texture_count(); tex++)
        {
            VkDescriptorImageInfo descriptor_image_info = {};
            descriptor_image_info.sampler = m_linear_sampler;
            descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
            descriptor_write.dstArrayElement = static_cast<uint32_t>(tex - 1);
            descriptor_write.descriptorCount = 1;
            descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

            mi::mdl::IType_texture::Shape shape = m_target_code->get_texture(tex)->get_shape();
            if (shape == mi::mdl::IType_texture::TS_2D)
            {
                descriptor_image_info.imageView = m_material_textures_2d[texture_2d_index++].image_view;
                descriptor_image_infos.push_back(descriptor_image_info);

                descriptor_write.dstBinding = g_binding_material_textures_2d;
                descriptor_write.pImageInfo = &descriptor_image_infos.back();
                descriptor_writes.push_back(descriptor_write);
            }
            else if (shape == mi::mdl::IType_texture::TS_3D || shape == mi::mdl::IType_texture::TS_BSDF_DATA)
            {
                descriptor_image_info.imageView = m_material_textures_3d[texture_3d_index++].image_view;
                descriptor_image_infos.push_back(descriptor_image_info);

                descriptor_write.dstBinding = g_binding_material_textures_3d;
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

        // Material argument block buffer
        if (m_argument_block_buffer.buffer)
        {
            VkDescriptorBufferInfo descriptor_buffer_info = {};
            descriptor_buffer_info.buffer = m_argument_block_buffer.buffer;
            descriptor_buffer_info.range = VK_WHOLE_SIZE;
            descriptor_buffer_infos.push_back(descriptor_buffer_info);

            VkWriteDescriptorSet descriptor_write = {};
            descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
            descriptor_write.dstBinding = g_binding_argument_block_buffer;
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

void Df_vulkan_app::create_query_pool()
{
    VkQueryPoolCreateInfo query_pool_create_info = {};
    query_pool_create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_pool_create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_pool_create_info.queryCount = m_image_count * 2;

    VK_CHECK(vkCreateQueryPool(m_device, &query_pool_create_info, nullptr, &m_query_pool));

    Temporary_command_buffer command_buffer(m_device, m_command_pool);
    command_buffer.begin();
    vkCmdResetQueryPool(command_buffer.get(), m_query_pool, 0, query_pool_create_info.queryCount);
    command_buffer.end_and_submit(m_graphics_queue);
}

void Df_vulkan_app::update_accumulation_image_descriptors()
{
    std::vector<VkWriteDescriptorSet> descriptor_writes;

    std::vector<VkDescriptorImageInfo> descriptor_image_infos;
    descriptor_image_infos.reserve(m_image_count * ACCUM_IMAGE_COUNT + ACCUM_IMAGE_COUNT);

    const uint32_t accum_image_bindings[] = {
        g_binding_beauty_buffer,
        g_binding_aux_albedo_diffuse_buffer,
        g_binding_aux_albedo_glossy_buffer,
        g_binding_aux_normal_buffer,
        g_binding_aux_roughness_buffer
    };

    for (uint32_t i = 0; i < m_image_count; i++)
    {
        VkDescriptorImageInfo descriptor_image_info = {};
        descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet descriptor_write = {};
        descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_write.dstSet = m_path_trace_descriptor_sets[i];
        descriptor_write.descriptorCount = 1;
        descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

        for (size_t j = 0; j < ACCUM_IMAGE_COUNT; j++)
        {
            descriptor_image_info.imageView = m_accum_images[j].image_view;
            descriptor_image_infos.push_back(descriptor_image_info);

            descriptor_write.dstBinding = accum_image_bindings[j];
            descriptor_write.pImageInfo = &descriptor_image_infos.back();
            descriptor_writes.push_back(descriptor_write);
        }
    }

    VkDescriptorImageInfo descriptor_info = {};
    descriptor_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet descriptor_write = {};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = m_display_descriptor_set;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorCount = 1;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;

    for (uint32_t i = 0; i < ACCUM_IMAGE_COUNT; i++)
    {
        descriptor_info.imageView = m_accum_images[i].image_view;
        descriptor_image_infos.push_back(descriptor_info);

        descriptor_write.dstBinding = i;
        descriptor_write.pImageInfo = &descriptor_image_infos.back();
        descriptor_writes.push_back(descriptor_write);
    }

    vkUpdateDescriptorSets(
        m_device, static_cast<uint32_t>(descriptor_writes.size()),
        descriptor_writes.data(), 0, nullptr);
}

void Df_vulkan_app::write_accum_images_to_files()
{
    auto export_pixels_rgba32f = [&](const std::vector<uint8_t>& pixels, const std::string& filename)
    {
        std::cout << "Saving image to " << filename << "\n";

        auto image = OIIO::ImageOutput::create(filename);
        if (!image)
        {
            std::cerr << "Error: Failed to create output image file for " << filename << "\n";
            exit_failure();
        }

        OIIO::ImageSpec spec(m_image_width, m_image_height, 3, OIIO::TypeDesc::FLOAT);
        image->open(filename, spec);
        image->write_image(OIIO::TypeDesc::FLOAT, pixels.data(), sizeof(float) * 4);
        image->close();
    };

    uint32_t image_bpp = get_image_format_bpp(g_accumulation_texture_format);
    std::vector<uint8_t> image_pixels[ACCUM_IMAGE_COUNT];
    for (uint32_t i = 0; i < ACCUM_IMAGE_COUNT; i++)
    {
        image_pixels[i] = copy_image_to_buffer(
            m_device, m_physical_device, m_command_pool, m_graphics_queue,
            m_accum_images[i].image, m_image_width, m_image_height, image_bpp,
            VK_IMAGE_LAYOUT_GENERAL, true);
    }

    std::vector<uint8_t> albedo_pixels(m_image_width * m_image_height * image_bpp);
    float* albedo_ptr = reinterpret_cast<float*>(albedo_pixels.data());
    float* albedo_diffuse_ptr = reinterpret_cast<float*>(image_pixels[ACCUM_IMAGE_AUX_ALBEDO_DIFFUSE].data());
    float* albedo_glossy_ptr = reinterpret_cast<float*>(image_pixels[ACCUM_IMAGE_AUX_ALBEDO_GLOSSY].data());
    for (size_t i = 0; i < albedo_pixels.size() / sizeof(float); i++)
        albedo_ptr[i] = albedo_diffuse_ptr[i] + albedo_glossy_ptr[i];

    std::string filename_base = m_options.output_file;
    std::string filename_ext;

    size_t dot_pos = m_options.output_file.rfind('.');
    if (dot_pos != std::string::npos)
    {
        filename_base = m_options.output_file.substr(0, dot_pos);
        filename_ext = m_options.output_file.substr(dot_pos);
    }

    export_pixels_rgba32f(image_pixels[ACCUM_IMAGE_BEAUTY], filename_base + filename_ext);
    export_pixels_rgba32f(albedo_pixels, filename_base + "_albedo" + filename_ext);
    export_pixels_rgba32f(image_pixels[ACCUM_IMAGE_AUX_ALBEDO_DIFFUSE], filename_base + "_albedo_diffuse" + filename_ext);
    export_pixels_rgba32f(image_pixels[ACCUM_IMAGE_AUX_ALBEDO_GLOSSY], filename_base + "_albedo_glossy" + filename_ext);
    export_pixels_rgba32f(image_pixels[ACCUM_IMAGE_AUX_NORMAL], filename_base + "_normal" + filename_ext);
    export_pixels_rgba32f(image_pixels[ACCUM_IMAGE_AUX_ROUGHNESS], filename_base + "_roughness" + filename_ext);
}


//------------------------------------------------------------------------------
// Command line helpers
//------------------------------------------------------------------------------
void print_usage(char const* prog_name)
{
    std::cout
        << "Usage: " << prog_name << " [options] [<material_name|full_mdle_path>]\n"
        << "Options:\n"
        << "  -h|--help                   print this text and exit\n"
        << "  --nowin                     don't show interactive display\n"
        << "  --res <res_x> <res_y>       resolution (default: 1024x768)\n"
        << "  --numimg <n>                swapchain image count (default: 3)\n"
        << "  --device <id>               run on supprted GPU <id>\n"
        << "  -o|--output <outputfile>    image file to write result in nowin mode (default: output.exr)\n"
        << "  --spp <num>                 samples per pixel, only used for --nowin (default: 4096)\n"
        << "  --spi <num>                 samples per render loop iteration (default: 8)\n"
        << "  --max_path_length <num>     maximum path length (default: 4)\n"
        << "  -f|--fov <fov>              the camera field of view in degrees (default: 96.0)\n"
        << "  --cam <x> <y> <z>           set the camera position (default: 0 0 3).\n"
        << "                              The camera will always look towards (0, 0, 0)\n"
        << "  -l|--light <x> <y> <z>      adds an omnidirectional light with the given position\n"
        << "             <r> <g> <b>      and intensity\n"
        << "  --hdr <path>                hdr image file used for the environment map\n"
        << "                              (default: nvidia/sdk_examples/resources/environment.hdr)\n"
        << "  --hdr_intensity <value>     intensity of the environment map (default: 1.0)\n"
        << "  --nocc                      don't compile the material using class compilation\n"
        << "  --tex_res <num>             size of the texture results cache\n"
        << "  --enable_ro_segment         enable the read-only data segment\n"
        << "  --disable_ssbo              disable use of an ssbo for constants\n"
        << "  --max_const_data <size>     set the maximum size of constants in bytes in the\n"
        << "                              generated code (requires read-only data segment or\n"
        << "                              ssbo, default 1024)\n"
        << "  -p|--mdl_path <path>        additional MDL search path, can occur multiple times\n"
        << "  --vkdebug                   enable the Vulkan validation layers\n"
        << "  --no_shader_opt             disables shader SPIR-V optimization\n"
        << "  --dump_glsl                 outputs the generated GLSL target code to a file\n"
        << "  --hide_gui                  hide the settings gui. Can be toggled with SPACE\n"
        << "  --allowed_scatter_mode <m>  limits the allowed scatter mode to \"none\", \"reflect\",\n"
        << "                              \"transmit\" or \"reflect_and_transmit\"\n"
        << "                              (default: restriction disabled)"
        << std::endl;

    exit(EXIT_FAILURE);
}

void parse_command_line(int argc, char* argv[], Options& options)
{
    options.additional_mdl_paths.push_back(get_samples_mdl_root());

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg[0] == '-')
        {
            if (arg == "--nowin")
                options.no_window = true;
            else if (arg == "--res" && i < argc - 2)
            {
                options.res_x = std::max(atoi(argv[++i]), 1);
                options.res_y = std::max(atoi(argv[++i]), 1);
            }
            else if (arg == "--numimg" && i < argc - 1)
                options.num_images = std::max(atoi(argv[++i]), 2);
            else if (arg == "--device" && i < argc - 1)
                options.device_index = std::max(atoi(argv[++i]), -1);
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
            else if (arg == "--cam" && i < argc - 3)
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
                options.light_enabled = true;
            }
            else if (arg == "--hdr" && i < argc - 1)
                options.hdr_file = argv[++i];
            else if (arg == "--hdr_intensity" && i < argc - 1)
                options.hdr_intensity = static_cast<float>(std::atof(argv[++i]));
            else if ((arg == "-p" || arg == "--mdl_path") && i < argc - 1)
                options.additional_mdl_paths.push_back(argv[++i]);
            else if (arg == "--nocc")
                options.use_class_compilation = false;
            else if (arg == "--tex_res")
                options.tex_results_cache_size = uint32_t(std::atoi(argv[++i]));
            else if (arg == "--enable_ro_segment")
                options.enable_ro_segment = true;
            else if (arg == "--disable_ssbo")
                options.disable_ssbo = true;
            else if (arg == "--max_const_data")
                options.max_const_data = uint32_t(std::atoi(argv[++i]));
            else if (arg == "--vkdebug")
                options.enable_validation_layers = true;
            else if (arg == "--no_shader_opt")
                options.enable_shader_optimization = false;
            else if (arg == "--dump_glsl")
                options.dump_glsl = true;
            else if (arg == "--hide_gui")
                options.hide_gui = true;
            else if (arg == "--allowed_scatter_mode" && i < argc - 1)
            {
                options.enable_bsdf_flags = true;
                std::string mode(argv[++i]);
                if (mode == "none")
                    options.allowed_scatter_mode = mi::mdl::DF_FLAGS_NONE;
                else if (mode == "reflect")
                    options.allowed_scatter_mode = mi::mdl::DF_FLAGS_ALLOW_REFLECT;
                else if (mode == "transmit")
                    options.allowed_scatter_mode = mi::mdl::DF_FLAGS_ALLOW_TRANSMIT;
                else if (mode == "reflect_and_transmit")
                    options.allowed_scatter_mode = mi::mdl::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
                else
                {
                    std::cout << "Unknown allowed_scatter_mode: \"" << mode << "\"" << std::endl;
                    print_usage(argv[0]);
                }
            }
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
    parse_command_line(argc, argv, options);

    // Access the MDL core compiler
    mi::base::Handle<mi::mdl::IMDL> mdl_compiler(load_mdl_compiler());
    check_success(mdl_compiler);

    {
        // Generate GLSL code for the material
        mi::mdl::ICode_generator::Target_language target_backend = mi::mdl::ICode_generator::TL_GLSL;

        // Configure compiler backend options
        mi::Uint32 backend_options = BACKEND_OPTIONS_ENABLE_AUX; // We always enable auxilary functions in this example
        if (options.enable_ro_segment)
            backend_options |= BACKEND_OPTIONS_ENABLE_RO_SEGMENT;

        std::unordered_map<std::string, std::string> additional_backend_options;
        additional_backend_options.emplace(MDL_JIT_OPTION_GLSL_VERSION, "450");
        additional_backend_options.emplace(MDL_JIT_OPTION_GLSL_PLACE_UNIFORMS_INTO_SSBO,
            (!options.disable_ssbo && !options.enable_ro_segment) ? "true" : "false");
        additional_backend_options.emplace(MDL_JIT_OPTION_GLSL_MAX_CONST_DATA, std::to_string(options.max_const_data));
        additional_backend_options.emplace(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_BINDING, std::to_string(g_binding_ro_data_buffer));
        additional_backend_options.emplace(MDL_JIT_OPTION_GLSL_UNIFORM_SSBO_SET, std::to_string(g_set_ro_data_buffer));
        additional_backend_options.emplace(MDL_JIT_OPTION_MAX_CONST_DATA, std::to_string(options.max_const_data));
        additional_backend_options.emplace(MDL_JIT_OPTION_LIBBSDF_FLAGS_IN_BSDF_DATA, options.enable_bsdf_flags ? "true" : "false");

        Material_backend_compiler material_be_compiler(
            mdl_compiler.get(),
            target_backend,
            options.tex_results_cache_size,
            backend_options,
            /*df_handle_mode=*/ "none",
            /*lambda_return_mode=*/ "default",
            additional_backend_options);

        for (const std::string& path : options.additional_mdl_paths)
            material_be_compiler.add_module_path(path.c_str());

        std::cout << "Generating GLSL code for material " << options.material_name << "\n";

        auto t0 = std::chrono::steady_clock::now();
        Material_instance material_instance(
            material_be_compiler.create_material_instance(options.material_name));
        auto t1 = std::chrono::steady_clock::now();
        check_success(material_instance);
        std::cout << "Create material instance: " << std::chrono::duration<float>(t1 - t0).count() << "s\n";

        mi::Uint32 flags = 0;
        t0 = std::chrono::steady_clock::now();
        mi::mdl::Dag_error_code error_code =
            material_be_compiler.initialize_material_instance(
                material_instance, {}, options.use_class_compilation, flags);
        t1 = std::chrono::steady_clock::now();
        check_success(error_code == mi::mdl::EC_NONE);
        std::cout << "Initialize material instance: " << std::chrono::duration<float>(t1 - t0).count() << "s\n";

        // Specify which functions to generate
        std::vector<Target_function_description> function_descs;
        function_descs.emplace_back("init", "mdl_init");
        function_descs.emplace_back("thin_walled", "mdl_thin_walled");
        function_descs.emplace_back("surface.scattering", "mdl_bsdf");
        function_descs.emplace_back("surface.emission.emission", "mdl_edf");
        function_descs.emplace_back("surface.emission.intensity", "mdl_emission_intensity");
        function_descs.emplace_back("volume.absorption_coefficient", "mdl_absorption_coefficient");

        // Try to determine if the material is thin walled so we can check
        // if backface functions need to be generated.
        const mi::mdl::DAG_node* dag_node = nullptr;
        const mi::mdl::IValue* value = nullptr;
        material_instance->lookup_sub_expression("thin_walled", dag_node, value);

        if (dag_node && dag_node->get_kind() == mi::mdl::DAG_node::EK_CONSTANT)
            value = mi::mdl::as<const mi::mdl::DAG_constant>(dag_node)->get_value();

        bool is_thin_walled_function = true;
        bool thin_walled_value = false;
        if (value && value->get_kind() == mi::mdl::IValue::VK_BOOL)
        {
            is_thin_walled_function = false;
            thin_walled_value = mi::mdl::as<const mi::mdl::IValue_bool>(value)->get_value();
        }

        // Back faces could be different for thin walled materials
        bool need_backface_bsdf = false;
        bool need_backface_edf = false;
        bool need_backface_emission_intensity = false;

        if (is_thin_walled_function || thin_walled_value)
        {
            // First, backface DFs are only considered for thin_walled materials

            // Second, we only need to generate new code if surface and backface are different
            need_backface_bsdf =
                material_instance->get_slot_hash(mi::mdl::IMaterial_instance::Slot::MS_SURFACE_BSDF_SCATTERING)
                != material_instance->get_slot_hash(mi::mdl::IMaterial_instance::Slot::MS_BACKFACE_BSDF_SCATTERING);
            need_backface_edf =
                material_instance->get_slot_hash(mi::mdl::IMaterial_instance::Slot::MS_SURFACE_EMISSION_EDF_EMISSION)
                != material_instance->get_slot_hash(mi::mdl::IMaterial_instance::Slot::MS_BACKFACE_EMISSION_EDF_EMISSION);
            need_backface_emission_intensity =
                material_instance->get_slot_hash(mi::mdl::IMaterial_instance::Slot::MS_SURFACE_EMISSION_INTENSITY)
                != material_instance->get_slot_hash(mi::mdl::IMaterial_instance::Slot::MS_SURFACE_EMISSION_INTENSITY);

            // Third, either the bsdf or the edf need to be non-default (black)
            const mi::mdl::DAG_node* scattering_dag_node = nullptr;
            const mi::mdl::IValue* scattering_value = nullptr;
            material_instance->lookup_sub_expression("backface.scattering", scattering_dag_node, scattering_value);

            if (scattering_dag_node && scattering_dag_node->get_kind() == mi::mdl::DAG_node::EK_CONSTANT)
                scattering_value = mi::mdl::as<const mi::mdl::DAG_constant>(scattering_dag_node)->get_value();

            const mi::mdl::DAG_node* emission_dag_node = nullptr;
            const mi::mdl::IValue* emission_value = nullptr;
            material_instance->lookup_sub_expression("backface.emission.emission", emission_dag_node, emission_value);

            if (emission_dag_node && emission_dag_node->get_kind() == mi::mdl::DAG_node::EK_CONSTANT)
                emission_value = mi::mdl::as<const mi::mdl::DAG_constant>(emission_dag_node)->get_value();

            if (scattering_value && emission_value
                && scattering_value->get_kind() == mi::mdl::IValue::VK_INVALID_REF
                && emission_value->get_kind() == mi::mdl::IValue::VK_INVALID_REF)
            {
                need_backface_bsdf = false;
                need_backface_edf = false;
                need_backface_emission_intensity = false;
            }
        }

        if (need_backface_bsdf)
            function_descs.emplace_back("backface.scattering", "mdl_backface_bsdf");

        if (need_backface_edf)
            function_descs.emplace_back("backface.emission.emission", "mdl_backface_edf");

        if (need_backface_emission_intensity)
            function_descs.emplace_back("backface.emission.intensity", "mdl_backface_emission_intensity");

        bool success = material_be_compiler.add_material(
            options.material_name,
            function_descs.data(),
            function_descs.size(),
            options.use_class_compilation);
        if (!success)
        {
            material_be_compiler.print_messages();
            exit_failure();
        }

        // Compile cutout_opacity also as standalone version to be used in the anyhit programs
        // to avoid costly precalculation of expressions only used by other expressions.
        // This can be especially useful for anyhit shaders in ray tracing pipelines.
        success = material_be_compiler.add_material_subexpr(
            options.material_name,
            "geometry.cutout_opacity",
            "mdl_standalone_cutout_opacity",
            options.use_class_compilation);
        if (!success)
        {
            material_be_compiler.print_messages();
            exit_failure();
        }

        // Generate GLSL code
        t0 = std::chrono::steady_clock::now();
        std::unique_ptr<Target_code> target_code(
            material_be_compiler.generate_target_code());
        t1 = std::chrono::steady_clock::now();
        std::cout << "Generate GLSL target code: " << std::chrono::duration<float>(t1 - t0).count() << "s\n";

        if (options.dump_glsl)
        {
            std::cout << "Dumping GLSL target code to target_code.glsl\n";
            std::ofstream file_stream("target_code.glsl");
            file_stream.write(target_code->get_src_code(), target_code->get_src_code_size());
        }

        // Start application
        Vulkan_base_app::Config app_config;
        app_config.window_title = "MDL Core DF Vulkan Example";
        app_config.image_width = options.res_x;
        app_config.image_height = options.res_y;
        app_config.image_count = options.num_images;
        app_config.device_index = options.device_index;
        app_config.headless = options.no_window;
        app_config.iteration_count = options.samples_per_pixel / options.samples_per_iteration;
        app_config.enable_validation_layers = options.enable_validation_layers;
        app_config.enable_descriptor_indexing = true;

        Df_vulkan_app app(target_code.get(), material_instance, function_descs[0].argument_block_index, mdl_compiler.get(), options);
        app.run(app_config);
    }

    // Free MDL core compiler and unload MDL core
    mdl_compiler = nullptr;
    check_success(unload());

    keep_console_open();
    return EXIT_SUCCESS;
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
