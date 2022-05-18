/******************************************************************************
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/shared/example_vulkan_shared.h
 //
 // Code shared by all Vulkan examples.

#ifndef EXAMPLE_VULKAN_SHARED_H
#define EXAMPLE_VULKAN_SHARED_H

#include <vector>
#include <iostream>

#include <vulkan/vulkan.h>
#include <glslang/Public/ShaderLang.h>

#include "example_shared.h"

#define terminate()      \
    do {                 \
        glfwTerminate(); \
        exit_failure();  \
    } while (0)

#define VK_CHECK(x)                                       \
    do {                                                  \
        VkResult err = x;                                 \
        if (err != VK_SUCCESS) {                          \
            std::cerr << "Vulkan error "                  \
                << mi::examples::vk::vkresult_to_str(err) \
                << " (" << err << ")"                     \
                << " in file " << __FILE__                \
                << ", line " << __LINE__ << ".\n";        \
            terminate();                                  \
        }                                                 \
    } while (0)

// Extensions
extern PFN_vkCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT;
extern PFN_vkDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT;

struct GLFWwindow;

namespace mi::examples::vk
{

// Compiles GLSL source code to SPIR-V which can be used to create Vulkan shader modules.
// Multiple shaders of the same type (e.g. EShLangFragment) are linked into a single
// SPIR-V module. Newer SPIR-V versions can be generated by changing the relevant
// parameters for TShader::setEnvClient and TShader::setEnvTarget.
class Glsl_compiler
{
public:
    Glsl_compiler(EShLanguage shader_type, const char* entry_point = "main")
        : m_shader_type(shader_type)
        , m_entry_point(entry_point)
    {
    }

    // Adds a list of #define to each shader's preamble parsed after this method is called.
    // Call this method BEFORE any call to add_shader.
    // Example: add_define({"MY_DEF=1", "MY_OTHER_DEF", "MY_THIRD_DEF 3"})
    void add_defines(const std::vector<std::string>& defines);

    // Parses the given shader source and adds it to the shader program
    // which can be compiled to SPIR-V by link_program.
    void add_shader(std::string_view source);

    // Links all previously added shaders and compiles the linked program to SPIR-V.
    std::vector<unsigned int> link_program(bool optimize = true);

private:
    class Simple_file_includer : public glslang::TShader::Includer
    {
    public:
        virtual ~Simple_file_includer() = default;

        virtual IncludeResult* includeSystem(const char* header_name,
            const char* includer_name, size_t inclusion_depth) override;
        virtual IncludeResult* includeLocal(const char* header_name,
            const char* includer_name, size_t inclusion_depth) override;
        virtual void releaseInclude(IncludeResult* include_result) override;
    };

private:
    static const EShMessages s_messages
        = static_cast<EShMessages>(EShMsgVulkanRules | EShMsgSpvRules);

    EShLanguage m_shader_type;
    std::string m_entry_point;
    std::vector<std::unique_ptr<glslang::TShader>> m_shaders;
    Simple_file_includer m_file_includer;
    std::string m_shader_preamble;
};


class Vulkan_example_app
{
public:
    struct Config
    {
        std::string window_title = "MDL SDK Vulkan Example";
        uint32_t image_width = 1024;
        uint32_t image_height = 768;
        uint32_t image_count = 3;
        std::vector<VkFormat> preferred_image_formats
            = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM };
        bool headless = false;
        uint32_t iteration_count = 1; // Headless mode only
        bool enable_validation_layers = false;
    };

public:
    Vulkan_example_app(
        mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
        mi::neuraylib::IImage_api* image_api)
    : m_mdl_impexp_api(mdl_impexp_api, mi::base::DUP_INTERFACE)
    , m_image_api(image_api, mi::base::DUP_INTERFACE)
    {
    }
    virtual ~Vulkan_example_app() = default;

    void run(const Config& config);

protected:
    // The callback for initializing all rendering related resources. Is called
    // after all resources (device, swapchain, etc.) of this base class are 
    // initialized.
    virtual void init_resources() {}

    // The callback for destroying all rendering related resources. Is called
    // before all resources (device, swapchain, etc.) of this base class are 
    // destroyed.
    virtual void cleanup_resources() {}

    // Render resources are split into framebuffer size dependent
    // and independent resources so that the correct resources are
    // recreated when the framebuffer is resized.
    virtual void recreate_size_dependent_resources() {}

    // The callback to update the appliction logic. Is called directly after
    // a new swapchain image is acquired and after waiting for the fence for
    // the current frame's command buffer.
    virtual void update(float elapsed_seconds, uint32_t image_index) = 0;

    // The callback to fill the current frame's command buffer. Is called directly
    // after vkBeginCommandBuffer and before vkEndCommandBuffer.
    virtual void render(VkCommandBuffer command_buffer, uint32_t image_index) = 0;

    // Called directly after the current command buffer is submitted, but before
    // the next frame is presented (before potential vsync).
    virtual void after_submit_callback(uint32_t image_index) {}

    // Called when a keyboard key is pressed or released.
    virtual void key_callback(int key, int action) {}

    // Called when a mouse button is pressed or released.
    virtual void mouse_button_callback(int button, int action) {}

    // Called when the mouse moves.
    virtual void mouse_move_callback(float pos_x, float pos_y) {}

    // Called when the scrolling event occurs (e.g. mouse wheel or touch pad).
    virtual void mouse_scroll_callback(float offset_x, float offset_y) {}

    // Called when the window is resized.
    virtual void resized_callback(uint32_t width, uint32_t height) {}

    // Request to save a screenshot the next frame.
    void request_screenshot() { m_screenshot_requested = true; }

    // Save the specified swapchain image to a file.
    void save_screenshot(uint32_t image_index, const char* filename) const;

protected:
    // MDL image interfaces.
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_mdl_impexp_api;
    mi::base::Handle<mi::neuraylib::IImage_api> m_image_api;

    GLFWwindow* m_window = nullptr;
    VkInstance m_instance = nullptr;
    VkSurfaceKHR m_surface = nullptr;
    VkPhysicalDevice m_physical_device = nullptr;
    VkDevice m_device = nullptr;

    uint32_t m_graphics_queue_family_index;
    uint32_t m_present_queue_family_index;
    VkQueue m_graphics_queue = nullptr;
    VkQueue m_present_queue = nullptr;

    // Framebuffer information for either the swapchain
    // images or headless framebuffer.
    VkFormat m_image_format;
    uint32_t m_image_width;
    uint32_t m_image_height;
    uint32_t m_image_count;

    // Swapchain is only created if a window is used.
    VkSwapchainKHR m_swapchain = nullptr;
    std::vector<VkImage> m_swapchain_images;
    std::vector<VkImageView> m_swapchain_image_views;
    VkSemaphore m_image_available_semaphore = nullptr;
    VkSemaphore m_render_finished_semaphore = nullptr;

    // For no window mode we have to handle device memory ourselfs
    std::vector<VkDeviceMemory> m_swapchain_device_memories;

    // Depth stencil buffer
    VkFormat m_depth_stencil_format;
    VkImage m_depth_stencil_image = nullptr;
    VkDeviceMemory m_depth_stencil_device_memory = nullptr;
    VkImageView m_depth_stencil_image_view = nullptr;

    std::vector<VkFence> m_frame_inflight_fences;
    VkCommandPool m_command_pool = nullptr;
    std::vector<VkCommandBuffer> m_command_buffers;
    std::vector<VkFramebuffer> m_framebuffers;
    VkRenderPass m_main_render_pass = nullptr;

private:
    void init(const Config& config);
    void cleanup();

    void init_window();
    void init_instance(
        const std::vector<const char*>& instance_extensions,
        const std::vector<const char*>& validation_layers);
    void pick_physical_device(const std::vector<const char*>& device_extensions);
    void init_device(
        const std::vector<const char*>& device_extensions,
        const std::vector<const char*>& validation_layers);
    void init_swapchain_for_window();
    void init_swapchain_for_headless();
    void init_depth_stencil_buffer();
    void init_render_pass();
    void init_framebuffers();
    void init_command_pool_and_buffers();
    void init_synchronization_objects();

    void recreate_swapchain_or_framebuffer_image();

    void render_loop_iteration(uint32_t image_index, double& last_frame_time);

    static void internal_key_callback(
        GLFWwindow* window, int key, int scancode, int action, int mods);
    static void internal_mouse_button_callback(
        GLFWwindow* window, int button, int action, int mods);
    static void internal_mouse_move_callback(GLFWwindow* window, double pos_x, double pos_y);
    static void internal_mouse_scroll_callback(GLFWwindow* window, double offset_x, double offset_y);
    static void internal_resize_callback(GLFWwindow* window, int width, int height);

    static void glfw_error_callback(int error_code, const char* description);

    static VKAPI_ATTR VkBool32 VKAPI_PTR debug_messenger_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_types,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        void* user_data);

private:
    Config m_config;
    VkDebugUtilsMessengerEXT m_debug_messenger = nullptr;
    bool m_framebuffer_resized = false;
    bool m_screenshot_requested = false;
};


class Staging_buffer
{
public:
    Staging_buffer(VkDevice device, VkPhysicalDevice physical_device,
        VkDeviceSize size, VkBufferUsageFlags usage);
    ~Staging_buffer();

    void* map_memory() const;
    void unmap_memory() const;

    VkBuffer get() const { return m_buffer; }

private:
    VkDevice m_device;
    VkBuffer m_buffer;
    VkDeviceMemory m_device_memory;
};


class Temporary_command_buffer
{
public:
    Temporary_command_buffer(VkDevice device, VkCommandPool command_pool);
    ~Temporary_command_buffer();

    void begin();
    void end_and_submit(VkQueue queue, bool wait = true);

    VkCommandBuffer get() const { return m_command_buffer; }

private:
    VkDevice m_device;
    VkCommandPool m_command_pool;
    VkCommandBuffer m_command_buffer;
};


// Extension helpers
bool load_debug_utils_extension(VkInstance instance);

bool check_instance_extensions_support(
    const std::vector<const char*>& requested_extensions);

bool check_device_extensions_support(VkPhysicalDevice device,
    const std::vector<const char*>& requested_extensions);

bool check_validation_layers_support(
    const std::vector<const char*>& requested_layers);


// Shader compilation helpers
VkShaderModule create_shader_module_from_file(
    VkDevice device, const char* shader_filename, EShLanguage shader_type,
    const std::vector<std::string>& defines = {});

VkShaderModule create_shader_module_from_sources(
    VkDevice device, const std::vector<std::string_view> shader_sources, EShLanguage shader_type,
    const std::vector<std::string>& defines = {});


// Memory allocation helpers
VkDeviceMemory allocate_and_bind_buffer_memory(
    VkDevice device, VkPhysicalDevice physical_device, VkBuffer buffer,
    VkMemoryPropertyFlags memory_property_flags);

VkDeviceMemory allocate_and_bind_image_memory(
    VkDevice device, VkPhysicalDevice physical_device, VkImage image,
    VkMemoryPropertyFlags memory_property_flags);

uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    uint32_t memory_type_bits_requirement,
    VkMemoryPropertyFlags required_properties);


// Format helpers
VkFormat find_supported_format(
    VkPhysicalDevice physical_device,
    const std::vector<VkFormat>& formats,
    VkImageTiling tiling,
    VkFormatFeatureFlags feature_flags);

bool has_stencil_component(VkFormat format);

uint32_t get_image_format_bpp(VkFormat format);

// Initialization helpers
VkRenderPass create_simple_color_only_render_pass(
    VkDevice device, VkFormat image_format, VkImageLayout final_layout);

VkRenderPass create_simple_render_pass(
    VkDevice device, VkFormat image_format,
    VkFormat depth_stencil_format, VkImageLayout final_layout);

VkSampler create_linear_sampler(VkDevice device);


// Misc helpers
std::vector<uint8_t> copy_image_to_buffer(
    VkDevice device, VkPhysicalDevice physical_device, VkCommandPool command_pool, VkQueue queue,
    VkImage image, uint32_t image_width, uint32_t image_height, uint32_t image_bpp,
    VkImageLayout image_layout, bool flip);

const char* vkresult_to_str(VkResult result);

// Convert some of the common formats to string.
const char* vkformat_to_str(VkFormat format);

// Default resource values for glslang.
// See: https://github.com/KhronosGroup/glslang/StandAlone/ResourceLimits.cpp
extern const TBuiltInResource g_default_built_in_resource;

} // namespace mi::examples::vk

#endif // EXAMPLE_VULKAN_SHARED_H