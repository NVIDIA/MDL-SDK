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

// examples/mdl_sdk/df_cuda/example_df_cuda.cpp
//
// Simple renderer using compiled BSDFs with a material parameter editor GUI.

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>

// shared example helpers
#include "example_df_cuda.h"
#include "lpe.h"

// Enable this to dump the generated PTX code to stdout.
// #define DUMP_PTX

#define OPENGL_INTEROP
#include "example_cuda_shared.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GL_DISPLAY_CUDA
#include "utils/gl_display.h"

#define terminate()          \
    do {                     \
        glfwTerminate();     \
        exit_failure();      \
    } while (0)

#define WINDOW_TITLE "MDL SDK DF CUDA Example"


/////////////////////////////
// Vector helper functions //
/////////////////////////////

inline float length(const float3 &d)
{
    return sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
}

inline float3 normalize(const float3 &d)
{
    const float inv_len = 1.0f / length(d);
    return make_float3(d.x * inv_len, d.y * inv_len, d.z * inv_len);
}

inline float3 operator/(const float3& d, float s)
{
    const float inv_s = 1.0f / s;
    return make_float3(d.x * inv_s, d.y * inv_s, d.z * inv_s);
}


/////////////////
// OpenGL code //
/////////////////

// Initialize OpenGL and create a window with an associated OpenGL context.
static GLFWwindow *init_opengl(std::string& version_string, int res_x, int res_y)
{
    // Initialize GLFW
    check_success(glfwInit());
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    version_string = "#version 330 core"; // see top comments in 'imgui_impl_opengl3.cpp'

    // Create an OpenGL window and a context
    GLFWwindow *window = glfwCreateWindow(
        res_x, res_y, WINDOW_TITLE, nullptr, nullptr);
    if (!window) {
        std::cerr << "Error creating OpenGL window!" << std::endl;
        terminate();
    }

    // Attach context to window
    glfwMakeContextCurrent(window);

    // Initialize GLEW to get OpenGL extensions
    GLenum res = glewInit();
    if (res != GLEW_OK) {
        std::cerr << "GLEW error: " << glewGetErrorString(res) << std::endl;
        terminate();
    }

    // Disable VSync
    glfwSwapInterval(0);

    check_success(glGetError() == GL_NO_ERROR);

    return window;
}


///////////////////////
// Application logic //
///////////////////////

// Context structure for window callback functions.
struct Window_context
{
    bool mouse_event, key_event;
    bool save_image;
    int zoom;

    int mouse_button;            // button from callback event plus one (0 = no event)
    int mouse_button_action;     // action from mouse button callback event
    int mouse_wheel_delta;
    bool moving;
    double move_start_x, move_start_y;
    double move_dx, move_dy;

    int material_index_delta;

    bool save_result;

    bool exposure_event;
    float exposure;
};

static std::string to_string(Display_buffer_options option)
{
    switch (option)
    {
        case DISPLAY_BUFFER_LPE: return "Selected LPE";
        case DISPLAY_BUFFER_ALBEDO: return "Albedo";
        case DISPLAY_BUFFER_NORMAL: return "Normal";
        default: return "";
    }
}

// GLFW scroll callback
static void handle_scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    if (yoffset > 0.0) {
        ctx->mouse_wheel_delta = 1; ctx->mouse_event = true;
    } else if (yoffset < 0.0) {
        ctx->mouse_wheel_delta = -1; ctx->mouse_event = true;
    }

    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}

// GLFW keyboard callback
static void handle_key(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    // Handle key press events
    if (action == GLFW_PRESS) {
        Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
        switch (key) {
            // Escape closes the window
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_DOWN:
            case GLFW_KEY_RIGHT:
            case GLFW_KEY_PAGE_DOWN:
                ctx->material_index_delta = 1;
                ctx->key_event = true;
                break;
            case GLFW_KEY_UP:
            case GLFW_KEY_LEFT:
            case GLFW_KEY_PAGE_UP:
                ctx->material_index_delta = -1;
                ctx->key_event = true;
                break;
            case GLFW_KEY_ENTER:
                ctx->save_result = true;
                break;
            case GLFW_KEY_KP_SUBTRACT:
                ctx->exposure--;
                ctx->exposure_event = true;
                break;
            case GLFW_KEY_KP_ADD:
                ctx->exposure++;
                ctx->exposure_event = true;
                break;
            default:
                break;
        }
    }

    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

// GLFW mouse button callback
static void handle_mouse_button(GLFWwindow *window, int button, int action, int mods)
{
    Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    ctx->mouse_button = button + 1;
    ctx->mouse_button_action = action;

    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

// GLFW mouse position callback
static void handle_mouse_pos(GLFWwindow *window, double xpos, double ypos)
{
    Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
    if (ctx->moving)
    {
        ctx->move_dx += xpos - ctx->move_start_x;
        ctx->move_dy += ypos - ctx->move_start_y;
        ctx->move_start_x = xpos;
        ctx->move_start_y = ypos;
        ctx->mouse_event = true;
    }
}

// Resize CUDA buffers for a given resolution
static void resize_buffers(CUdeviceptr *buffer_cuda, int width, int height)
{
    // Allocate CUDA buffer
    if (*buffer_cuda)
        check_cuda_success(cuMemFree(*buffer_cuda));

    if (width == 0 || height == 0)
        *buffer_cuda = 0;
    else
        check_cuda_success(cuMemAlloc(buffer_cuda, width * height * sizeof(float3)));
}

// Helper for create_environment()
static float build_alias_map(
    const float *data,
    const unsigned int size,
    Env_accel *accel)
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

// Create environment map texture and acceleration data for importance sampling
static void create_environment(
    cudaTextureObject_t *env_tex,
    cudaArray_t *env_tex_data,
    CUdeviceptr *env_accel,
    uint2 *res,
    mi::base::Handle<mi::neuraylib::ITransaction> transaction,
    mi::base::Handle<mi::neuraylib::IImage_api> image_api,
    const char *envmap_name)
{
    // Load environment texture
    mi::base::Handle<mi::neuraylib::IImage>image(
        transaction->create<mi::neuraylib::IImage>("Image"));
    check_success(image->reset_file(envmap_name) == 0);

    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas());
    const mi::Uint32 rx = canvas->get_resolution_x();
    const mi::Uint32 ry = canvas->get_resolution_y();
    res->x = rx;
    res->y = ry;

    // Check, whether we need to convert the image
    char const *image_type = image->get_type();
    if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0)
        canvas = image_api->convert(canvas.get(), "Color");

    // Copy the image data to a CUDA array
    const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    check_cuda_success(cudaMallocArray(env_tex_data, &channel_desc, rx, ry));

    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
    const float *pixels = static_cast<const float *>(tile->get_data());

    check_cuda_success(cudaMemcpy2DToArray(
        *env_tex_data, 0, 0, pixels,
        rx * sizeof(float4), rx * sizeof(float4), ry, cudaMemcpyHostToDevice));

    // Create a CUDA texture
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = *env_tex_data;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0]   = cudaAddressModeWrap;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;  // don't sample beyond poles of env sphere
    tex_desc.addressMode[2]   = cudaAddressModeWrap;
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;

    check_cuda_success(cudaCreateTextureObject(env_tex, &res_desc, &tex_desc, nullptr));

    // Create importance sampling data
    Env_accel *env_accel_host = static_cast<Env_accel *>(malloc(rx * ry * sizeof(Env_accel)));
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
            const unsigned int idx4 =  idx * 4;
            importance_data[idx] =
                area * std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2]));
        }
    }
    const float inv_env_integral = 1.0f / build_alias_map(importance_data, rx * ry, env_accel_host);
    free(importance_data);
    for (unsigned int i = 0; i < rx * ry; ++i) {
        const unsigned int idx4 = i * 4;
        env_accel_host[i].pdf =
            std::max(pixels[idx4], std::max(pixels[idx4 + 1], pixels[idx4 + 2])) * inv_env_integral;
    }

    *env_accel = gpu_mem_dup(env_accel_host, rx * ry * sizeof(Env_accel));
    free(env_accel_host);
}


static void upload_lpe_state_machine(
    Kernel_params& kernel_params,
    LPE_state_machine& lpe_state_machine)
{
    uint32_t num_trans = lpe_state_machine.get_transition_count();
    uint32_t num_states = lpe_state_machine.get_state_count();
    kernel_params.lpe_num_transitions = num_trans;
    kernel_params.lpe_num_states = num_states;

    // free old data
    if (kernel_params.lpe_state_table)
        check_cuda_success(cuMemFree(reinterpret_cast<CUdeviceptr>(kernel_params.lpe_state_table)));
    if (kernel_params.lpe_final_mask)
        check_cuda_success(cuMemFree(reinterpret_cast<CUdeviceptr>(kernel_params.lpe_final_mask)));

    // state table
    CUdeviceptr state_table = 0;
    check_cuda_success(cuMemAlloc(&state_table, num_states * num_trans * sizeof(uint32_t)));
    check_cuda_success(cuMemcpyHtoD(state_table, lpe_state_machine.get_state_table().data(),
                       num_states * num_trans * sizeof(uint32_t)));
    kernel_params.lpe_state_table = reinterpret_cast<uint32_t*>(state_table);

    // final state masks
    CUdeviceptr final_mask = 0;
    check_cuda_success(cuMemAlloc(&final_mask, num_states * sizeof(uint32_t)));
    check_cuda_success(cuMemcpyHtoD(final_mask, lpe_state_machine.get_final_state_masks().data(),
                       num_states * sizeof(uint32_t)));
    kernel_params.lpe_final_mask = reinterpret_cast<uint32_t*>(final_mask);

    // tag ID for light sources as they don't store tags in this examples
    kernel_params.default_gtag = lpe_state_machine.handle_to_global_tag("");
    kernel_params.point_light_gtag = lpe_state_machine.handle_to_global_tag("point_light");
    kernel_params.env_gtag = lpe_state_machine.handle_to_global_tag("env");
}

// Save current result image to disk
static void save_result(
    const CUdeviceptr cuda_buffer,
    const unsigned int width,
    const unsigned int height,
    const std::string &filename,
    mi::base::Handle<mi::neuraylib::IImage_api> image_api,
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api)
{
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        image_api->create_canvas("Rgb_fp", width, height));
    mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile(0, 0));
    float3 *data = static_cast<float3 *>(tile->get_data());
    check_cuda_success(cuMemcpyDtoH(data, cuda_buffer, width * height * sizeof(float3)));

    // assuming EXR and HDR are the only linear color space formats we are using
    // other formats are stored in sRGB (approximated by gamma 2.2)
    if (!mi::examples::strings::ends_with(filename, ".exr") &&
        !mi::examples::strings::ends_with(filename, ".hdr"))
            image_api->adjust_gamma(canvas.get(), 2.2f);

    mdl_impexp_api->export_canvas(filename.c_str(), canvas.get());
}

// Application options
struct Options {
    int cuda_device;
    float gui_scale;
    bool opengl;
    bool use_class_compilation;
    bool no_aa;
    bool enable_derivatives;
    bool fold_ternary_on_df;
    bool enable_auxiliary_output;
    unsigned int res_x, res_y;
    unsigned int iterations;
    unsigned int samples_per_iteration;
    unsigned int mdl_test_type;
    unsigned int max_path_length;
    float fov;
    float exposure;
    float3 cam_pos;
    float3 light_pos;
    float3 light_intensity;

    std::string hdrfile;
    float hdr_rot;
    std::string outputfile;
    std::vector<std::string> material_names;

    // Default constructor, sets default values.
    Options()
    : cuda_device(0)
    , gui_scale(1.0f)
    , opengl(true)
    , use_class_compilation(true)
    , no_aa(false)
    , enable_derivatives(false)
    , fold_ternary_on_df(false)
    , enable_auxiliary_output(true)
    , res_x(1024)
    , res_y(1024)
    , iterations(4096)
    , samples_per_iteration(8)
    , mdl_test_type(MDL_TEST_MIS)
    , max_path_length(4)
    , fov(96.0f)
    , exposure(0.0f)
    , cam_pos(make_float3(0, 0, 3))
    , light_pos(make_float3(10, 0, 5))
    , light_intensity(make_float3(0, 0, 0))
    , hdrfile("nvidia/sdk_examples/resources/environment.hdr")
    , hdr_rot(0.0f)
    , outputfile("output.exr")
    , material_names()
    {}
};

// Possible enum values if any.
struct Enum_value {
    std::string name;
    int         value;

    Enum_value(const std::string &name, int value)
    : name(name), value(value)
    {
    }
};

// Info for an enum type.
struct Enum_type_info {
    std::vector<Enum_value> values;

    // Adds a enum value and its integer value to the enum type info.
    void add(const std::string &name, int value) {
        values.push_back(Enum_value(name, value));
    }
};

// Material parameter information structure.
class Param_info
{
public:
    enum Param_kind
    {
        PK_UNKNOWN,
        PK_FLOAT,
        PK_FLOAT2,
        PK_FLOAT3,
        PK_COLOR,
        PK_ARRAY,
        PK_BOOL,
        PK_INT,
        PK_ENUM,
        PK_STRING,
        PK_TEXTURE,
        PK_LIGHT_PROFILE,
        PK_BSDF_MEASUREMENT
    };

    Param_info(
        mi::Size index,
        char const *name,
        char const *display_name,
        char const *group_name,
        Param_kind kind,
        Param_kind array_elem_kind,
        mi::Size   array_size,
        mi::Size   array_pitch,
        char *data_ptr,
        const Enum_type_info *enum_info = nullptr)
    : m_index(index)
    , m_name(name)
    , m_display_name(display_name)
    , m_group_name(group_name)
    , m_kind(kind)
    , m_array_elem_kind(array_elem_kind)
    , m_array_size(array_size)
    , m_array_pitch(array_pitch)
    , m_data_ptr(data_ptr)
    , m_range_min(-100), m_range_max(100)
    , m_enum_info(enum_info)
    {
    }

    // Get data as T&.
    template<typename T>
    T &data() { return *reinterpret_cast<T *>(m_data_ptr); }

    // Get data as const T&.
    template<typename T>
    const T &data() const { return *reinterpret_cast<const T *>(m_data_ptr); }

    const char * &display_name()     { return m_display_name; }
    const char *display_name() const { return m_display_name; }

    const char * &group_name()     { return m_group_name; }
    const char *group_name() const { return m_group_name; }

    Param_kind kind() const { return m_kind; }

    Param_kind array_elem_kind() const { return m_array_elem_kind; }
    mi::Size array_size() const        { return m_array_size; }
    mi::Size array_pitch() const       { return m_array_pitch; }

    float &range_min()      { return m_range_min; }
    float range_min() const { return m_range_min; }
    float &range_max()      { return m_range_max; }
    float range_max() const { return m_range_max; }

    const Enum_type_info *enum_info() const { return m_enum_info; }

private:
    mi::Size             m_index;
    char const           *m_name;
    char const           *m_display_name;
    char const           *m_group_name;
    Param_kind           m_kind;
    Param_kind           m_array_elem_kind;
    mi::Size             m_array_size;
    mi::Size             m_array_pitch;   // the distance between two array elements
    char                 *m_data_ptr;
    float                m_range_min, m_range_max;
    const Enum_type_info *m_enum_info;
};

// Material information structure.
class Material_info
{
public:
    Material_info(char const *name)
    : m_name(name)
    {}

    // Add the parameter information as last entry of the corresponding group, or to the
    // end of the list, if no group name is available.
    void add_sorted_by_group(const Param_info &info) {
        bool group_found = false;
        if (info.group_name() != nullptr) {
            for (std::list<Param_info>::iterator it = params().begin(); it != params().end(); ++it)
            {
                const bool same_group =
                    it->group_name() != nullptr && strcmp(it->group_name(), info.group_name()) == 0;
                if (group_found && !same_group) {
                    m_params.insert(it, info);
                    return;
                }
                if (same_group)
                    group_found = true;
            }
        }
        m_params.push_back(info);
    }

    // Add a new enum type to the list of used enum types.
    void add_enum_type(const std::string name, std::shared_ptr<Enum_type_info> enum_info) {
        enum_types[name] = enum_info;
    }

    // Lookup enum type info for a given enum type absolute MDL name.
    const Enum_type_info *get_enum_type(const std::string name) {
        Enum_type_map::const_iterator it = enum_types.find(name);
        if (it != enum_types.end())
            return it->second.get();
        return nullptr;
    }

    // Get the name of the material.
    char const *name() const { return m_name; }

    // Get the parameters of this material.
    std::list<Param_info> &params() { return m_params; }

private:
    // name of the material
    char const *m_name;

    // parameters of the material
    std::list<Param_info> m_params;

    typedef std::map<std::string, std::shared_ptr<Enum_type_info> > Enum_type_map;

    // used enum types of the material
    Enum_type_map enum_types;
};

// Helper class to handle Resource tables of the target code.
class Resource_table
{
    typedef std::map<std::string, unsigned> Resource_id_map;
public:
    enum Kind {
        RESOURCE_TEXTURE,
        RESOURCE_LIGHT_PROFILE,
        RESOURCE_BSDF_MEASUREMENT
    };

    // Constructor.
    Resource_table(
        mi::base::Handle<mi::neuraylib::ITarget_code const> target_code,
        mi::base::Handle<mi::neuraylib::ITransaction>       transaction,
        Kind                                                kind)
    : m_max_len(0u)
    {
        read_resources(target_code, transaction, kind);
    }

    // Get the length of the longest URL in the resource table.
    size_t get_max_length() const { return m_max_len; }

    // Get all urls.
    std::vector<std::string> const &get_urls() const { return m_urls; }

private:
    void read_resources(
        mi::base::Handle<mi::neuraylib::ITarget_code const> target_code,
        mi::base::Handle<mi::neuraylib::ITransaction>       transaction,
        Kind                                                kind)
    {
        m_urls.push_back("<unset>");
        switch (kind) {
        case RESOURCE_TEXTURE:
            for (mi::Size i = 1, n = target_code->get_texture_count(); i < n; ++i) {
                const char *s = target_code->get_texture(i);
                mi::base::Handle<mi::neuraylib::ITexture const> tex(
                    transaction->access<mi::neuraylib::ITexture>(s));
                char const *url = nullptr;
                if (char const *img = tex->get_image()) {
                    mi::base::Handle<mi::neuraylib::IImage const> image(
                        transaction->access<mi::neuraylib::IImage>(img));
                    url = image->get_filename();
                }
                if (url == nullptr)
                    url = s;
                size_t l = strlen(url);
                if (l > m_max_len)
                    m_max_len = l;
                m_resource_map[s] = (unsigned)i;
                m_urls.push_back(url);
            }
            break;
        case RESOURCE_LIGHT_PROFILE:
            for (mi::Size i = 1, n = target_code->get_light_profile_count(); i < n; ++i) {
                const char *s = target_code->get_light_profile(i);
                mi::base::Handle<mi::neuraylib::ILightprofile const> lp(
                    transaction->access<mi::neuraylib::ILightprofile>(s));
                char const *url = lp->get_filename();
                if (url == nullptr)
                    url = s;
                size_t l = strlen(url);
                if (l > m_max_len)
                    m_max_len = l;
                m_resource_map[s] = (unsigned)i;
                m_urls.push_back(url);
            }
            break;
        case RESOURCE_BSDF_MEASUREMENT:
            for (mi::Size i = 1, n = target_code->get_bsdf_measurement_count(); i < n; ++i) {
                const char *s = target_code->get_bsdf_measurement(i);
                mi::base::Handle<mi::neuraylib::IBsdf_measurement const> bm(
                    transaction->access<mi::neuraylib::IBsdf_measurement>(s));
                char const *url = bm->get_filename();
                if (url == nullptr)
                    url = s;
                size_t l = strlen(url);
                if (l > m_max_len)
                    m_max_len = l;
                m_resource_map[s] = (unsigned)i;
                m_urls.push_back(url);
            }
            break;
        }
    }

private:
    Resource_id_map          m_resource_map;
    std::vector<std::string> m_urls;
    size_t                   m_max_len;
};

// Helper class to handle the string table of a target code.
class String_constant_table
{
    typedef std::map<std::string, unsigned> String_map;
public:
    // Constructor.
    String_constant_table(mi::base::Handle<mi::neuraylib::ITarget_code const> target_code)
    {
        get_all_strings(target_code);
    }

    // Get the ID for a given string, return 0 if the string does not exist in the table.
    unsigned get_id_for_string(const char *name) {
        String_map::const_iterator it(m_string_constants_map.find(name));
        if (it != m_string_constants_map.end())
            return it->second;

        // the user adds a sting that is NOT in the code and we have not seen so far, add it
        // and assign a new id
        unsigned n_id = unsigned(m_string_constants_map.size() + 1);

        m_string_constants_map[name] = n_id;
        m_strings.reserve((n_id + 63) & ~63);
        m_strings.push_back(name);

        size_t l = strlen(name);
        if (l > m_max_len)
            m_max_len = l;
        return n_id;
    }

    // Get the length of the longest string in the string constant table.
    size_t get_max_length() const { return m_max_len; }

    // Get the string for a given ID, or nullptr if this ID does not exists.
    const char *get_string(unsigned id) {
        if (id == 0 || id - 1 >= m_strings.size())
            return nullptr;
        return m_strings[id - 1].c_str();
    }

private:
    // Get all string constants used inside a target code and their maximum length.
    void get_all_strings(
        mi::base::Handle<mi::neuraylib::ITarget_code const> target_code)
    {
        m_max_len = 0;
        // ignore the 0, it is the "Not-a-known-string" entry
        m_strings.reserve(target_code->get_string_constant_count());
        for (mi::Size i = 1, n = target_code->get_string_constant_count(); i < n; ++i) {
            const char *s = target_code->get_string_constant(i);
            size_t l = strlen(s);
            if (l > m_max_len)
                m_max_len = l;
            m_string_constants_map[s] = (unsigned)i;
            m_strings.push_back(s);
        }
    }

private:
    String_map               m_string_constants_map;
    std::vector<std::string> m_strings;
    size_t                   m_max_len;
};

// Update the camera kernel parameters.
static void update_camera(
    Kernel_params &kernel_params,
    double phi,
    double theta,
    float base_dist,
    int zoom)
{
    kernel_params.cam_dir.x = float(-sin(phi) * sin(theta));
    kernel_params.cam_dir.y = float(-cos(theta));
    kernel_params.cam_dir.z = float(-cos(phi) * sin(theta));

    kernel_params.cam_right.x = float(cos(phi));
    kernel_params.cam_right.y = 0.0f;
    kernel_params.cam_right.z = float(-sin(phi));

    kernel_params.cam_up.x = float(-sin(phi) * cos(theta));
    kernel_params.cam_up.y = float(sin(theta));
    kernel_params.cam_up.z = float(-cos(phi) * cos(theta));

    const float dist = float(base_dist * pow(0.95, double(zoom)));
    kernel_params.cam_pos.x = -kernel_params.cam_dir.x * dist;
    kernel_params.cam_pos.y = -kernel_params.cam_dir.y * dist;
    kernel_params.cam_pos.z = -kernel_params.cam_dir.z * dist;
}

// Add a combobox for the given resource parameter to the GUI
static bool handle_resource(
    Param_info           &param,
    Resource_table const &res_table)
{
    bool changed = false;
    std::vector<std::string> const &urls = res_table.get_urls();
    int id = param.data<int>();
    std::string cur_url = urls[id];

    if (ImGui::BeginCombo(param.display_name(), cur_url.c_str())) {
        for (size_t i = 0, n = urls.size(); i < n; ++i) {
            const std::string &name = urls[i];
            bool is_selected = (cur_url == name);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                param.data<int>() = int(i);
                changed = true;
            }
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    return changed;
}

// Progressively render scene
static void render_scene(
    const Options &options,
    mi::base::Handle<mi::neuraylib::ITransaction>         transaction,
    mi::base::Handle<mi::neuraylib::IImage_api>           image_api,
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>      mdl_impexp_api,
    mi::base::Handle<mi::neuraylib::ITarget_code const>   target_code,
    Material_compiler::Material_definition_list const    &material_defs,
    Material_compiler::Compiled_material_list const      &compiled_materials,
    std::vector<size_t> const                            &arg_block_indices,
    std::vector<Df_cuda_material> const                  &material_bundle,
    LPE_state_machine                                    &lpe_state_machine)
{
    Window_context window_context;
    memset(&window_context, 0, sizeof(Window_context));

    mi::examples::mdl::GL_display *gl_display = nullptr;
    GLFWwindow *window = nullptr;
    int width = -1;
    int height = -1;

    if (options.opengl) {
        // Init OpenGL window
        std::string version_string;
        window = init_opengl(version_string, int(options.res_x), int(options.res_y));
        glfwSetWindowUserPointer(window, &window_context);
        glfwSetKeyCallback(window, handle_key);
        glfwSetScrollCallback(window, handle_scroll);
        glfwSetCursorPosCallback(window, handle_mouse_pos);
        glfwSetMouseButtonCallback(window, handle_mouse_button);
        glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGui_ImplGlfw_InitForOpenGL(window, false);
        ImGui_ImplOpenGL3_Init(version_string.c_str());
        ImGui::GetIO().IniFilename = nullptr;       // disable creating imgui.ini
        ImGui::StyleColorsDark();
        ImGui::GetStyle().Alpha = 0.7f;
        ImGui::GetStyle().ScaleAllSizes(options.gui_scale);

        gl_display = new mi::examples::mdl::GL_display(int(options.res_x), int(options.res_y));
    }

    // Initialize CUDA
    CUcontext cuda_context = init_cuda(options.cuda_device, options.opengl);

    CUdeviceptr accum_buffer = 0;
    CUdeviceptr aux_albedo_buffer = 0; // buffer for auxiliary output
    CUdeviceptr aux_normal_buffer = 0; //

    if (!options.opengl) {
        width = options.res_x;
        height = options.res_y;
        check_cuda_success(cuMemAlloc(&accum_buffer, width * height * sizeof(float3)));
        check_cuda_success(cuMemAlloc(&aux_albedo_buffer, width * height * sizeof(float3)));
        check_cuda_success(cuMemAlloc(&aux_normal_buffer, width * height * sizeof(float3)));
    }

    // Setup initial CUDA kernel parameters
    Kernel_params kernel_params;
    memset(&kernel_params, 0, sizeof(Kernel_params));
    kernel_params.cam_focal = 1.0f / tanf(options.fov / 2 * float(2 * M_PI / 360));
    kernel_params.light_pos = options.light_pos;
    kernel_params.light_intensity = fmaxf(
        options.light_intensity.x, fmaxf(options.light_intensity.y, options.light_intensity.z));
    kernel_params.light_color = kernel_params.light_intensity > 0.0f
        ? options.light_intensity / kernel_params.light_intensity
        : make_float3(1.0f, 0.9f, 0.5f);
    kernel_params.env_intensity = 1.0f;
    kernel_params.iteration_start = 0;
    kernel_params.iteration_num = options.samples_per_iteration;
    kernel_params.mdl_test_type = options.mdl_test_type;
    kernel_params.max_path_length = options.max_path_length;
    kernel_params.exposure_scale = powf(2.0f, options.exposure);
    kernel_params.disable_aa = options.no_aa;
    kernel_params.use_derivatives = options.enable_derivatives;
    kernel_params.enable_auxiliary_output = options.enable_auxiliary_output;
    kernel_params.display_buffer_index = 0;

    kernel_params.lpe_ouput_expression = 0;
    kernel_params.lpe_state_table = nullptr;
    kernel_params.lpe_final_mask = nullptr;

    kernel_params.current_material = 0;
    kernel_params.geometry = material_bundle[kernel_params.current_material].contains_hair_bsdf ?
        GT_HAIR : GT_SPHERE;

    // Setup camera
    float base_dist = length(options.cam_pos);
    double theta, phi;
    {
        const float3 inv_dir = normalize(options.cam_pos);
        phi = atan2(inv_dir.x, inv_dir.z);
        theta = acos(inv_dir.y);
    }

    update_camera(kernel_params, phi, theta, base_dist, window_context.zoom);

    // Build the full CUDA kernel with all the generated code
    std::vector<mi::base::Handle<const mi::neuraylib::ITarget_code> > target_codes;
    target_codes.push_back(target_code);
    CUfunction  cuda_function;
    char const *ptx_name = options.enable_derivatives ?
        "example_df_cuda_derivatives.ptx" : "example_df_cuda.ptx";
    CUmodule    cuda_module = build_linked_kernel(
        target_codes,
        (mi::examples::io::get_executable_folder() + "/" + ptx_name).c_str(),
        "render_scene_kernel",
        &cuda_function);

    // copy materials of the scene to the device
    CUdeviceptr material_buffer = 0;
    check_cuda_success(cuMemAlloc(&material_buffer,
                                  material_bundle.size() * sizeof(Df_cuda_material)));

    check_cuda_success(cuMemcpyHtoD(material_buffer, material_bundle.data(),
                                    material_bundle.size() * sizeof(Df_cuda_material)));
    kernel_params.material_buffer = reinterpret_cast<Df_cuda_material*>(material_buffer);

    // Setup environment map and acceleration
    CUdeviceptr env_accel;
    cudaArray_t env_tex_data;
    create_environment(
        &kernel_params.env_tex, &env_tex_data, &env_accel, &kernel_params.env_size, transaction,
        image_api, options.hdrfile.c_str());
    kernel_params.env_accel = reinterpret_cast<Env_accel *>(env_accel);
    kernel_params.env_rotation = options.hdr_rot / 180.0f * float(M_PI);

    // Setup GPU runtime of the LPE state machine
    upload_lpe_state_machine(kernel_params, lpe_state_machine);

    // Setup file name for nogl mode
    std::string next_filename_base;
    std::string filename_base, filename_ext;
    size_t dot_pos = options.outputfile.rfind('.');
    if (dot_pos == std::string::npos) {
        filename_base = options.outputfile;
    } else {
        filename_base = options.outputfile.substr(0, dot_pos);
        filename_ext = options.outputfile.substr(dot_pos);
    }
    if (options.material_names.size() > 1)
        next_filename_base = filename_base + "-0";
    else
        next_filename_base = filename_base;

    // Scope for material context resources
    {
        // Prepare the needed data of all target codes for the GPU
        Material_gpu_context material_gpu_context(options.enable_derivatives);
        if (!material_gpu_context.prepare_target_code_data(
                transaction.get(), image_api.get(), target_code.get(), arg_block_indices))
            terminate();
        kernel_params.tc_data = reinterpret_cast<Target_code_data *>(
            material_gpu_context.get_device_target_code_data_list());
        kernel_params.arg_block_list = reinterpret_cast<char const **>(
            material_gpu_context.get_device_target_argument_block_list());

        String_constant_table constant_table(target_code);
        Resource_table texture_table(target_code, transaction, Resource_table::RESOURCE_TEXTURE);
        Resource_table lp_table(target_code, transaction, Resource_table::RESOURCE_LIGHT_PROFILE);
        Resource_table bm_table(
            target_code, transaction, Resource_table::RESOURCE_BSDF_MEASUREMENT);

        // Collect information about the arguments of the compiled materials
        std::vector<Material_info> mat_infos;
        for (size_t i = 0, num_mats = compiled_materials.size(); i < num_mats; ++i) {
            // Get the compiled material and the parameter annotations
            mi::neuraylib::ICompiled_material const *cur_mat = compiled_materials[i].get();
            mi::neuraylib::IMaterial_definition const *cur_def = material_defs[i].get();
            mi::base::Handle<mi::neuraylib::IAnnotation_list const> anno_list(
                cur_def->get_parameter_annotations());

            // Get the target argument block and its layout
            size_t arg_block_index = material_gpu_context.get_bsdf_argument_block_index(i);
            mi::base::Handle<mi::neuraylib::ITarget_value_layout const> layout(
                material_gpu_context.get_argument_block_layout(arg_block_index));
            mi::base::Handle<mi::neuraylib::ITarget_argument_block> arg_block(
                material_gpu_context.get_argument_block(arg_block_index));
            char *arg_block_data = arg_block != nullptr ? arg_block->get_data() : nullptr;

            Material_info mat_info(cur_def->get_mdl_name());
            for (mi::Size j = 0, num_params = cur_mat->get_parameter_count(); j < num_params; ++j) {
                const char *name = cur_mat->get_parameter_name(j);
                if (name == nullptr) continue;

                // Determine the type of the argument
                mi::base::Handle<mi::neuraylib::IValue const> arg(cur_mat->get_argument(j));
                mi::neuraylib::IValue::Kind kind = arg->get_kind();

                Param_info::Param_kind param_kind            = Param_info::PK_UNKNOWN;
                Param_info::Param_kind param_array_elem_kind = Param_info::PK_UNKNOWN;
                mi::Size               param_array_size      = 0;
                mi::Size               param_array_pitch     = 0;
                const Enum_type_info   *enum_type            = nullptr;

                switch (kind) {
                case mi::neuraylib::IValue::VK_FLOAT:
                    param_kind = Param_info::PK_FLOAT;
                    break;
                case mi::neuraylib::IValue::VK_COLOR:
                    param_kind = Param_info::PK_COLOR;
                    break;
                case mi::neuraylib::IValue::VK_BOOL:
                    param_kind = Param_info::PK_BOOL;
                    break;
                case mi::neuraylib::IValue::VK_INT:
                    param_kind = Param_info::PK_INT;
                    break;
                case mi::neuraylib::IValue::VK_VECTOR:
                    {
                        mi::base::Handle<mi::neuraylib::IValue_vector const> val(
                            arg.get_interface<mi::neuraylib::IValue_vector const>());
                        mi::base::Handle<mi::neuraylib::IType_vector const> val_type(
                            val->get_type());
                        mi::base::Handle<mi::neuraylib::IType_atomic const> elem_type(
                            val_type->get_element_type());
                        if (elem_type->get_kind() == mi::neuraylib::IType::TK_FLOAT) {
                            switch (val_type->get_size()) {
                            case 2: param_kind = Param_info::PK_FLOAT2; break;
                            case 3: param_kind = Param_info::PK_FLOAT3; break;
                            default: assert(false || "Vector Size invalid or unhandled.");
                            }
                        }
                    }
                    break;
                case mi::neuraylib::IValue::VK_ARRAY:
                    {
                        mi::base::Handle<mi::neuraylib::IValue_array const> val(
                            arg.get_interface<mi::neuraylib::IValue_array const>());
                        mi::base::Handle<mi::neuraylib::IType_array const> val_type(
                            val->get_type());
                        mi::base::Handle<mi::neuraylib::IType const> elem_type(
                            val_type->get_element_type());

                        // we currently only support arrays of some values
                        switch (elem_type->get_kind()) {
                        case mi::neuraylib::IType::TK_FLOAT:
                            param_array_elem_kind = Param_info::PK_FLOAT;
                            break;
                        case mi::neuraylib::IType::TK_COLOR:
                            param_array_elem_kind = Param_info::PK_COLOR;
                            break;
                        case mi::neuraylib::IType::TK_BOOL:
                            param_array_elem_kind = Param_info::PK_BOOL;
                            break;
                        case mi::neuraylib::IType::TK_INT:
                            param_array_elem_kind = Param_info::PK_INT;
                            break;
                        case mi::neuraylib::IType::TK_VECTOR:
                            {
                                mi::base::Handle<mi::neuraylib::IType_vector const> val_type(
                                    elem_type.get_interface<
                                        mi::neuraylib::IType_vector const>());
                                mi::base::Handle<mi::neuraylib::IType_atomic const> velem_type(
                                    val_type->get_element_type());
                                if (velem_type->get_kind() == mi::neuraylib::IType::TK_FLOAT) {
                                    switch (val_type->get_size()) {
                                    case 2:
                                        param_array_elem_kind = Param_info::PK_FLOAT2;
                                        break;
                                    case 3:
                                        param_array_elem_kind = Param_info::PK_FLOAT3;
                                        break;
                                    default:
                                        assert(false || "Vector Size invalid or unhandled.");
                                    }
                                }
                            }
                            break;
                        default:
                            assert(false || "Array element type invalid or unhandled.");
                        }
                        if (param_array_elem_kind != Param_info::PK_UNKNOWN) {
                            param_kind = Param_info::PK_ARRAY;
                            param_array_size = val_type->get_size();

                            // determine pitch of array if there are at least two elements
                            if (param_array_size > 1) {
                                mi::neuraylib::Target_value_layout_state array_state(
                                    layout->get_nested_state(j));
                                mi::neuraylib::Target_value_layout_state next_elem_state(
                                    layout->get_nested_state(1, array_state));

                                mi::neuraylib::IValue::Kind kind;
                                mi::Size param_size;
                                mi::Size start_offset = layout->get_layout(
                                    kind, param_size, array_state);
                                mi::Size next_offset = layout->get_layout(
                                    kind, param_size, next_elem_state);
                                param_array_pitch = next_offset - start_offset;
                            }
                        }
                    }
                    break;
                case mi::neuraylib::IValue::VK_ENUM:
                    {
                        mi::base::Handle<mi::neuraylib::IValue_enum const> val(
                            arg.get_interface<mi::neuraylib::IValue_enum const>());
                        mi::base::Handle<mi::neuraylib::IType_enum const> val_type(
                            val->get_type());

                        // prepare info for this enum type if not seen so far
                        const Enum_type_info *info = mat_info.get_enum_type(val_type->get_symbol());
                        if (info == nullptr) {
                            std::shared_ptr<Enum_type_info> p(new Enum_type_info());

                            for (mi::Size i = 0, n = val_type->get_size(); i < n; ++i) {
                                p->add(val_type->get_value_name(i), val_type->get_value_code(i));
                            }
                            mat_info.add_enum_type(val_type->get_symbol(), p);
                            info = p.get();
                        }
                        enum_type = info;

                        param_kind = Param_info::PK_ENUM;
                    }
                    break;
                case mi::neuraylib::IValue::VK_STRING:
                    param_kind = Param_info::PK_STRING;
                    break;
                case mi::neuraylib::IValue::VK_TEXTURE:
                    param_kind = Param_info::PK_TEXTURE;
                    break;
                case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
                    param_kind = Param_info::PK_LIGHT_PROFILE;
                    break;
                case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
                    param_kind = Param_info::PK_BSDF_MEASUREMENT;
                    break;
                default:
                    // Unsupported? -> skip
                    continue;
                }

                // Get the offset of the argument within the target argument block
                mi::neuraylib::Target_value_layout_state state(layout->get_nested_state(j));
                mi::neuraylib::IValue::Kind kind2;
                mi::Size param_size;
                mi::Size offset = layout->get_layout(kind2, param_size, state);
                check_success(kind == kind2);

                Param_info param_info(
                    j,
                    name,
                    name,
                    /*group_name=*/ nullptr,
                    param_kind,
                    param_array_elem_kind,
                    param_array_size,
                    param_array_pitch,
                    arg_block_data + offset,
                    enum_type);

                // Check for annotation info
                mi::base::Handle<mi::neuraylib::IAnnotation_block const> anno_block(
                    anno_list->get_annotation_block(name));
                if (anno_block) {
                    mi::neuraylib::Annotation_wrapper annos(anno_block.get());
                    mi::Size anno_index =
                        annos.get_annotation_index("::anno::soft_range(float,float)");
                    if (anno_index == mi::Size(-1)) {
                        anno_index = annos.get_annotation_index("::anno::hard_range(float,float)");
                    }
                    if (anno_index != mi::Size(-1)) {
                        annos.get_annotation_param_value(anno_index, 0, param_info.range_min());
                        annos.get_annotation_param_value(anno_index, 1, param_info.range_max());
                    }
                    anno_index = annos.get_annotation_index("::anno::display_name(string)");
                    if (anno_index != mi::Size(-1)) {
                        annos.get_annotation_param_value(anno_index, 0, param_info.display_name());
                    }
                    anno_index = annos.get_annotation_index("::anno::in_group(string)");
                    if (anno_index != mi::Size(-1)) {
                        annos.get_annotation_param_value(anno_index, 0, param_info.group_name());
                    }
                }

                mat_info.add_sorted_by_group(param_info);
            }
            mat_infos.push_back(mat_info);
        }

        std::chrono::duration<double> state_update_time( 0.0 );
        std::chrono::duration<double> render_time( 0.0 );
        std::chrono::duration<double> display_time( 0.0 );
        char stats_text[128];
        int last_update_frames = -1;
        auto last_update_time = std::chrono::steady_clock::now();
        const std::chrono::duration<double> update_min_interval( 0.5 );

        // Main render loop
        while (true)
        {
            std::chrono::time_point<std::chrono::steady_clock> t0;

            if (!options.opengl)
            {
                kernel_params.resolution.x = width;
                kernel_params.resolution.y = height;
                kernel_params.accum_buffer = reinterpret_cast<float3 *>(accum_buffer);
                kernel_params.albedo_buffer = reinterpret_cast<float3 *>(aux_albedo_buffer);
                kernel_params.normal_buffer = reinterpret_cast<float3 *>(aux_normal_buffer);


                // Check if desired number of samples is reached
                if (kernel_params.iteration_start >= options.iterations) {
                    std::cout << "rendering done" << std::endl;

                    save_result(
                        accum_buffer, width, height,
                        next_filename_base + filename_ext,
                        image_api, mdl_impexp_api);

                    save_result(
                        aux_albedo_buffer, width, height,
                        next_filename_base +  "_albedo" + filename_ext,
                        image_api, mdl_impexp_api);

                    save_result(
                        aux_normal_buffer, width, height,
                        next_filename_base + "_normal" + filename_ext,
                        image_api, mdl_impexp_api);

                    std::cout << std::endl;

                    // All materials have been rendered? -> done
                    if (kernel_params.current_material + 1 >= material_bundle.size())
                        break;

                    if (material_bundle[kernel_params.current_material].contains_hair_bsdf == 0)
                        kernel_params.geometry = GT_SPHERE;
                    else
                        kernel_params.geometry = GT_HAIR;

                    // Start new image with next material
                    kernel_params.iteration_start = 0;
                    ++kernel_params.current_material;
                    next_filename_base =
                        filename_base + "-" + to_string(kernel_params.current_material);
                }

                std::cout
                    << "rendering iterations " << kernel_params.iteration_start << " to "
                    << kernel_params.iteration_start + kernel_params.iteration_num << std::endl;
            }
            else
            {
                t0 = std::chrono::steady_clock::now();

                // Check for termination
                if (glfwWindowShouldClose(window))
                    break;

                // Poll for events and process them
                glfwPollEvents();

                // Check if buffers need to be resized
                int nwidth, nheight;
                glfwGetFramebufferSize(window, &nwidth, &nheight);
                if (nwidth != width || nheight != height)
                {
                    width = nwidth;
                    height = nheight;

                    gl_display->resize(width, height);

                    resize_buffers(
                        &accum_buffer, width, height);
                    kernel_params.accum_buffer = reinterpret_cast<float3 *>(accum_buffer);

                    resize_buffers(&aux_albedo_buffer, width, height);
                    kernel_params.albedo_buffer = reinterpret_cast<float3 *>(aux_albedo_buffer);


                    resize_buffers(&aux_normal_buffer, width, height);
                    kernel_params.normal_buffer = reinterpret_cast<float3 *>(aux_normal_buffer);

                    kernel_params.resolution.x = width;
                    kernel_params.resolution.y = height;
                    kernel_params.iteration_start = 0;
                }

                // Don't render anything, if minimized
                if (width == 0 || height == 0) {
                    // Wait until something happens
                    glfwWaitEvents();
                    continue;
                }

                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                // Create material parameter editor window

                ImGui::SetNextWindowPos(ImVec2(10, 100), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(
                    ImVec2(360 * options.gui_scale, 600 * options.gui_scale),
                    ImGuiCond_FirstUseEver);
                ImGui::Begin("Settings");
                ImGui::SetWindowFontScale(options.gui_scale);
                ImGui::PushItemWidth(-200 * options.gui_scale);
                if (options.use_class_compilation)
                    ImGui::Text("CTRL + Click to manually enter numbers");
                else
                    ImGui::Text("Parameter editing requires class compilation.");

                if (kernel_params.enable_auxiliary_output)
                {
                    ImGui::Dummy(ImVec2(0.0f, 3.0f));
                    ImGui::Text("Display options");
                    ImGui::Separator();

                    std::string current_lpe_name = lpe_state_machine.get_expression_name(
                        kernel_params.lpe_ouput_expression);
                    if (ImGui::BeginCombo("LPE", current_lpe_name.c_str()))
                    {
                        for (uint32_t i = 0; i < lpe_state_machine.get_expression_count(); ++i)
                        {
                            const char* name = lpe_state_machine.get_expression_name(i);
                            bool is_selected = (i == kernel_params.lpe_ouput_expression);
                            if (ImGui::Selectable(name, is_selected))
                            {
                                kernel_params.lpe_ouput_expression = i;
                                kernel_params.iteration_start = 0;
                            }
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }

                    std::string current_display_buffer =
                        to_string((Display_buffer_options) kernel_params.display_buffer_index);
                    if (ImGui::BeginCombo("buffer", current_display_buffer.c_str()))
                    {
                        for (unsigned i = 0; i < (unsigned) DISPLAY_BUFFER_COUNT; ++i)
                        {
                            const std::string &name = to_string((Display_buffer_options) i);
                            bool is_selected = (current_display_buffer == name);
                            if (ImGui::Selectable(name.c_str(), is_selected))
                            {
                                kernel_params.display_buffer_index = i;
                                kernel_params.iteration_start = 0;
                            }
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                }

                ImGui::Dummy(ImVec2(0.0f, 3.0f));
                ImGui::Text("Light parameters");
                ImGui::Separator();

                if (ImGui::ColorEdit3("Point Light Color", &kernel_params.light_color.x))
                    kernel_params.iteration_start = 0;

                if (ImGui::SliderFloat("Point Light Intensity",
                    &kernel_params.light_intensity, 0.0f, 50000.0f))
                        kernel_params.iteration_start = 0;

                if (ImGui::SliderFloat("Environment Intensity Scale",
                    &kernel_params.env_intensity, 0.0f, 10.0f))
                        kernel_params.iteration_start = 0;

                float env_rot_degree = kernel_params.env_rotation / float(M_PI) * 180.0f;
                if (ImGui::SliderFloat("Environment Rotation",
                    &env_rot_degree, 0.0f, 360.0f))
                {
                    // wrap in case of negative input
                    // we don't want fmodf behavior for negative values
                    env_rot_degree -= floorf(env_rot_degree / 360.0f) * 360.f;
                    kernel_params.env_rotation = fmodf(env_rot_degree, 360.0f) / 180.0f * float(M_PI);
                    kernel_params.iteration_start = 0;
                }

                ImGui::Dummy(ImVec2(0.0f, 3.0f));
                ImGui::Text("Material parameters");
                ImGui::Separator();

                Material_info &mat_info = mat_infos[
                    material_bundle[kernel_params.current_material].compiled_material_index];

                // Print material name
                ImGui::Text("%s", mat_info.name());

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
                        (param.group_name() &&
                            (!group_name || strcmp(group_name, param.group_name()) != 0)))
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
                            ImGui::Indent(16.0f * options.gui_scale);
                            char *ptr = &param.data<char>();
                            for (mi::Size i = 0, n = param.array_size(); i < n; ++i) {
                                std::string idx_str = to_string(i);
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
                                    assert(false || "Array element type invalid or unhandled.");
                                }
                                ptr += param.array_pitch();
                            }
                            ImGui::Unindent(16.0f * options.gui_scale);
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
                        {
                            std::vector<char> buf;

                            size_t max_len = constant_table.get_max_length();
                            max_len = max_len > 63 ? max_len + 1 : 64;

                            buf.resize(max_len);

                            // fill the current value
                            unsigned curr_index = param.data<unsigned>();
                            const char *opt = constant_table.get_string(curr_index);
                            strcpy(buf.data(), opt != nullptr ? opt : "");

                            if (ImGui::InputText(
                                param.display_name(),
                                buf.data(), buf.size(),
                                ImGuiInputTextFlags_EnterReturnsTrue))
                            {
                                unsigned id = constant_table.get_id_for_string(buf.data());

                                param.data<unsigned>() = id;
                                changed = true;
                            }
                        }
                        break;
                    case Param_info::PK_TEXTURE:
                        changed |= handle_resource(param, texture_table);
                        break;
                    case Param_info::PK_LIGHT_PROFILE:
                        changed |= handle_resource(param, lp_table);
                        break;
                    case Param_info::PK_BSDF_MEASUREMENT:
                        changed |= handle_resource(param, bm_table);
                        break;
                    case Param_info::PK_UNKNOWN:
                        break;
                    }

                    ImGui::PopID();
                }

                if (options.enable_derivatives) {
                    ImGui::Separator();
                    bool b = kernel_params.use_derivatives != 0;
                    if (ImGui::Checkbox("Use derivatives", &b)) {
                        kernel_params.iteration_start = 0;
                        kernel_params.use_derivatives = b;
                    }
                }

                ImGui::PopItemWidth();
                ImGui::End();

                // If any material argument changed, update the target argument block on the device
                if (changed) {
                    material_gpu_context.update_device_argument_block(
                        material_bundle[kernel_params.current_material].argument_block_index);
                    kernel_params.iteration_start = 0;
                }

                // Handle events
                Window_context *ctx =
                    static_cast<Window_context*>(glfwGetWindowUserPointer(window));
                if (ctx->save_result && !ImGui::GetIO().WantCaptureKeyboard) {
                    save_result(
                        accum_buffer,
                        width, height,
                        options.outputfile,
                        image_api, mdl_impexp_api);

                    save_result(
                        aux_albedo_buffer,
                        width, height,
                        filename_base + "_albedo" + filename_ext,
                        image_api, mdl_impexp_api);

                    save_result(
                        aux_normal_buffer,
                        width, height,
                        filename_base + "_normal" + filename_ext,
                        image_api, mdl_impexp_api);
                }
                if (ctx->exposure_event && !ImGui::GetIO().WantCaptureKeyboard) {
                    kernel_params.exposure_scale = powf(2.0f, ctx->exposure);
                }
                if (ctx->key_event && !ImGui::GetIO().WantCaptureKeyboard) {
                    kernel_params.iteration_start = 0;

                    // Update change material
                    const unsigned num_materials = unsigned(material_bundle.size());
                    kernel_params.current_material = (kernel_params.current_material +
                        ctx->material_index_delta + num_materials) % num_materials;
                    ctx->material_index_delta = 0;

                    if (material_bundle[kernel_params.current_material].contains_hair_bsdf == 0)
                        kernel_params.geometry = GT_SPHERE;
                    else
                        kernel_params.geometry = GT_HAIR;
                }
                if (ctx->mouse_button - 1 == GLFW_MOUSE_BUTTON_LEFT) {
                    // Only accept button press when not hovering GUI window
                    if (ctx->mouse_button_action == GLFW_PRESS &&
                            !ImGui::GetIO().WantCaptureMouse) {
                        ctx->moving = true;
                        glfwGetCursorPos(window, &ctx->move_start_x, &ctx->move_start_y);
                    }
                    else
                        ctx->moving = false;
                }
                if (ctx->mouse_wheel_delta && !ImGui::GetIO().WantCaptureMouse) {
                    ctx->zoom += ctx->mouse_wheel_delta;
                }
                if (ctx->mouse_event && !ImGui::GetIO().WantCaptureMouse) {
                    kernel_params.iteration_start = 0;

                    // Update camera
                    phi -= ctx->move_dx * 0.001 * M_PI;
                    theta -= ctx->move_dy * 0.001 * M_PI;
                    theta = std::max(theta, 0.00 * M_PI);
                    theta = std::min(theta, 1.00 * M_PI);
                    ctx->move_dx = ctx->move_dy = 0.0;

                    update_camera(kernel_params, phi, theta, base_dist, ctx->zoom);
                }

                // Clear all events
                ctx->save_result = false;
                ctx->key_event = false;
                ctx->mouse_event = false;
                ctx->exposure_event = false;
                ctx->mouse_wheel_delta = 0;
                ctx->mouse_button = 0;

                auto t1 = std::chrono::steady_clock::now();
                state_update_time += t1 - t0;
                t0 = t1;

                // Map GL buffer for access with CUDA
                kernel_params.display_buffer =
                    reinterpret_cast<unsigned int *>(gl_display->map(0));
            }


            // Launch kernel
            dim3 threads_per_block(16, 16);
            dim3 num_blocks((width + 15) / 16, (height + 15) / 16);
            void *params[] = { &kernel_params };
            check_cuda_success(cuLaunchKernel(
                cuda_function,
                num_blocks.x, num_blocks.y, num_blocks.z,
                threads_per_block.x, threads_per_block.y, threads_per_block.z,
                0, nullptr, params, nullptr));


            kernel_params.iteration_start += kernel_params.iteration_num;

            // Make sure, any debug::print()s are written to the console
            check_cuda_success(cuStreamSynchronize(0));

            if (options.opengl)
            {
                // Unmap GL buffer
                gl_display->unmap(0);

                auto t1 = std::chrono::steady_clock::now();
                render_time += t1 - t0;
                t0 = t1;

                // Render GL buffer
                gl_display->update_display();

                t1 = std::chrono::steady_clock::now();
                display_time += t1 - t0;

                // Render stats window
                ImGui::SetNextWindowPos(ImVec2(10, 10));
                ImGui::Begin("##notitle", nullptr,
                    ImGuiWindowFlags_NoDecoration |
                    ImGuiWindowFlags_AlwaysAutoResize |
                    ImGuiWindowFlags_NoSavedSettings |
                    ImGuiWindowFlags_NoFocusOnAppearing |
                    ImGuiWindowFlags_NoNav);

                // Update stats only every 0.5s
                ++last_update_frames;
                if (t1 - last_update_time > update_min_interval || last_update_frames == 0) {
                    typedef std::chrono::duration<double, std::milli> durationMs;

                    snprintf(stats_text, sizeof(stats_text),
                        "%5.1f fps\n\n"
                        "state update: %8.1f ms\n"
                        "render:       %8.1f ms\n"
                        "display:      %8.1f ms\n",
                        last_update_frames / std::chrono::duration<double>(
                            t1 - last_update_time).count(),
                        (durationMs(state_update_time) / last_update_frames).count(),
                        (durationMs(render_time) / last_update_frames).count(),
                        (durationMs(display_time) / last_update_frames).count());

                    last_update_time = t1;
                    last_update_frames = 0;
                    state_update_time = render_time = display_time =
                        std::chrono::duration<double>::zero();
                }

                ImGui::TextUnformatted(stats_text);
                ImGui::End();

                // Show the GUI
                ImGui::Render();
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                // Swap front and back buffers
                glfwSwapBuffers(window);
            }
        }
    }

    // Cleanup CUDA
    check_cuda_success(cudaDestroyTextureObject(kernel_params.env_tex));
    check_cuda_success(cudaFreeArray(env_tex_data));
    check_cuda_success(cuMemFree(env_accel));
    check_cuda_success(cuMemFree(accum_buffer));
    check_cuda_success(cuMemFree(aux_albedo_buffer));
    check_cuda_success(cuMemFree(aux_normal_buffer));
    check_cuda_success(cuMemFree(material_buffer));
    check_cuda_success(cuMemFree(reinterpret_cast<CUdeviceptr>(kernel_params.lpe_state_table)));
    check_cuda_success(cuMemFree(reinterpret_cast<CUdeviceptr>(kernel_params.lpe_final_mask)));
    check_cuda_success(cuModuleUnload(cuda_module));
    uninit_cuda(cuda_context);

    // Cleanup OpenGL
    if (options.opengl) {
        delete gl_display;
        gl_display = nullptr;
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

// Returns true, if the string str starts with the given prefix, false otherwise.
bool starts_with(std::string const &str, std::string const &prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

// Create application material representation for use in our CUDA kernel
Df_cuda_material create_cuda_material(
    size_t target_code_index,
    size_t compiled_material_index,
    std::vector<mi::neuraylib::Target_function_description> const& descs,
    bool use_hair_bsdf)
{
    Df_cuda_material mat;

    // shared by all generated functions of the same material
    // used here to alter the materials parameter set
    mat.compiled_material_index = static_cast<unsigned int>(compiled_material_index);

    // Note: the same argument_block_index is filled into all function descriptions of a
    //       material, if any function uses it
    mat.argument_block_index = static_cast<unsigned int>(descs[0].argument_block_index);

    if (!use_hair_bsdf)
    {
        // identify the BSDF function by target_code_index (i'th link unit)
        // and the function_index inside this target_code.
        // same for the EDF and the intensity expression.
        mat.bsdf.x = static_cast<unsigned int>(target_code_index);
        mat.bsdf.y = static_cast<unsigned int>(descs[0].function_index);

        mat.edf.x = static_cast<unsigned int>(target_code_index);
        mat.edf.y = static_cast<unsigned int>(descs[1].function_index);

        mat.emission_intensity.x = static_cast<unsigned int>(target_code_index);
        mat.emission_intensity.y = static_cast<unsigned int>(descs[2].function_index);

        mat.volume_absorption.x = static_cast<unsigned int>(target_code_index);
        mat.volume_absorption.y = static_cast<unsigned int>(descs[3].function_index);

        mat.thin_walled.x = static_cast<unsigned int>(target_code_index);
        mat.thin_walled.y = static_cast<unsigned int>(descs[4].function_index);
    }
    else
    {
        mat.bsdf.x = static_cast<unsigned int>(target_code_index);
        mat.bsdf.y = static_cast<unsigned int>(descs[5].function_index);
        mat.contains_hair_bsdf = 1;
    }

    // init tag maps with zeros (optional)
    memset(mat.bsdf_mtag_to_gtag_map, 0, MAX_DF_HANDLES * sizeof(unsigned int));
    memset(mat.edf_mtag_to_gtag_map, 0, MAX_DF_HANDLES * sizeof(unsigned int));
    return mat;
}

void create_cuda_material_handles(
    Df_cuda_material& mat,
    const mi::neuraylib::ITarget_code* target_code,
    LPE_state_machine& lpe_state_machine)
{
    // fill tag ID list.
    // allows to map from local per material Tag IDs to global per scene Tag IDs
    // Note, calling 'LPE_state_machine::handle_to_global_tag(...)' registers the string handles
    // present in the MDL in our 'scene'
    mat.bsdf_mtag_to_gtag_map_size = static_cast<unsigned int>(
        target_code->get_callable_function_df_handle_count(mat.bsdf.y));
    for (mi::Size i = 0; i < mat.bsdf_mtag_to_gtag_map_size; ++i)
        mat.bsdf_mtag_to_gtag_map[i] = lpe_state_machine.handle_to_global_tag(
            target_code->get_callable_function_df_handle(mat.bsdf.y, i));

    // same for all other distribution functions
    mat.edf_mtag_to_gtag_map_size = static_cast<unsigned int>(
        target_code->get_callable_function_df_handle_count(mat.edf.y));
    for (mi::Size i = 0; i < mat.edf_mtag_to_gtag_map_size; ++i)
        mat.edf_mtag_to_gtag_map[i] = lpe_state_machine.handle_to_global_tag(
            target_code->get_callable_function_df_handle(mat.edf.y, i));
}

// checks if a compiled material contains none-invalid hair BSDF
bool contains_hair_bsdf(const mi::neuraylib::ICompiled_material* compiled_material)
{
    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> body(
        compiled_material->get_body());

    mi::base::Handle<const mi::neuraylib::IExpression_list> body_args(body->get_arguments());
    for (mi::Size i = 0, n = body_args->get_size(); i < n; ++i)
    {
        const char* name = body_args->get_name(i);
        if (strcmp(name, "hair") == 0)
        {
            mi::base::Handle<const mi::neuraylib::IExpression> hair_exp(
                body_args->get_expression(i));

            if (hair_exp->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
                return true;

            mi::base::Handle<const mi::neuraylib::IExpression_constant> hair_exp_const(
                hair_exp->get_interface<const mi::neuraylib::IExpression_constant>());

            mi::base::Handle<const mi::neuraylib::IValue> hair_exp_const_value(
                hair_exp_const->get_value());

            return hair_exp_const_value->get_kind() != mi::neuraylib::IValue::VK_INVALID_DF;
        }
    }
    return true;
}

static void usage(const char *name)
{
    std::cout
        << "usage: " << name << " [options] [<material_name1|full_mdle_path1> ...]\n"
        << "-h|--help                   print this text and exit\n"
        << "-v|--version                print the MDL SDK version string and exit\n"
        << "--device <id>               run on CUDA device <id> (default: 0)\n"
        << "--nogl                      don't open interactive display\n"
        << "--nocc                      don't use class-compilation\n"
        << "--noaux                     don't generate code for albedo and normal buffers\n"
        << "--gui_scale <factor>        GUI scaling factor (default: 1.0)\n"
        << "--res <res_x> <res_y>       resolution (default: 1024x1024)\n"
        << "--hdr <filename>            HDR environment map "
           "(default: nvidia/sdk_examples/resources/environment.hdr)\n"
        << "--hdr_rot <degrees>         rotation of the environment in degrees (default: 0.0)\n"
        << "-o <outputfile>             image file to write result to (default: output.exr).\n"
        << "                            With multiple materials \"-<material index>\" will be\n"
        << "                            added in front of the extension\n"
        << "--spp <num>                 samples per pixel, only active for --nogl (default: 4096)\n"
        << "--spi <num>                 samples per render call (default: 8)\n"
        << "-t <type>                   0: eval, 1: sample, 2: mis, 3: mis + pdf, 4: no env\n"
        << "                            (default: 2)\n"
        << "-e <exposure>               exposure for interactive display (default: 0.0)\n"
        << "-f <fov>                    the camera field of view in degree (default: 96.0)\n"
        << "-p <x> <y> <z>              set the camera position (default 0 0 3).\n"
        << "                            The camera will always look towards (0, 0, 0).\n"
        << "-l <x> <y> <z> <r> <g> <b>  add an isotropic point light with given coordinates and "
           "intensity (flux)\n"
        << "--mdl_path <path>           MDL search path, can occur multiple times.\n"
        << "--max_path_length <num>     maximum path length, default 4 (up to one total internal\n"
        << "                            reflection), clamped to 2..100\n"
        << "--noaa                      disable pixel oversampling\n"
        << "-d                          enable use of derivatives\n"
        << "--fold_ternary_on_df        fold all ternary operators on *df types (default: false)\n"
        << "\n"
        << "Note: material names can end with an '*' as a wildcard\n"
        << "      and alternatively, full MDLE file paths can be passed as material name\n";

    exit(EXIT_FAILURE);
}

int MAIN_UTF8(int argc, char* argv[])
{
    // Parse commandline options
    Options options;
    mi::examples::mdl::Configure_options configure_options;
    bool print_version_and_exit = false;

    for (int i = 1; i < argc; ++i) {
        const char *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "--device") == 0 && i < argc - 1) {
                options.cuda_device = atoi(argv[++i]);
            } else if (strcmp(opt, "--nogl") == 0) {
                options.opengl = false;
            } else if (strcmp(opt, "--nocc") == 0) {
                options.use_class_compilation = false;
            } else if (strcmp(opt, "--noaux") == 0) {
                options.enable_auxiliary_output = false;
            } else if (strcmp(opt, "--gui_scale") == 0 && i < argc - 1) {
                options.gui_scale = static_cast<float>(atof(argv[++i]));
            } else if (strcmp(opt, "--res") == 0 && i < argc - 2) {
                options.res_x = std::max(atoi(argv[++i]), 1);
                options.res_y = std::max(atoi(argv[++i]), 1);
            } else if (strcmp(opt, "--hdr") == 0 && i < argc - 1) {
                options.hdrfile = argv[++i];
            } else if (strcmp(opt, "--hdr_rot") == 0 && i < argc - 1) {
                options.hdr_rot = static_cast<float>(atof(argv[++i]));
                // wrap in case of negative input
                // we don't want fmodf behavior for negative values
                options.hdr_rot -= floorf(options.hdr_rot / 360.0f) * 360.f;
                options.hdr_rot = fmodf(options.hdr_rot, 360.0f);
            } else if (strcmp(opt, "-o") == 0 && i < argc - 1) {
                options.outputfile = argv[++i];
            } else if (strcmp(opt, "--spp") == 0 && i < argc - 1) {
                options.iterations = std::max(atoi(argv[++i]), 1);
            } else if (strcmp(opt, "--spi") == 0 && i < argc - 1) {
                options.samples_per_iteration = std::max(atoi(argv[++i]), 1);
            } else if (strcmp(opt, "-t") == 0 && i < argc - 1) {
                const int type = atoi(argv[++i]);
                if (type < 0 || type >= MDL_TEST_COUNT) {
                    std::cout << "Invalid type for \"-t\" option!" << std::endl;
                    usage(argv[0]);
                }
                options.mdl_test_type = type;
            } else if (strcmp(opt, "-e") == 0 && i < argc - 1) {
                options.exposure = static_cast<float>(atof(argv[++i]));
            } else if (strcmp(opt, "-f") == 0 && i < argc - 1) {
                options.fov = static_cast<float>(atof(argv[++i]));
            } else if (strcmp(opt, "-p") == 0 && i < argc - 3) {
                options.cam_pos.x = static_cast<float>(atof(argv[++i]));
                options.cam_pos.y = static_cast<float>(atof(argv[++i]));
                options.cam_pos.z = static_cast<float>(atof(argv[++i]));
            } else if (strcmp(opt, "-l") == 0 && i < argc - 6) {
                options.light_pos.x = static_cast<float>(atof(argv[++i]));
                options.light_pos.y = static_cast<float>(atof(argv[++i]));
                options.light_pos.z = static_cast<float>(atof(argv[++i]));
                options.light_intensity.x = static_cast<float>(atof(argv[++i]));
                options.light_intensity.y = static_cast<float>(atof(argv[++i]));
                options.light_intensity.z = static_cast<float>(atof(argv[++i]));
            } else if (strcmp(opt, "--mdl_path") == 0 && i < argc - 1) {
                configure_options.additional_mdl_paths.push_back(argv[++i]);
            } else if (strcmp(opt, "--max_path_length") == 0 && i < argc - 1) {
                options.max_path_length = std::min(std::max(atoi(argv[++i]), 2), 100);
            } else if (strcmp(opt, "--noaa") == 0) {
                options.no_aa = true;
            } else if (strcmp(opt, "-d") == 0) {
                options.enable_derivatives = true;
            } else if (strcmp(opt, "--fold_ternary_on_df") == 0) {
                options.fold_ternary_on_df = true;
            } else if (strcmp(opt, "-v") == 0 || strcmp(opt, "--version") == 0) {
                print_version_and_exit = true;
            } else {
                if (strcmp(opt, "-h") != 0 && strcmp(opt, "--help") != 0)
                    std::cout << "Unknown option: \"" << opt << "\"" << std::endl;
                usage(argv[0]);
            }
        }
        else
            options.material_names.push_back(std::string(opt));
    }

    // Use default material, if none was provided via command line
    if (options.material_names.empty())
        options.material_names.push_back("::nvidia::sdk_examples::tutorials::example_df");

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Handle the --version flag
    if (print_version_and_exit) {

        // print library version information.
        mi::base::Handle<const mi::neuraylib::IVersion> version(
            neuray->get_api_component<const mi::neuraylib::IVersion>());
        fprintf(stdout, "%s\n", version->get_string());

        // free the handles and unload the MDL SDK
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
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

    // LPE state machine for rendering into multiple buffers
    LPE_state_machine lpe_state_machine;
    lpe_state_machine.handle_to_global_tag("point_light");  // register handles before building
    lpe_state_machine.handle_to_global_tag("env");          // the state machine

     // register other handles in the scene, e.g.: for object instances
    lpe_state_machine.handle_to_global_tag("sphere");       // for illustration, not used currently

    // Add some common and custom LPEs
    lpe_state_machine.add_expression("Beauty", LPE::create_common(LPE::Common::Beauty));

    lpe_state_machine.add_expression("Diffuse", LPE::create_common(LPE::Common::Diffuse));
    lpe_state_machine.add_expression("Glossy", LPE::create_common(LPE::Common::Glossy));
    lpe_state_machine.add_expression("Specular", LPE::create_common(LPE::Common::Specular));
    lpe_state_machine.add_expression("SSS", LPE::create_common(LPE::Common::SSS));
    lpe_state_machine.add_expression("Transmission", LPE::create_common(LPE::Common::Transmission));

    lpe_state_machine.add_expression("Beauty-Env", LPE::sequence({
        LPE::camera(),
        LPE::zero_or_more(LPE::any_scatter()),
        LPE::light("env") }));  // only light with the name 'env'

    lpe_state_machine.add_expression("Beauty-PointLight", LPE::sequence({
        LPE::camera(),
        LPE::zero_or_more(LPE::any_scatter()),
        LPE::light("point_light") })); // only light with the name 'point_light'

    lpe_state_machine.add_expression("Beauty-Emission", LPE::sequence({
        LPE::camera(),
        LPE::zero_or_more(LPE::any_scatter()),
        LPE::emission() })); // only emission


    lpe_state_machine.add_expression("Beauty-Base", LPE::sequence({
        LPE::camera(),
        LPE::zero_or_more(LPE::any_scatter("base")),
        LPE::light()})); // no emission

    lpe_state_machine.add_expression("Beauty-Coat", LPE::sequence({
        LPE::camera(),
        LPE::zero_or_more(LPE::any_scatter("coat")),
        LPE::light()})); // no emission

    lpe_state_machine.add_expression("Beauty-^Coat", LPE::sequence({
        LPE::camera(),
        LPE::zero_or_more(LPE::any_scatter("coat", false)),
        LPE::any_light()})); // emission or light source

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
        {
            // Initialize the material compiler with 16 result buffer slots ("texture results")
            Material_compiler mc(
                mdl_impexp_api.get(),
                mdl_backend_api.get(),
                mdl_factory.get(),
                transaction.get(),
                16,
                options.enable_derivatives,
                options.fold_ternary_on_df,
                options.enable_auxiliary_output,
                /*df_handle_mode=*/ "pointer");

            // List of materials in the scene
            std::vector<Df_cuda_material> material_bundle;

            // Select the functions to translate
            std::vector<mi::neuraylib::Target_function_description> descs;
            descs.push_back(
                mi::neuraylib::Target_function_description("surface.scattering"));
            descs.push_back(
                mi::neuraylib::Target_function_description("surface.emission.emission"));
            descs.push_back(
                mi::neuraylib::Target_function_description("surface.emission.intensity"));
            descs.push_back(
                mi::neuraylib::Target_function_description("volume.absorption_coefficient"));
            descs.push_back(
                mi::neuraylib::Target_function_description("thin_walled"));
            descs.push_back(
                mi::neuraylib::Target_function_description("hair"));

            // Generate code for all materials
            std::vector<std::string> used_material_names;
            for (size_t i = 0; i < options.material_names.size(); ++i) {
                std::string& opt_material_name = options.material_names[i];

                // split module and material name
                std::string module_qualified_name, material_simple_name;
                if (!mi::examples::mdl::parse_cmd_argument_material_name(
                    opt_material_name, module_qualified_name, material_simple_name, true))
                        exit_failure("Provided material name '%s' is invalid.",
                            opt_material_name.c_str());

                // Is this a material name pattern? (not applicable to mdle)
                if (!mi::examples::strings::ends_with(module_qualified_name, ".mdle") &&
                    opt_material_name.size() > 1 &&
                    opt_material_name.back() == '*') {

                    // prepare the pattern for matching
                    std::string pattern = opt_material_name.substr(0, opt_material_name.size() - 1);
                    if (!starts_with(pattern, "::"))
                        pattern = "::" + pattern;

                    // load the module
                    std::string module_db_name = mc.load_module(module_qualified_name);

                    // iterate over all materials in that module
                    mi::base::Handle<const mi::neuraylib::IModule> loaded_module(
                        transaction->access<const mi::neuraylib::IModule>(module_db_name.c_str()));

                    for (mi::Size j = 0, n = loaded_module->get_material_count(); j < n; ++j) {

                        // get the j`th material
                        const char* material_db_name = loaded_module->get_material(j);
                        mi::base::Handle<const mi::neuraylib::IMaterial_definition> mat_def(
                            transaction->access<const mi::neuraylib::IMaterial_definition>(
                                material_db_name));

                        // make sure the material name starts with the pattern
                        std::string material_qualified_name = mat_def->get_mdl_name();
                        if (!mi::examples::strings::starts_with(material_qualified_name, pattern))
                            continue;

                        std::cout << "Adding material \"" << material_qualified_name  << std::endl;

                        // Add functions of the material to the link unit
                        check_success(mc.add_material(
                            module_qualified_name, mat_def->get_mdl_simple_name(),
                            descs.data(), descs.size(),
                            options.use_class_compilation));

                        mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
                            mc.get_compiled_materials().back());

                        // Create application material representation
                        material_bundle.push_back(create_cuda_material(
                            0, material_bundle.size(), descs,
                            contains_hair_bsdf(compiled_material.get())));
                        used_material_names.push_back(material_qualified_name);
                    }
                } else {
                    std::string material_qualified_name = module_qualified_name + "::" + material_simple_name;
                    std::cout << "Adding material \"" << material_qualified_name << std::endl;

                    // Add functions of the material to the link unit
                    check_success(mc.add_material(
                        module_qualified_name, material_simple_name,
                        descs.data(), descs.size(),
                        options.use_class_compilation));

                    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
                        mc.get_compiled_materials().back());

                    // Create application material representation
                    material_bundle.push_back(create_cuda_material(
                        0, material_bundle.size(), descs,
                        contains_hair_bsdf(compiled_material.get())));
                    used_material_names.push_back(material_qualified_name);
                }
            }

            // Update the material names with the actually used names
            options.material_names = used_material_names;

            // Generate the CUDA PTX code for the link unit
            mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(
                mc.generate_cuda_ptx());

            // convert handles to tag IDs
            for (auto& mat : material_bundle)
                create_cuda_material_handles(mat, target_code.get(), lpe_state_machine);

            // Acquire image API needed to prepare the textures
            mi::base::Handle<mi::neuraylib::IImage_api> image_api(
                neuray->get_api_component<mi::neuraylib::IImage_api>());

            // when all scene elements that have handles are loaded and all handles as well as
            // light path expressions are registered, the state machine can be constructed.
            lpe_state_machine.build();

            // Render
            render_scene(
                options,
                transaction,
                image_api,
                mdl_impexp_api,
                target_code,
                mc.get_material_defs(),
                mc.get_compiled_materials(),
                mc.get_argument_block_indices(),
                material_bundle,
                lpe_state_machine);
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

