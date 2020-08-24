/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_core/df_cuda/example_df_cuda.cpp
//
// Simple renderer using compiled BSDFs with a material parameter editor GUI.

#include <iostream>
#include <string>
#include <vector>
#include <list>
#define _USE_MATH_DEFINES
#include <math.h>

#include "example_df_cuda.h"

// Enable this to dump the generated PTX code to stdout.
// #define DUMP_PTX

#define OPENGL_INTEROP
#include "example_cuda_shared.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define terminate()          \
    do {                     \
        glfwTerminate();     \
        keep_console_open(); \
        exit(EXIT_FAILURE);  \
    } while (0)

#define WINDOW_TITLE "MDL Core DF Example"


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


/////////////////
// OpenGL code //
/////////////////

// Initialize OpenGL and create a window with an associated OpenGL context.
static GLFWwindow *init_opengl(unsigned res_x, unsigned res_y, std::string& version_string)
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
        int(res_x), int(res_y), WINDOW_TITLE, nullptr, nullptr);
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

static void dump_info(GLuint shader, const char* text)
{
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if (length > 0) {
        GLchar *log = new GLchar[length + 1];
        glGetShaderInfoLog(shader, length + 1, nullptr, log);
        std::cerr << text << log << std::endl;
        delete [] log;
    } else {
        std::cerr << text << std::endl;
    }
}

static void add_shader(GLenum shader_type, const std::string& source_code, GLuint program)
{
    const GLchar* src_buffers[1] = { source_code.c_str() };
    GLuint shader = glCreateShader(shader_type);
    check_success(shader);
    glShaderSource(shader, 1, src_buffers, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        dump_info(shader,"Error compiling the fragment shader: ");
        terminate();
    }
    glAttachShader(program, shader);
    check_success(glGetError() == GL_NO_ERROR);
}


// Create a shader program with a fragment shader.
static GLuint create_shader_program()
{
    GLint success;
    GLuint program = glCreateProgram();

    const char *vert =
        "#version 330\n"
        "in vec3 Position;\n"
        "out vec2 TexCoord;\n"
        "void main() {\n"
        "    gl_Position = vec4(Position, 1.0);\n"
        "    TexCoord = 0.5 * Position.xy + vec2(0.5);\n"
        "}\n";
    add_shader(GL_VERTEX_SHADER, vert, program);

    const char *frag =
        "#version 330\n"
        "in vec2 TexCoord;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D TexSampler;\n"
        "void main() {\n"
        "    FragColor = texture(TexSampler, TexCoord);\n"
        "}\n";
    add_shader(GL_FRAGMENT_SHADER, frag, program);

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        dump_info(program, "Error linking the shader program: ");
        terminate();
    }

#if !defined(__APPLE__)
    glValidateProgram(program);
    glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
    if (!success) {
        dump_info(program, "Error validating the shader program: ");
        terminate();
    }
#endif

    glUseProgram(program);
    check_success(glGetError() == GL_NO_ERROR);

    return program;
}

// Create a quad filling the whole screen.
static GLuint create_quad(GLuint program, GLuint* vertex_buffer)
{
    static const float3 vertices[6] = {
        { -1.f, -1.f, 0.0f },
        {  1.f, -1.f, 0.0f },
        { -1.f,  1.f, 0.0f },
        {  1.f, -1.f, 0.0f },
        {  1.f,  1.f, 0.0f },
        { -1.f,  1.f, 0.0f }
    };

    glGenBuffers(1, vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, *vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    const GLint pos_index = glGetAttribLocation(program, "Position");
    glEnableVertexAttribArray(pos_index);
    glVertexAttribPointer(
        pos_index, 3, GL_FLOAT, GL_FALSE, sizeof(float3), 0);

    check_success(glGetError() == GL_NO_ERROR);

    return vertex_array;
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

// Resize OpenGL and CUDA buffers for a given resolution
static void resize_buffers(
    CUdeviceptr *accum_buffer_cuda,
    CUgraphicsResource *display_buffer_cuda, int width, int height, GLuint display_buffer)
{
    // Allocate GL display buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, display_buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    check_success(glGetError() == GL_NO_ERROR);

    // Register GL display buffer to CUDA
    if (*display_buffer_cuda)
        check_cuda_success(cuGraphicsUnregisterResource(*display_buffer_cuda));

    if (width == 0 || height == 0)
        *display_buffer_cuda = 0;
    else
        check_cuda_success(
            cuGraphicsGLRegisterBuffer(
                display_buffer_cuda, display_buffer, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

    // Allocate CUDA accumulation buffer
    if (*accum_buffer_cuda)
        check_cuda_success(cuMemFree(*accum_buffer_cuda));

    if (width == 0 || height == 0)
        *accum_buffer_cuda = 0;
    else
        check_cuda_success(cuMemAlloc(accum_buffer_cuda, width * height * sizeof(float3)));
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
    const char *envmap_name,
    mi::mdl::IMDL *mdl_compiler)
{
    // Load environment texture
    Texture_data env(envmap_name, mdl_compiler->create_entity_resolver(nullptr));
    check_success(env.is_valid());

    const mi::Uint32 rx = env.get_width();
    const mi::Uint32 ry = env.get_height();
    res->x = rx;
    res->y = ry;

    FIBITMAP *dib = env.get_dib();

    float *pixels;
    float4 *own_buf = nullptr;

    // Check, whether we need to convert the image
    if (FreeImage_GetImageType(dib) == FIT_RGBF) {
        // This example expects, that there is no additional padding per image line
        check_success(FreeImage_GetPitch(dib) == unsigned(env.get_width() * 3 * sizeof(float)));

        // Implement conversion of RGBF to RGBAF on our own, because FreeImage_ConvertToRGBAF
        // clamps the values to 1.0.
        const float3 *dib_pixels = reinterpret_cast<float3 *>(FreeImage_GetBits(dib));
        own_buf = (float4 *) malloc(rx * ry * sizeof(float4));
        for(size_t i = 0, n = rx * ry; i < n; ++i) {
            own_buf[i].x = dib_pixels[i].x;
            own_buf[i].y = dib_pixels[i].y;
            own_buf[i].z = dib_pixels[i].z;
            own_buf[i].w = 1;
        }

        pixels = reinterpret_cast<float *>(own_buf);
    } else if (FreeImage_GetImageType(dib) == FIT_RGBAF) {
        // This example expects, that there is no additional padding per image line
        check_success(FreeImage_GetPitch(dib) == unsigned(env.get_width() * 4 * sizeof(float)));

        // We can use the image data directly
        pixels = reinterpret_cast<float *>(FreeImage_GetBits(dib));
    } else {
        check_success(!"Only RGBF and RGBAF environments are supported.");
    }

    // Copy the image data to a CUDA array
    const cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    check_cuda_success(cudaMallocArray(env_tex_data, &channel_desc, rx, ry));
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
    free(own_buf);
}

// Save current result image to disk
static void save_result(
    const CUdeviceptr accum_buffer,
    const unsigned int width,
    const unsigned int height,
    const std::string &filename)
{
    float3 *data = static_cast<float3 *>(malloc(width * height * sizeof(float3)));
    if (data == nullptr) return;

    check_cuda_success(cuMemcpyDtoH(data, accum_buffer, width * height * sizeof(float3)));

    export_image_rgbf(filename.c_str(), width, height, data);

    free(data);
}


// Application options
struct Options {
    float gui_scale;
    bool opengl;
    bool use_class_compilation;
    bool no_aa;
    bool enable_derivatives;
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
    std::string outputfile;
    std::vector<std::string> material_names;
    std::vector<std::string> mdl_paths;

    // Default constructor, sets default values.
    Options()
    : gui_scale(1.0f)
    , opengl(true)
    , use_class_compilation(true)
    , no_aa(false)
    , enable_derivatives(false)
    , res_x(1024)
    , res_y(1024)
    , iterations(4096)
    , samples_per_iteration(8)
    , mdl_test_type(MDL_TEST_MIS)
    , max_path_length(4)
    , fov(96.0f)
    , exposure(0.0f)
    , cam_pos(make_float3(0, 0, 3))
    , light_pos(make_float3(0, 0, 0))
    , light_intensity(make_float3(0, 0, 0))
    , hdrfile("nvidia/sdk_examples/resources/environment.hdr")
    , outputfile("output.exr")
    , material_names()
    , mdl_paths()
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
        PK_BOOL,
        PK_ENUM,
        PK_STRING
    };

    Param_info(
        size_t index,
        char const *name,
        char const *display_name,
        char const *group_name,
        Param_kind kind,
        char *data_ptr,
        const Enum_type_info *enum_info = nullptr)
    : m_index(index)
    , m_name(name)
    , m_display_name(display_name)
    , m_group_name(group_name)
    , m_kind(kind)
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

    float &range_min()      { return m_range_min; }
    float range_min() const { return m_range_min; }
    float &range_max()      { return m_range_max; }
    float range_max() const { return m_range_max; }

    const Enum_type_info *enum_info() const { return m_enum_info; }

private:
    size_t                m_index;
    char const           *m_name;
    char const           *m_display_name;
    char const           *m_group_name;
    Param_kind           m_kind;
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
            for (std::list<Param_info>::iterator it = params().begin(); it != params().end(); ++it) {
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

// Type trait to get the value type for a given type.
template<typename T> struct Value_trait     { /* error */ };
template<> struct Value_trait<float>        { typedef mi::mdl::IValue_float IVALUE_TYPE; };
template<> struct Value_trait<char const *> { typedef mi::mdl::IValue_string IVALUE_TYPE; };

template<typename T>
bool get_annotation_argument_value(mi::mdl::DAG_call const *anno, int index, T &res)
{
    mi::mdl::DAG_constant const *dag_const =
        mi::mdl::as<mi::mdl::DAG_constant>(anno->get_argument(index));
    if (dag_const == nullptr)
        return false;

    typedef typename Value_trait<T>::IVALUE_TYPE IValue_type;

    IValue_type const *val = mi::mdl::as<IValue_type>(dag_const->get_value());
    if (val == nullptr)
        return false;

    res = val->get_value();
    return true;
}

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

// Progressively render scene
static void render_scene(
    const Options &options,
    std::unique_ptr<Ptx_code> target_code,
    mi::mdl::IMDL *mdl_compiler,
    const std::vector<Df_cuda_material>& material_bundle)
{
    Window_context window_context;
    memset(&window_context, 0, sizeof(Window_context));

    GLuint display_buffer = 0;
    GLuint display_tex = 0;
    GLuint program = 0;
    GLuint quad_vertex_buffer = 0;
    GLuint quad_vao = 0;
    GLFWwindow *window = nullptr;
    int width = -1;
    int height = -1;

    if (options.opengl) {
        // Init OpenGL window
        std::string version_string;
        window = init_opengl(options.res_x, options.res_y, version_string);
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
        ImGui::GetStyle().ScaleAllSizes(options.gui_scale);

        glGenBuffers(1, &display_buffer);
        glGenTextures(1, &display_tex);
        check_success(glGetError() == GL_NO_ERROR);

        // Create shader program
        program = create_shader_program();

        // Create scene data
        quad_vao = create_quad(program, &quad_vertex_buffer);
    }

    // Initialize CUDA
    CUcontext cuda_context = init_cuda(options.opengl);

    CUdeviceptr accum_buffer = 0;
    CUgraphicsResource display_buffer_cuda = nullptr;
    if (!options.opengl) {
        width = options.res_x;
        height = options.res_y;
        check_cuda_success(cuMemAlloc(&accum_buffer, width * height * sizeof(float3)));
    }

    // Setup initial CUDA kernel parameters
    Kernel_params kernel_params;
    memset(&kernel_params, 0, sizeof(Kernel_params));
    kernel_params.cam_focal = 1.0f / tanf(options.fov / 2 * float(2 * M_PI / 360));
    kernel_params.light_pos = options.light_pos;
    kernel_params.light_intensity = options.light_intensity;
    kernel_params.iteration_start = 0;
    kernel_params.iteration_num = options.samples_per_iteration;
    kernel_params.mdl_test_type = options.mdl_test_type;
    kernel_params.max_path_length = options.max_path_length;
    kernel_params.exposure_scale = powf(2.0f, options.exposure);
    kernel_params.disable_aa = options.no_aa;
    kernel_params.use_derivatives = options.enable_derivatives;

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
    std::vector<std::unique_ptr<Ptx_code> > target_codes;
    target_codes.push_back(std::move(target_code));
    CUfunction  cuda_function;
    char const *ptx_name = options.enable_derivatives ?
        "example_df_cuda_derivatives.ptx" : "example_df_cuda.ptx";
    CUmodule    cuda_module = build_linked_kernel(
        target_codes,
        (get_executable_folder() + "/" + ptx_name).c_str(),
        "render_sphere_kernel",
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
        &kernel_params.env_tex, &env_tex_data, &env_accel, &kernel_params.env_size,
        options.hdrfile.c_str(), mdl_compiler);
    kernel_params.env_accel = reinterpret_cast<Env_accel *>(env_accel);

    // Setup file name for nogl mode
    std::string next_filename;
    std::string filename_base, filename_ext;
    if (options.material_names.size() > 1) {
        size_t dot_pos = options.outputfile.rfind(".");
        if (dot_pos == std::string::npos)
            filename_base = options.outputfile;
        else {
            filename_base = options.outputfile.substr(0, dot_pos);
            filename_ext = options.outputfile.substr(dot_pos);
        }
        next_filename = filename_base + "-0" + filename_ext;
    } else
        next_filename = options.outputfile;

    // Scope for material context resources
    {
        // Prepare the needed data of all target codes for the GPU
        Material_gpu_context material_gpu_context(options.enable_derivatives);
        if (!material_gpu_context.prepare_target_code_data(target_codes[0].get())) {
            fprintf(stderr, "Error: preparing data for GPU failed\n");
            terminate();
        }
        kernel_params.tc_data = reinterpret_cast<Target_code_data *>(
            material_gpu_context.get_device_target_code_data_list());
        kernel_params.arg_block_list = reinterpret_cast<char const **>(
            material_gpu_context.get_device_target_argument_block_list());

        String_constant_table &constant_table(target_codes[0]->get_string_constant_table());

        // Collect information about the arguments of the compiled materials
        std::vector<Material_info> mat_infos;

        for (size_t i = 0, num_mats = target_codes[0]->get_material_instance_count();
                i < num_mats; ++i) {
            Material_instance const &cur_inst = target_codes[0]->get_material_instance(i);

            // Get the target argument block and its layout
            size_t arg_block_index = material_gpu_context.get_bsdf_argument_block_index(i);
            mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> layout(
                material_gpu_context.get_argument_block_layout(arg_block_index));
            Argument_block *arg_block = material_gpu_context.get_argument_block(arg_block_index);
            char *arg_block_data = arg_block != nullptr ? arg_block->get_data() : nullptr;

            Material_info mat_info(cur_inst.get_dag_material_name());
            for (size_t j = 0, num_params = cur_inst->get_parameter_count(); j < num_params; ++j) {
                const char *name = cur_inst->get_parameter_name(j);
                if (name == nullptr) continue;

                // Determine the type of the argument
                mi::mdl::IValue const *arg = cur_inst->get_parameter_default(j);
                mi::mdl::IValue::Kind kind = arg->get_kind();

                Param_info::Param_kind param_kind = Param_info::PK_UNKNOWN;
                const Enum_type_info *enum_type = nullptr;

                switch (kind) {
                case mi::mdl::IValue::VK_FLOAT:
                    param_kind = Param_info::PK_FLOAT;
                    break;
                case mi::mdl::IValue::VK_RGB_COLOR:
                    param_kind = Param_info::PK_COLOR;
                    break;
                case mi::mdl::IValue::VK_BOOL:
                    param_kind = Param_info::PK_BOOL;
                    break;
                case mi::mdl::IValue::VK_VECTOR:
                    {
                        mi::mdl::IValue_vector const *val = mi::mdl::as<mi::mdl::IValue_vector>(arg);
                        mi::mdl::IType_vector const *val_type = val->get_type();
                        mi::mdl::IType_atomic const *elem_type = val_type->get_element_type();
                        if (elem_type->get_kind() == mi::mdl::IType::TK_FLOAT) {
                            switch (val_type->get_size()) {
                            case 2: param_kind = Param_info::PK_FLOAT2; break;
                            case 3: param_kind = Param_info::PK_FLOAT3; break;
                            }
                        }
                    }
                    break;
                case mi::mdl::IValue::VK_ENUM:
                    {
                        const mi::mdl::IValue_enum *val = mi::mdl::as<mi::mdl::IValue_enum>(arg);
                        const mi::mdl::IType_enum  *val_type = val->get_type();

                        // prepare info for this enum type if not seen so far
                        const char *e_name = val_type->get_symbol()->get_name();
                        const Enum_type_info *info = mat_info.get_enum_type(e_name);
                        if (info == nullptr) {
                            std::shared_ptr<Enum_type_info> p(new Enum_type_info());

                            for (int i = 0, n = val_type->get_value_count(); i < n; ++i) {
                                const mi::mdl::ISymbol *e_sym = nullptr;
                                int                    e_code = 0;
                                val_type->get_value(i, e_sym, e_code);

                                p->add(e_sym->get_name(), e_code);
                            }
                            mat_info.add_enum_type(e_name, p);
                            info = p.get();
                        }
                        enum_type = info;

                        param_kind = Param_info::PK_ENUM;
                    }
                    break;
                case mi::mdl::IValue::VK_STRING:
                    param_kind = Param_info::PK_STRING;
                    break;
                default:
                    // Unsupported? -> skip
                    continue;
                }

                // Get the offset of the argument within the target argument block
                mi::mdl::IGenerated_code_value_layout::State state(layout->get_nested_state(j));
                mi::mdl::IValue::Kind kind2;
                size_t param_size;
                size_t offset = layout->get_layout(kind2, param_size, state);
                check_success(kind == kind2);

                Param_info param_info(
                    j, name, name, /*group_name=*/nullptr, param_kind, arg_block_data + offset,
                    enum_type);

                // Check for annotation info
                size_t dag_param_index = cur_inst.get_dag_parameter_index(name);
                if (dag_param_index != ~0) {
                    bool has_soft_range = false;
                    size_t anno_count = cur_inst.get_dag_parameter_annotation_count(dag_param_index);
                    for (size_t anno_ind = 0; anno_ind < anno_count; ++anno_ind) {
                        if (mi::mdl::DAG_call const *anno = mi::mdl::as<mi::mdl::DAG_call>(
                                cur_inst.get_dag_parameter_annotation(dag_param_index, anno_ind))) {
                            switch (anno->get_semantic()) {
                            case mi::mdl::IDefinition::DS_SOFT_RANGE_ANNOTATION:
                                has_soft_range = true;
                                get_annotation_argument_value(anno, 0, param_info.range_min());
                                get_annotation_argument_value(anno, 1, param_info.range_max());
                                break;
                            case mi::mdl::IDefinition::DS_HARD_RANGE_ANNOTATION:
                                if (!has_soft_range) {
                                    get_annotation_argument_value(anno, 0, param_info.range_min());
                                    get_annotation_argument_value(anno, 1, param_info.range_max());
                                }
                                break;
                            case mi::mdl::IDefinition::DS_DISPLAY_NAME_ANNOTATION:
                                get_annotation_argument_value(anno, 0, param_info.display_name());
                                break;
                            case mi::mdl::IDefinition::DS_IN_GROUP_ANNOTATION:
                                get_annotation_argument_value(anno, 0, param_info.group_name());
                                break;
                            default:
                                break;
                            }
                        }
                    }
                }

                mat_info.add_sorted_by_group(param_info);
            }
            mat_infos.push_back(mat_info);
        }

        // Main render loop
        while (true)
        {
            double start_time = 0.0;

            if (!options.opengl)
            {
                kernel_params.resolution.x = width;
                kernel_params.resolution.y = height;
                kernel_params.accum_buffer = reinterpret_cast<float3 *>(accum_buffer);

                // Check if desired number of samples is reached
                if (kernel_params.iteration_start >= options.iterations) {
                    std::cout << "rendering done" << std::endl;

                    save_result(accum_buffer, width, height, options.outputfile);

                    std::cout << std::endl;

                    // All materials have been rendered? -> done
                    if (kernel_params.current_material + 1 >= material_bundle.size())
                        break;

                    // Start new image with next material
                    kernel_params.iteration_start = 0;
                    ++kernel_params.current_material;
                    next_filename = filename_base + "-" + to_string(kernel_params.current_material)
                        + filename_ext;
                }

                std::cout
                    << "rendering iterations " << kernel_params.iteration_start << " to "
                    << kernel_params.iteration_start + kernel_params.iteration_num << std::endl;
            }
            else
            {
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

                    resize_buffers(
                        &accum_buffer, &display_buffer_cuda, width, height, display_buffer);
                    kernel_params.accum_buffer = reinterpret_cast<float3 *>(accum_buffer);

                    glViewport(0, 0, width, height);

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
                ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(
                    ImVec2(360 * options.gui_scale, 350 * options.gui_scale),
                    ImGuiCond_FirstUseEver);
                ImGui::Begin("Material parameters");
                ImGui::SetWindowFontScale(options.gui_scale);
                ImGui::PushItemWidth(-200 * options.gui_scale);
                if (options.use_class_compilation)
                    ImGui::Text("CTRL + Click to manually enter numbers");
                else
                    ImGui::Text("Parameter editing requires class compilation.");

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
                        if (param.group_name())
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

                start_time = glfwGetTime();

                // Handle events
                Window_context *ctx =
                    static_cast<Window_context*>(glfwGetWindowUserPointer(window));
                if (ctx->save_result && !ImGui::GetIO().WantCaptureKeyboard) {
                    save_result(accum_buffer, width, height, options.outputfile);
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

                // Map GL buffer for access with CUDA
                check_cuda_success(cuGraphicsMapResources(1, &display_buffer_cuda, /*stream=*/0));
                CUdeviceptr p;
                size_t size_p;
                check_cuda_success(
                    cuGraphicsResourceGetMappedPointer(&p, &size_p, display_buffer_cuda));
                kernel_params.display_buffer = reinterpret_cast<unsigned int *>(p);
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
                check_cuda_success(cuGraphicsUnmapResources(1, &display_buffer_cuda, /*stream=*/0));

                // Update texture
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, display_buffer);
                glBindTexture(GL_TEXTURE_2D, display_tex);
                glTexImage2D(
                    GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                check_success(glGetError() == GL_NO_ERROR);

                // Render the quad
                glClear(GL_COLOR_BUFFER_BIT);
                glBindVertexArray(quad_vao);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                check_success(glGetError() == GL_NO_ERROR);

                // Show the GUI
                ImGui::Render();
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                // Swap front and back buffers
                glfwSwapBuffers(window);

                // Update window title
                const double fps =
                    double(kernel_params.iteration_num) / (glfwGetTime() - start_time);
                glfwSetWindowTitle(
                    window, (std::string(WINDOW_TITLE) +
                             " (iterations/s: " + to_string(fps) + ")").c_str());
            }
        }
    }

    // Cleanup CUDA
    check_cuda_success(cudaDestroyTextureObject(kernel_params.env_tex));
    check_cuda_success(cudaFreeArray(env_tex_data));
    check_cuda_success(cuMemFree(env_accel));
    check_cuda_success(cuMemFree(accum_buffer));
    check_cuda_success(cuMemFree(material_buffer));
    check_cuda_success(cuModuleUnload(cuda_module));
    uninit_cuda(cuda_context);

    // Cleanup OpenGL
    if (options.opengl) {
        glDeleteVertexArrays(1, &quad_vao);
        glDeleteBuffers(1, &quad_vertex_buffer);
        glDeleteProgram(program);
        check_success(glGetError() == GL_NO_ERROR);
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
    std::vector<Target_function_description> const& descs)
{
    Df_cuda_material mat;

    // shared by all generated functions of the same material
    // used here to alter the materials parameter set
    mat.compiled_material_index = static_cast<unsigned int>(compiled_material_index);

    // Note: the same argument_block_index is filled into all function descriptions of a
    //       material, if any function uses it
    mat.argument_block_index = static_cast<unsigned int>(descs[0].argument_block_index);

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

    return mat;
}

static void usage(const char *name)
{
    std::cout
        << "usage: " << name << " [options] [<material_name1> ...]\n"
        << "-h                          print this text\n"
        << "--nogl                      don't open interactive display\n"
        << "--nocc                      don't use class-compilation\n"
        << "--gui_scale <factor>        GUI scaling factor (default: 1.0)\n"
        << "--res <res_x> <res_y>       resolution (default: 1024x1024)\n"
        << "--hdr <filename>            HDR environment map "
           "(default: nvidia/sdk_examples/resources/environment.hdr)\n"
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
        << "\n"
        << "Note: material names can end with an '*' as a wildcard\n";

    exit(EXIT_FAILURE);
}

int MAIN_UTF8(int argc, char* argv[])
{
    // Parse commandline options
    Options options;
    options.mdl_paths.push_back(get_samples_mdl_root());

    bool use_default_window_size = true;

    for (int i = 1; i < argc; ++i) {
        const char *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "--nogl") == 0) {
                options.opengl = false;
            } else if (strcmp(opt, "--nocc") == 0) {
                options.use_class_compilation = false;
            } else if (strcmp(opt, "--gui_scale") == 0 && i < argc - 1) {
                options.gui_scale = static_cast<float>(atof(argv[++i]));
            } else if (strcmp(opt, "--res") == 0 && i < argc - 2) {
                options.res_x = std::max(atoi(argv[++i]), 1);
                options.res_y = std::max(atoi(argv[++i]), 1);
                use_default_window_size = false;
            } else if (strcmp(opt, "--hdr") == 0 && i < argc - 1) {
                options.hdrfile = argv[++i];
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
                options.mdl_paths.push_back(argv[++i]);
            } else if (strcmp(opt, "--max_path_length") == 0 && i < argc - 1) {
                options.max_path_length = std::min(std::max(atoi(argv[++i]), 2), 100);
            } else if (strcmp(opt, "--noaa") == 0) {
                options.no_aa = true;
            } else if (strcmp(opt, "-d") == 0) {
                options.enable_derivatives = true;
            } else {
                std::cout << "Unknown option: \"" << opt << "\"" << std::endl;
                usage(argv[0]);
            }
        }
        else
            options.material_names.push_back(std::string(opt));
    }

    if (options.opengl && use_default_window_size) {
        options.res_x = 1024;
        options.res_y = 768;
    }

    // Access the MDL Core compiler
    mi::base::Handle<mi::mdl::IMDL> mdl_compiler(load_mdl_compiler());
    check_success(mdl_compiler);

    FreeImage_Initialise();

    // Use default material, if non was provided via command line
    if (options.material_names.empty())
        options.material_names.push_back("::nvidia::sdk_examples::tutorials::example_df");

    {
        // Initialize the material compiler with 16 result buffer slots ("texture results")
        Material_ptx_compiler mc(
            mdl_compiler.get(),
            16,
            options.enable_derivatives,
            /*df_handle_mode*/ "none");
        for (std::size_t i = 0; i < options.mdl_paths.size(); ++i)
            mc.add_module_path(options.mdl_paths[i].c_str());

        // List of materials in the scene
        std::vector<Df_cuda_material> material_bundle;

        // Select the functions to translate
        std::vector<Target_function_description> descs;
        descs.push_back(
            Target_function_description("surface.scattering"));
        descs.push_back(
            Target_function_description("surface.emission.emission"));
        descs.push_back(
            Target_function_description("surface.emission.intensity"));
        descs.push_back(
            Target_function_description("volume.absorption_coefficient"));

        // Generate code for all materials
        bool success = true;
        std::vector<std::string> used_material_names;
        for (size_t i = 0; i < options.material_names.size(); ++i) {
            std::string material_name(options.material_names[i]);
            if (!starts_with(material_name, "::")) material_name = "::" + material_name;

            // Is this a material name pattern?
            if (material_name.size() > 1 && material_name.back() == '*') {
                std::string pattern = material_name.substr(0, material_name.size() - 1);

                std::vector<std::string> module_materials(mc.get_material_names(
                    mc.get_module_name(material_name)));

                for (size_t j = 0, n = module_materials.size(); j < n; ++j) {
                    material_name = module_materials[j];

                    // make sure the material name starts with the pattern
                    if (!starts_with(material_name, pattern))
                        continue;

                    std::cout << "Adding material \"" << material_name << "\"..." << std::endl;

                    // Add functions of the material to the link unit
                    if (!mc.add_material(
                            material_name,
                            descs.data(), descs.size(),
                            options.use_class_compilation)) {
                        std::cout << "Failed!" << std::endl;
                        success = false;
                    }

                    // Create application material representation
                    material_bundle.push_back(create_cuda_material(
                        0, material_bundle.size(), descs));
                    used_material_names.push_back(material_name);
                }
            } else {
                std::cout << "Adding material \"" << material_name << "\"..." << std::endl;

                // Add functions of the material to the link unit
                if (!mc.add_material(
                        material_name,
                        descs.data(), descs.size(),
                        options.use_class_compilation)) {
                    std::cout << "Failed!" << std::endl;
                    success = false;
                }

                // Create application material representation
                material_bundle.push_back(create_cuda_material(
                    0, material_bundle.size(), descs));
                used_material_names.push_back(material_name);
            }
        }

        if (!success) {
            // Print any compiler messages, if available
            mc.print_messages();
        } else {
            // Generate the CUDA PTX code for the link unit.
            std::unique_ptr<Ptx_code> target_code(mc.generate_cuda_ptx());

            if (target_code.get()) {
                // Render
                render_scene(options, std::move(target_code), mdl_compiler.get(), material_bundle);
            }
        }
    }

    FreeImage_DeInitialise();

    // Free MDL compiler before shutting down MDL Core
    mdl_compiler = 0;

    // Unload MDL Core
    check_success(unload());

    keep_console_open();
    return EXIT_SUCCESS;
}

// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
