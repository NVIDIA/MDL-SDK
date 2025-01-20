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

 // examples/mdl_sdk/df_native/example_df_native.cpp
 //
 // Simple CPU renderer using compiled BSDFs with a material parameter editor GUI.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <mi/math/matrix.h>
#include <mi/math/vector.h>

#if MI_PLATFORM_MACOSX
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "example_shared.h"
#include "example_shared_backends.h"
#include "texture_support_native.h"


///////////////////////////////////////////////////////////////////////////////
// Global Constants
///////////////////////////////////////////////////////////////////////////////

struct
{
    const float DIRAC = -1.f;
    const float PI = static_cast<float>(M_PI);
    const mi::mdl::tct_float3 zeros_float3 = { 0.f, 0.f, 0.f };
    const mi::mdl::tct_float3 ones_float3 = { 1.f, 1.f, 1.f };
    const mi::mdl::tct_float3 tangent_u[1] = { {1.0f, 0.0f, 0.0f} };
    const mi::mdl::tct_float3 tangent_v[1] = { { 0.0f, 1.0f, 0.0f} };
    const mi::mdl::tct_float4 identity[4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    };
} Constants;

///////////////////////////////////////////////////////////////////////////////
// Random Number Generator
///////////////////////////////////////////////////////////////////////////////

unsigned tea(unsigned N, unsigned val0, unsigned val1)
{
    unsigned v0 = val0;
    unsigned v1 = val1;
    unsigned s0 = 0;

    for (unsigned n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Generate random uint in [0, 2^24)
unsigned lcg(unsigned& prev)
{
    const unsigned LCG_A = 1664525u;
    const unsigned LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
float rnd(unsigned& prev)
{
    const unsigned next = lcg(prev);
    return ((float)next / (float)0x01000000);
}

///////////////////////////////////////////////////////////////////////////////
// OpenGL code
///////////////////////////////////////////////////////////////////////////////
#define terminate()          \
    do {                     \
        glfwTerminate();     \
        keep_console_open(); \
        exit(EXIT_FAILURE);  \
    } while (0)

static int gl_bind_index = 0;

// Initialize OpenGL and create a window with an associated OpenGL context
GLFWwindow* init_opengl(unsigned res_x, unsigned res_y, std::string& version_string)
{
    // Initialize GLFW
    check_success(glfwInit());
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    version_string = "#version 330 core"; // see top comments in 'imgui_impl_opengl3.cpp'

    // Create an OpenGL window and a context
    GLFWwindow* window = glfwCreateWindow(
        int(res_x), int(res_y), "MDL Core df_native Example", nullptr, nullptr);
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

void dump_info(GLuint shader, const char* text)
{
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    if (length > 0)
    {
        std::unique_ptr<GLchar[]> log = std::make_unique<GLchar[]>(length + 1);
        glGetShaderInfoLog(shader, length + 1, nullptr, log.get());
        std::cerr << text << log.get() << std::endl;
    }
    else
    {
        std::cerr << text << std::endl;
    }
}

void add_shader(GLenum shader_type, const std::string& source_code, GLuint program)
{
    const GLchar* src_buffers[1] = { source_code.c_str() };
    GLuint shader = glCreateShader(shader_type);
    check_success(shader);
    glShaderSource(shader, 1, src_buffers, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        dump_info(shader, "Error compiling the fragment shader: ");
        terminate();
    }
    glAttachShader(program, shader);
    check_success(glGetError() == GL_NO_ERROR);
}

// Create a shader program with a fragment shader
GLuint create_shader_program()
{
    GLint success;
    GLuint program = glCreateProgram();

    const char* vert =
        "#version 330\n"
        "in vec3 Position;\n"
        "out vec2 TexCoord;\n"
        "void main() {\n"
        "    gl_Position = vec4(Position, 1.0);\n"
        "    TexCoord = 0.5 * Position.xy + vec2(0.5);\n"
        "}\n";
    add_shader(GL_VERTEX_SHADER, vert, program);

    const char* frag =
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

#if !defined(MI_PLATFORM_MACOSX)
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

// Create a quad filling the whole screen
GLuint create_quad(GLuint program, GLuint* vertex_buffer)
{
    const mi::mdl::tct_float3 vertices[6] = {
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
        pos_index, 3, GL_FLOAT, GL_FALSE, sizeof(mi::mdl::tct_float3), 0);

    check_success(glGetError() == GL_NO_ERROR);

    return vertex_array;
}

///////////////////////////////////////////////////////////////////////////////
// Window Handling
///////////////////////////////////////////////////////////////////////////////

// Window context structure for window keys/mouse event callback functions
struct Window_context
{
    bool mouse_event = false;
    bool key_event = false;

    // For environment
    float env_intensity = 0.f;

    // For omni light movement
    float omni_theta = 0.f;
    float omni_phi = 0.f;
    float omni_intensity = 0.f;

    // For camera movement
    int mouse_button = 0;            // button from callback event plus one (0 = no event)
    int mouse_button_action = 0;     // action from mouse button callback event
    int mouse_wheel_delta = 0;
    bool moving = false;
    double move_start_x= 0., move_start_y = 0.;
    double move_dx = 0., move_dy = 0.;
    int zoom = 0;

    // Image output
    bool save_sreenshot = false;

    // GLFW keyboard callback
    static void handle_key(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        // Handle key press events
        if (action == GLFW_PRESS)
        {
            Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));

            if (mods & GLFW_MOD_CONTROL)
            {
                switch (key)
                {
                case GLFW_KEY_MINUS:
                case GLFW_KEY_KP_SUBTRACT:
                    ctx->env_intensity -= 0.05f;
                    if (ctx->env_intensity < 0.f) ctx->env_intensity = 0.f;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_KP_ADD:
                case GLFW_KEY_EQUAL:
                    ctx->env_intensity += 0.05f;
                    ctx->key_event = true;
                    break;
                }
            }
            else
            {
                switch (key)
                {
                    // Escape closes the window
                case GLFW_KEY_ESCAPE:
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    break;
                case GLFW_KEY_DOWN:
                    ctx->omni_theta += 0.05f * Constants.PI;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_UP:
                    ctx->omni_theta -= 0.05f * Constants.PI;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_LEFT:
                    ctx->omni_phi -= 0.05f * Constants.PI;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_RIGHT:
                    ctx->omni_phi += 0.05f * Constants.PI;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_MINUS:
                case GLFW_KEY_KP_SUBTRACT:
                    ctx->omni_intensity -= 1000.f;
                    if (ctx->omni_intensity < 0.f) ctx->omni_intensity = 0.f;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_KP_ADD:
                case GLFW_KEY_EQUAL:
                    ctx->omni_intensity += 1000.f;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_ENTER:
                    ctx->save_sreenshot = true;
                    break;
                default:
                    break;
                }
            }
        }

        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
    }

    // GLFW mouse button callback
    static void handle_mouse_button(GLFWwindow* window, int button, int action, int mods)
    {
        Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
        ctx->mouse_button = button + 1;
        ctx->mouse_button_action = action;

        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
    }

    // GLFW mouse position callback
    static void handle_mouse_pos(GLFWwindow* window, double xpos, double ypos)
    {
        Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
        if (ctx->moving)
        {
            ctx->move_dx += xpos - ctx->move_start_x;
            ctx->move_dy += ypos - ctx->move_start_y;
            ctx->move_start_x = xpos;
            ctx->move_start_y = ypos;
            ctx->mouse_event = true;
        }
    }

    // GLFW scroll callback
    static void handle_scroll(GLFWwindow* window, double xoffset, double yoffset)
    {
        Window_context* ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
        if (yoffset > 0.0)
        {
            ctx->mouse_wheel_delta = 1;
            ctx->mouse_event = true;
        }
        else if (yoffset < 0.0)
        {
            ctx->mouse_wheel_delta = -1;
            ctx->mouse_event = true;
        }

        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Vector Helper Functions
///////////////////////////////////////////////////////////////////////////////
float int_as_float(uint32_t v)
{
    union
    {
        uint32_t bit;
        float    value;
    } temp;

    temp.bit = v;
    return temp.value;
}

uint32_t float_as_int(float v)
{
    union
    {
        uint32_t bit;
        float    value;
    } temp;

    temp.value = v;
    return temp.bit;
}

void clamp(mi::mdl::tct_float3& d, float min = 0.f, float max = 1.f)
{
    float* pd = &d.x;
    for (int i = 0; i < 3; ++i)
    {
        if (pd[i] < min)
            pd[i] = min;
        else if (pd[i] > max)
            pd[i] = max;
    }
}

float length(const mi::mdl::tct_float3& d)
{
    return sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
}

float dot(const mi::mdl::tct_float3& a, const mi::mdl::tct_float3& b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

mi::mdl::tct_float3 normalize(const mi::mdl::tct_float3& d)
{
    const float dotprod = dot(d, d);

    if (dotprod > 0.f)
    {
        const float inv_len = 1.0f / sqrtf(dotprod);
        return mi::mdl::tct_float3({ d.x * inv_len, d.y * inv_len, d.z * inv_len });
    }
    else
    {
        return d;
    }
}

mi::mdl::tct_float3 operator-(const mi::mdl::tct_float3& a)
{
    return mi::mdl::tct_float3({ -a.x, -a.y, -a.z });
}

mi::mdl::tct_float3 operator+(const mi::mdl::tct_float3& a, const mi::mdl::tct_float3& b)
{
    return mi::mdl::tct_float3({ a.x + b.x, a.y + b.y, a.z + b.z });
}

mi::mdl::tct_float3 operator-(const mi::mdl::tct_float3& a, const mi::mdl::tct_float3& b)
{
    return mi::mdl::tct_float3({ a.x - b.x, a.y - b.y, a.z - b.z });
}

mi::mdl::tct_float3 operator*(const mi::mdl::tct_float3& a, const mi::mdl::tct_float3& b)
{
    return mi::mdl::tct_float3({ a.x * b.x, a.y * b.y, a.z * b.z });
}

mi::mdl::tct_float3 operator*(const mi::mdl::tct_float3& d, float s)
{
    return mi::mdl::tct_float3({ d.x * s, d.y * s, d.z * s });
}

mi::mdl::tct_float3 operator/(const mi::mdl::tct_float3& d, float s)
{
    const float inv_s = 1.0f / s;
    return mi::mdl::tct_float3({ d.x * inv_s, d.y * inv_s, d.z * inv_s });
}

mi::mdl::tct_float3& operator+=(mi::mdl::tct_float3& a, float s)
{
    a.x += s;
    a.y += s;
    a.z += s;
    return a;
}

mi::mdl::tct_float3& operator+=(mi::mdl::tct_float3& a, const mi::mdl::tct_float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

mi::mdl::tct_float3& operator*=(mi::mdl::tct_float3& a, float s)
{
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}

mi::mdl::tct_float3& operator*=(mi::mdl::tct_float3& a, const mi::mdl::tct_float3& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

///////////////////////////////////////////////////////////////////////////////
// Command Line Options
///////////////////////////////////////////////////////////////////////////////

// Command line options structure
struct Options
{
    // Print command line usage to console and terminate the application.
    void usage(char const* prog_name)
    {
        std::cout
            << "Usage: " << prog_name << " [options] [<material_name>]\n"
            << "Options:\n"
            << "  -h|--help                  print this text and exit\n"
            << "  -v|--version               print the MDL SDK version string and exit\n"
            << "  --res <x> <y>              resolution (default: 700x520)\n"
            << "  --hdr <filename>           environment map\n"
            << "                             (default: nvidia/sdk_examples/resources/environment.hdr)\n"
            << "  --nocc                     don't use class compilation\n"
            << "  --allowed_scatter_mode <m> limits the allowed scatter mode to \"none\", \"reflect\", "
            << "\"transmit\" or \"reflect_and_transmit\" (default: restriction disabled)\n"
            << "  -d                         enable use of derivatives\n"
            << "  --nogui                    don't open interactive display\n"
            << "  --spp                      samples per pixel (default: 100) for output image when nogui\n"
            << "  -o <outputfile>            image file to write result to\n"
            << "                             (default: example_native.png)\n"
            << "  -p|--mdl_path <path>       mdl search path, can occur multiple times\n"
            << "  --single_threaded          render on one thread only"
            << "\n"
            << "Viewport controls:\n"
            << "  Mouse               Camera rotation, zoom\n"
            << "  Arrow keys, (+/-)   Omni-light rotation, intensity\n"
            << "  CTRL + (+/-)        Environment intensity\n"
            << "  ENTER               Screenshot\n"
            << std::endl;

        exit(EXIT_FAILURE);
    }

    // Parse command line options
    void parse(int argc, char* argv[])
    {
        for (int i = 1; i < argc; ++i)
        {
            char const* opt = argv[i];
            if (opt[0] == '-')
            {
                if (strcmp(opt, "--nogui") == 0)
                {
                    no_gui = true;
                }
                else if (strcmp(opt, "--spp") == 0 && i < argc - 1)
                {
                    iterations = std::max(atoi(argv[++i]), 1);
                }
                else if (strcmp(opt, "-o") == 0 && i < argc - 1)
                {
                    outputfile = argv[++i];
                }
                else if (strcmp(opt, "--res") == 0 && i < argc - 2)
                {
                    res_x = std::max(atoi(argv[++i]), 1);
                    res_y = std::max(atoi(argv[++i]), 1);
                }
                else if (strcmp(opt, "--max_path_length") == 0 && i < argc - 1)
                {
                    max_ray_length = std::max(atoi(argv[++i]), 0);
                }
                else if (strcmp(opt, "--hdr") == 0 && i < argc - 1)
                {
                    env_map = argv[++i];
                }
                else if (strcmp(opt, "--hdr_scale") == 0 && i < argc - 1)
                {
                    env_scale = static_cast<float>(atof(argv[++i]));
                }
                else if (strcmp(opt, "-f") == 0 && i < argc - 1)
                {
                    cam_fov = static_cast<float>(atof(argv[++i]));
                }
                else if (strcmp(opt, "--pos") == 0 && i < argc - 3)
                {
                    cam_pos.x = static_cast<float>(atof(argv[++i]));
                    cam_pos.y = static_cast<float>(atof(argv[++i]));
                    cam_pos.z = static_cast<float>(atof(argv[++i]));
                }
                else if (strcmp(opt, "-l") == 0 && i < argc - 6)
                {
                    light_pos.x = static_cast<float>(atof(argv[++i]));
                    light_pos.y = static_cast<float>(atof(argv[++i]));
                    light_pos.z = static_cast<float>(atof(argv[++i]));
                    light_intensity.x = static_cast<float>(atof(argv[++i]));
                    light_intensity.y = static_cast<float>(atof(argv[++i]));
                    light_intensity.z = static_cast<float>(atof(argv[++i]));
                }
                else if (strcmp(opt, "--nocc") == 0)
                {
                    use_class_compilation = false;
                }
                else if (strcmp(opt, "--allowed_scatter_mode") == 0 && i < argc - 1)
                {
                    enable_bsdf_flags = true;
                    char const* mode = argv[++i];
                    if (strcmp(mode, "none") == 0)
                    {
                        allowed_scatter_mode = mi::mdl::DF_FLAGS_NONE;
                    }
                    else if (strcmp(mode, "reflect") == 0)
                    {
                        allowed_scatter_mode = mi::mdl::DF_FLAGS_ALLOW_REFLECT;
                    }
                    else if (strcmp(mode, "transmit") == 0)
                    {
                        allowed_scatter_mode = mi::mdl::DF_FLAGS_ALLOW_TRANSMIT;
                    }
                    else if (strcmp(mode, "reflect_and_transmit") == 0)
                    {
                        allowed_scatter_mode =
                            mi::mdl::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
                    }
                    else
                    {
                        std::cout << "Unknown allowed_scatter_mode: \"" << mode << "\"" << std::endl;
                        usage(argv[0]);
                    }
                }
                else if (strcmp(opt, "-d") == 0)
                {
                    enable_derivatives = true;
                }
                else if ((strcmp(opt, "--mdl_path") == 0 || strcmp(opt, "-p") == 0) &&
                    i < argc - 1)
                {
                    mdl_paths.push_back(argv[++i]);
                }
                else if (strcmp(opt, "--single_threaded") == 0)
                {
                    single_threaded = true;
                }
                else
                {
                    if (strcmp(opt, "-h") != 0 && strcmp(opt, "--help") != 0)
                        std::cout << "Unknown option: \"" << opt << "\"" << std::endl;
                    usage(argv[0]);
                }
            }
            else
            {
                material_name = opt;
            }
        }
    }

    // Don't open OpenGL GUI
    bool no_gui = false;

    // Number of iterations for output images
    size_t iterations = 100;

    // A result output file name
    std::string outputfile = "example_df_native.png";

    // The resolution of the display / image
    unsigned res_x = 1024, res_y = 1024;

    // Path-tracer max ray-length
    int max_ray_length = 6;

    // Environment map filename and scale
    std::string env_map = "nvidia/sdk_examples/resources/environment.hdr";
    float env_scale = 1.f;

    // Camera position and FOV
    mi::mdl::tct_float3 cam_pos = { 0.f, 0.f, 3.f };
    float cam_fov = 86.f;

    // Light position and intensity
    mi::mdl::tct_float3 light_pos = { 10.f, 5.f, 0.f };
    mi::mdl::tct_float3 light_intensity = { 1.0f, 0.902f, 0.502f };

    // BSDF flags
    bool enable_bsdf_flags = false;
    mi::mdl::Df_flags allowed_scatter_mode = mi::mdl::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;

    // Whether class compilation should be used for the materials
    bool use_class_compilation = true;

    // Whether derivative support should be enabled
    bool enable_derivatives = false;

    // If true, render on one thread only
    bool single_threaded = false;

    // Additional search paths
    std::vector<std::string> mdl_paths;

    // Material to use
    std::string material_name;
};

///////////////////////////////////////////////////////////////////////////////
// Scene Render Context
///////////////////////////////////////////////////////////////////////////////

// Viewport buffers for progressive rendering
enum VP_channel
{
    VPCH_ILLUM = 0,
    // More channel types can be added here
    VPCH_NB_CHANNELS
};

struct VP_buffers
{
    std::shared_ptr<mi::mdl::tct_float3[]> accum_buffer;
    // More channel buffers can be added here
};

// Surface intersection info
struct Isect_info
{
    mi::mdl::tct_float3 pos;    // surface position
    mi::mdl::tct_float3 normal; // surface normal
    mi::mdl::tct_float3 uvw;    // uvw coordinates
    mi::mdl::tct_float3 tan_u;  // tangent vector in u direction
    mi::mdl::tct_float3 tan_v;  // tangent vector in v direction
};

// Render context
struct Render_context
{
    // Render options
    int max_ray_length = 6;
    bool use_derivatives;
    mi::mdl::Df_flags bsdf_data_flags = mi::mdl::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;

    // Scene data
    // Environment color
    struct Environment
    {
        mi::mdl::tct_float3 color = { 0.53f, 0.81f, 0.92f }; // used when no environment map is set
        float intensity = 1.f;

        struct Alias_map
        {
            unsigned int alias;
            float        q;
        };

        std::shared_ptr<Alias_map[]> alias_map;

        float               inv_integral;
        mi::Uint32_2        map_size;
        std::shared_ptr<std::vector<float>>  env_data;
        const float*        map_pixels = nullptr;

        // Load environment map from file
        void load(std::string filename, mi::mdl::IMDL* mdl_compiler)
        {
            Texture_data env_tex(filename.c_str(), mdl_compiler->create_entity_resolver(nullptr));
            check_success(env_tex.is_valid());

            const mi::Uint32 rx = env_tex.get_width();
            const mi::Uint32 ry = env_tex.get_height();

            env_data = std::make_shared<std::vector<float>>(4 * rx * ry, 0.0f);
            std::shared_ptr<OIIO::ImageInput> image(env_tex.get_image());
            mi::Sint32 bytes_per_row = 4 * rx * sizeof(float);
            image->read_image(
                /*subimage*/ 0,
                /*miplevel*/ 0,
                /*chbegin*/ 0,
                /*chend*/ 4,
                OIIO::TypeDesc::FLOAT,
                env_data->data() + (ry - 1) * 4 * rx,
                /*xstride*/ 4 * sizeof(float),
                /*ystride*/ -bytes_per_row,
                /*zstride*/ OIIO::AutoStride);

            if (image->spec().nchannels <= 3)
                for (size_t i = 0, n = env_data->size(); i < n; i += 4)
                    (*env_data)[i + 3] = 1.0f;

            map_size.x = rx;
            map_size.y = ry;
            map_pixels = reinterpret_cast<float*>(env_data->data());

            //Build environment importance sampling data
            build_alias_map();
        }

        // Build environment importance sampling data
        void build_alias_map()
        {
            const mi::Uint32 rx = map_size.x;
            const mi::Uint32 ry = map_size.y;
            alias_map.reset(new Alias_map[rx * ry]);

            std::unique_ptr<float[]> importance_data = std::make_unique<float[]>(rx * ry);
            float cos_theta0 = 1.0f;
            const float step_phi = 2.f * Constants.PI / static_cast<float>(rx);
            const float step_theta = Constants.PI / static_cast<float>(ry);
            for (unsigned int y = 0; y < ry; ++y)
            {
                const float theta1 = static_cast<float>(y + 1) * step_theta;
                const float cos_theta1 = std::cos(theta1);
                const float area = (cos_theta0 - cos_theta1) * step_phi;
                cos_theta0 = cos_theta1;

                for (unsigned int x = 0; x < rx; ++x)
                {
                    const unsigned int idx = y * rx + x;
                    const unsigned int idx4 = idx * 4;
                    importance_data[idx] =
                        area * std::max(map_pixels[idx4], std::max(map_pixels[idx4 + 1], map_pixels[idx4 + 2]));
                }
            }

            // Build alias map
            // Create qs (normalized)
            size_t size = rx * ry;
            float sum = 0.0f;
            for (unsigned int i = 0; i < size; ++i)
                sum += importance_data[i];

            for (unsigned int i = 0; i < size; ++i)
                alias_map[i].q = (static_cast<float>(size) * importance_data[i] / sum);

            // Create partition table
            std::unique_ptr<unsigned int[]> partition_table = std::make_unique<unsigned int[]>(size);
            unsigned int s = 0u, large = size;
            for (unsigned int i = 0; i < size; ++i)
                partition_table[(alias_map[i].q < 1.0f) ? (s++) : (--large)] = alias_map[i].alias = i;

            // Create alias map
            for (s = 0; s < large && large < size; ++s)
            {
                const unsigned int j = partition_table[s], k = partition_table[large];
                alias_map[j].alias = k;
                alias_map[k].q += alias_map[j].q - 1.0f;
                large = (alias_map[k].q < 1.0f) ? (large + 1u) : large;
            }

            inv_integral = 1.0f / sum;
        }
    }env;

    // Perspective camera
    struct Camera
    {
        float focal;
        float aspect;
        float zoom;
        mi::mdl::tct_float2 inv_res;
        mi::mdl::tct_float3 pos;
        mi::mdl::tct_float3 dir;
        mi::mdl::tct_float3 right;
        mi::mdl::tct_float3 up;

        // Update camera settings
        void update(
            float phi,
            float theta,
            float base_dist,
            int zoom)
        {
            dir.x = -sinf(phi) * sinf(theta);
            dir.y = -cosf(theta);
            dir.z = -cosf(phi) * sinf(theta);

            right.x = cosf(phi);
            right.y = 0.0f;
            right.z = -sinf(phi);

            up.x = -sinf(phi) * cosf(theta);
            up.y = sinf(theta);
            up.z = -cosf(phi) * cosf(theta);

            const float dist = base_dist * powf(0.95f, static_cast<float>(zoom));
            pos.x = -dir.x * dist;
            pos.y = -dir.y * dist;
            pos.z = -dir.z * dist;
        }
    } cam;

    // Omni light
    struct Omni
    {
        mi::mdl::tct_float3 color = { 1.0f, 0.902f, 0.502f };
        mi::mdl::tct_float3 dir = normalize(Constants.ones_float3);
        float distance = 11.18f;
        float intensity = 0.f;

        // Update light direction
        void update(
            float phi,
            float theta,
            float intensity)
        {
            dir.x = sinf(theta) * sinf(phi);
            dir.y = cosf(theta);
            dir.z = sinf(theta) * cosf(phi);

            intensity = intensity;
        }

    } omni_light;

    // Sphere object
    struct Sphere
    {
        mi::mdl::tct_float3 center = Constants.zeros_float3;
        float   radius = 1.f;
    } sphere;

    // A single raytracing ray
    struct Ray
    {
        mi::mdl::tct_float3 p0;
        mi::mdl::tct_float3 dir;
        mi::mdl::tct_float3 weight = { 1.f, 1.f, 1.f };
        int level = 0;
        float last_pdf = -1.f;
        bool is_inside = false;

        // Ray to sphere intersection
        bool isect(const Sphere& sphere, Isect_info& isect_info)
        {
            mi::mdl::tct_float3 oc = p0 - sphere.center;
            float b = 2.f * dot(oc, dir);
            float c = dot(oc, oc) - sphere.radius * sphere.radius;
            float disc = b * b - 4.f * c;

            // No intersection
            if (disc <= 0.f)
                return false;

            disc = sqrtf(disc);

            // First hit
            float t = (-b - disc) * 0.5f;
            if (t <= 0.f)
            {
                // Try second hit
                t = (-b + disc) * 0.5f;
                // Sphere behind ray?
                if (t <= 0.f)
                    return false;
            }

            isect_info.pos = p0 + dir * t;
            isect_info.normal = normalize(isect_info.pos - sphere.center);

            // Compute uvw coordinates
            const float phi = atan2f(isect_info.normal.x, isect_info.normal.z);
            const float theta = acosf(isect_info.normal.y);

            isect_info.uvw.x = phi / Constants.PI + 1.f;
            isect_info.uvw.y = 1.f - theta / Constants.PI;
            isect_info.uvw.z = 0.f;

            // Compute surface derivatives
            const float pi_rad = Constants.PI * sphere.radius;
            const float sp = sinf(phi);
            const float cp = cosf(phi);
            const float st = sinf(theta);

            isect_info.tan_u.x = cp * st * pi_rad;
            isect_info.tan_u.y = 0.f;
            isect_info.tan_u.z = -sp * st * pi_rad;
            isect_info.tan_u = normalize(isect_info.tan_u);

            isect_info.tan_v.x = -sp * isect_info.normal.y * pi_rad;
            isect_info.tan_v.y = st * pi_rad;
            isect_info.tan_v.z = -cp * isect_info.normal.y * pi_rad;
            isect_info.tan_v = normalize(isect_info.tan_v);

            return true;
        }

        // Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
        void offset_ray(const mi::mdl::tct_float3& n)
        {
            const float origin = 1.0f / 32.0f;
            const float float_scale = 1.0f / 65536.0f;
            const float int_scale = 256.0f;

            const mi::Sint32_3 of_i(
                static_cast<int>(int_scale * n.x),
                static_cast<int>(int_scale * n.y),
                static_cast<int>(int_scale * n.z));

            mi::mdl::tct_float3 p_i(
                { int_as_float(float_as_int(p0.x) + ((p0.x < 0.0f) ? -of_i.x : of_i.x)),
                  int_as_float(float_as_int(p0.y) + ((p0.y < 0.0f) ? -of_i.y : of_i.y)),
                  int_as_float(float_as_int(p0.z) + ((p0.z < 0.0f) ? -of_i.z : of_i.z)) });

            p0.x = abs(p0.x) < origin ? p0.x + float_scale * n.x : p_i.x;
            p0.y = abs(p0.y) < origin ? p0.y + float_scale * n.y : p_i.y;
            p0.z = abs(p0.z) < origin ? p0.z + float_scale * n.z : p_i.z;
        }
    };

    // MDL Backend execution
    mi::mdl::Shading_state_material shading_state;
    mi::mdl::Shading_state_material_with_derivs shading_state_derivs;
    mi::base::Handle<mi::mdl::IGenerated_code_lambda_function> exe_code;

    // Material argument block
    std::shared_ptr<Argument_block> argument_block;

    // Custom material handle
    Texture_handler* tex_handler = nullptr;
    Texture_handler_deriv* tex_handler_deriv = nullptr;
    mi::Size argument_block_index;

    // Index to selected material expressions
    mi::Size init_function_index;
    mi::Size surface_bsdf_function_index;
    mi::Size surface_edf_function_index;
    mi::Size surface_emission_intensity_function_index;
    mi::Size backface_bsdf_function_index;
    mi::Size backface_edf_function_index;
    mi::Size backface_emission_intensity_function_index;
    mi::Size cutout_opacity_function_index;
    mi::Size thin_walled_function_index;

    Render_context(bool use_derivatives)
        : use_derivatives(use_derivatives)
    {

        // Init constant parameters of material shader state
        shading_state.animation_time = 0.f;
        shading_state.text_coords = nullptr;
        shading_state.tangent_u = Constants.tangent_u;
        shading_state.tangent_v = Constants.tangent_v;
        shading_state.text_results = nullptr;
        shading_state.ro_data_segment = nullptr;
        shading_state.world_to_object = &Constants.identity[0];
        shading_state.object_to_world = &Constants.identity[0];
        shading_state.object_id = 0;
        shading_state.meters_per_scene_unit = 1.f;

        shading_state_derivs.animation_time = 0.f;
        shading_state_derivs.text_coords = nullptr;
        shading_state_derivs.tangent_u = Constants.tangent_u;
        shading_state_derivs.tangent_v = Constants.tangent_v;
        shading_state_derivs.text_results = nullptr;
        shading_state_derivs.ro_data_segment = nullptr;
        shading_state_derivs.world_to_object = &Constants.identity[0];
        shading_state_derivs.object_to_world = &Constants.identity[0];
        shading_state_derivs.object_id = 0;
        shading_state_derivs.meters_per_scene_unit = 1.f;
    }

    // Evaluate the environment map for a given ray direction
    mi::mdl::tct_float3 evaluate_environment(float& pdf, const mi::mdl::tct_float3& dir)
    {
         // Use environment map?
        if (env.map_pixels)
        {
            const float u = atan2f(dir.z, dir.x) * (0.5f / Constants.PI) + 0.5f;
            const float v = acosf(fmax(fminf(-dir.y, 1.0f), -1.0f)) / Constants.PI;

            size_t x = mi::math::min(static_cast<mi::Uint32>(u * env.map_size.x), env.map_size.x - 1u);
            size_t y = mi::math::min(static_cast<mi::Uint32>(v * env.map_size.y), env.map_size.y - 1u);

            const float* pixel = env.map_pixels + ((y * env.map_size.x + x) * 4);

            pdf = std::max(pixel[0], std::max(pixel[1], pixel[2])) * env.inv_integral;

            return mi::mdl::tct_float3({ pixel[0], pixel[1], pixel[2] }) * env.intensity;
        }
        else
        {
            pdf = 1.f;
            return env.color * env.intensity;
        }
    }

    // Importance sampling the environment map
    mi::mdl::tct_float3 sample_environment(mi::mdl::tct_float3& light_dir, float& light_pdf, unsigned& seed)
    {
        mi::mdl::tct_float3 xi;
        xi.x = rnd(seed);
        xi.y = rnd(seed);
        xi.z = rnd(seed);

        // Importance sample the environment using an alias map
        const unsigned int size = env.map_size.x * env.map_size.y;
        const unsigned int idx =
            mi::math::min(static_cast<unsigned>(xi.x * static_cast<float>(size)), size - 1);
        unsigned int env_idx;
        float xi_y = xi.y;
        if (xi_y < env.alias_map[idx].q)
        {
            env_idx = idx;
            xi_y /= env.alias_map[idx].q;
        }
        else
        {
            env_idx = env.alias_map[idx].alias;
            xi_y = (xi_y - env.alias_map[idx].q) / (1.0f - env.alias_map[idx].q);
        }

        const unsigned int py = env_idx / env.map_size.x;
        const unsigned int px = env_idx % env.map_size.x;

        // Uniformly sample spherical area of pixel
        const float u = static_cast<float>(px + xi_y) / static_cast<float>(env.map_size.x);
        const float phi = u * 2.0f * Constants.PI - Constants.PI;
        const float sin_phi = sinf(phi);
        const float cos_phi = cosf(phi);
        const float step_theta = Constants.PI / static_cast<float>(env.map_size.y);
        const float theta0 = static_cast<float>(py) * step_theta;
        const float cos_theta = cosf(theta0) * (1.0f - xi.z) + cosf(theta0 + step_theta) * xi.z;
        const float theta = acosf(cos_theta);
        const float sin_theta = sinf(theta);
        light_dir = mi::mdl::tct_float3({ cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta });

        // Lookup filtered beauty
        const float v = theta / Constants.PI;

        size_t x = mi::math::min(static_cast<mi::Uint32>(u * env.map_size.x), env.map_size.x - 1u);
        size_t y = mi::math::min(static_cast<mi::Uint32>(v * env.map_size.y), env.map_size.y - 1u);

        const float* pix = env.map_pixels + ((y * env.map_size.x + x) * 4);
        light_pdf = mi::math::max(pix[0], mi::math::max(pix[1], pix[2])) * env.inv_integral;
        return mi::mdl::tct_float3({ pix[0], pix[1], pix[2] }) * env.intensity;
    }

    // Sample scene lights (omni + environment map)
    mi::mdl::tct_float3 sample_lights(const mi::mdl::tct_float3& pos, mi::mdl::tct_float3& light_dir, float& light_pdf, unsigned& seed)
    {
        float p_select_light = 1.0f;
        if (omni_light.intensity > 0.f)
        {
            // Keep it simple and use either point light or environment light, each with the same
            // probability. If the environment factor is zero, we always use the point light
            p_select_light = env.intensity > 0.0f ? 0.5f : 1.0f;

            // In general, you would select the light depending on the importance of it
            // e.g. by incorporating their luminance

            // Randomly select one of the lights
            if (rnd(seed) <= p_select_light)
            {
                light_pdf = Constants.DIRAC; // infinity

                // Compute light direction and distance
                light_dir = omni_light.dir * omni_light.distance - pos;

                const float inv_distance2 = 1.0f / dot(light_dir, light_dir);
                light_dir *= sqrtf(inv_distance2);

                return omni_light.color *
                    (omni_light.intensity * inv_distance2 * 0.25f / (Constants.PI * p_select_light));
            }

            // Probability to select the environment instead
            p_select_light = (1.0f - p_select_light);
        }

        // Light from the environment map
        mi::mdl::tct_float3 radiance = sample_environment(light_dir, light_pdf, seed);

        // Return radiance over pdf
        light_pdf *= p_select_light;
        return radiance / light_pdf;
    }
};

///////////////////////////////////////////////////////////////////////////////
// Recursive raytracing
///////////////////////////////////////////////////////////////////////////////
bool trace_ray(mi::mdl::tct_float3 vp_sample[3], Render_context& rc, Render_context::Ray& ray, unsigned& seed)
{
    if (ray.level >= rc.max_ray_length)
        return false;

    ray.level++;

    Isect_info isect_info;

    // Ray hits sphere?
    if (ray.isect(rc.sphere, isect_info))
    {
        mi::mdl::Shading_state_material* shading_state = nullptr;
        mi::mdl::Texture_handler_base* tex_handler = nullptr;
        mi::mdl::tct_float4 text_results[128];

        // Update material shader state
        if (rc.use_derivatives) {
            // FIXME: compute dx, dy
            mi::mdl::tct_deriv_float3 position = {
                isect_info.pos,     // value component
                { 0.0f, 0.0f, 0.0f },   // dx component
                { 0.0f, 0.0f, 0.0f }    // dy component
            };

            // FIXME: compute dx, dy
            mi::mdl::tct_deriv_float3 texture_coords[1] = {
                {
                    isect_info.uvw,     // value component
                    { 0.0f, 0.0f, 0.0f },   // dx component
                    { 0.0f, 0.0f, 0.0f }    // dy component
                }
            };

            rc.shading_state_derivs.position = position;
            rc.shading_state_derivs.normal = ray.is_inside ? -isect_info.normal : isect_info.normal;
            rc.shading_state_derivs.geom_normal = rc.shading_state_derivs.normal;
            rc.shading_state_derivs.text_coords = texture_coords;
            rc.shading_state_derivs.tangent_u = &isect_info.tan_u;
            rc.shading_state_derivs.tangent_v = &isect_info.tan_v;
            rc.shading_state_derivs.text_results = text_results;

            shading_state =
                reinterpret_cast<mi::mdl::Shading_state_material*>(&rc.shading_state_derivs);
            tex_handler =
                reinterpret_cast<mi::mdl::Texture_handler_base*>(rc.tex_handler_deriv);

        }
        else {
            rc.shading_state.position = isect_info.pos;
            rc.shading_state.normal = ray.is_inside ? -isect_info.normal : isect_info.normal;
            rc.shading_state.geom_normal = rc.shading_state.normal;
            rc.shading_state.text_coords = &isect_info.uvw;
            rc.shading_state.tangent_u = &isect_info.tan_u;
            rc.shading_state.tangent_v = &isect_info.tan_v;
            rc.shading_state.text_results = text_results;

            shading_state = &rc.shading_state;
            tex_handler = rc.tex_handler;
        }

        // Beware: the layout of the structs *is different*
        mi::mdl::tct_float3& normal =
            rc.use_derivatives ? rc.shading_state_derivs.normal : rc.shading_state.normal;
        mi::mdl::tct_float3& geom_normal =
            rc.use_derivatives ? rc.shading_state_derivs.geom_normal : rc.shading_state.geom_normal;

        // Return code to check if the code execution succeeded
        bool ret_code;

        // shader initialization for the current hit point
        ret_code = rc.exe_code->run_init(
            rc.init_function_index,
            shading_state,
            tex_handler,
            rc.argument_block ? rc.argument_block->get_data() : nullptr);
        assert(ret_code && "execute_bsdf_init failed");

        // Evaluate material surface emission contribution
        {
            uint64_t edf_function_index = rc.surface_edf_function_index;

            mi::mdl::Edf_evaluate_data<mi::mdl::DF_HSM_NONE> eval_data;
            eval_data.k1 = -ray.dir;

            ret_code = rc.exe_code->run_generic(
                edf_function_index + 1, // edf_function_index corresponds to 'sample'
                                        // edf_function_index + 1 corresponds to 'evaluate'
                &eval_data,
                shading_state,
                tex_handler,
                rc.argument_block ? rc.argument_block->get_data() : nullptr);
            assert(ret_code && "execute_edf_evaluate failed");


            // Emission contribution is only valid for positive pdf
            if (eval_data.pdf > 1.e-6f)
            {
                uint64_t emission_intensity_function_index = rc.surface_emission_intensity_function_index;

                mi::mdl::tct_float3 intensity({ 1.f, 1.f, 1.f });
                ret_code = rc.exe_code->run_generic(
                    emission_intensity_function_index,
                    &intensity,
                    shading_state,
                    tex_handler,
                    rc.argument_block ? rc.argument_block->get_data() : nullptr);
                assert(ret_code && "execute emission intensity function failed");

                vp_sample[VPCH_ILLUM] += eval_data.edf * intensity * ray.weight;
            }
        }

        uint64_t surface_bsdf_function_index = rc.surface_bsdf_function_index;

        // Evaluate scene lights contribution
        {
            mi::mdl::tct_float3 light_dir;
            float light_pdf = 0.f;
            mi::mdl::tct_float3 radiance_over_pdf = rc.sample_lights(isect_info.pos, light_dir, light_pdf, seed);

            bool light_culled = !(
                (ray.level < rc.max_ray_length) &&
                (light_pdf != 0.0f) &&
                ((dot(normal, light_dir) > 0.f) != (ray.is_inside)));

            if (!light_culled)
            {
                mi::mdl::Bsdf_evaluate_data<mi::mdl::DF_HSM_NONE> eval_data;
                if (ray.is_inside)
                {
                    eval_data.ior1 = -Constants.ones_float3;
                    eval_data.ior2 = Constants.ones_float3;
                }
                else
                {
                    eval_data.ior1 = Constants.ones_float3;
                    eval_data.ior2 = -Constants.ones_float3;
                }

                eval_data.k1 = -ray.dir;
                eval_data.k2 = light_dir;
                eval_data.bsdf_diffuse = Constants.zeros_float3;
                eval_data.bsdf_glossy = Constants.zeros_float3;
                eval_data.flags = rc.bsdf_data_flags;

                ret_code = rc.exe_code->run_generic(
                    surface_bsdf_function_index + 1, // surface_bsdf_function_index corresponds to 'sample'
                                                     // surface_bsdf_function_index + 1 corresponds to 'eval'
                    &eval_data,
                    shading_state,
                    tex_handler,
                    rc.argument_block ? rc.argument_block->get_data() : nullptr);
                assert(ret_code && "execute_edf_evaluate failed");


                if (eval_data.pdf > 1.e-6f)
                {
                    const float mis_weight = (light_pdf == Constants.DIRAC)
                        ? 1.0f : light_pdf / (light_pdf + eval_data.pdf);

                    vp_sample[VPCH_ILLUM] += (eval_data.bsdf_diffuse + eval_data.bsdf_glossy) * (radiance_over_pdf * ray.weight) * mis_weight;
                }
            }
        }

        // Sample material bsdf contribution
        {
            mi::mdl::Bsdf_sample_data sample_data;  // input/output data for sample
            if (ray.is_inside)
            {
                sample_data.ior1 = -Constants.ones_float3;
                sample_data.ior2 = Constants.ones_float3;
            }
            else
            {
                sample_data.ior1 = Constants.ones_float3;
                sample_data.ior2 = -Constants.ones_float3;
            }
            sample_data.k1 = -ray.dir;  // outgoing direction
            sample_data.xi.x = rnd(seed);
            sample_data.xi.y = rnd(seed);
            sample_data.xi.z = rnd(seed);
            sample_data.xi.w = rnd(seed);
            sample_data.flags = rc.bsdf_data_flags;

            ret_code = rc.exe_code->run_generic(
                surface_bsdf_function_index,  // surface_bsdf_function_index corresponds to 'sample'
                                              // surface_bsdf_function_index + 1 corresponds to 'eval'
                &sample_data,
                shading_state,
                tex_handler,
                rc.argument_block ? rc.argument_block->get_data() : nullptr);
            assert(ret_code && "execute_edf_evaluate failed");


            if (sample_data.event_type != mi::mdl::BSDF_EVENT_ABSORB)
            {
                if ((sample_data.event_type & mi::mdl::BSDF_EVENT_SPECULAR) != 0)
                    ray.last_pdf = -1.0f;
                else
                    ray.last_pdf = sample_data.pdf;

                // There is a scattering event, trace either the reflection or transmission ray
                ray.weight *= sample_data.bsdf_over_pdf;
                ray.p0 = isect_info.pos;
                ray.dir = normalize(sample_data.k2);

                // Medium change?
                if (sample_data.event_type & mi::mdl::BSDF_EVENT_TRANSMISSION)
                {
                    ray.offset_ray(-geom_normal);
                    ray.is_inside = !ray.is_inside;
                }
                else
                {
                    ray.offset_ray(geom_normal);
                }

                mi::mdl::tct_float3 scat_color[3] =
                    { Constants.zeros_float3, Constants.zeros_float3, Constants.zeros_float3 };
                trace_ray(scat_color, rc, ray, seed);
                vp_sample[VPCH_ILLUM] += scat_color[VPCH_ILLUM];
            }
        }
        return true;

    }
    // Ray hits environment
    else
    {
        float pdf = 1.f;
        vp_sample[VPCH_ILLUM] = rc.evaluate_environment(pdf, ray.dir) * ray.weight;

        // Account multi importance sampling for environment
        if (ray.level > 1 && ray.last_pdf > 0.f)
        {
            // Point light selection probability
            if (rc.omni_light.intensity > 0.f)
                pdf *= 0.5f;

            vp_sample[VPCH_ILLUM] *= ray.last_pdf / (ray.last_pdf + pdf);
        }

        return false;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Scene Rendering
///////////////////////////////////////////////////////////////////////////////

void render_scene(
    Render_context rc,
    size_t frame_nb,
    VP_buffers* vp_buffers,
    unsigned char* dst,
    size_t ymin,
    size_t ymax,
    size_t width,
    size_t height,
    unsigned char channels)
{
    if (ymax > height)
        ymax = height;

    Render_context::Ray ray;
    size_t pixel_offset = ymin * width * channels;
    size_t vp_idx = ymin * width;

    for (size_t y = ymin; y < ymax; ++y)
    {
        // Random sequence initialization
        unsigned seed = tea(16, y * width, frame_nb);

        for (size_t x = 0; x < width; ++x, ++vp_idx)
        {
            mi::mdl::tct_float3 vp_sample[3] =
                { Constants.zeros_float3, Constants.zeros_float3, Constants.zeros_float3 };

            float x_rnd = rnd(seed);
            float y_rnd = rnd(seed);

            mi::mdl::tct_float2 screen_pos(
                { (x + x_rnd) * rc.cam.inv_res.x,
                (y + y_rnd) * rc.cam.inv_res.y });

            float r = (2.0f * screen_pos.x - 1.0f);
            float u = (2.0f * screen_pos.y - 1.0f);

            ray.p0 = rc.cam.pos;
            ray.dir = normalize(rc.cam.dir * rc.cam.focal +
                rc.cam.right * r + rc.cam.up * (rc.cam.aspect * u));
            ray.weight = Constants.ones_float3;
            ray.is_inside = false;
            ray.level = 0;
            ray.last_pdf = -1.f;

            //Trace camera ray
            trace_ray(vp_sample, rc, ray, seed);

            // Update progressive rendering viewport buffer
            if (frame_nb == 1)
            {
                vp_buffers->accum_buffer[vp_idx] = vp_sample[VPCH_ILLUM];
            }
            else
            {
                vp_buffers->accum_buffer[vp_idx] =
                    (vp_buffers->accum_buffer[vp_idx] * static_cast<float>(frame_nb - 1) + vp_sample[VPCH_ILLUM]) * (1.f / frame_nb);
                vp_sample[VPCH_ILLUM] = vp_buffers->accum_buffer[vp_idx];
            }

            if (dst)
            {
                // Apply gamma correction
                vp_sample[VPCH_ILLUM].x = powf(vp_sample[VPCH_ILLUM].x, 1.f / 2.2f);
                vp_sample[VPCH_ILLUM].y = powf(vp_sample[VPCH_ILLUM].y, 1.f / 2.2f);
                vp_sample[VPCH_ILLUM].z = powf(vp_sample[VPCH_ILLUM].z, 1.f / 2.2f);

                // Write final pixel
                vp_sample[VPCH_ILLUM] *= 255.f;
                clamp(vp_sample[VPCH_ILLUM], 0.f, 255.f);
                dst[pixel_offset++] = static_cast<unsigned char>(vp_sample[VPCH_ILLUM].x);
                dst[pixel_offset++] = static_cast<unsigned char>(vp_sample[VPCH_ILLUM].y);
                dst[pixel_offset++] = static_cast<unsigned char>(vp_sample[VPCH_ILLUM].z);
                dst[pixel_offset++] = 255u;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Material arguments information
///////////////////////////////////////////////////////////////////////////////

// Possible enum values if any
struct Enum_value
{
    std::string name;
    int         value;

    Enum_value(const std::string& name, int value)
        : name(name), value(value)
    {
    }
};

// Info for an enum type
struct Enum_type_info
{
    std::vector<Enum_value> values;

    // Adds a enum value and its integer value to the enum type info
    void add(const std::string& name, int value)
    {
        values.push_back(Enum_value(name, value));
    }
};

// Material parameter information structure
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
        char const* name,
        char const* display_name,
        char const* group_name,
        Param_kind kind,
        char* data_ptr,
        const Enum_type_info* enum_info = nullptr)
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

    // Get data as T&
    template<typename T>
    T& data() { return *reinterpret_cast<T*>(m_data_ptr); }

    // Get data as const T&
    template<typename T>
    const T& data() const { return *reinterpret_cast<const T*>(m_data_ptr); }

    const char*& display_name() { return m_display_name; }
    const char* display_name() const { return m_display_name; }

    const char*& group_name() { return m_group_name; }
    const char* group_name() const { return m_group_name; }

    Param_kind kind() const { return m_kind; }

    float& range_min() { return m_range_min; }
    float range_min() const { return m_range_min; }
    float& range_max() { return m_range_max; }
    float range_max() const { return m_range_max; }

    const Enum_type_info* enum_info() const { return m_enum_info; }

private:
    size_t                m_index;
    char const* m_name;
    char const* m_display_name;
    char const* m_group_name;
    Param_kind            m_kind;
    char* m_data_ptr;
    float                 m_range_min, m_range_max;
    const Enum_type_info* m_enum_info;
};

// Material information structure
class Material_info
{
public:
    Material_info(char const* name)
        : m_name(name)
    {}

    // Add the parameter information as last entry of the corresponding group,
    // or to the end of the list, if no group name is available
    void add_sorted_by_group(const Param_info& info) {
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

    // Add a new enum type to the list of used enum types
    void add_enum_type(const std::string name, std::shared_ptr<Enum_type_info> enum_info) {
        enum_types[name] = enum_info;
    }

    // Lookup enum type info for a given enum type absolute MDL name
    const Enum_type_info* get_enum_type(const std::string name) {
        Enum_type_map::const_iterator it = enum_types.find(name);
        if (it != enum_types.end())
            return it->second.get();
        return nullptr;
    }

    // Get the name of the material
    char const* name() const { return m_name; }

    // Get the parameters of this material
    std::list<Param_info>& params() { return m_params; }

private:
    // Name of the material
    char const* m_name;

    // Parameters of the material
    std::list<Param_info> m_params;

    typedef std::map<std::string, std::shared_ptr<Enum_type_info> > Enum_type_map;

    // Used enum types of the material
    Enum_type_map enum_types;
};

// Type trait to get the value type for a given type.
template<typename T> struct Value_trait { /* error */ };
template<> struct Value_trait<float> { typedef mi::mdl::IValue_float IVALUE_TYPE; };
template<> struct Value_trait<char const*> { typedef mi::mdl::IValue_string IVALUE_TYPE; };

template<typename T>
bool get_annotation_argument_value(mi::mdl::DAG_call const* anno, int index, T& res)
{
    mi::mdl::DAG_constant const* dag_const =
        mi::mdl::as<mi::mdl::DAG_constant>(anno->get_argument(index));
    if (dag_const == nullptr)
        return false;

    typedef typename Value_trait<T>::IVALUE_TYPE IValue_type;

    IValue_type const* val = mi::mdl::as<IValue_type>(dag_const->get_value());
    if (val == nullptr)
        return false;

    res = val->get_value();
    return true;
}

void collect_material_paramaters_info(
    Material_info& mat_info,
    mi::Size argument_block_index,
    Render_context& rc,
    const Target_code* target_code)
{

    if (argument_block_index != mi::Size(-1))
    {
        // We create our own copy of the argument data block, so we can modify the material parameters
        rc.argument_block = std::make_shared<Argument_block>(*target_code->get_argument_block(argument_block_index));
        rc.argument_block_index = argument_block_index;
    }

    // Scope for material context resources
    Material_instance const& cur_inst = target_code->get_material_instance(0);

    // Get the target argument block and its layout
    mi::base::Handle<mi::mdl::IGenerated_code_value_layout const> layout(
        rc.exe_code->get_captured_arguments_layout(rc.argument_block_index));

    char* arg_block_data = rc.argument_block ? rc.argument_block->get_data() : nullptr;

    for (size_t j = 0, num_params = cur_inst->get_parameter_count(); j < num_params; ++j)
    {
        const char* name = cur_inst->get_parameter_name(j);
        if (name == nullptr) continue;

        // Determine the type of the argument
        mi::mdl::IValue const* arg = cur_inst->get_parameter_default(j);
        mi::mdl::IValue::Kind kind = arg->get_kind();

        Param_info::Param_kind param_kind = Param_info::PK_UNKNOWN;
        const Enum_type_info* enum_type = nullptr;

        switch (kind)
        {
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
            mi::mdl::IValue_vector const* val = mi::mdl::as<mi::mdl::IValue_vector>(arg);
            mi::mdl::IType_vector const* val_type = val->get_type();
            mi::mdl::IType_atomic const* elem_type = val_type->get_element_type();
            if (elem_type->get_kind() == mi::mdl::IType::TK_FLOAT)
            {
                switch (val_type->get_size())
                {
                case 2: param_kind = Param_info::PK_FLOAT2; break;
                case 3: param_kind = Param_info::PK_FLOAT3; break;
                }
            }
        }
        break;
        case mi::mdl::IValue::VK_ENUM:
        {
            const mi::mdl::IValue_enum* val = mi::mdl::as<mi::mdl::IValue_enum>(arg);
            const mi::mdl::IType_enum* val_type = val->get_type();

            // prepare info for this enum type if not seen so far
            const char* e_name = val_type->get_symbol()->get_name();
            const Enum_type_info* info = mat_info.get_enum_type(e_name);
            if (info == nullptr)
            {
                std::shared_ptr<Enum_type_info> p = std::make_shared<Enum_type_info>();

                for (size_t i = 0, n = val_type->get_value_count(); i < n; ++i)
                {
                    const mi::mdl::IType_enum::Value* e_val = val_type->get_value(i);
                    const mi::mdl::ISymbol* e_sym = e_val->get_symbol();
                    int e_code = e_val->get_code();

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
        if (dag_param_index != ~0)
        {
            bool has_soft_range = false;
            size_t anno_count = cur_inst.get_dag_parameter_annotation_count(dag_param_index);
            for (size_t anno_ind = 0; anno_ind < anno_count; ++anno_ind)
            {
                if (mi::mdl::DAG_call const* anno = mi::mdl::as<mi::mdl::DAG_call>(
                    cur_inst.get_dag_parameter_annotation(dag_param_index, anno_ind)))
                {
                    switch (anno->get_semantic())
                    {
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

}

// Update material parameter editor window
bool update_parmater_editor_window(
    Material_info& mat_info,
    String_constant_table& constant_table,
    const Options& options)
{
    ImGui::SetNextWindowPos(ImVec2(10, 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(
        ImVec2(360.f, 600.f),
        ImGuiCond_FirstUseEver);
    ImGui::Begin("Material parameters");
    ImGui::SetWindowFontScale(1.f);
    ImGui::PushItemWidth(-200.f);
    if (options.use_class_compilation)
        ImGui::Text("CTRL + Click to manually enter numbers");
    else
        ImGui::Text("Parameter editing requires class compilation.");

    // Print material name
    ImGui::Text("%s", mat_info.name());

    bool changed = false;
    const char* group_name = nullptr;
    int id = 0;

    for (std::list<Param_info>::iterator it = mat_info.params().begin(),
        end = mat_info.params().end(); it != end; ++it, ++id)
    {
        Param_info& param = *it;

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
        switch (param.kind())
        {
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

            const Enum_type_info* info = param.enum_info();
            for (size_t i = 0, n = info->values.size(); i < n; ++i)
            {
                if (info->values[i].value == value)
                {
                    curr_value = info->values[i].name;
                    break;
                }
            }

            if (ImGui::BeginCombo(param.display_name(), curr_value.c_str()))
            {
                for (size_t i = 0, n = info->values.size(); i < n; ++i)
                {
                    const std::string& name = info->values[i].name;
                    bool is_selected = (curr_value == name);
                    if (ImGui::Selectable(info->values[i].name.c_str(), is_selected))
                    {
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

            // Fill the current value
            unsigned curr_index = param.data<unsigned>();
            const char* opt = constant_table.get_string(curr_index);
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

    ImGui::PopItemWidth();
    ImGui::End();

    return changed;
}
///////////////////////////////////////////////////////////////////////////////
// Main Function
///////////////////////////////////////////////////////////////////////////////

int MAIN_UTF8(int argc, char* argv[])
{
    // Parse command line options
    Options options;
    options.mdl_paths.push_back(get_samples_mdl_root());
    options.parse(argc, argv);

    // Create render context
    Render_context rc(options.enable_derivatives);

    // Access the MDL Core compiler
    mi::base::Handle<mi::mdl::IMDL> mdl_compiler(load_mdl_compiler());
    check_success(mdl_compiler);

    // Set a default material, when not provided in the command line
    if (options.material_name.empty())
        options.material_name = "::nvidia::sdk_examples::tutorials::example_df";

    // Configure compiler backend options
    mi::Uint32 backend_options =
        options.enable_derivatives ? BACKEND_OPTIONS_ENABLE_DERIVATIVES : BACKEND_OPTIONS_NONE;

    // Use bsdf flags in BSDF data struct
    if (options.enable_bsdf_flags)
    {
        backend_options |= BACKEND_OPTIONS_ENABLE_BSDF_FLAGS;
        rc.bsdf_data_flags = options.allowed_scatter_mode;
    }

    // Initialize the material compiler with 16 result buffer slots ("texture results")
    Material_backend_compiler mc(
        mdl_compiler.get(),
        /*target_backend*/ mi::mdl::ICode_generator::TL_NATIVE,
        /*num_texture_results*/ 16,
        backend_options,
        /*df_handle_mode=*/ "none",
        /*lambda_return_mode=*/ "sret");
    for (std::size_t i = 0; i < options.mdl_paths.size(); ++i)
        mc.add_module_path(options.mdl_paths[i].c_str());

    // Select some functions to translate
    std::vector<Target_function_description> descs;
    descs.push_back(
        Target_function_description("init"));
    descs.push_back(
        Target_function_description("surface.scattering"));
    descs.push_back(
        Target_function_description("surface.emission.emission"));
    descs.push_back(
        Target_function_description("surface.emission.intensity"));

    // Generate code for the material
    std::cout << "Adding material \"" << options.material_name << "\"..." << std::endl;

    // Add functions of the material to the link unit
    if (!mc.add_material(
        options.material_name,
        descs.data(), descs.size(),
        options.use_class_compilation))
    {
        std::cout << "Failed!" << std::endl;
        // Print any compiler messages, if available
        mc.print_messages();
        exit(EXIT_FAILURE);
    }

    // Collect material sub-expression indices
    rc.init_function_index = descs[0].function_index;;
    rc.surface_bsdf_function_index = descs[1].function_index;
    rc.surface_edf_function_index = descs[2].function_index;
    rc.surface_emission_intensity_function_index = descs[3].function_index;

    // Generate the native target code
    Target_code* target_code = mc.generate_target_code();
    rc.exe_code = target_code->get_code_lambda();

    // Collect material arguments information for the class compilation mode
    Material_info mat_info(target_code->get_material_instance(0).get_dag_material_name());
    if (options.use_class_compilation)
        collect_material_paramaters_info(
            mat_info,
            descs[0].argument_block_index,
            rc,
            target_code
        );

    String_constant_table& constant_table(target_code->get_string_constant_table());

    // Setup custom texture handler
    Texture_handler tex_handler;
    Texture_handler_deriv tex_handler_deriv;

    // Prepare textures
    std::vector<Texture> textures;
    for (mi::Size i = 1; i < target_code->get_texture_count(); ++i)
    {
        const Texture_data* tex = target_code->get_texture(i);
        assert(tex->is_valid());

        // Get BSDF texture data if exists
        if (tex->get_shape() == mi::mdl::IType_texture::TS_BSDF_DATA)
        {
            textures.push_back(Texture(
                tex->get_bsdf_data(),
                tex->get_width(), tex->get_height(), tex->get_depth(),
                tex->get_pixel_type()));
        }
        // Get texture image data
        else
        {
            textures.push_back(tex->get_image());
        }
    }

    if (options.enable_derivatives)
    {
        tex_handler_deriv.vtable = &tex_deriv_vtable;
        tex_handler_deriv.num_textures = target_code->get_texture_count() - 1;
        tex_handler_deriv.textures = textures.data();

        rc.tex_handler_deriv = &tex_handler_deriv;
    }
    else
    {
        tex_handler.vtable = &tex_vtable;
        tex_handler.num_textures = target_code->get_texture_count() - 1;
        tex_handler.textures = textures.data();

        rc.tex_handler = &tex_handler;
    }

    // Create window context
    Window_context window_context;

    // setup render data
    size_t window_width = -1, window_height = -1;
    size_t frame_nb = 0; // frame counter

    // Viewport buffers for progressive rendering
    VP_buffers vp_buffers;

    // Setup file name for nogl mode
    std::string filename_base, filename_ext;
    size_t dot_pos = options.outputfile.rfind('.');
    if (dot_pos == std::string::npos)
    {
        filename_base = options.outputfile;
    }
    else
    {
        filename_base = options.outputfile.substr(0, dot_pos);
        filename_ext = options.outputfile.substr(dot_pos);
    }

    // Single/Multi-threading rendering?
    const int num_threads =
        options.single_threaded ? 1 : std::thread::hardware_concurrency();
    std::cout << "Rendering on " << num_threads << " threads.\n";

    // Render options
    rc.max_ray_length = options.max_ray_length;

    // Load/Setup environment map
    window_context.env_intensity = rc.env.intensity = options.env_scale;
    rc.env.load(options.env_map, mdl_compiler.get());

    // Setup omni light
    rc.omni_light.intensity = std::max(std::max(options.light_intensity.x, options.light_intensity.y), options.light_intensity.y);
    if (rc.omni_light.intensity > 0.f)
        rc.omni_light.color = options.light_intensity / rc.omni_light.intensity;
    else
        rc.omni_light.color = options.light_intensity;
    rc.omni_light.distance = length(options.light_pos);
    rc.omni_light.dir = normalize(options.light_pos);

    window_context.omni_phi = atan2f(rc.omni_light.dir.x, rc.omni_light.dir.z);
    window_context.omni_theta = acosf(rc.omni_light.dir.y);
    window_context.omni_intensity = rc.omni_light.intensity;

    rc.omni_light.update(window_context.omni_phi, window_context.omni_theta, window_context.omni_intensity);

    // Setup initial camera
    float base_dist = length(options.cam_pos);
    float theta, phi;

    const mi::mdl::tct_float3 inv_dir = normalize(options.cam_pos);
    phi = atan2f(inv_dir.x, inv_dir.z);
    theta = acosf(inv_dir.y);

    rc.cam.focal = 1.0f / tanf(options.cam_fov * Constants.PI / 360.f);
    rc.cam.update(phi, theta, base_dist, window_context.zoom);

    // Render to image?
    if (options.no_gui)
    {
        window_width = options.res_x;
        window_height = options.res_y;

        frame_nb = 0;
        vp_buffers.accum_buffer.reset(new mi::mdl::tct_float3[window_width * window_height]);

        // Update camera parameters
        rc.cam.inv_res.x = 1.0f / static_cast<float>(window_width);
        rc.cam.inv_res.y = 1.0f / static_cast<float>(window_height);
        rc.cam.aspect = static_cast<float>(window_height)
            / static_cast<float>(window_width);

        {
            //Render loop
            while (frame_nb < options.iterations)
            {
                frame_nb++;

                if (options.single_threaded)
                {
                    render_scene(rc, frame_nb, &vp_buffers,
                        nullptr, 0, window_height, window_width, window_height, 4);
                }
                else
                {
                    // Preparing render threads
                    std::vector<std::thread> threads;

                    // window lines per thread
                    size_t lpt = window_height / num_threads + (window_height % num_threads != 0 ? 1 : 0);

                    // Launch render threads
                    for (int i = 0; i < num_threads; ++i)
                        threads.emplace_back(std::thread(render_scene, rc, frame_nb, &vp_buffers,
                            nullptr, lpt * i, lpt * (i + 1), window_width, window_height, 4));

                    // Wait for threads to finish
                    for (auto& th : threads)
                        if (th.joinable())
                            th.join();
                }
            }
        }

        // Save screenshot
        export_image_rgbf(std::string(filename_base + filename_ext).c_str(),
            window_width,
            window_height,
            vp_buffers.accum_buffer.get());
    }
    else // Interactive renderer
    {
        GLuint pixel_buffer_object_ids[2] = { 0, 0 };
        std::unique_ptr<GLubyte[]> image_data;

        GLuint display_tex = 0;
        GLuint program = 0;
        GLuint quad_vertex_buffer = 0;
        GLuint quad_vao = 0;

        // Init OpenGL window
        std::string version_string;
        GLFWwindow* window = init_opengl(options.res_x, options.res_y, version_string);

        glfwSetWindowUserPointer(window, &window_context);
        glfwSetKeyCallback(window, Window_context::handle_key);
        glfwSetScrollCallback(window, Window_context::handle_scroll);
        glfwSetCursorPosCallback(window, Window_context::handle_mouse_pos);
        glfwSetMouseButtonCallback(window, Window_context::handle_mouse_button);
        glfwSetCharCallback(window, ImGui_ImplGlfw_CharCallback);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGui_ImplGlfw_InitForOpenGL(window, false);
        ImGui_ImplOpenGL3_Init(version_string.c_str());
        ImGui::GetIO().IniFilename = nullptr;       // disable creating imgui.ini
        ImGui::StyleColorsDark();
        ImGui::GetStyle().Alpha = 0.7f;
        ImGui::GetStyle().ScaleAllSizes(/*options.gui_scale*/1.f);

        glGenTextures(1, &display_tex);
        check_success(glGetError() == GL_NO_ERROR);

        // Create shader program
        program = create_shader_program();

        // Create scene data
        quad_vao = create_quad(program, &quad_vertex_buffer);

        std::chrono::duration<double> state_update_time(0.0);
        std::chrono::duration<double> render_time(0.0);
        std::chrono::duration<double> display_time(0.0);
        char stats_text[128];
        int last_update_frames = -1;
        auto last_update_time = std::chrono::steady_clock::now();
        const std::chrono::duration<double> update_min_interval(0.5);

        // Render loop
        while (true)
        {
            std::chrono::time_point<std::chrono::steady_clock> t0 =
                std::chrono::steady_clock::now();

            // Check for termination
            if (glfwWindowShouldClose(window))
                break;

            // Poll for events and process them
            glfwPollEvents();

            // Check if buffers need to be resized
            int nwidth, nheight;
            glfwGetFramebufferSize(window, &nwidth, &nheight);

            // Get the window size and resize the image if necessary
            if (window_width != nwidth || window_height != nheight)
            {
                window_width = nwidth;
                window_height = nheight;

                //Resize OGL display buffer
                {
                    size_t new_buffer_size = window_width * window_height * 4;
                    glViewport(0, 0, window_width, window_height);

                    // Free the old image data
                    if (image_data)
                    {
                        glDeleteBuffers(2, pixel_buffer_object_ids);

                        glDeleteTextures(1, &display_tex);
                        glGenTextures(1, &display_tex);
                    }

                    image_data = std::make_unique<GLubyte[]>(new_buffer_size);
                    std::fill_n(image_data.get(), new_buffer_size, 0);

                    glBindTexture(GL_TEXTURE_2D, display_tex);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)image_data.get());
                    glBindTexture(GL_TEXTURE_2D, 0);

                    glGenBuffers(2, pixel_buffer_object_ids);
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_object_ids[0]);
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, new_buffer_size, 0, GL_STREAM_DRAW);
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_object_ids[1]);
                    glBufferData(GL_PIXEL_UNPACK_BUFFER, new_buffer_size, 0, GL_STREAM_DRAW);
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

                    if (glGetError() != GL_NO_ERROR)
                        exit(EXIT_FAILURE);
                }

                frame_nb = 0;
                vp_buffers.accum_buffer.reset(new mi::mdl::tct_float3[window_width * window_height]);

                // Update camera parameters
                rc.cam.inv_res.x = 1.0f / static_cast<float>(window_width);
                rc.cam.inv_res.y = 1.0f / static_cast<float>(window_height);
                rc.cam.aspect = static_cast<float>(window_height)
                    / static_cast<float>(window_width);
            }

            // Don't render anything, if minimized
            if (window_width == 0 || window_height == 0)
            {
                // Wait until something happens
                glfwWaitEvents();
                continue;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Update material parameter editor window
            bool changed = 
                update_parmater_editor_window(mat_info, constant_table, options);

            // Handle key input events
            if (window_context.key_event && !ImGui::GetIO().WantCaptureMouse)
            {
                // Update environment
                rc.env.intensity = window_context.env_intensity;

                // Update light
                rc.omni_light.update(window_context.omni_phi, window_context.omni_theta, window_context.omni_intensity);
            }

            // Handle save screenshot event
            if (window_context.save_sreenshot && !ImGui::GetIO().WantCaptureMouse)
            {
                // Save screenshot
                export_image_rgbf(std::string(filename_base + filename_ext).c_str(),
                    window_width,
                    window_height,
                    vp_buffers.accum_buffer.get());
            }

            // Handle mouse input events
            if (window_context.mouse_button - 1 == GLFW_MOUSE_BUTTON_LEFT)
            {
                // Only accept button press when not hovering GUI window
                if (window_context.mouse_button_action == GLFW_PRESS &&
                    !ImGui::GetIO().WantCaptureMouse)
                {
                    window_context.moving = true;
                    glfwGetCursorPos(window, &window_context.move_start_x, &window_context.move_start_y);
                }
                else
                {
                    window_context.moving = false;
                }
            }

            if (window_context.mouse_wheel_delta && !ImGui::GetIO().WantCaptureMouse)
            {
                window_context.zoom += window_context.mouse_wheel_delta;
            }

            if (window_context.mouse_event && !ImGui::GetIO().WantCaptureMouse)
            {
                // Update camera
                phi -= static_cast<float>(window_context.move_dx) * 0.001f * Constants.PI;
                theta -= static_cast<float>(window_context.move_dy) * 0.001f * Constants.PI;

                if (theta < 0.f)
                    theta = 0.f;
                else if (theta > Constants.PI)
                    theta = Constants.PI;

                window_context.move_dx = window_context.move_dy = 0.0;

                rc.cam.update(phi, theta, base_dist, window_context.zoom);
            }

            if (window_context.key_event || window_context.mouse_event || changed)
                frame_nb = 0;

            // Clear all events
            window_context.key_event = false;
            window_context.mouse_event = false;
            window_context.mouse_wheel_delta = 0;
            window_context.mouse_button = 0;
            window_context.save_sreenshot = false;

            ++frame_nb;

            auto t1 = std::chrono::steady_clock::now();
            state_update_time += t1 - t0;
            t0 = t1;

            // Map the buffer, update the image data and un-map afterwards
            // Make sure this is as fast as possible
            unsigned char* dst_image_data = nullptr;
            {
                gl_bind_index = (gl_bind_index + 1) % 2;
                int nextIndex = (gl_bind_index + 1) % 2;

                glBindTexture(GL_TEXTURE_2D, display_tex);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_object_ids[gl_bind_index]);

                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_object_ids[nextIndex]);

                glBufferData(GL_PIXEL_UNPACK_BUFFER, window_width * window_height * 4, 0, GL_STREAM_DRAW);
                dst_image_data = (unsigned char*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
            }

            if (options.single_threaded)
            {
                render_scene(rc, frame_nb, &vp_buffers,
                    dst_image_data, 0, window_height, window_width, window_height, 4);
            }
            else
            {
                // Preparing render threads
                std::vector<std::thread> threads;

                // window lines per thread
                size_t lpt = window_height / num_threads + (window_height % num_threads != 0 ? 1 : 0);

                // Launch render threads
                for (int i = 0; i < num_threads; ++i)
                    threads.emplace_back(render_scene, rc, frame_nb, &vp_buffers,
                        dst_image_data, lpt * i, lpt * (i + 1), window_width, window_height, 4);

                // Wait for threads to finish
                for (auto& th : threads)
                    if (th.joinable())
                        th.join();
            }

            t1 = std::chrono::steady_clock::now();
            render_time += t1 - t0;
            t0 = t1;

            // Render the updated image to screen
            {
                // Unmap display buffer
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);

                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, display_tex);

                // Render the quad
                glClear(GL_COLOR_BUFFER_BIT);
                glBindVertexArray(quad_vao);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                if (glGetError() != GL_NO_ERROR)
                    exit(EXIT_FAILURE);

                // Unbind texture
                glBindTexture(GL_TEXTURE_2D, 0);
            }

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
            if (t1 - last_update_time > update_min_interval || last_update_frames == 0)
            {
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

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            // Swap front and back buffers
            glfwSwapBuffers(window);
        }

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

    exit(EXIT_SUCCESS);
}
// Convert command line arguments to UTF8 on Windows
COMMANDLINE_TO_UTF8
