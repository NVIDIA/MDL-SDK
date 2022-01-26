/******************************************************************************
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <iomanip>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "example_shared.h"
#include "texture_support_native.h"

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GL_DISPLAY_NATIVE
#include <utils/gl_display.h>

#if MI_PLATFORM_MACOSX
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include "utils/profiling.h"
using namespace mi::examples::profiling;

#define USE_PARALLEL_RENDERING
#define ADD_EXTRA_TIMERS

///////////////////////////////////////////////////////////////////////////////
// Global Constants
///////////////////////////////////////////////////////////////////////////////

static struct
{
    const float DIRAC = -1.f;
    const float PI = static_cast<float>(M_PI);
    const mi::Float32_3_struct tangent_u[1] = { {1.0f, 0.0f, 0.0f} };
    const mi::Float32_3_struct tangent_v[1] = { { 0.0f, 1.0f, 0.0f} };
    const mi::Float32_3_4 identity = mi::Float32_3_4(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f);
} Constants;

///////////////////////////////////////////////////////////////////////////////
// Random Number Generator
///////////////////////////////////////////////////////////////////////////////

inline unsigned tea(unsigned N, unsigned val0, unsigned val1)
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
inline unsigned lcg(unsigned &prev)
{
    const unsigned LCG_A = 1664525u;
    const unsigned LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
inline float rnd(unsigned &prev)
{
    const unsigned next = lcg(prev);
    return ((float)next / (float)0x01000000);
}

///////////////////////////////////////////////////////////////////////////////
// Window Handling
///////////////////////////////////////////////////////////////////////////////

// Window context structure for window keys/mouse event callback functions.
struct Window_context
{
    bool mouse_event, key_event;

    // for environment
    float env_intensity;

    // for omni light movement
    float omni_theta;
    float omni_phi;
    float omni_intensity;

    // for camera movement
    int mouse_button;            // button from callback event plus one (0 = no event)
    int mouse_button_action;     // action from mouse button callback event
    int mouse_wheel_delta;
    bool moving;
    double move_start_x, move_start_y;
    double move_dx, move_dy;
    int zoom;

    // image output
    bool save_sreenshot;

    Window_context()
        : mouse_event(false)
        , key_event(false)
        , env_intensity(0.0f)
        , omni_theta(0.0f)
        , omni_phi(0.0f)
        , omni_intensity(0.0f)
        , mouse_button(0)
        , mouse_button_action(0)
        , mouse_wheel_delta(0)
        , moving(false)
        , move_start_x(0.0)
        , move_start_y(0.0)
        , move_dx(0.0)
        , move_dy(0.0)
        , zoom(0)
        , save_sreenshot(false)
    {}

    // GLFW keyboard callback
    static void handle_key(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        // Handle key press events
        if (action == GLFW_PRESS)
        {
            Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));

            if (mods&GLFW_MOD_CONTROL)
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
                    ctx->omni_theta += 0.05f*Constants.PI;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_UP:
                    ctx->omni_theta -= 0.05f*Constants.PI;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_LEFT:
                    ctx->omni_phi -= 0.05f*Constants.PI;
                    ctx->key_event = true;
                    break;
                case GLFW_KEY_RIGHT:
                    ctx->omni_phi += 0.05f*Constants.PI;
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

    // GLFW scroll callback
    static void handle_scroll(GLFWwindow *window, double xoffset, double yoffset)
    {
        Window_context *ctx = static_cast<Window_context*>(glfwGetWindowUserPointer(window));
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
static inline float int_as_float(uint32_t v)
{
    union
    {
        uint32_t bit;
        float    value;
    } temp;

    temp.bit = v;
    return temp.value;
}

static inline uint32_t float_as_int(float v)
{
    union
    {
        uint32_t bit;
        float    value;
    } temp;

    temp.value = v;
    return temp.bit;
}

inline void clamp(mi::Float32_3 &d, float min = 0.f, float max = 1.f)
{
    for (int i = 0; i < 3; ++i)
    {
        if (d[i] < min)
            d[i] = min;
        else if (d[i] > max)
            d[i] = max;
    }
}

inline float length(const mi::Float32_3 &d)
{
    return sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
}

inline float dot(const mi::Float32_3 &a, const mi::Float32_3 &b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

inline mi::Float32_3 normalize(const mi::Float32_3 &d)
{
    const float inv_len = 1.0f / length(d);
    return mi::Float32_3(d.x * inv_len, d.y * inv_len, d.z * inv_len);
}

inline mi::Float32_3 operator+(const mi::Float32_3& a, const mi::Float32_3& b)
{
    return mi::Float32_3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline mi::Float32_3 operator-(const mi::Float32_3& a, const mi::Float32_3& b)
{
    return mi::Float32_3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline mi::Float32_3 operator*(const mi::Float32_3& a, const mi::Float32_3& b)
{
    return mi::Float32_3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline mi::Float32_3 operator*(const mi::Float32_3& d, float s)
{
    return mi::Float32_3(d.x * s, d.y * s, d.z * s);
}

inline mi::Float32_3 operator/(const mi::Float32_3& d, float s)
{
    const float inv_s = 1.0f / s;
    return mi::Float32_3(d.x * inv_s, d.y * inv_s, d.z * inv_s);
}

///////////////////////////////////////////////////////////////////////////////
// Command Line Options
///////////////////////////////////////////////////////////////////////////////

// Command line options structure.
struct Options
{
    // Don't open OpenGL GUI
    bool no_gui;

    // Number of iterations for output images
    size_t iterations;

    // A result output file name.
    std::string outputfile;
    bool output_auxiliary; // output albedo and normal auxiliary buffers.

    // The resolution of the display / image.
    unsigned res_x, res_y;

    // Path-tracer max ray-length
    int max_ray_length;

    // Environment map filename and scale
    std::string env_map;
    float env_scale;

    // Camera position and FOV
    mi::Float32_3 cam_pos;
    float cam_fov;

    // Light position and intensity
    mi::Float32_3 light_pos;
    mi::Float32_3 light_intensity;

    // Whether class compilation should be used for the materials.
    bool use_class_compilation;

    // Whether the custom texture runtime should be used.
    bool use_custom_tex_runtime;

    // Whether normals should be adapted.
    bool use_adapt_normal;

    // Whether derivative support should be enabled.
    // This example does not support derivatives in combination with the custom texture runtime.
    bool enable_derivatives;

    // Material to use.
    std::string material_name;

    Options()
        : no_gui(false)
        , iterations(100)
        , outputfile("example_df_native.png")
        , output_auxiliary(false)
        , res_x(700)
        , res_y(520)
        , max_ray_length(6)
        , env_map("nvidia/sdk_examples/resources/environment.hdr")
        , env_scale(1.f)
        , cam_pos(0.f, 0.f, 3.f)
        , cam_fov(86.f)
        , light_pos(10.f, 5.f, 0.f)
        , light_intensity(1.0f, 0.902f, 0.502f)
        , use_class_compilation(false)
        , use_custom_tex_runtime(false)
        , use_adapt_normal(false)
        , enable_derivatives(false)
    {}
};

///////////////////////////////////////////////////////////////////////////////
// Scene Render Context
///////////////////////////////////////////////////////////////////////////////

// Viewport buffers for progressive rendering
enum VP_channel
{
    VPCH_ILLUM = 0,
    VPCH_ALBEDO,
    VPCH_NORMAL,
    VPCH_NB_CHANNELS
};

struct VP_buffers
{
    mi::Float32_3 *accum_buffer;
    mi::Float32_3 *albedo_buffer;
    mi::Float32_3 *normal_buffer;

    VP_buffers()
        : accum_buffer(nullptr)
        , albedo_buffer(nullptr)
        , normal_buffer(nullptr)
    {}

    ~VP_buffers()
    {
        if (accum_buffer)
            delete[] accum_buffer;
        if (albedo_buffer)
            delete[] albedo_buffer;
        if (normal_buffer)
            delete[] normal_buffer;
    }
};


// Surface intersection info
struct Isect_info
{
    mi::Float32_3 pos;    // surface position
    mi::Float32_3 normal; // surface normal
    mi::Float32_3 uvw;    // uvw coordinates
    mi::Float32_3 tan_u;  // tangent vector in u direction
    mi::Float32_3 tan_v;  // tangent vector in u direction
};

// Render context
struct Render_context
{
    // render options
    int max_ray_length;
    bool render_auxiliary;

    // scene data
    // environment color
    struct Environment
    {
        mi::Float32_3 color; // used when no environment map is set
        float intensity;
        mi::base::Handle<const mi::neuraylib::ICanvas> map;
        struct Alias_map
        {
            unsigned int alias;
            float        q;
        } *alias_map;

        float inv_integral;
        mi::Uint32_2 map_size;
        const float *map_pixels;
    }env;

    // Perspective camera
    struct Camera
    {
        float focal;
        float aspect;
        float zoom;
        mi::Float32_2 inv_res;
        mi::Float32_3 pos;
        mi::Float32_3 dir;
        mi::Float32_3 right;
        mi::Float32_3 up;
    } cam;

    // Omni light
    struct Omni
    {
        mi::Float32_3 color;
        mi::Float32_3 dir;
        float distance;
        float intensity;
    } omni_light;

    // sphere object
    struct Sphere
    {
        mi::Float32_3 center;
        float   radius;
    } sphere;

    // MDL cutout opacity
    struct Cutout
    {
        bool is_constant;
        float constant_opacity;
    } cutout;

    // MDL thin_walled
    struct Thin_walled
    {
        bool is_constant;
        bool is_thin_walled;
    } thin_walled;


    // single raytracing ray
    struct Ray
    {
        mi::Float32_3 p0;
        mi::Float32_3 dir;
        mi::Float32_3 weight;
        int level;
        float last_pdf;
        bool is_inside;

        Ray()
            : weight(1.f)
            , level(0)
            , last_pdf(-1.f)
            , is_inside(false)
        {};

        //-------------------------------------------------------------------------------------------------
        // Avoiding self intersections (see Ray Tracing Gems, Ch. 6)
        //-------------------------------------------------------------------------------------------------
        inline void offset_ray(const mi::Float32_3 &n)
        {
            const float origin = 1.0f / 32.0f;
            const float float_scale = 1.0f / 65536.0f;
            const float int_scale = 256.0f;

            const mi::Sint32_3 of_i(
                static_cast<int>(int_scale * n.x),
                static_cast<int>(int_scale * n.y),
                static_cast<int>(int_scale * n.z));

            mi::Float32_3 p_i(
                int_as_float(float_as_int(p0.x) + ((p0.x < 0.0f) ? -of_i.x : of_i.x)),
                int_as_float(float_as_int(p0.y) + ((p0.y < 0.0f) ? -of_i.y : of_i.y)),
                int_as_float(float_as_int(p0.z) + ((p0.z < 0.0f) ? -of_i.z : of_i.z)));

            p0.x = abs(p0.x) < origin ? p0.x + float_scale * n.x : p_i.x;
            p0.y = abs(p0.y) < origin ? p0.y + float_scale * n.y : p_i.y;
            p0.z = abs(p0.z) < origin ? p0.z + float_scale * n.z : p_i.z;
        }
    };

    // MDL Backend execution
    mi::neuraylib::Shading_state_material shading_state;
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code;
    Texture_handler *tex_handler;
    uint64_t surface_bsdf_function_index;
    uint64_t surface_edf_function_index;
    uint64_t surface_emission_intensity_function_index;
    uint64_t backface_bsdf_function_index;
    uint64_t backface_edf_function_index;
    uint64_t backface_emission_intensity_function_index;
    uint64_t cutout_opacity_function_index;
    uint64_t thin_walled_function_index;

    Render_context()
        : target_code(nullptr)
        , tex_handler(nullptr)
    {
        max_ray_length = 6;
        render_auxiliary = false;
        env.color = mi::Float32_3(0.53f, 0.81f, 0.92f);
        env.intensity = 1.0f;
        env.alias_map = nullptr;

        omni_light.color = mi::Float32_3(1.0f, 0.902f, 0.502f);
        omni_light.dir = normalize(mi::Float32_3(1.f, 1.f, 1.f));
        omni_light.distance = 11.18f;
        omni_light.intensity = 0.0f;

        sphere.center = mi::Float32_3(0.f); // sphere in the origin
        sphere.radius = 1.f;

        cutout.is_constant = false;
        cutout.constant_opacity = 1.f;

        thin_walled.is_constant = true;
        thin_walled.is_thin_walled = false;

        // init constant parameters of material shader state
        shading_state.animation_time = 0.f;
        shading_state.tangent_u = Constants.tangent_u;
        shading_state.tangent_v = Constants.tangent_v;
        shading_state.text_results = nullptr;
        shading_state.ro_data_segment = nullptr;
        shading_state.world_to_object = &Constants.identity[0];
        shading_state.object_to_world = &Constants.identity[0];
        shading_state.object_id = 0;
        shading_state.meters_per_scene_unit = 1.f;
    }

    // Free resources owned by the render context.
    void uninit()
    {
        if (env.alias_map) {
            free(env.alias_map);
            env.alias_map = nullptr;
        }
        target_code = nullptr;
    }

    inline void update_light(
        float phi,
        float theta,
        float intensity)
    {
        omni_light.dir.x = sinf(theta) * sinf(phi);
        omni_light.dir.y = cosf(theta);
        omni_light.dir.z = sinf(theta) * cosf(phi);

        omni_light.intensity = intensity;
    }

    inline void update_camera(
        float phi,
        float theta,
        float base_dist,
        int zoom)
    {
        cam.dir.x = -sinf(phi) * sinf(theta);
        cam.dir.y = -cosf(theta);
        cam.dir.z = -cosf(phi) * sinf(theta);

        cam.right.x = cosf(phi);
        cam.right.y = 0.0f;
        cam.right.z = -sinf(phi);

        cam.up.x = -sinf(phi) * cosf(theta);
        cam.up.y = sinf(theta);
        cam.up.z = -cosf(phi) * cosf(theta);

        const float dist = base_dist * powf(0.95f, static_cast<float>(zoom));
        cam.pos.x = -cam.dir.x * dist;
        cam.pos.y = -cam.dir.y * dist;
        cam.pos.z = -cam.dir.z * dist;
    }

    // Ray to sphere intersection
    inline bool isect(const Ray &ray, const Sphere &sphere, Isect_info& isect_info)
    {
        mi::Float32_3 oc = ray.p0 - sphere.center;
        float b = 2.f * dot(oc, ray.dir);
        float c = dot(oc, oc) - sphere.radius * sphere.radius;
        float disc = b * b - 4.f * c;

        // no intersection
        if (disc <= 0.f)
            return false;

        disc = sqrtf(disc);

        //first hit
        float t = (-b - disc) * 0.5f;
        if (t <= 0.f)
        {
            //try second hit
            t = (-b + disc) * 0.5f;
            //sphere behind ray?
            if (t <= 0.f)
                return false;
        }

        isect_info.pos = ray.p0 + ray.dir*t;
        isect_info.normal = normalize(isect_info.pos - sphere.center);

        // compute uvw coordinates
        const float phi = atan2f(isect_info.normal.x, isect_info.normal.z);
        const float theta = acosf(isect_info.normal.y);

        isect_info.uvw.x = phi / Constants.PI + 1.f;
        isect_info.uvw.y = 1.f - theta / Constants.PI;
        isect_info.uvw.z = 0.f;

        // compute surface derivatives
        const float pi_rad = Constants.PI*sphere.radius;
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

    // build environment importance sampling data
    void build_alias_map()
    {
        const mi::Uint32 rx = env.map_size.x;
        const mi::Uint32 ry = env.map_size.y;
        env.alias_map = static_cast<Render_context::Environment::Alias_map *>(
            malloc(rx * ry * sizeof(Render_context::Environment::Alias_map)));
        float *importance_data = static_cast<float *>(malloc(rx * ry * sizeof(float)));
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
                    area * std::max(env.map_pixels[idx4], std::max(env.map_pixels[idx4 + 1], env.map_pixels[idx4 + 2]));
            }
        }

        // build alias map
        // create qs (normalized)
        size_t size = rx * ry;
        float sum = 0.0f;
        for (unsigned int i = 0; i < size; ++i)
            sum += importance_data[i];

        for (unsigned int i = 0; i < size; ++i)
            env.alias_map[i].q = (static_cast<float>(size) * importance_data[i] / sum);

        // create partition table
        unsigned int *partition_table = static_cast<unsigned int *>(
            malloc(size * sizeof(unsigned int)));
        unsigned int s = 0u, large = size;
        for (unsigned int i = 0; i < size; ++i)
            partition_table[(env.alias_map[i].q < 1.0f) ? (s++) : (--large)] = env.alias_map[i].alias = i;

        // create alias map
        for (s = 0; s < large && large < size; ++s)
        {
            const unsigned int j = partition_table[s], k = partition_table[large];
            env.alias_map[j].alias = k;
            env.alias_map[k].q += env.alias_map[j].q - 1.0f;
            large = (env.alias_map[k].q < 1.0f) ? (large + 1u) : large;
        }

        free(partition_table);

        env.inv_integral = 1.0f / sum;
        free(importance_data);
    }

    // evaluate the environment map for a given ray direction
    inline mi::Float32_3 evaluate_environment(float& pdf, const mi::Float32_3& dir)
    {
        // use environment map?
        if (env.map.is_valid_interface())
        {
            const float u = atan2f(dir.z, dir.x) * (0.5f / Constants.PI) + 0.5f;
            const float v = acosf(fmax(fminf(-dir.y, 1.0f), -1.0f)) / Constants.PI;

            size_t x = mi::math::min(static_cast<mi::Uint32>(u * env.map_size.x), env.map_size.x - 1u);
            size_t y = mi::math::min(static_cast<mi::Uint32>(v * env.map_size.y), env.map_size.y - 1u);

            const float *pixel = env.map_pixels + ((y*env.map_size.x + x) * 4);

            pdf = std::max(pixel[0], std::max(pixel[1], pixel[2])) * env.inv_integral;

            return mi::Float32_3(pixel[0], pixel[1], pixel[2])*env.intensity;
        }
        else
        {
            pdf = 1.f;
            return env.color*env.intensity;
        }
    }

    // importance sampling the environment map
    mi::Float32_3 sample_environment(mi::Float32_3& light_dir, float &light_pdf, unsigned &seed)
    {
        mi::Float32_3 xi;
        xi.x = rnd(seed);
        xi.y = rnd(seed);
        xi.z = rnd(seed);

        // importance sample the environment using an alias map
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

        // uniformly sample spherical area of pixel
        const float u = static_cast<float>(px + xi_y) / static_cast<float>(env.map_size.x);
        const float phi = u * 2.0f * Constants.PI - Constants.PI;
        const float sin_phi = sinf(phi);
        const float cos_phi = cosf(phi);
        const float step_theta = Constants.PI / static_cast<float>(env.map_size.y);
        const float theta0 = static_cast<float>(py)* step_theta;
        const float cos_theta = cosf(theta0) * (1.0f - xi.z) + cosf(theta0 + step_theta) * xi.z;
        const float theta = acosf(cos_theta);
        const float sin_theta = sinf(theta);
        light_dir = mi::Float32_3(cos_phi * sin_theta, -cos_theta, sin_phi * sin_theta);

        // lookup filtered beauty
        const float v = theta / Constants.PI;

        size_t x = mi::math::min(static_cast<mi::Uint32>(u * env.map_size.x), env.map_size.x - 1u);
        size_t y = mi::math::min(static_cast<mi::Uint32>(v * env.map_size.y), env.map_size.y - 1u);

        const float *pix = env.map_pixels + ((y*env.map_size.x + x) * 4);
        light_pdf = mi::math::max(pix[0], mi::math::max(pix[1], pix[2])) * env.inv_integral;
        return mi::Float32_3(pix[0], pix[1], pix[2])*env.intensity;
    }

    // sample scene lights (omni + environment map)
    mi::Float32_3 sample_lights(const mi::Float32_3 &pos, mi::Float32_3& light_dir, float& light_pdf, unsigned &seed)
    {
        float p_select_light = 1.0f;
        if (omni_light.intensity > 0.f)
        {
            // keep it simple and use either point light or environment light, each with the same
            // probability. If the environment factor is zero, we always use the point light
            // Note: see also miss shader
            p_select_light = env.intensity > 0.0f ? 0.5f : 1.0f;

            // in general, you would select the light depending on the importance of it
            // e.g. by incorporating their luminance

            // randomly select one of the lights
            if (rnd(seed) <= p_select_light)
            {
                light_pdf = Constants.DIRAC; // infinity

                // compute light direction and distance
                light_dir = omni_light.dir*omni_light.distance - pos;

                const float inv_distance2 = 1.0f / dot(light_dir, light_dir);
                light_dir *= sqrtf(inv_distance2);

                return omni_light.color *
                    (omni_light.intensity * inv_distance2 * 0.25f / (Constants.PI * p_select_light));
            }

            // probability to select the environment instead
            p_select_light = (1.0f - p_select_light);
        }

        // light from the environment map
        mi::Float32_3 radiance = sample_environment(light_dir, light_pdf, seed);

        // return radiance over pdf
        light_pdf *= p_select_light;
        return radiance / light_pdf;
    }

};


///////////////////////////////////////////////////////////////////////////////
// MDL Material Helper Functions
///////////////////////////////////////////////////////////////////////////////

// Creates an instance of the given material.
void create_material_instance(
    mi::neuraylib::IMdl_factory* mdl_factory,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
    mi::neuraylib::IMdl_execution_context* context,
    const char* material_name,
    const char* instance_name)
{
    // split module and material name
    std::string module_name, material_simple_name;
    if (!mi::examples::mdl::parse_cmd_argument_material_name(
        material_name, module_name, material_simple_name, true))
        exit_failure();

    // Load the module.
    mdl_impexp_api->load_module(transaction, module_name.c_str(), context);
    if (!print_messages(context))
        exit_failure("Loading module '%s' failed.", module_name.c_str());

    // Get the database name for the module we loaded
    mi::base::Handle<const mi::IString> module_db_name(
        mdl_factory->get_db_module_name(module_name.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
    if (!module)
        exit_failure("Failed to access the loaded module.");

    // Attach the material name
    std::string material_db_name
        = std::string(module_db_name->get_c_str()) + "::" + material_simple_name;
    material_db_name = mi::examples::mdl::add_missing_material_signature(
        module.get(), material_db_name);
    if (material_db_name.empty())
        exit_failure("Failed to find the material %s in the module %s.",
            material_simple_name.c_str(), module_name.c_str());

    // Get the material definition from the database
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
    if (!material_definition)
        exit_failure("Accessing definition '%s' failed.", material_db_name.c_str());

    // Create a material instance from the material definition with the default arguments.
    // Assuming the material has defaults for all parameters.
    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(
        material_definition->create_function_call(0, &result));
    if (result != 0)
        exit_failure("Instantiating '%s' failed.", material_db_name.c_str());

    transaction->store(material_instance.get(), instance_name);
}

// Compiles the given material instance in the given compilation modes and stores it in the DB.
void compile_material_instance(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_execution_context* context,
    const char* instance_name,
    const char* compiled_material_name,
    bool class_compilation)
{
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance(
        transaction->access<mi::neuraylib::IMaterial_instance>(instance_name));
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance->create_compiled_material(flags, context));
    check_success(print_messages(context));

    transaction->store(compiled_material.get(), compiled_material_name);
}

// Generate and execute native CPU code for a subexpression of a given compiled material.
void generate_native(
    Render_context& render_context,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_backend_api* mdl_backend_api,
    mi::neuraylib::IMdl_execution_context* context,
    const char* compiled_material_name,
    bool use_custom_tex_runtime,
    bool use_adapt_normal,
    bool enable_derivatives)
{
    Timing timing("generate target code");

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        transaction->access<mi::neuraylib::ICompiled_material>(compiled_material_name));

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

    // has material a constant cutout opacity?
    render_context.cutout.is_constant =
        compiled_material->get_cutout_opacity(&render_context.cutout.constant_opacity);

    // has material a constant thin_walled property?
    mi::base::Handle<mi::neuraylib::IExpression const> thin_walled(
        compiled_material->lookup_sub_expression("thin_walled"));

    render_context.thin_walled.is_constant = false;
    render_context.thin_walled.is_thin_walled = false;
    if (thin_walled->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
        mi::base::Handle<mi::neuraylib::IExpression_constant const> thin_walled_const(
            thin_walled->get_interface<mi::neuraylib::IExpression_constant const>());
        mi::base::Handle<mi::neuraylib::IValue_bool const> thin_walled_bool(
            thin_walled_const->get_value<mi::neuraylib::IValue_bool>());

        render_context.thin_walled.is_constant = true;
        render_context.thin_walled.is_thin_walled = thin_walled_bool->get_value();
    }

    // back faces could be different for thin walled materials
    bool need_backface_bsdf = false;
    bool need_backface_edf = false;
    bool need_backface_emission_intensity = false;
    if (!render_context.thin_walled.is_constant || render_context.thin_walled.is_thin_walled)
    {
        // first, backfaces dfs are only considered for thin_walled materials

        // second, we only need to generate new code if surface and backface are different
        need_backface_bsdf =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_SCATTERING) !=
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_SCATTERING);
        need_backface_edf =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_EDF_EMISSION) !=
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_EDF_EMISSION);
        need_backface_emission_intensity =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_INTENSITY) !=
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_INTENSITY);

        // third, either the bsdf or the edf need to be non-default (black)
        mi::base::Handle<mi::neuraylib::IExpression const> scattering_expr(
            compiled_material->lookup_sub_expression("backface.scattering"));
        mi::base::Handle<mi::neuraylib::IExpression const> emission_expr(
            compiled_material->lookup_sub_expression("backface.emission.emission"));

        if (scattering_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT &&
            emission_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
        {
            mi::base::Handle<mi::neuraylib::IExpression_constant const> scattering_expr_constant(
                scattering_expr->get_interface<mi::neuraylib::IExpression_constant>());
            mi::base::Handle<mi::neuraylib::IValue const> scattering_value(
                scattering_expr_constant->get_value());

            mi::base::Handle<mi::neuraylib::IExpression_constant const> emission_expr_constant(
                emission_expr->get_interface<mi::neuraylib::IExpression_constant>());
            mi::base::Handle<mi::neuraylib::IValue const> emission_value(
                emission_expr_constant->get_value());

            if (scattering_value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF &&
                emission_value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF)
            {
                need_backface_bsdf = false;
                need_backface_edf = false;
                need_backface_emission_intensity = false;
            }
        }
    }

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
#endif

    mi::base::Handle<mi::neuraylib::IMdl_backend> be_native(
        mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_NATIVE));

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
#endif

    check_success(be_native->set_option("num_texture_spaces", "1") == 0);

    if (render_context.render_auxiliary)
        check_success(be_native->set_option("enable_auxiliary", "on") == 0);

    if (use_custom_tex_runtime)
        check_success(be_native->set_option("use_builtin_resource_handler", "off") == 0);

    if (enable_derivatives)
        check_success(be_native->set_option("texture_runtime_with_derivs", "on") == 0);

    if (use_adapt_normal)
        check_success(be_native->set_option("use_renderer_adapt_normal", "on") == 0);

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();
#endif

    mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(
        be_native->create_link_unit(transaction, context));
    check_success(print_messages(context));

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t6 = std::chrono::steady_clock::now();
#endif

    // select expressions to generate code for
    std::vector<mi::neuraylib::Target_function_description> descs;
    descs.push_back(mi::neuraylib::Target_function_description("surface.scattering"));
    descs.push_back(mi::neuraylib::Target_function_description("surface.emission.emission"));
    descs.push_back(mi::neuraylib::Target_function_description("surface.emission.intensity"));

    size_t backface_scattering_index = ~0;
    if (need_backface_bsdf)
    {
        backface_scattering_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("backface.scattering"));
    }

    size_t backface_edf_index = ~0;
    if (need_backface_edf)
    {
        backface_edf_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("backface.emission.emission"));
    }

    size_t backface_emission_intensity_index = ~0;
    if (need_backface_emission_intensity)
    {
        backface_emission_intensity_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("backface.emission.intensity"));
    }

    size_t cutout_opacity_desc_index = ~0;
    if (!render_context.cutout.is_constant)
    {
        cutout_opacity_desc_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("geometry.cutout_opacity"));
    }

    size_t thin_walled_desc_index = ~0;
    if (!render_context.thin_walled.is_constant)
    {
        thin_walled_desc_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("thin_walled"));
    }

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();
#endif

    // add the material to the link unit
    link_unit->add_material(
        compiled_material.get(),
        descs.data(), descs.size(),
        context);
    check_success(print_messages(context));

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t8 = std::chrono::steady_clock::now();
#endif

    // translate link unit
    mi::base::Handle<const mi::neuraylib::ITarget_code> code_native(
        be_native->translate_link_unit(link_unit.get(), context));
    check_success(print_messages(context));
    check_success(code_native);

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t9 = std::chrono::steady_clock::now();
#endif

    // update render context
    render_context.target_code = code_native;
    render_context.surface_bsdf_function_index = descs[0].function_index;
    render_context.surface_edf_function_index = descs[1].function_index;
    render_context.surface_emission_intensity_function_index = descs[2].function_index;

    render_context.backface_bsdf_function_index = need_backface_bsdf
        ? descs[backface_scattering_index].function_index : descs[0].function_index;

    render_context.backface_edf_function_index = need_backface_edf
        ? descs[backface_edf_index].function_index : descs[1].function_index;

    render_context.backface_emission_intensity_function_index = need_backface_emission_intensity
        ? descs[backface_emission_intensity_index].function_index : descs[2].function_index;

    render_context.cutout_opacity_function_index = !render_context.cutout.is_constant
        ? descs[cutout_opacity_desc_index].function_index : ~0;

    render_context.thin_walled_function_index = !render_context.thin_walled.is_constant
        ? descs[thin_walled_desc_index].function_index : ~0;

#ifdef ADD_EXTRA_TIMERS
    std::chrono::steady_clock::time_point t10 = std::chrono::steady_clock::now();
#endif

#ifdef ADD_EXTRA_TIMERS
    std::chrono::duration<double> et = t10 - t1;
    printf("GTC |||| Total time                 : %f seconds.\n", et.count());

    et = t2 - t1;
    printf("GTC | Compiled material DB          : %f seconds.\n", et.count());

    et = t3 - t2;
    printf("GTC | Mateial properties inspection : %f seconds.\n", et.count());

    et = t4 - t3;
    printf("GTC | Native backend                : %f seconds.\n", et.count());

    et = t5 - t4;
    printf("GTC | Material flags                : %f seconds.\n", et.count());

    et = t6 - t5;
    printf("GTC | Create link unit              : %f seconds.\n", et.count());

    et = t7 - t6;
    printf("GTC | Material expressions selection: %f seconds.\n", et.count());

    et = t8 - t7;
    printf("GTC | Add material to link unit     : %f seconds.\n", et.count());

    et = t9 - t8;
    printf("GTC | Translate link unit           : %f seconds.\n", et.count());

    et = t10 - t9;
    printf("GTC | RC update                     : %f seconds.\n", et.count());
#endif
}

// Prepare the textures for our own texture runtime.
bool prepare_textures(
    std::vector<Texture>& textures,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IImage_api* image_api,
    const mi::neuraylib::ITarget_code* target_code)
{
    for (mi::Size i = 1 /*skip invalid texture*/; i < target_code->get_texture_count(); ++i)
    {
        mi::base::Handle<const mi::neuraylib::ITexture> texture(
            transaction->access<const mi::neuraylib::ITexture>(
                target_code->get_texture(i)));
        mi::base::Handle<const mi::neuraylib::IImage> image(
            transaction->access<mi::neuraylib::IImage>(texture->get_image()));
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
        char const* image_type = image->get_type(0, 0);

        if (image->is_uvtile() || image->is_animated()) {
            std::cerr << "The example does not support uvtile and/or animated textures!" << std::endl;
            return false;
        }

        // For simplicity, the texture access functions are only implemented for float4 and gamma
        // is pre-applied here (all images are converted to linear space).

        // Convert to linear color space if necessary
        if (texture->get_effective_gamma(0, 0) != 1.0f) {
            // Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.
            mi::base::Handle<mi::neuraylib::ICanvas> gamma_canvas(
                image_api->convert(canvas.get(), "Color"));
            gamma_canvas->set_gamma(texture->get_effective_gamma(0, 0));
            image_api->adjust_gamma(gamma_canvas.get(), 1.0f);
            canvas = gamma_canvas;
        }
        else if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0) {
            // Convert to expected format
            canvas = image_api->convert(canvas.get(), "Color");
        }
        textures.push_back(Texture(canvas));
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Recursive raytracing
///////////////////////////////////////////////////////////////////////////////
void trace_ray(std::vector<mi::Float32_3> &vp_sample, Render_context &rc, Render_context::Ray &ray, unsigned seed)
{
    if (ray.level >= rc.max_ray_length)
        return;

    ray.level++;

    Isect_info isect_info;

    // ray hits sphere?
    if (rc.isect(ray, rc.sphere, isect_info))
    {
        // update material shader state
        rc.shading_state.position = isect_info.pos;
        rc.shading_state.normal = ray.is_inside ? -isect_info.normal : isect_info.normal;
        rc.shading_state.geom_normal = rc.shading_state.normal;
        rc.shading_state.text_coords = &isect_info.uvw;
        rc.shading_state.tangent_u = &isect_info.tan_u;
        rc.shading_state.tangent_v = &isect_info.tan_v;

        // evaluate material cutout opacity
        float cutout_opacity = rc.cutout.constant_opacity;
        if (!rc.cutout.is_constant)
        {
            rc.target_code->execute(
                rc.cutout_opacity_function_index,
                reinterpret_cast<mi::neuraylib::Shading_state_material&>(rc.shading_state),
                rc.tex_handler,
                /*arg_block_data=*/ nullptr,
                &cutout_opacity);
        }

        // it's the surface cutted out?. Then skip the surface and send a ray through
        if (cutout_opacity < rnd(seed))
        {
            ray.p0 = isect_info.pos;
            ray.offset_ray(ray.is_inside ? isect_info.normal : -isect_info.normal);
            ray.is_inside = !ray.is_inside;
            ray.level--;
            trace_ray(vp_sample, rc, ray, seed);
        }
        else
        {
            // evaluate thin_walled state
            bool is_thin_walled = rc.thin_walled.is_thin_walled;
            if (!rc.thin_walled.is_constant)
            {
                rc.target_code->execute(
                    rc.thin_walled_function_index,
                    reinterpret_cast<mi::neuraylib::Shading_state_material&>(rc.shading_state),
                    rc.tex_handler,
                    /*arg_block_data=*/ nullptr,
                    &is_thin_walled);
            }

            // evaluate material surface emission contribution
            {
                // restore material shader state normal
                rc.shading_state.normal = ray.is_inside ?
                    -isect_info.normal : isect_info.normal;

                uint64_t edf_function_index = ray.is_inside ? rc.backface_edf_function_index : rc.surface_edf_function_index;
                // shader initialization for the current hit point
                rc.target_code->execute_bsdf_init(
                    edf_function_index,
                    rc.shading_state,
                    rc.tex_handler,
                    nullptr);

                mi::neuraylib::Edf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;
                eval_data.k1 = -ray.dir;

                // evaluate material surface edf
                rc.target_code->execute_edf_evaluate(
                    edf_function_index + 2, // edf_function_index corresponds to 'init'
                                                       // edf_function_index+2 to 'evaluate'
                    &eval_data,
                    rc.shading_state,
                    rc.tex_handler,
                    /*arg_block_data=*/ nullptr);

                // emission contribution is only valid for positive pdf
                if (eval_data.pdf > 1.e-6f)
                {
                    mi::Float32_3 intensity(1.f);
                    rc.target_code->execute(
                        rc.surface_emission_intensity_function_index,
                        reinterpret_cast<mi::neuraylib::Shading_state_material&>(rc.shading_state),
                        rc.tex_handler,
                        /*arg_block_data=*/ nullptr,
                        &intensity);

                    vp_sample[VPCH_ILLUM] += static_cast<mi::Float32_3>(eval_data.edf)*intensity*ray.weight;
                }
            }

            // restore material shader state normal
            rc.shading_state.normal = ray.is_inside ?
                -isect_info.normal : isect_info.normal;

            uint64_t surface_bsdf_function_index =
                ray.is_inside ? rc.backface_bsdf_function_index : rc.surface_bsdf_function_index;

            // shader initialization for the current hit point
            rc.target_code->execute_bsdf_init(
                surface_bsdf_function_index,
                rc.shading_state,
                rc.tex_handler,
                nullptr);

            // get auxiliarity data
            if (rc.render_auxiliary && ray.level == 1)
            {
                mi::neuraylib::Bsdf_auxiliary_data<mi::neuraylib::DF_HSM_NONE> aux_data;
                if (ray.is_inside && !is_thin_walled)
                {
                    aux_data.ior1 = mi::Float32_3(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
                    aux_data.ior2 = mi::Float32_3(1.0f);
                }
                else
                {
                    aux_data.ior1 = mi::Float32_3(1.0f);
                    aux_data.ior2 = mi::Float32_3(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
                }
                aux_data.k1 = -ray.dir;

                rc.target_code->execute_bsdf_auxiliary(
                    surface_bsdf_function_index + 4,    // bsdf_function_index corresponds to 'init'
                                                        // bsdf_function_index+4 to 'auxiliary'
                    &aux_data,
                    rc.shading_state,
                    rc.tex_handler,
                    nullptr);

                vp_sample[VPCH_ALBEDO] = aux_data.albedo;
                vp_sample[VPCH_NORMAL] = aux_data.normal;
            }

            // evaluate scene lights contribution
            mi::Float32_3 light_dir;
            float light_pdf = 0.f;
            mi::Float32_3 radiance_over_pdf = rc.sample_lights(isect_info.pos, light_dir, light_pdf, seed);

            if (ray.level < rc.max_ray_length && light_pdf != 0.0f && ((dot(rc.shading_state.normal, light_dir) > 0.f) != ray.is_inside))
            {
                mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> eval_data;
                if (ray.is_inside && !is_thin_walled)
                {
                    eval_data.ior1 = mi::Float32_3(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
                    eval_data.ior2 = mi::Float32_3(1.0f);
                }
                else
                {
                    eval_data.ior1 = mi::Float32_3(1.0f);
                    eval_data.ior2 = mi::Float32_3(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
                }

                eval_data.k1 = -ray.dir;
                eval_data.k2 = light_dir;
                eval_data.bsdf_diffuse = mi::Float32_3(0.f);
                eval_data.bsdf_glossy = mi::Float32_3(0.f);

                // evaluate material surface bsdf
                rc.target_code->execute_bsdf_evaluate(
                    surface_bsdf_function_index + 2,    // bsdf_function_index corresponds to 'init'
                                                        // bsdf_function_index+2 to 'evaluate'
                    &eval_data,
                    rc.shading_state,
                    rc.tex_handler,
                    /*arg_block_data=*/ nullptr);

                if (eval_data.pdf > 1.e-6f)
                {
                    const float mis_weight = (light_pdf == Constants.DIRAC)
                        ? 1.0f : light_pdf / (light_pdf + eval_data.pdf);

                    vp_sample[VPCH_ILLUM] += (eval_data.bsdf_diffuse + eval_data.bsdf_glossy)*(radiance_over_pdf*ray.weight)*mis_weight;
                }
            }

            // sample material bsdf contribution
            {
                mi::neuraylib::Bsdf_sample_data sample_data;  // input/output data for sample
                if (ray.is_inside && !is_thin_walled)
                {
                    sample_data.ior1 = mi::Float32_3(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
                    sample_data.ior2 = mi::Float32_3(1.0f);
                }
                else
                {
                    sample_data.ior1 = mi::Float32_3(1.0f);
                    sample_data.ior2 = mi::Float32_3(MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR);
                }
                sample_data.k1 = -ray.dir;                    // outgoing direction
                sample_data.xi.x = rnd(seed);
                sample_data.xi.y = rnd(seed);
                sample_data.xi.z = rnd(seed);
                sample_data.xi.w = rnd(seed);

                rc.target_code->execute_bsdf_sample(
                    surface_bsdf_function_index + 1,         // bsdf_function_index corresponds to 'init'
                                                             // bsdf_function_index+1 to 'sample'
                    &sample_data,   // input/output
                    rc.shading_state,
                    rc.tex_handler,
                    /*arg_block_data=*/ nullptr);

                if (sample_data.event_type != mi::neuraylib::BSDF_EVENT_ABSORB)
                {
                    if ((sample_data.event_type & mi::neuraylib::BSDF_EVENT_SPECULAR) != 0)
                        ray.last_pdf = -1.0f;
                    else
                        ray.last_pdf = sample_data.pdf;

                    // there is a scattering event, trace either the reflection or transmision ray
                    ray.weight *= static_cast<mi::Float32_3>(sample_data.bsdf_over_pdf);
                    ray.p0 = isect_info.pos;
                    ray.dir = normalize(sample_data.k2);

                    // medium change?
                    if (sample_data.event_type&mi::neuraylib::BSDF_EVENT_TRANSMISSION)
                    {
                        ray.offset_ray(-mi::Float32_3(rc.shading_state.geom_normal));
                        ray.is_inside = !ray.is_inside;
                    }
                    else
                    {
                        ray.offset_ray(rc.shading_state.geom_normal);
                    }

                    std::vector<mi::Float32_3> scat_color = {mi::Float32_3(0.f)};
                    trace_ray(scat_color, rc, ray, seed);
                    vp_sample[VPCH_ILLUM] += scat_color[VPCH_ILLUM];
                }
            }
        }
    }
    // ray hits environment
    else
    {
        float pdf;
        vp_sample[VPCH_ILLUM] = rc.evaluate_environment(pdf, ray.dir)*ray.weight;

        // account multi importance sampling for environment
        if (ray.level > 1 && ray.last_pdf > 0.f)
        {
            // point light selection probability
            if (rc.omni_light.intensity > 0.f)
                pdf *= 0.5f;

            vp_sample[VPCH_ILLUM] *= ray.last_pdf / (ray.last_pdf + pdf);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Scene Rendering
///////////////////////////////////////////////////////////////////////////////

void render_scene(
    Render_context rc,
    size_t frame_nb,
    VP_buffers *vp_buffers,
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
        // random sequence initialization
        unsigned seed = tea(16, y*width, frame_nb);

        for (size_t x = 0; x < width; ++x, ++vp_idx)
        {
            std::vector<mi::Float32_3> vp_sample =
                { mi::Float32_3(0.f), mi::Float32_3(0.f), mi::Float32_3(0.f) };

            float x_rnd = rnd(seed);
            float y_rnd = rnd(seed);

            mi::Float32_2 screen_pos(
                (x + x_rnd)*rc.cam.inv_res.x,
                (y + y_rnd)*rc.cam.inv_res.y);

            float r = (2.0f * screen_pos.x - 1.0f);
            float u = (2.0f * screen_pos.y - 1.0f);

            ray.p0 = rc.cam.pos;
            ray.dir = normalize(rc.cam.dir * rc.cam.focal +
                rc.cam.right * r + rc.cam.up * (rc.cam.aspect * u));
            ray.weight = 1.f;
            ray.is_inside = false;
            ray.level = 0;
            ray.last_pdf = -1.f;

            //trace camera ray
            trace_ray(vp_sample, rc, ray, seed);

            // update progressive rendering viewport buffer
            if (frame_nb == 1)
            {
                vp_buffers->accum_buffer[vp_idx] = vp_sample[VPCH_ILLUM];

                if(rc.render_auxiliary)
                {
                    vp_buffers->albedo_buffer[vp_idx] = vp_sample[VPCH_ALBEDO];
                    vp_buffers->normal_buffer[vp_idx] = vp_sample[VPCH_NORMAL];
                }
            }
            else
            {
                vp_buffers->accum_buffer[vp_idx] =
                    (vp_buffers->accum_buffer[vp_idx] * static_cast<float>(frame_nb - 1) + vp_sample[VPCH_ILLUM]) * (1.f / frame_nb);
                vp_sample[VPCH_ILLUM] = vp_buffers->accum_buffer[vp_idx];

                if (rc.render_auxiliary)
                {
                    vp_buffers->albedo_buffer[vp_idx] =
                        (vp_buffers->albedo_buffer[vp_idx] * static_cast<float>(frame_nb - 1) + vp_sample[VPCH_ALBEDO]) * (1.f / frame_nb);
                    vp_sample[VPCH_ALBEDO] = vp_buffers->albedo_buffer[vp_idx];

                    vp_buffers->normal_buffer[vp_idx] =
                        (vp_buffers->normal_buffer[vp_idx] * static_cast<float>(frame_nb - 1) + vp_sample[VPCH_NORMAL]) * (1.f / frame_nb);
                    vp_sample[VPCH_NORMAL] = vp_buffers->normal_buffer[vp_idx];
                }
            }

            if (dst)
            {
                // apply gamma correction
                vp_sample[VPCH_ILLUM].x = powf(vp_sample[VPCH_ILLUM].x, 1.f / 2.2f);
                vp_sample[VPCH_ILLUM].y = powf(vp_sample[VPCH_ILLUM].y, 1.f / 2.2f);
                vp_sample[VPCH_ILLUM].z = powf(vp_sample[VPCH_ILLUM].z, 1.f / 2.2f);

                // write final pixel
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

// Save current result image to disk
static void save_screenshot(
    const mi::Float32_3* image_buffer,
    const unsigned int width,
    const unsigned int height,
    const std::string &filename,
    mi::base::Handle<mi::neuraylib::IImage_api> image_api,
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api)
{
    mi::base::Handle<mi::neuraylib::ICanvas> canvas(
        image_api->create_canvas("Float32<3>", width, height));
    mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile());
    memcpy(tile->get_data(), image_buffer, width*height * sizeof(mi::Float32_3));
    mdl_impexp_api->export_canvas(filename.c_str(), canvas.get(), 100U, true);
}

///////////////////////////////////////////////////////////////////////////////
// Main Function
///////////////////////////////////////////////////////////////////////////////

// Print command line usage to console and terminate the application.
void usage(char const *prog_name)
{
    std::cout
        << "Usage: " << prog_name << " [options] [<material_name>]\n"
        << "Options:\n"
        << "  -h|--help              print this text and exit\n"
        << "  -v|--version           print the MDL SDK version string and exit\n"
        << "  --res <x> <y>          resolution (default: 700x520)\n"
        << "  --hdr <filename>       environment map\n"
        << "                         (default: nvidia/sdk_examples/resources/environment.hdr)\n"
        << "  --cc                   use class compilation\n"
        << "  --cr                   use custom texture runtime\n"
        << "  --an                   use adapt normal function\n"
        << "  --nogui                don't open interactive display\n"
        << "  --spp                  samples per pixel (default: 100) for output image when nogui\n"
        << "  -o <outputfile>        image file to write result to\n"
        << "                         (default: example_native.png)\n"
        << "  -oaux                  output albedo and normal auxiliary buffers.\n"
        << "  -p|--mdl_path <path>   mdl search path, can occur multiple times\n"
        << "\n"
        << "Viewport controls:\n"
        << "  Mouse               Camera rotation, zoom\n"
        << "  Arrow keys, (+/-)   Omni-light rotation, intensity\n"
        << "  CTRL + (+/-)        Environment intensity\n"
        << "  ENTER               Screenshot\n"
        << std::endl;

    exit_failure();
}

int MAIN_UTF8(int argc, char *argv[])
{
    // Parse command line options
    Render_context rc;
    Options options;
    mi::examples::mdl::Configure_options configure_options;
    configure_options.add_example_search_path = false;

    bool print_version_and_exit = false;

    for (int i = 1; i < argc; ++i)
    {
        char const *opt = argv[i];
        if (opt[0] == '-')
        {
            if (strcmp(opt, "--nogui") == 0)
            {
                options.no_gui = true;
            }
            else if (strcmp(opt, "--spp") == 0 && i < argc - 1)
            {
                options.iterations = std::max(atoi(argv[++i]), 1);
            }
            else if (strcmp(opt, "-o") == 0 && i < argc - 1)
            {
                options.outputfile = argv[++i];
            }
            else if (strcmp(opt, "-oaux") == 0)
            {
                options.output_auxiliary = true;
            }
            else if (strcmp(opt, "--res") == 0 && i < argc - 2)
            {
                options.res_x = std::max(atoi(argv[++i]), 1);
                options.res_y = std::max(atoi(argv[++i]), 1);
            }
            else if (strcmp(opt, "--max_path_length") == 0 && i < argc - 1)
            {
                options.max_ray_length = std::max(atoi(argv[++i]), 0);
            }
            else if (strcmp(opt, "--hdr") == 0 && i < argc - 1)
            {
                options.env_map = argv[++i];
            }
            else if (strcmp(opt, "--hdr_scale") == 0 && i < argc - 1)
            {
                options.env_scale = static_cast<float>(atof(argv[++i]));
            }
            else if (strcmp(opt, "-f") == 0 && i < argc - 1)
            {
                options.cam_fov = static_cast<float>(atof(argv[++i]));
            }
            else if (strcmp(opt, "-p") == 0 && i < argc - 3)
            {
                options.cam_pos.x = static_cast<float>(atof(argv[++i]));
                options.cam_pos.y = static_cast<float>(atof(argv[++i]));
                options.cam_pos.z = static_cast<float>(atof(argv[++i]));
            }
            else if (strcmp(opt, "-l") == 0 && i < argc - 6)
            {
                options.light_pos.x = static_cast<float>(atof(argv[++i]));
                options.light_pos.y = static_cast<float>(atof(argv[++i]));
                options.light_pos.z = static_cast<float>(atof(argv[++i]));
                options.light_intensity.x = static_cast<float>(atof(argv[++i]));
                options.light_intensity.y = static_cast<float>(atof(argv[++i]));
                options.light_intensity.z = static_cast<float>(atof(argv[++i]));
            }
            else if (strcmp(opt, "--cc") == 0)
            {
                options.use_class_compilation = true;
            }
            else if (strcmp(opt, "--cr") == 0)
            {
                options.use_custom_tex_runtime = true;
            }
            else if (strcmp(opt, "--an") == 0)
            {
                options.use_adapt_normal = true;
            }
            else if ((strcmp(opt, "--mdl_path") == 0 || strcmp(opt, "-p") == 0) &&
                i < argc - 1)
            {
                configure_options.additional_mdl_paths.push_back(argv[++i]);
            }
            else if (strcmp(opt, "-v") == 0 || strcmp(opt, "--version") == 0)
            {
                print_version_and_exit = true;
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
            options.material_name = opt;
        }
    }

    // Use default material, if none was provided via command line
    configure_options.add_example_search_path = true;

    if (options.material_name.empty())
        options.material_name = "::nvidia::sdk_examples::tutorials::example_df";

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Handle the --version flag
    if (print_version_and_exit)
    {
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

    // Enable/Disable auxiliary buffers
    rc.render_auxiliary = options.output_auxiliary;

    {
        // Create a transaction
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());

        // Acquire image API needed to prepare the textures
        mi::base::Handle<mi::neuraylib::IImage_api> image_api(
            neuray->get_api_component<mi::neuraylib::IImage_api>());

        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());

        {
            mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(
                neuray->get_api_component<mi::neuraylib::IMdl_factory>());

            mi::base::Handle<mi::neuraylib::IMdl_backend_api> mdl_backend_api(
                neuray->get_api_component<mi::neuraylib::IMdl_backend_api>());

            mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
                mdl_factory->create_execution_context());

            // Load the MDL module and create a material instance
            std::string instance_name = "material instance";
            create_material_instance(
                mdl_factory.get(),
                transaction.get(),
                mdl_impexp_api.get(),
                context.get(),
                options.material_name.c_str(),
                instance_name.c_str());

            // Compile the material instance in instance compilation mode
            std::string instance_compilation_name
                = std::string("instance compilation of ") + instance_name;
            // Compile the material instance
            std::string compilation_name
                = std::string("compilation of ") + instance_name;
            compile_material_instance(
                transaction.get(),
                context.get(),
                instance_name.c_str(),
                compilation_name.c_str(),
                options.use_class_compilation);

            // Generate target code for some material expression and update render context
            generate_native(
                rc,
                transaction.get(),
                mdl_backend_api.get(),
                context.get(),
                compilation_name.c_str(),
                options.use_custom_tex_runtime,
                options.use_adapt_normal,
                options.enable_derivatives);
        }

        // Setup custom texture handler, if requested
        std::vector<Texture>                  textures;
        Texture_handler                       tex_handler = { 0 };
        mi::neuraylib::Texture_handler_vtable tex_only_adapt_normal_vtable = { 0 };

        if (options.use_custom_tex_runtime)
        {
            check_success(prepare_textures(
                textures, transaction.get(), image_api.get(), rc.target_code.get()));

            tex_handler.vtable = &tex_vtable;
            tex_handler.num_textures = rc.target_code->get_texture_count() - 1;
            tex_handler.textures = textures.data();

            rc.tex_handler = &tex_handler;
        }
        else if (options.use_adapt_normal)
        {
            // only set the m_adapt_normal entry in the vtable of the texture handler object

            tex_only_adapt_normal_vtable.m_adapt_normal = adapt_normal;

            tex_handler.vtable = &tex_only_adapt_normal_vtable;

            rc.tex_handler = &tex_handler;
        }

        // create window context
        Window_context window_context;

        // setup render data
        // ------------------------------------------------------------------------
        size_t window_width = 0, window_height = 0;
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


#ifdef USE_PARALLEL_RENDERING
        // get number of physical/virtual threads available.
#ifdef MI_PLATFORM_WINDOWS
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        const int num_threads = sysinfo.dwNumberOfProcessors;
#elif MI_PLATFORM_MACOSX
        int num_threads;
        size_t len = sizeof(num_threads);
        sysctlbyname("hw.logicalcpu", &num_threads, &len, NULL, 0);
#else // LINUX // ARCH_64BIT
        const int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
        std::cout << "Rendering on " << num_threads << " threads.\n";
#endif // USE_PARALLEL_RENDERING

        // render options
        rc.max_ray_length = options.max_ray_length;

        // load/setup environment map
        window_context.env_intensity = rc.env.intensity = options.env_scale;

        mi::base::Handle<mi::neuraylib::IImage> image(
            transaction->create<mi::neuraylib::IImage>("Image"));
        check_success(image->reset_file(options.env_map.c_str()) == 0);

        rc.env.map = image->get_canvas(0, 0, 0);
        rc.env.map_size.x = rc.env.map->get_resolution_x();
        rc.env.map_size.y = rc.env.map->get_resolution_y();

        // Check, whether we need to convert the image
        char const *image_type = image->get_type(0, 0);
        if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0)
            rc.env.map = image_api->convert(rc.env.map.get(), "Color");

        rc.env.map_pixels = reinterpret_cast<const float*>(
                mi::base::make_handle(rc.env.map->get_tile())->get_data());
        rc.build_alias_map();

        // setup omni light
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

        rc.update_light(window_context.omni_phi, window_context.omni_theta, window_context.omni_intensity);

        // setup initial camera
        float base_dist = length(options.cam_pos);
        float theta, phi;

        const mi::Float32_3 inv_dir = normalize(options.cam_pos);
        phi = atan2f(inv_dir.x, inv_dir.z);
        theta = acosf(inv_dir.y);

        rc.cam.focal = 1.0f / tanf(options.cam_fov * Constants.PI / 360.f);
        rc.update_camera(phi, theta, base_dist, window_context.zoom);

        // render to image?
        if (options.no_gui)
        {
            window_width = options.res_x;
            window_height = options.res_y;

            frame_nb = 0;
            if (vp_buffers.accum_buffer)
                delete[] vp_buffers.accum_buffer;

            vp_buffers.accum_buffer = new mi::Float32_3[window_width*window_height];

            if(options.output_auxiliary)
            {
                if (vp_buffers.albedo_buffer)
                    delete[] vp_buffers.albedo_buffer;
                if (vp_buffers.normal_buffer)
                    delete[] vp_buffers.normal_buffer;

                vp_buffers.albedo_buffer = new mi::Float32_3[window_width*window_height];
                vp_buffers.normal_buffer = new mi::Float32_3[window_width*window_height];
            }

            // update camera parameters
            rc.cam.inv_res.x = 1.0f / static_cast<float>(window_width);
            rc.cam.inv_res.y = 1.0f / static_cast<float>(window_height);
            rc.cam.aspect = static_cast<float>(window_height)
                / static_cast<float>(window_width);

            {
                Timing timing("rendering");

                //render loop
                while (frame_nb < options.iterations)
                {
                    frame_nb++;
#ifdef USE_PARALLEL_RENDERING
                    // preparing render threads
                    std::vector<std::thread> threads;
                    size_t lpt = window_height / num_threads +
                        (window_height % num_threads != 0 ? 1 : 0); // lines per thread

                    // Launch render threads
                    for (int i = 0; i < num_threads; ++i)
                        threads.push_back(std::thread(render_scene, rc, frame_nb, &vp_buffers,
                            nullptr, lpt * i, lpt * (i + 1), window_width, window_height, 4));

                    // wait for threads to finish
                    for (int i = 0; i < num_threads; ++i)
                        threads[i].join();

                    threads.clear();
#else
                    render_scene(rc, frame_nb, &vp_buffers,
                        nullptr, 0, window_height, window_width, window_height, 4);
#endif
                }
            }

            // save screenshot
            save_screenshot(vp_buffers.accum_buffer, window_width, window_height,
                filename_base + filename_ext, image_api, mdl_impexp_api);

            if (options.output_auxiliary)
            {
                save_screenshot(vp_buffers.albedo_buffer, window_width, window_height,
                    filename_base + "_albedo" + filename_ext, image_api, mdl_impexp_api);
                save_screenshot(vp_buffers.normal_buffer, window_width, window_height,
                    filename_base + "_normal" + filename_ext, image_api, mdl_impexp_api);
            }
        }
        else // interactive renderer
        {
            // create the main window
            if (!glfwInit())
                exit(EXIT_FAILURE);

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

            mi::examples::mdl::GL_window::Description window_desc;
            window_desc.width = options.res_x;
            window_desc.height = options.res_y;
            window_desc.title = "MDL Native Rendering";
            window_desc.no_gui = false;

            mi::examples::mdl::GL_window gl_window(window_desc);

            gl_window.set_window_user_pointer(&window_context);
            gl_window.set_key_callback(Window_context::handle_key);
            gl_window.set_mouse_button_callback(Window_context::handle_mouse_button);
            gl_window.set_cursor_pos_callback(Window_context::handle_mouse_pos);
            gl_window.set_scroll_callback(Window_context::handle_scroll);

            if (GLEW_OK != glewInit())
                exit(EXIT_FAILURE);

            glfwSwapInterval(1);

            // create a display, this allows to render a buffer to screen
            mi::examples::mdl::GL_display gl_display(window_desc.width, window_desc.height);

            // setup GUI
            // ------------------------------------------------------------------------

            // init the GUI system in terms of styles and fonts
            mi::examples::gui::Root* gui = window_desc.no_gui ? nullptr : gl_window.get_gui();
            if (gui)
            {
                gui->initialize();
                // add panels and other controls here
            }

            // render loop
            while (gl_window.update())
            {
                // get the window size and resize the image if necessary
                if (window_width != gl_window.get_width() || window_height != gl_window.get_height())
                {
                    window_width = gl_window.get_width();
                    window_height = gl_window.get_height();

                    frame_nb = 0;
                    if (vp_buffers.accum_buffer)
                        delete[] vp_buffers.accum_buffer;

                    vp_buffers.accum_buffer = new mi::Float32_3[window_width*window_height];

                    if (options.output_auxiliary)
                    {
                        if (vp_buffers.albedo_buffer)
                            delete[] vp_buffers.albedo_buffer;
                        if (vp_buffers.normal_buffer)
                            delete[] vp_buffers.normal_buffer;

                        vp_buffers.albedo_buffer = new mi::Float32_3[window_width*window_height];
                        vp_buffers.normal_buffer = new mi::Float32_3[window_width*window_height];
                    }

                    // update camera parameters
                    rc.cam.inv_res.x = 1.0f / static_cast<float>(window_width);
                    rc.cam.inv_res.y = 1.0f / static_cast<float>(window_height);
                    rc.cam.aspect = static_cast<float>(window_height)
                        / static_cast<float>(window_width);
                }

                // handle key input events
                if (window_context.key_event && !ImGui::GetIO().WantCaptureMouse)
                {
                    // update environment
                    rc.env.intensity = window_context.env_intensity;

                    // Update light
                    rc.update_light(window_context.omni_phi, window_context.omni_theta, window_context.omni_intensity);
                }

                // handle save screenshot event
                if (window_context.save_sreenshot && !ImGui::GetIO().WantCaptureMouse)
                {
                    save_screenshot(vp_buffers.accum_buffer, window_width, window_height,
                        filename_base + filename_ext, image_api, mdl_impexp_api);

                    if (options.output_auxiliary)
                    {
                        save_screenshot(vp_buffers.albedo_buffer, window_width, window_height,
                            filename_base + "_albedo" + filename_ext, image_api, mdl_impexp_api);
                        save_screenshot(vp_buffers.normal_buffer, window_width, window_height,
                            filename_base + "_normal" + filename_ext, image_api, mdl_impexp_api);
                    }
                }

                // handle mouse input events
                if (window_context.mouse_button - 1 == GLFW_MOUSE_BUTTON_LEFT)
                {
                    // Only accept button press when not hovering GUI window
                    if (window_context.mouse_button_action == GLFW_PRESS &&
                        !ImGui::GetIO().WantCaptureMouse)
                    {
                        window_context.moving = true;
                        glfwGetCursorPos(gl_window.get_window(), &window_context.move_start_x, &window_context.move_start_y);
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

                    rc.update_camera(phi, theta, base_dist, window_context.zoom);
                }

                if (window_context.key_event || window_context.mouse_event)
                    frame_nb = 0;

                // Clear all events
                window_context.key_event = false;
                window_context.mouse_event = false;
                window_context.mouse_wheel_delta = 0;
                window_context.mouse_button = 0;
                window_context.save_sreenshot = false;

                ++frame_nb;
                gl_display.resize(window_width, window_height);

                // handle resize on the application side (resize rendering buffer, restart)
                if (gui)
                {
                    // begin a new frame for the GUI and update the controls
                    gui->new_frame();   // required even when the main GUI is not rendered
                    gui->update(/*transaction*/ nullptr);    // update GUI elements

                    // process events
                    mi::examples::gui::Event e = gui->process_event();
                    while (e.is_valid())
                    {
                        /* handle custom application events here */
                        e = gui->process_event();
                    }
                }

                // map the buffer, update the image data and un-map afterwards
                // make sure this is as fast as possible
                unsigned char* dst_image_data = gl_display.map();

#ifdef USE_PARALLEL_RENDERING
                // preparing render threads
                std::vector<std::thread> threads;
                size_t lpt = window_height / num_threads +
                    (window_height % num_threads != 0 ? 1 : 0); // lines per thread

                  //Launch render threads
                for (int i = 0; i < num_threads; ++i)
                    threads.push_back(std::thread(render_scene, rc, frame_nb, &vp_buffers,
                        dst_image_data, lpt*i, lpt*(i + 1), window_width, window_height, 4));

                // wait for threads to finish
                for (int i = 0; i < num_threads; ++i)
                    threads[i].join();

                threads.clear();
#else
                render_scene(rc, frame_nb, &vp_buffers,
                    dst_image_data, 0, window_height, window_width, window_height, 4);
#endif

                gl_display.unmap();

                // render the updated image to screen
                gl_display.update_display();

                // render GUI on top
                if (gui)
                    gui->render(nullptr);

                // finish the frame
                gl_window.present_back_buffer();
            }
            glfwTerminate();
        }


        // free environment image
        image = nullptr;

        transaction->commit();
    }

    // Uninitialize the render context
    rc.uninit();

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
