/******************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/execution_glsl/example_execution_glsl.cpp
//
// Introduces the execution of generated code for compiled materials for
// the GLSL backend and shows how to manually bake a material
// sub-expression to a texture.

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "example_shared.h"
#include "example_glsl_shared.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Enable this to dump the generated GLSL code to stdout.
//#define DUMP_GLSL

char const* vertex_shader_filename   = "example_execution_glsl.vert";
char const* fragment_shader_filename = "example_execution_glsl.frag";

// Command line options structure.
struct Options {
    // If true, no interactive display will be used.
    bool no_window;

    // If true, SSBO will be used (OpenGL 4.3 required).
    bool use_ssbo;

    // If true, MDL ::base noise functions will be remapped to cheap GLSL implementations.
    bool remap_noise_functions;

    // An result output file name for non-interactive mode.
    std::string outputfile;

    // The pattern number representing the combination of materials to display.
    int material_pattern;

    // The resolution of the display / image.
    unsigned res_x, res_y;

    // The constructor.
    Options()
        : no_window(false)
#if defined(MI_PLATFORM_MACOSX) || defined(MI_ARCH_ARM_64)
        , use_ssbo(false)
#else
        , use_ssbo(true)
#endif
        , remap_noise_functions(true)
        , outputfile("output.png")
        , material_pattern(7)
        , res_x(1024)
        , res_y(768)
    {
    }
};

// Struct representing a vertex of a scene object.
struct Vertex {
    mi::Float32_3_struct position;
    mi::Float32_2_struct tex_coord;
};

//------------------------------------------------------------------------------
//
// OpenGL code
//
//------------------------------------------------------------------------------

// Error callback for GLFW.
static void handle_glfw_error(int error_code, const char* description)
{
    std::cerr << "GLFW error (code: " << error_code << "): \"" << description << "\"\n";
}

// Initialize OpenGL and create a window with an associated OpenGL context.
static GLFWwindow *init_opengl(Options const &options)
{
    glfwSetErrorCallback(handle_glfw_error);

    // Initialize GLFW
    check_success(glfwInit());

    if (options.use_ssbo) {
        // SSBO requires GLSL 4.30
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    } else {
        // else GLSL 3.30 is sufficient
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    }
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Hide window in no-window mode
    if (options.no_window)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    // Create an OpenGL window and a context
    GLFWwindow *window = glfwCreateWindow(
        options.res_x, options.res_y,
        "MDL SDK GLSL Execution Example - Switch pattern with keys 1 - 7", nullptr, nullptr);
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

    // Enable VSync
    glfwSwapInterval(1);

    check_gl_success();

    return window;
}

// Generate GLSL source code for a function executing an MDL subexpression function
// selected by a given id.
static std::string generate_glsl_switch_func(
    const mi::base::Handle<const mi::neuraylib::ITarget_code>& target_code)
{
    // Note: The "State" struct must be in sync with the struct in example_execution_glsl.frag and
    //       the code generated by the MDL SDK (see dumped code when enabling DUMP_GLSL).

    std::string src =
        "#version 330 core\n"
        "struct State {\n"
        "    vec3 normal;\n"
        "    vec3 geom_normal;\n"
        "    vec3 position;\n"
        "    float animation_time;\n"
        "    vec3 text_coords[1];\n"
        "    vec3 tangent_u[1];\n"
        "    vec3 tangent_v[1];\n"
        "    int ro_data_segment_offset;\n"
        "    mat4 world_to_object;\n"
        "    mat4 object_to_world;\n"
        "    int object_id;\n"
        "    float meters_per_scene_unit;\n"
        "    int arg_block_offset;\n"
        "};\n"
        "\n"
        "int get_mdl_num_mat_subexprs() { return " +
        to_string(target_code->get_callable_function_count()) +
        "; }\n"
        "\n";

    std::string switch_func =
        "vec3 mdl_mat_subexpr(int id, State state) {\n"
        "    switch(id) {\n";

    // Create one switch case for each callable function in the target code
    for (size_t i = 0, num_target_codes = target_code->get_callable_function_count();
          i < num_target_codes;
          ++i)
    {
        std::string func_name(target_code->get_callable_function(i));

        // Add prototype declaration
        src += target_code->get_callable_function_prototype(
            i, mi::neuraylib::ITarget_code::SL_GLSL);
        src += '\n';

        switch_func += "        case " + to_string(i) + ": return " + func_name + "(state);\n";
    }

    switch_func +=
        "        default: return vec3(0);\n"
        "    }\n"
        "}\n";

    return src + "\n" + switch_func;
}

// Create the shader program with a fragment shader.
static GLuint create_shader_program(
    bool use_ssbo,
    bool remap_noise_functions,
    unsigned max_materials,
    unsigned max_textures,
    const mi::base::Handle<const mi::neuraylib::ITarget_code>& target_code)
{
    GLint success;

    GLuint program = glCreateProgram();

    add_shader(GL_VERTEX_SHADER,
        read_text_file(
            mi::examples::io::get_executable_folder() + "/" + vertex_shader_filename), program);

    std::stringstream sstr;
    sstr << (use_ssbo ? "#version 430 core\n" : "#version 330 core\n");
    sstr << "#define MAX_MATERIALS " << to_string(max_materials) << "\n";
    sstr << "#define MAX_TEXTURES "  << to_string(max_textures) << "\n";
    sstr << read_text_file(
        mi::examples::io::get_executable_folder() + "/" + fragment_shader_filename);
    add_shader(GL_FRAGMENT_SHADER, sstr.str() , program);

    std::string code(target_code->get_code());
    if (remap_noise_functions) {
        code.append(read_text_file(
            mi::examples::io::get_executable_folder() + "/" + "noise_no_lut.glsl"));
    }

    add_shader(GL_FRAGMENT_SHADER, code, program);

    // Generate GLSL switch function for the generated functions
    std::string glsl_switch_func = generate_glsl_switch_func(target_code);

#ifdef DUMP_GLSL
    std::cout << "Dumping GLSL code for the \"mdl_mat_subexpr\" switch function:\n\n"
        << glsl_switch_func << std::endl;
#endif

    add_shader(GL_FRAGMENT_SHADER, glsl_switch_func.c_str(), program);

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        dump_program_info(program, "Error linking the shader program: ");
        terminate();
    }

    glUseProgram(program);
    check_gl_success();

    return program;
}

// Create a quad filling the whole screen.
static GLuint create_quad(GLuint program, GLuint* vertex_buffer)
{
    static Vertex const vertices[6] = {
        { { -1.f, -1.f, 0.0f }, { 0.f, 0.f } },
        { {  1.f, -1.f, 0.0f }, { 1.f, 0.f } },
        { { -1.f,  1.f, 0.0f }, { 0.f, 1.f } },
        { {  1.f, -1.f, 0.0f }, { 1.f, 0.f } },
        { {  1.f,  1.f, 0.0f }, { 1.f, 1.f } },
        { { -1.f,  1.f, 0.0f }, { 0.f, 1.f } }
    };

    glGenBuffers(1, vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, *vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    // Get locations of vertex shader inputs
    GLint   pos_index              = glGetAttribLocation(program, "Position");
    GLint   tex_coord_index        = glGetAttribLocation(program, "TexCoord");
    check_success(pos_index >= 0 && tex_coord_index >= 0);

    glEnableVertexAttribArray(pos_index);
    glVertexAttribPointer(
        pos_index, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);

    glEnableVertexAttribArray(tex_coord_index);
    glVertexAttribPointer(
        tex_coord_index, 2, GL_FLOAT, GL_FALSE,
        sizeof(Vertex), reinterpret_cast<const GLvoid*>(sizeof(mi::Float32_3_struct)));

    check_gl_success();

    return vertex_array;
}


//------------------------------------------------------------------------------
//
// Material_opengl_context class
//
//------------------------------------------------------------------------------

// Helper class responsible for making textures and read-only data available to OpenGL
// by generating and managing a list of Material_data objects.
class Material_opengl_context
{
public:
    Material_opengl_context(GLuint program, bool use_ssbo)
    : m_program(program)
    , m_use_ssbo(use_ssbo)
    , m_next_storage_block_binding(0)
    {}

    // Free all acquired resources.
    ~Material_opengl_context();

    // Prepare the needed material data of the given target code.
    bool prepare_material_data(
        mi::base::Handle<mi::neuraylib::ITransaction>       transaction,
        mi::base::Handle<mi::neuraylib::IImage_api>         image_api,
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code);

    // Sets all collected material data in the OpenGL program.
    bool set_material_data(unsigned max_textures);

private:
    // Sets the read-only data segments in the current OpenGL program object.
    void set_mdl_readonly_data(mi::base::Handle<const mi::neuraylib::ITarget_code> target_code);

    // Prepare the texture identified by the texture_index for use by the texture access functions
    // in the OpenGL program.
    bool prepare_texture(
        mi::base::Handle<mi::neuraylib::ITransaction>       transaction,
        mi::base::Handle<mi::neuraylib::IImage_api>         image_api,
        mi::base::Handle<const mi::neuraylib::ITarget_code> code_ptx,
        mi::Size                                            texture_index,
        GLuint                                              texture_array);

private:
    // The OpenGL program to prepare.
    GLuint m_program;

    bool m_use_ssbo;

    std::vector<GLuint> m_texture_objects;

    std::vector<int> m_material_texture_starts;

    std::vector<GLuint> m_buffer_objects;
    GLuint m_next_storage_block_binding;
};

// Free all acquired resources.
Material_opengl_context::~Material_opengl_context()
{
    if (m_buffer_objects.size() > 0)
        glDeleteBuffers(GLsizei(m_buffer_objects.size()), &m_buffer_objects[0]);

    if (m_texture_objects.size() > 0)
        glDeleteTextures(GLsizei(m_texture_objects.size()), &m_texture_objects[0]);

    check_gl_success();
}

// Sets the read-only data segments in the current OpenGL program object.
void Material_opengl_context::set_mdl_readonly_data(
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code)
{
    mi::Size num_uniforms = target_code->get_ro_data_segment_count();
    if (num_uniforms == 0) return;

    if (m_use_ssbo) {
        // WITH SSBO
        size_t cur_buffer_offs = m_buffer_objects.size();
        m_buffer_objects.insert(m_buffer_objects.end(), num_uniforms, 0);

        glGenBuffers(GLsizei(num_uniforms), &m_buffer_objects[cur_buffer_offs]);

        for (mi::Size i = 0; i < num_uniforms; ++i) {
            mi::Size segment_size = target_code->get_ro_data_segment_size(i);
            char const *segment_data = target_code->get_ro_data_segment_data(i);

#ifdef DUMP_GLSL
            std::cout << "Dump ro segment data " << i << " \""
                << target_code->get_ro_data_segment_name(i) << "\" (size = "
                << segment_size << "):\n" << std::hex;

            for (int j = 0; j < 16 && j < segment_size; ++j) {
                std::cout << "0x" << (unsigned int)(unsigned char)segment_data[j] << ", ";
            }
            std::cout << std::dec << std::endl;
#endif

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_buffer_objects[cur_buffer_offs + i]);
            glBufferData(
                GL_SHADER_STORAGE_BUFFER, GLsizeiptr(segment_size), segment_data, GL_STATIC_DRAW);

            GLuint block_index = glGetProgramResourceIndex(
                m_program, GL_SHADER_STORAGE_BLOCK, target_code->get_ro_data_segment_name(i));
            glShaderStorageBlockBinding(m_program, block_index, m_next_storage_block_binding);
            glBindBufferBase(
                GL_SHADER_STORAGE_BUFFER,
                m_next_storage_block_binding,
                m_buffer_objects[cur_buffer_offs + i]);

            ++m_next_storage_block_binding;

            check_gl_success();
        }
    } else {
        // WITHOUT SSBO
        std::vector<char const*> uniform_names;
        for (mi::Size i = 0; i < num_uniforms; ++i) {
#ifdef DUMP_GLSL
            mi::Size segment_size = target_code->get_ro_data_segment_size(i);
            const char* segment_data = target_code->get_ro_data_segment_data(i);

            std::cout << "Dump ro segment data " << i << " \""
                << target_code->get_ro_data_segment_name(i) << "\" (size = "
                << segment_size << "):\n" << std::hex;

            for (int i = 0; i < 16 && i < segment_size; ++i) {
                std::cout << "0x" << (unsigned int)(unsigned char)segment_data[i] << ", ";
            }
            std::cout << std::dec << std::endl;
#endif

            uniform_names.push_back(target_code->get_ro_data_segment_name(i));
        }

        std::vector<GLuint> uniform_indices(num_uniforms, 0);
        glGetUniformIndices(
            m_program, GLsizei(num_uniforms), &uniform_names[0], &uniform_indices[0]);

        for (mi::Size i = 0; i < num_uniforms; ++i) {
            // uniforms may have been removed, if they were not used
            if (uniform_indices[i] == GL_INVALID_INDEX)
                continue;

            GLint uniform_type = 0;
            GLuint index = GLuint(uniform_indices[i]);
            glGetActiveUniformsiv(m_program, 1, &index, GL_UNIFORM_TYPE, &uniform_type);

#ifdef DUMP_GLSL
            std::cout << "Uniform type of " << uniform_names[i]
                << ": 0x" << std::hex << uniform_type << std::dec << std::endl;
#endif

            mi::Size segment_size = target_code->get_ro_data_segment_size(i);
            const char* segment_data = target_code->get_ro_data_segment_data(i);

            GLint uniform_location = glGetUniformLocation(m_program, uniform_names[i]);

            switch (uniform_type) {

// For bool, the data has to be converted to int, first
#define CASE_TYPE_BOOL(type, func, num)                            \
    case type: {                                                   \
        GLint *buf = new GLint[segment_size];                      \
        for (mi::Size j = 0; j < segment_size; ++j)                \
            buf[j] = GLint(segment_data[j]);                       \
        func(uniform_location, GLsizei(segment_size / num), buf);  \
        delete[] buf;                                              \
        break;                                                     \
    }

                CASE_TYPE_BOOL(GL_BOOL,      glUniform1iv, 1)
                CASE_TYPE_BOOL(GL_BOOL_VEC2, glUniform2iv, 2)
                CASE_TYPE_BOOL(GL_BOOL_VEC3, glUniform3iv, 3)
                CASE_TYPE_BOOL(GL_BOOL_VEC4, glUniform4iv, 4)

#define CASE_TYPE(type, func, num, elemtype)                                      \
    case type:                                                                    \
        func(uniform_location, GLsizei(segment_size/(num * sizeof(elemtype))),    \
            (const elemtype*)segment_data);                                       \
        break

                CASE_TYPE(GL_INT,             glUniform1iv, 1, GLint);
                CASE_TYPE(GL_INT_VEC2,        glUniform2iv, 2, GLint);
                CASE_TYPE(GL_INT_VEC3,        glUniform3iv, 3, GLint);
                CASE_TYPE(GL_INT_VEC4,        glUniform4iv, 4, GLint);
                CASE_TYPE(GL_FLOAT,           glUniform1fv, 1, GLfloat);
                CASE_TYPE(GL_FLOAT_VEC2,      glUniform2fv, 2, GLfloat);
                CASE_TYPE(GL_FLOAT_VEC3,      glUniform3fv, 3, GLfloat);
                CASE_TYPE(GL_FLOAT_VEC4,      glUniform4fv, 4, GLfloat);
                CASE_TYPE(GL_DOUBLE,          glUniform1dv, 1, GLdouble);
                CASE_TYPE(GL_DOUBLE_VEC2,     glUniform2dv, 2, GLdouble);
                CASE_TYPE(GL_DOUBLE_VEC3,     glUniform3dv, 3, GLdouble);
                CASE_TYPE(GL_DOUBLE_VEC4,     glUniform4dv, 4, GLdouble);

#define CASE_TYPE_MAT(type, func, num, elemtype)                                  \
    case type:                                                                    \
        func(uniform_location, GLsizei(segment_size/(num * sizeof(elemtype))),    \
            false, (const elemtype*)segment_data);                                \
        break

                CASE_TYPE_MAT(GL_FLOAT_MAT2_ARB,  glUniformMatrix2fv,   4,  GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT2x3,    glUniformMatrix2x3fv, 6,  GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT3x2,    glUniformMatrix3x2fv, 6,  GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT2x4,    glUniformMatrix2x4fv, 8,  GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT4x2,    glUniformMatrix4x2fv, 8,  GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT3_ARB,  glUniformMatrix3fv,   9,  GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT3x4,    glUniformMatrix3x4fv, 12, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT4x3,    glUniformMatrix4x3fv, 12, GLfloat);
                CASE_TYPE_MAT(GL_FLOAT_MAT4_ARB,  glUniformMatrix4fv,   16, GLfloat);
                CASE_TYPE_MAT(GL_DOUBLE_MAT2,     glUniformMatrix2dv,   4,  GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT2x3,   glUniformMatrix2x3dv, 6,  GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT3x2,   glUniformMatrix3x2dv, 6,  GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT2x4,   glUniformMatrix2x4dv, 8,  GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT4x2,   glUniformMatrix4x2dv, 8,  GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT3,     glUniformMatrix3dv,   9,  GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT3x4,   glUniformMatrix3x4dv, 12, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT4x3,   glUniformMatrix4x3dv, 12, GLdouble);
                CASE_TYPE_MAT(GL_DOUBLE_MAT4,     glUniformMatrix4dv,   16, GLdouble);

                default:
                    std::cerr << "Unsupported uniform type: 0x"
                        << std::hex << uniform_type << std::dec << std::endl;
                    terminate();
                    break;
            }

            check_gl_success();
        }
    }
}

// Prepare the texture identified by the texture_index for use by the texture access functions
// on the GPU.
bool Material_opengl_context::prepare_texture(
    mi::base::Handle<mi::neuraylib::ITransaction>       transaction,
    mi::base::Handle<mi::neuraylib::IImage_api>         image_api,
    mi::base::Handle<const mi::neuraylib::ITarget_code> code,
    mi::Size                                            texture_index,
    GLuint                                              texture_obj)
{
    // Get access to the texture data by the texture database name from the target code.
    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        transaction->access<mi::neuraylib::ITexture>(code->get_texture(texture_index)));
    mi::base::Handle<const mi::neuraylib::IImage> image(
        transaction->access<mi::neuraylib::IImage>(texture->get_image()));
    mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
    mi::Uint32 tex_width = canvas->get_resolution_x();
    mi::Uint32 tex_height = canvas->get_resolution_y();
    mi::Uint32 tex_layers = canvas->get_layers_size();
    char const *image_type = image->get_type(0, 0);

    if (image->is_uvtile() || image->is_animated()) {
        std::cerr << "The example does not support uvtile and/or animated textures!" << std::endl;
        return false;
    }
    if (tex_layers != 1) {
        std::cerr << "The example doesn't support layered images!" << std::endl;
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
    } else if (strcmp(image_type, "Color") != 0 && strcmp(image_type, "Float32<4>") != 0) {
        // Convert to expected format
        canvas = image_api->convert(canvas.get(), "Color");
    }

    // This example supports only 2D textures
    mi::neuraylib::ITarget_code::Texture_shape texture_shape =
        code->get_texture_shape(texture_index);
    if (texture_shape == mi::neuraylib::ITarget_code::Texture_shape_2d) {
        mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile());
        mi::Float32 const *data = static_cast<mi::Float32 const *>(tile->get_data());

        glBindTexture(GL_TEXTURE_2D, texture_obj);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, tex_width, tex_height, 0,  GL_RGBA, GL_FLOAT, data);

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    check_gl_success();

    return true;
}

// Prepare the needed material data of the given target code.
bool Material_opengl_context::prepare_material_data(
    mi::base::Handle<mi::neuraylib::ITransaction>       transaction,
    mi::base::Handle<mi::neuraylib::IImage_api>         image_api,
    mi::base::Handle<const mi::neuraylib::ITarget_code> target_code)
{
    // Handle the read-only data segments if necessary
    set_mdl_readonly_data(target_code);

    // Handle the textures if there are more than just the invalid texture
    size_t cur_tex_offs = m_texture_objects.size();
    m_material_texture_starts.push_back(GLuint(cur_tex_offs));

    mi::Size num_textures = target_code->get_texture_count();
    if (num_textures > 1) {
        m_texture_objects.insert(m_texture_objects.end(), num_textures - 1, 0);

        glGenTextures(GLsizei(num_textures - 1), &m_texture_objects[cur_tex_offs]);

        // Loop over all textures skipping the first texture,
        // which is always the MDL invalid texture
        for (mi::Size i = 1; i < num_textures; ++i) {
            if (!prepare_texture(
                    transaction, image_api, target_code,
                    i, m_texture_objects[cur_tex_offs + i - 1]))
                return false;
        }
    }

    return true;
}

// Sets all collected material data in the OpenGL program.
bool Material_opengl_context::set_material_data(unsigned max_textures)
{
    GLsizei total_textures = GLsizei(m_texture_objects.size());

    if (total_textures > GLsizei(max_textures)) {
        fprintf( stderr, "Number of required textures (%d) is not supported (max: %d)\n",
            total_textures, max_textures);
        return false;
    }

    if (m_use_ssbo) {
        if (glfwExtensionSupported("GL_ARB_bindless_texture")) {
            if (total_textures > 0) {
                std::vector<GLuint64> texture_handles;
                texture_handles.resize(total_textures);
                for (GLsizei i = 0; i < total_textures; ++i) {
                    texture_handles[i] = glGetTextureHandleARB(m_texture_objects[i]);
                    glMakeTextureHandleResidentARB(texture_handles[i]);
                }

                glUniformHandleui64vARB(
                    glGetUniformLocation(m_program, "material_texture_samplers_2d"),
                    total_textures,
                    &texture_handles[0]);

                glUniform1iv(
                    glGetUniformLocation(m_program, "material_texture_starts"),
                    GLsizei(m_material_texture_starts.size()),
                    &m_material_texture_starts[0]);
            }
        } else if (glfwExtensionSupported("GL_NV_bindless_texture")) {
            if (total_textures > 0) {
                std::vector<GLuint64> texture_handles;
                texture_handles.resize(total_textures);
                for (GLsizei i = 0; i < total_textures; ++i) {
                    texture_handles[i] = glGetTextureHandleNV(m_texture_objects[i]);
                    glMakeTextureHandleResidentNV(texture_handles[i]);
                }

                glUniformHandleui64vARB(
                    glGetUniformLocation(m_program, "material_texture_samplers_2d"),
                    total_textures,
                    &texture_handles[0]);

                glUniform1iv(
                    glGetUniformLocation(m_program, "material_texture_starts"),
                    GLsizei(m_material_texture_starts.size()),
                    &m_material_texture_starts[0]);
            }
        } else {
            fprintf(stderr, "Sample requires Bindless Textures, "
                "that are not supported by the current system.\n");
            return false;
        }
    }

    // Check for any errors. If you get an error, check whether MAX_TEXTURES and MAX_MATERIALS
    // in example_execution_glsl.frag still fit to your needs.
    return glGetError() == GL_NO_ERROR;
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
        bool use_ssbo,
        bool remap_noise_functions,
        mi::neuraylib::IMdl_impexp_api* mdl_impexp_api,
        mi::neuraylib::IMdl_backend_api* mdl_backend_api,
        mi::neuraylib::IMdl_factory* mdl_factory,
        mi::neuraylib::ITransaction* transaction);

    // Generates GLSL target code for a subexpression of a given material.
    // path is the path of the sub-expression.
    // fname is the function name in the generated code.
    bool add_material_subexpr(
        const std::string& qualified_module_name,
        const std::string& material_db_name,
        const char* path,
        const char* fname);

    // Generates GLSL target code for a subexpression of a given compiled material.
    mi::base::Handle<const mi::neuraylib::ITarget_code> generate_glsl();

private:
    // Creates an instance of the given material.
    mi::neuraylib::IFunction_call* create_material_instance(
        const std::string& qualified_module_name,
        const std::string& material_db_name);

    // Compiles the given material instance in the given compilation modes.
    mi::neuraylib::ICompiled_material* compile_material_instance(
        mi::neuraylib::IFunction_call* material_instance,
        bool class_compilation);

private:
    mi::base::Handle<mi::neuraylib::IMdl_factory> m_factory;
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> m_mdl_impexp_api;
    mi::base::Handle<mi::neuraylib::IMdl_backend> m_be_glsl;
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> m_context;
    mi::base::Handle<mi::neuraylib::ILink_unit> m_link_unit;

    // This selects SSBO (Shader Storage Buffer Objects) mode for passing uniforms and MDL
    // const data.
    // Should not be disabled unless you only use materials with very small const data.
    // In this example, this would only apply to execution_material_2, because the others are using
    // lookup tables for noise functions.
    bool m_use_ssbo;

    // If enabled, the GLSL backend will remap these functions
    //   float ::base::perlin_noise(float4 pos)
    //   float ::base::mi_noise(float3 pos)
    //   float ::base::mi_noise(int3 pos)
    //   ::base::worley_return ::base::worley_noise(float3 pos, float jitter, int metric)
    //
    // to lut-free alternatives. When enabled, you can avoid to set the USE_SSBO define for this
    // example.
    bool m_remap_noise_functions;
};

// Creates an instance of the given material.
mi::neuraylib::IFunction_call* Material_compiler::create_material_instance(
    const std::string& qualified_module_name,
    const std::string& material_db_name)
{
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        m_transaction->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str()));
    if (!material_definition)
        exit_failure("Failed to access material definition '%s'.", material_db_name.c_str());

    // Create a material instance from the material definition
    // with the default arguments.
    mi::Sint32 result;
    mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(
        material_definition->create_function_call(0, &result));
    check_success(result == 0);
    if (result != 0)
        exit_failure("Failed to instantiate material '%s'.", material_db_name.c_str());

    material_instance->retain();
    return material_instance.get();
}

// Compiles the given material instance in the given compilation modes.
mi::neuraylib::ICompiled_material *Material_compiler::compile_material_instance(
    mi::neuraylib::IFunction_call* material_instance,
    bool class_compilation)
{
    mi::Uint32 flags = class_compilation
        ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
        : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance2(
        material_instance->get_interface<mi::neuraylib::IMaterial_instance>());
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance2->create_compiled_material(flags, m_context.get()));
    check_success(print_messages(m_context.get()));

    compiled_material->retain();
    return compiled_material.get();
}

// Generates GLSL target code for a subexpression of a given compiled material.
mi::base::Handle<const mi::neuraylib::ITarget_code> Material_compiler::generate_glsl()
{
    mi::base::Handle<const mi::neuraylib::ITarget_code> code_glsl(
        m_be_glsl->translate_link_unit(m_link_unit.get(), m_context.get()));
    check_success(print_messages(m_context.get()));
    check_success(code_glsl);

#ifdef DUMP_GLSL
    std::cout << "Dumping GLSL code:\n\n" << code_glsl->get_code() << std::endl;
#endif

    return code_glsl;
}

// Generates GLSL target code for a subexpression of a given material.
// path is the path of the sub-expression.
// fname is the function name in the generated code.
bool Material_compiler::add_material_subexpr(
    const std::string& qualified_module_name,
    const std::string& material_db_name,
    const char* path,
    const char* fname)
{
    // Load the given module and create a material instance
    mi::base::Handle<mi::neuraylib::IFunction_call> material_instance(
        create_material_instance(qualified_module_name, material_db_name));

    // Compile the material instance in instance compilation mode
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        compile_material_instance(material_instance.get(), /*class_compilation=*/false));

    m_link_unit->add_material_expression(compiled_material.get(), path, fname,
        m_context.get());
    return print_messages(m_context.get());
}

// Constructor.
Material_compiler::Material_compiler(
    bool use_ssbo,
    bool remap_noise_functions,
    mi::neuraylib::IMdl_impexp_api *mdl_impexp_api,
    mi::neuraylib::IMdl_backend_api *mdl_backend_api,
    mi::neuraylib::IMdl_factory *mdl_factory,
    mi::neuraylib::ITransaction *transaction)
    : m_factory(mi::base::make_handle_dup(mdl_factory))
    , m_mdl_impexp_api(mi::base::make_handle_dup(mdl_impexp_api))
    , m_be_glsl(mdl_backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_GLSL))
    , m_transaction(mi::base::make_handle_dup(transaction))
    , m_context(mdl_factory->create_execution_context())
    , m_link_unit()
    , m_use_ssbo(use_ssbo)
    , m_remap_noise_functions(remap_noise_functions)
{
    check_success(m_be_glsl->set_option("num_texture_spaces", "1") == 0);

    if (m_use_ssbo) {
        // SSBO requires GLSL 4.30
        check_success(m_be_glsl->set_option("glsl_version", "430") == 0);

#if 0
        check_success(m_be_glsl->set_option("glsl_max_const_data", "0") == 0);
#endif
        check_success(m_be_glsl->set_option("glsl_place_uniforms_into_ssbo", "on") == 0);
    } else {
        // GLSL 3.30 is sufficient
        check_success(m_be_glsl->set_option("glsl_version", "330") == 0);

#if 0
        check_success(m_be_glsl->set_option("glsl_max_const_data", "1024") == 0);
#endif
        check_success(m_be_glsl->set_option("glsl_place_uniforms_into_ssbo", "off") == 0);
    }

    if (m_remap_noise_functions) {
        // remap noise functions that access the constant tables
        check_success(m_be_glsl->set_option("glsl_remap_functions",
            "_ZN4base12perlin_noiseEu6float4=noise_float4"
            ",_ZN4base12worley_noiseEu6float3fi=noise_worley"
            ",_ZN4base8mi_noiseEu6float3=noise_mi_float3"
            ",_ZN4base8mi_noiseEu4int3=noise_mi_int3") == 0);
    }

    // After we set the options, we can create the link unit
    m_link_unit = mi::base::make_handle(m_be_glsl->create_link_unit(transaction, m_context.get()));
}


//------------------------------------------------------------------------------
//
// Application logic
//
//------------------------------------------------------------------------------

// Context structure for window callback functions.
struct Window_context
{
    // A number from 1 to 7 specifying the material pattern to display.
    int material_pattern;
};

// GLFW callback handler for keyboard inputs.
void handle_key(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/)
{
    // Handle key press events
    if (action == GLFW_PRESS) {
        // Map keypad numbers to normal numbers
        if (GLFW_KEY_KP_0 <= key && key <= GLFW_KEY_KP_9)
            key += GLFW_KEY_0 - GLFW_KEY_KP_0;

        switch (key) {
            // Escape closes the window
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;

            // Numbers 1 - 7 select the different material patterns
            case GLFW_KEY_1:
            case GLFW_KEY_2:
            case GLFW_KEY_3:
            case GLFW_KEY_4:
            case GLFW_KEY_5:
            case GLFW_KEY_6:
            case GLFW_KEY_7:
            {
                Window_context *ctx = static_cast<Window_context*>(
                    glfwGetWindowUserPointer(window));
                ctx->material_pattern = key - GLFW_KEY_0;
                break;
            }

            default:
                break;
        }
    }
}

// GLFW callback handler for framebuffer resize events (when window size or resolution changes).
void handle_framebuffer_size(GLFWwindow* /*window*/, int width, int height)
{
    glViewport(0, 0, width, height);
}

// Initializes OpenGL, creates the shader program and the scene and executes the animation loop.
void show_and_animate_scene(
    mi::base::Handle<mi::neuraylib::ITransaction>        transaction,
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api>     mdl_impexp_api,
    mi::base::Handle<mi::neuraylib::IImage_api>          image_api,
    mi::base::Handle<const mi::neuraylib::ITarget_code>  target_code,
    Options const                                       &options)
{
    Window_context window_context = { options.material_pattern };

    // Init OpenGL window
    GLFWwindow *window = init_opengl(options);

    // If SSBO is disabled, reduce the supported amount of materials and textures.
    unsigned max_materials = options.use_ssbo ? 64 : 16;
    unsigned max_textures  = options.use_ssbo ? 32 : 16;

    // Create shader program
    GLuint program = create_shader_program(
        options.use_ssbo, options.remap_noise_functions, max_materials, max_textures, target_code);

    // Create scene data
    GLuint quad_vertex_buffer;
    GLuint quad_vao = create_quad(program, &quad_vertex_buffer);

    // Scope for material context resources
    {
        // Prepare the needed material data of all target codes for the fragment shader
        Material_opengl_context material_opengl_context(program, options.use_ssbo);
        check_success(material_opengl_context.prepare_material_data(
                transaction, image_api, target_code));
        check_success(material_opengl_context.set_material_data(max_textures));

        // Get locations of uniform parameters for fragment shader
        GLint   material_pattern_index = glGetUniformLocation(program, "material_pattern");
        GLint   animation_time_index   = glGetUniformLocation(program, "animation_time");

        if (!options.no_window) {
            GLfloat animation_time = 0;
            double  last_frame_time = glfwGetTime();

            glfwSetWindowUserPointer(window, &window_context);
            glfwSetKeyCallback(window, handle_key);
            glfwSetFramebufferSizeCallback(window, handle_framebuffer_size);

            // Loop until the user closes the window
            while (!glfwWindowShouldClose(window))
            {
                // Update animation time
                double cur_frame_time = glfwGetTime();
                animation_time += GLfloat(cur_frame_time - last_frame_time);
                last_frame_time = cur_frame_time;

                // Set uniform frame parameters
                glUniform1i(material_pattern_index, window_context.material_pattern);
                glUniform1f(animation_time_index, animation_time);

                // Render the scene
                glClear(GL_COLOR_BUFFER_BIT);
                glBindVertexArray(quad_vao);
                glDrawArrays(GL_TRIANGLES, 0, 6);

                // Swap front and back buffers
                glfwSwapBuffers(window);

                // Poll for events and process them
                glfwPollEvents();
            }
        } else {  // no_window
            // Set up frame buffer
            GLuint frame_buffer = 0, color_buffer = 0;
            glGenFramebuffers(1, &frame_buffer);
            glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer);
            glGenRenderbuffers(1, &color_buffer);
            glBindRenderbuffer(GL_RENDERBUFFER, color_buffer);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, options.res_x, options.res_y);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_RENDERBUFFER, color_buffer);
            check_gl_success();

            // Set uniform frame parameters
            glUniform1i(material_pattern_index, window_context.material_pattern);
            glUniform1f(animation_time_index, 0.f);

            // Render the scene
            glClear(GL_COLOR_BUFFER_BIT);
            glViewport(0, 0, options.res_x, options.res_y);
            check_gl_success();
            glBindVertexArray(quad_vao);
            check_gl_success();
            glDrawArrays(GL_TRIANGLES, 0, 6);
            check_gl_success();

            // Create a canvas and copy the result image to it
            mi::base::Handle<mi::neuraylib::ICanvas> canvas(
                image_api->create_canvas("Rgba", options.res_x, options.res_y));
            mi::base::Handle<mi::neuraylib::ITile> tile(canvas->get_tile());
            glReadPixels(0, 0, options.res_x, options.res_y,
                GL_RGBA, GL_UNSIGNED_BYTE, tile->get_data());

            // Save the image to disk
            mdl_impexp_api->export_canvas(options.outputfile.c_str(), canvas.get());

            // Cleanup frame buffer
            glDeleteRenderbuffers(1, &color_buffer);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteFramebuffers(1, &frame_buffer);
        }
    }

    // Cleanup OpenGL
    glDeleteVertexArrays(1, &quad_vao);
    glDeleteBuffers(1, &quad_vertex_buffer);
    glDeleteProgram(program);
    check_gl_success();
    glfwDestroyWindow(window);
    glfwTerminate();
}


static void usage(char const *prog_name, bool default_ssbo)
{
    std::cout
        << "Usage: " << prog_name << " [options] [<material_pattern>]\n"
        << "Options:\n"
        << "  --nowin             don't show interactive display\n"
        << "  --res <x> <y>       resolution (default: 1024x768)\n"
        << "  --with-ssbo         Enable SSBO" << (default_ssbo ? " (default)\n" : "\n")
        << "  --no-ssbo           Disable SSBO" << (!default_ssbo ? " (default)\n" : "\n")
        << "  --no-noise-remap    Do not remap MDL ::base noise functions\n"
        << "  -o <outputfile>     image file to write result in nowin mode (default: output.png)\n"
        << "  <material_pattern>  a number from 1 to 7 choosing which material combination to use"
        << std::endl;
    exit_failure();
}

//------------------------------------------------------------------------------
//
// Main function
//
//------------------------------------------------------------------------------

int MAIN_UTF8(int argc, char* argv[])
{
    // Parse command line options
    Options options;

    bool default_ssbo = options.use_ssbo;

    for (int i = 1; i < argc; ++i) {
        char const *opt = argv[i];
        if (opt[0] == '-') {
            if (strcmp(opt, "--nowin") == 0) {
                options.no_window = true;
            } else if (strcmp(opt, "-o") == 0) {
                if (i < argc - 1) {
                    options.outputfile = argv[++i];
                } else {
                    usage(argv[0], default_ssbo);
                }
            } else if (strcmp(opt, "--with-ssbo") == 0) {
                options.use_ssbo = true;
            } else if (strcmp(opt, "--no-ssbo") == 0) {
                options.use_ssbo = false;
            } else if (strcmp(opt, "--no-noise-remap") == 0) {
                options.remap_noise_functions = false;
            } else if (strcmp(opt, "--res") == 0) {
                if (i < argc - 2) {
                    options.res_x = std::max(atoi(argv[++i]), 1);
                    options.res_y = std::max(atoi(argv[++i]), 1);
                } else {
                    usage(argv[0], default_ssbo);
                }
            } else {
                usage(argv[0], default_ssbo);
            }
        } else {
            options.material_pattern = atoi(opt);
            if (options.material_pattern < 1 || options.material_pattern > 7) {
                std::cerr << "Invalid material_pattern parameter." << std::endl;
                usage(argv[0], default_ssbo);
            }
        }
    }

    printf("SSBO Extension is       : %s\n",
        options.use_ssbo ? "enabled" : "disabled");
    printf("Noise function remap is : %s\n",
        options.remap_noise_functions ? "enabled" : "disabled");

    // Access the MDL SDK
    mi::base::Handle<mi::neuraylib::INeuray> neuray(mi::examples::mdl::load_and_get_ineuray());
    if (!neuray.is_valid_interface())
        exit_failure("Failed to load the SDK.");

    // Configure the MDL SDK
    mi::examples::mdl::Configure_options configure_options;
    if (!mi::examples::mdl::configure(neuray.get(), configure_options))
        exit_failure("Failed to initialize the SDK.");

    // Start the MDL SDK
    mi::Sint32 ret = neuray->start();
    if (ret != 0)
        exit_failure("Failed to initialize the SDK. Result code: %d", ret);

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

        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdl_factory->create_execution_context());

        // Load the module.
        std::string module_name = "::nvidia::sdk_examples::tutorials";
        mdl_impexp_api->load_module(transaction.get(), module_name.c_str(), context.get());
        if (!print_messages(context.get()))
            exit_failure("Loading module '%s' failed.", module_name.c_str());

        // Get the database name for the module we loaded
        mi::base::Handle<const mi::IString> module_db_name(
            mdl_factory->get_db_module_name(module_name.c_str()));
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str()));
        if (!module)
            exit_failure("Failed to access the loaded module.");

        // Set up the materials.
        std::vector<std::string> material_simple_names;
        std::vector<std::string> fnames;
        if (options.use_ssbo || options.remap_noise_functions) {
            material_simple_names.push_back("example_execution1");
            fnames.push_back("tint");
        }
        material_simple_names.push_back("example_execution2");
        fnames.push_back("tint_2");
        if (options.use_ssbo || options.remap_noise_functions) {
            material_simple_names.push_back("example_execution3");
            fnames.push_back("tint_3");
        }

        // Construct material DB names.
        size_t n = material_simple_names.size();
        std::vector<std::string> material_db_names(n);
        for (size_t i = 0; i < n; ++i) {
            material_db_names[i]
                = std::string(module_db_name->get_c_str()) + "::" + material_simple_names[i];
            material_db_names[i] = mi::examples::mdl::add_missing_material_signature(
                module.get(), material_db_names[i]);
            if (material_db_names[i].empty())
                exit_failure("Failed to find the material %s in the module %s.",
                    material_simple_names[i].c_str(), module_name.c_str());
        }
        module.reset();

        // Add material sub-expressions of different materials to the link unit.
        Material_compiler mc(
            options.use_ssbo, options.remap_noise_functions,
            mdl_impexp_api.get(), mdl_backend_api.get(), mdl_factory.get(), transaction.get());
        for (size_t i = 0; i < n; ++i) {
            mc.add_material_subexpr(
                module_name, material_db_names[i], "surface.scattering.tint", fnames[i].c_str());
        }

        // Generate the GLSL code for the link unit.
        mi::base::Handle<const mi::neuraylib::ITarget_code> target_code(mc.generate_glsl());

        // Acquire image API needed to prepare the textures
        mi::base::Handle<mi::neuraylib::IImage_api> image_api(
            neuray->get_api_component<mi::neuraylib::IImage_api>());

        show_and_animate_scene(transaction, mdl_impexp_api, image_api, target_code, options);

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
