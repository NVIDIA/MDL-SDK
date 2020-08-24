/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/shared/utils/gl_display.h
 //
 // Code shared by all examples

#ifndef EXAMPLE_SHARED_UTILS_DISPLAY_H
#define EXAMPLE_SHARED_UTILS_DISPLAY_H

// To be defined before including this header
// #define GL_DISPLAY_CUDA
// #define GL_DISPLAY_NATIVE

#include <memory>
#include <chrono>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#if defined(GL_DISPLAY_CUDA)
    #include <cuda.h>
#endif

#include "../gui/gui.h"
#include "../gui/gui_api_interface_glfw.h"

namespace mi { namespace examples { namespace mdl
{
    /// Helper class providing a display buffer which can be mapped by CUDA or to main memory
    /// and which can be rendered to the screen using OpenGL.
    class GL_display
    {
    public:
        /// Constructor.
        ///
        /// \param width   The width of the display buffer.
        /// \param height  The height of the display buffer.
        GL_display(unsigned width, unsigned height)
        : m_display_tex(0)
        , m_program(0)
        , m_quad_vertex_buffer(0)
        , m_quad_vao(0)
        , m_width(0)
        , m_height(0)
#if defined(GL_DISPLAY_CUDA)
        , m_display_buffer(0)
        , m_cuda_resource(nullptr)
#elif defined(GL_DISPLAY_NATIVE)
        , m_pixel_buffer_object_ids {0, 0}
        , m_image_data(nullptr)
#endif
        {
#if defined(GL_DISPLAY_CUDA)
            glGenBuffers(1, &m_display_buffer);
#elif defined(GL_DISPLAY_NATIVE)

#endif
            glGenTextures(1, &m_display_tex);
            if (glGetError() != GL_NO_ERROR)
                exit_failure("Creating GL_display failed");

            // Create shader program
            create_shader_program();

            // Create scene data
            create_quad();

            resize(width, height);
        }

        /// Destructor.
        ~GL_display()
        {
            glDeleteVertexArrays(1, &m_quad_vao);
            glDeleteBuffers(1, &m_quad_vertex_buffer);
            glDeleteProgram(m_program);
        }

        /// Resize the display buffer.
        ///
        /// \param new_width   The new width of the display buffer.
        /// \param new_height  The new height of the display buffer.
        /// \return            True if the size changed
        bool resize(unsigned new_width, unsigned new_height)
        {
            size_t old_buffer_size = m_width * m_height * 4;
            size_t new_buffer_size = new_width * new_height * 4;

            m_width = new_width;
            m_height = new_height;
            glViewport(0, 0, m_width, m_height);

#if defined(GL_DISPLAY_CUDA)

            // Allocate GL display buffer
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_display_buffer);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4, nullptr, GL_DYNAMIC_COPY);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            // Register GL displayer buffer with CUDA
            if (m_cuda_resource)
                cuGraphicsUnregisterResource(m_cuda_resource);

            if (m_width == 0 || m_height == 0)
                m_cuda_resource = 0;
            else
                cuGraphicsGLRegisterBuffer(
                    &m_cuda_resource,
                    m_display_buffer,
                    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

#elif defined(GL_DISPLAY_NATIVE)

            if (m_image_data && new_buffer_size == old_buffer_size)
                return false;

            // free the old image data
            if (m_image_data)
            {
                delete[] m_image_data;
                m_image_data = nullptr;

                glDeleteBuffers(2, m_pixel_buffer_object_ids);

                glDeleteTextures(1, &m_display_tex);
                glGenTextures(1, &m_display_tex);
            }

            m_image_data = new GLubyte[new_buffer_size];
            memset(m_image_data, 0, new_buffer_size);

            glBindTexture(GL_TEXTURE_2D, m_display_tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, new_width, new_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)m_image_data);
            glBindTexture(GL_TEXTURE_2D, 0);

            glGenBuffers(2, m_pixel_buffer_object_ids);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixel_buffer_object_ids[0]);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, new_buffer_size, 0, GL_STREAM_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixel_buffer_object_ids[1]);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, new_buffer_size, 0, GL_STREAM_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#endif
            if (glGetError() != GL_NO_ERROR)
                exit_failure("Resizing display buffer failed");

            return new_buffer_size != old_buffer_size;
        }

#if defined(GL_DISPLAY_CUDA)
        /// Map the display buffer for CUDA device code.
        CUdeviceptr map(CUstream stream)
        {
            CUresult err = cuGraphicsMapResources(1, &m_cuda_resource, stream);
            if (err != CUDA_SUCCESS)
                exit_failure("Mapping display resource failed");

            CUdeviceptr device_ptr;
            size_t buffer_size;
            err = cuGraphicsResourceGetMappedPointer(&device_ptr, &buffer_size, m_cuda_resource);
            if (err != CUDA_SUCCESS)
                exit_failure("Getting mapped display pointer failed");
            return device_ptr;
        }

        /// Unmap the display buffer.
        void unmap(CUstream stream)
        {
            CUresult err = cuGraphicsUnmapResources(1, &m_cuda_resource, stream);
            if (err != CUDA_SUCCESS)
                exit_failure("Unmapping display resource failed");
        }
#elif defined(GL_DISPLAY_NATIVE)
        unsigned char* map()
        {
            static int index = 0;
            index = (index + 1) % 2;
            int nextIndex = (index + 1) % 2;

            glBindTexture(GL_TEXTURE_2D, m_display_tex);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixel_buffer_object_ids[index]);

            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixel_buffer_object_ids[nextIndex]);

            glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4, 0, GL_STREAM_DRAW);
            return (unsigned char*) glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        }

        void unmap()
        {
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
#endif

        /// Render the display buffer to screen.
        void update_display()
        {
#if defined(GL_DISPLAY_CUDA)

            // Update texture
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_display_buffer);
            glBindTexture(GL_TEXTURE_2D, m_display_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            if (glGetError() != GL_NO_ERROR)
                exit_failure("Updating display texture failed");

#elif defined(GL_DISPLAY_NATIVE)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, m_display_tex);
#endif

            // Render the quad
            glClear(GL_COLOR_BUFFER_BIT);
            glBindVertexArray(m_quad_vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            if (glGetError() != GL_NO_ERROR)
                exit_failure("Rendering display quad failed");

            // unbind texture
            glBindTexture(GL_TEXTURE_2D, 0);
        }

    private:
        /// Dump any OpenGL log messages to the console.
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

        /// Compile a shader and attach it to the given program.
        static void add_shader(
            GLenum shader_type, const std::string& source_code, GLuint program)
        {
            const GLchar* src_buffers[1] = { source_code.c_str() };
            GLuint shader = glCreateShader(shader_type);
            if (!shader)
                exit_failure("Creating shader failed");
            glShaderSource(shader, 1, src_buffers, nullptr);
            glCompileShader(shader);

            GLint success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                dump_info(shader,"Error compiling the fragment shader: ");
                exit_failure();
            }
            glAttachShader(program, shader);
            if (glGetError() != GL_NO_ERROR)
                exit_failure("Attaching shader failed");
        }

        /// Create a shader program with a fragment shader, which just copies a texture.
        void create_shader_program()
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
                exit_failure();
            }

        #if !defined(__APPLE__)
            glValidateProgram(program);
            glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
            if (!success) {
                dump_info(program, "Error validating the shader program: ");
                exit_failure();
            }
        #endif

            glUseProgram(program);
            if (glGetError() != GL_NO_ERROR)
                exit_failure("Creating shader program failed");

            m_program = program;
        }

        /// Create a quad filling the whole screen.
        void create_quad()
        {
            static const float vertices[6 * 3] = {
                -1.f, -1.f, 0.0f,
                 1.f, -1.f, 0.0f,
                -1.f,  1.f, 0.0f,
                 1.f, -1.f, 0.0f,
                 1.f,  1.f, 0.0f,
                -1.f,  1.f, 0.0f
            };

            glGenBuffers(1, &m_quad_vertex_buffer);
            glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            glGenVertexArrays(1, &m_quad_vao);
            glBindVertexArray(m_quad_vao);

            const GLint pos_index = glGetAttribLocation(m_program, "Position");
            glEnableVertexAttribArray(pos_index);
            glVertexAttribPointer(
                pos_index, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0);

            if (glGetError() != GL_NO_ERROR)
                exit_failure("Creating quad failed");
        }

    private:
        GLuint             m_display_tex;
        GLuint             m_program;
        GLuint             m_quad_vertex_buffer;
        GLuint             m_quad_vao;
        unsigned           m_width;
        unsigned           m_height;

#if defined(GL_DISPLAY_CUDA)
        GLuint             m_display_buffer;
        CUgraphicsResource m_cuda_resource;
#elif defined(GL_DISPLAY_NATIVE)
        GLuint m_pixel_buffer_object_ids[2];
        GLubyte* m_image_data = nullptr;
#endif


    };

    // --------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------

    static void _gl_window_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    // --------------------------------------------------------------------------------------------

    /// Helper class that handles the window management for simple OpenGL examples.
    class GL_window
    {
    public:
        struct Description
        {
            size_t width;
            size_t height;
            std::string title;
            bool no_gui;
        };

        struct Update_args
        {
            size_t frame_number;
            double elapsed_time_in_seconds;
        };

        explicit GL_window(const Description& desc)
            : m_window(nullptr)
            , m_width(desc.width)
            , m_height(desc.height)
            , m_gui(nullptr)
            , m_update_args{0, 0.0}
        {
            m_window = glfwCreateWindow(m_width, m_height, desc.title.c_str(), NULL, NULL);
            if (!m_window)
            {
                glfwTerminate();
                exit(EXIT_FAILURE);
            }

            glfwSetKeyCallback(m_window, _gl_window_key_callback);
            glfwMakeContextCurrent(m_window);

            // create platform specific UI boilerplate
            if (!desc.no_gui)
            {
                std::unique_ptr<mi::examples::gui::Api_interface_glfw> api_interface(
                    new mi::examples::gui::Api_interface_glfw(m_window));
                m_gui = new mi::examples::gui::Root(std::move(api_interface));
            }
        }

        ~GL_window()
        {
            if (m_gui)
                delete m_gui;
            glfwDestroyWindow(m_window);
        }

        bool update()
        {
            static std::chrono::high_resolution_clock::time_point last;

            if (m_update_args.frame_number == 0)
            {
                last = std::chrono::high_resolution_clock::now();
                m_update_args.elapsed_time_in_seconds = 0.0f;
            }
            else
            {
                auto now = std::chrono::high_resolution_clock::now();
                m_update_args.elapsed_time_in_seconds = (now - last).count() * 1e-9;
                last = now;
            }
            m_update_args.frame_number++;

            if (glfwWindowShouldClose(m_window))
                return false;

            int width, height;
            glfwGetFramebufferSize(m_window, &width, &height);
            m_width = width;
            m_height = height;
            return true;
        }

        const Update_args& get_update_args() const { return m_update_args; }

        size_t get_width() const { return m_width; }
        size_t get_height() const { return m_height; }

        void present_back_buffer()
        {
            // present to back buffer
            glfwSwapBuffers(m_window);
            glfwPollEvents();
        }

        // get the windows main UI instance
        mi::examples::gui::Root* get_gui() { return m_gui; }

    private:
        GLFWwindow* m_window;
        size_t m_width;
        size_t m_height;
        mi::examples::gui::Root* m_gui;
        Update_args m_update_args;
    };
}}}

#endif
