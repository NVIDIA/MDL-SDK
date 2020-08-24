/******************************************************************************
 * Copyright 2019 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/

 // examples/mdl_sdk/shared/gui.h

#ifndef EXAMPLE_SHARED_GUI_API_INTERFACE_GLFW_H
#define EXAMPLE_SHARED_GUI_API_INTERFACE_GLFW_H

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include "gui/gui.h"

namespace mi { namespace examples { namespace gui
{
    class Api_interface_glfw : public Api_interface
    {
    public:
        struct Render_context_glfw : public Render_context
        {
        };

        explicit Api_interface_glfw(
            GLFWwindow* window)
            : m_window(window)
            , m_window_width(0)
            , m_window_height(0)
        {
            ImGui_ImplGlfw_InitForOpenGL(window, false);
            ImGui_ImplOpenGL3_Init(NULL);
        }

        virtual ~Api_interface_glfw()
        {
            ImGui_ImplGlfw_Shutdown();
            ImGui_ImplOpenGL3_Shutdown();
        }

        void new_frame() override
        {
            int width, height;
            glfwGetFramebufferSize(m_window, &width, &height);
            m_window_width = width;
            m_window_height = height;

            ImGui_ImplGlfw_NewFrame();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui::NewFrame();
        }

        void render(Render_context* context) override
        {
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        void get_window_size(size_t& out_width, size_t& out_height) const override
        {
            out_width = m_window_width;
            out_height = m_window_height;
        }


    private:
        GLFWwindow* m_window;
        size_t m_window_width;
        size_t m_window_height;
    };
}}}
#endif
