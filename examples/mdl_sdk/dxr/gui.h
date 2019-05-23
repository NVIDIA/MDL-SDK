/******************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/dxr/gui.h

#ifndef MDL_D3D12_GUI_H
#define MDL_D3D12_GUI_H

#include "mdl_d3d12/common.h"

namespace mdl_d3d12
{
    class Base_application;
    struct Update_args;
    struct Render_args;
    class Scene;
    class Scene_node;
}


class Camera_controls
{
public:
    explicit Camera_controls(mdl_d3d12::Scene_node* node = nullptr);
    virtual ~Camera_controls() = default;

    bool update(const mdl_d3d12::Update_args& args);
    void set_target(mdl_d3d12::Scene_node* node);

    float movement_speed;
    float rotation_speed;
    
private:
    bool m_left_mouse_button_held;
    bool m_middle_mouse_button_held;
    int32_t m_mouse_move_start_x;
    int32_t m_mouse_move_start_y;
    mdl_d3d12::Scene_node* m_target;
};

class Gui
{
public:
    explicit Gui(mdl_d3d12::Base_application* app);
    virtual ~Gui();

    void resize(size_t width, size_t height);
    bool update(mdl_d3d12::Scene* scene, const mdl_d3d12::Update_args& args, bool show_gui);
    void render(mdl_d3d12::D3DCommandList* command_list, const mdl_d3d12::Render_args& args);

    mdl_d3d12::Scene_node* get_selected_camera() const;

private:
    mdl_d3d12::Base_application* m_app;
    mdl_d3d12::ComPtr<ID3D12DescriptorHeap> m_ui_heap;

    std::string m_selected_material;

    std::unordered_map<std::string, mdl_d3d12::Scene_node*> m_camera_map;
    std::string m_selected_camera;
    Camera_controls m_camera_controls;

};

namespace ImGui
{
    bool SliderUint(
        const char* label, 
        uint32_t* v, 
        uint32_t v_min, 
        uint32_t v_max, 
        const char* format = "%d");
}

#endif
