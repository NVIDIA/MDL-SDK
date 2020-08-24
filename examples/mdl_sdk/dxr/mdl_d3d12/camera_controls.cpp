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

#include "camera_controls.h"

#include "base_application.h"
#include "window.h"
#include "scene.h"
#include <imgui.h>

namespace mi { namespace examples { namespace mdl_d3d12
{

Camera_controls::Camera_controls(mdl_d3d12::Base_application* app, mdl_d3d12::Scene_node* node)
    : m_app(app)
    , movement_speed(1.0f)
    , rotation_speed(1.0f)
    , m_left_mouse_button_held(false)
    , m_middle_mouse_button_held(false)
    , m_right_mouse_button_held(false)
    , m_mouse_move_start_x(0)
    , m_mouse_move_start_y(0)
    , m_target(nullptr)
    , m_has_focus(false)
{
    set_target(node);
}

// ------------------------------------------------------------------------------------------------

bool Camera_controls::update(const mdl_d3d12::Update_args & args)
{
    if (m_target == nullptr)
        return false;

    ImGuiIO &io = ImGui::GetIO();
    bool camera_changed = false;

    float delta_theta = 0.0f;
    float delta_phi = 0.0f;

    float delta_right = 0.0f;
    float delta_up = 0.0f;
    float delta_forward = 0.0f;

    bool has_focus = m_app->get_window()->has_focus();
    bool orbit_mode = false;

    // lost focus
    if (m_has_focus && !has_focus)
    {
        for (size_t k = 0; k < 512; k++) {
            io.KeysDown[k] = false;
        }
    }
    m_has_focus = has_focus;

    if (!io.WantCaptureKeyboard && has_focus)
    {
        if (io.KeysDown['W'])
        {
            delta_forward += float(args.elapsed_time * 4.0f * movement_speed);
            camera_changed = true;
        }

        if (io.KeysDown['S'])
        {
            delta_forward -= float(args.elapsed_time * 4.0f * movement_speed);
            camera_changed = true;
        }

        if (io.KeysDown['A'])
        {
            delta_right -= float(args.elapsed_time * 4.0f * movement_speed);
            camera_changed = true;
        }

        if (io.KeysDown['D'])
        {
            delta_right += float(args.elapsed_time * 4.0f * movement_speed);
            camera_changed = true;
        }

        if (io.KeysDown['R'])
        {
            delta_up += float(args.elapsed_time * 4.0f * movement_speed);
            camera_changed = true;
        }

        if (io.KeysDown['F'])
        {
            delta_up -= float(args.elapsed_time * 4.0f * movement_speed);
            camera_changed = true;
        }
    }

    if (!io.WantCaptureMouse)
    {
        // zooming (actually moving in view direction)
        if (int(io.MouseWheel))
        {
            delta_forward += int(io.MouseWheel) * 0.5f * movement_speed;
            camera_changed = true;
        }

        // rotation around focus
        if (io.KeyAlt && io.MouseDown[0] && !io.MouseDown[1] && !io.MouseDown[2])
        {
            orbit_mode = true;

            if (!m_left_mouse_button_held)
            {
                m_left_mouse_button_held = true;
                m_mouse_move_start_x = int32_t(io.MousePos.x);
                m_mouse_move_start_y = int32_t(io.MousePos.y);
            }
            else
            {
                int32_t move_dx = int32_t(io.MousePos.x) - m_mouse_move_start_x;
                int32_t move_dy = int32_t(io.MousePos.y) - m_mouse_move_start_y;
                if (abs(move_dx) > 0 || abs(move_dy) > 0)
                {
                    m_mouse_move_start_x = int32_t(io.MousePos.x);
                    m_mouse_move_start_y = int32_t(io.MousePos.y);

                    // Update camera
                    delta_phi += float(move_dx * 0.003f * rotation_speed);
                    delta_theta += float(move_dy * 0.003f * rotation_speed);
                    camera_changed = true;
                }
            }
        }
        else
            m_left_mouse_button_held = false;


        // rotation of camera
        if (!io.MouseDown[0] && io.MouseDown[1] && !io.MouseDown[2])
        {
            if (!m_right_mouse_button_held)
            {
                m_right_mouse_button_held = true;
                m_mouse_move_start_x = int32_t(io.MousePos.x);
                m_mouse_move_start_y = int32_t(io.MousePos.y);
            }
            else
            {
                int32_t move_dx = int32_t(io.MousePos.x) - m_mouse_move_start_x;
                int32_t move_dy = int32_t(io.MousePos.y) - m_mouse_move_start_y;
                if (abs(move_dx) > 0 || abs(move_dy) > 0)
                {
                    m_mouse_move_start_x = int32_t(io.MousePos.x);
                    m_mouse_move_start_y = int32_t(io.MousePos.y);

                    // Update camera
                    delta_phi += float(move_dx * 0.003f * rotation_speed);
                    delta_theta += float(move_dy * 0.003f * rotation_speed);
                    camera_changed = true;
                }
            }
        }
        else
            m_right_mouse_button_held = false;

        // panning
        if (!io.MouseDown[0] && !io.MouseDown[1] && io.MouseDown[2])
        {
            if (!m_middle_mouse_button_held)
            {
                m_middle_mouse_button_held = true;
                m_mouse_move_start_x = int32_t(io.MousePos.x);
                m_mouse_move_start_y = int32_t(io.MousePos.y);
            }
            else
            {
                int32_t move_dx = int32_t(io.MousePos.x) - m_mouse_move_start_x;
                int32_t move_dy = int32_t(io.MousePos.y) - m_mouse_move_start_y;
                if (abs(move_dx) > 0 || abs(move_dy) > 0)
                {
                    m_mouse_move_start_x = int32_t(io.MousePos.x);
                    m_mouse_move_start_y = int32_t(io.MousePos.y);

                    // Update camera
                    delta_right += float(move_dx * -0.01f * movement_speed);
                    delta_up += float(move_dy * 0.01f * movement_speed);
                    camera_changed = true;
                }
            }
        }
        else
            m_middle_mouse_button_held = false;
    }

    // apply changes to the node transformation
    if (camera_changed)
    {
        Transform trafo;
        if(!Transform::try_from_matrix(m_target->get_global_transformation(), trafo))
           return camera_changed;

        if (orbit_mode)
        {
            DirectX::XMVECTOR pos = DirectX::XMLoadFloat3(&trafo.translation);
            DirectX::XMVECTOR target{ 0, 0, 0, 1 };

            // compute distance to target
            DirectX::XMVECTOR dir = DirectX::XMVectorSubtract(target, pos);
            DirectX::XMVECTOR distance = DirectX::XMVector3Length(dir);

            // compute the actual focus point (camera does probably not face the target directly)
            DirectX::XMVECTOR forward =
                DirectX::XMVector3Rotate({ 0.0f, 0.0f, -1.0f, 0.0f }, trafo.rotation);
            target = DirectX::XMVectorMultiplyAdd(forward, distance, pos);

            // rotate around that focus point
            dir = DirectX::XMVectorSubtract(pos, target);
            distance = DirectX::XMVector3Length(dir);
            dir = DirectX::XMVector3Normalize(dir);

            // rotate around world-up
            DirectX::XMVECTOR rot = DirectX::XMQuaternionRotationAxis({ 0.0f, 1.0f, 0.0f, 0.0f }, -delta_phi);
            DirectX::XMVECTOR new_dir = DirectX::XMVector3Rotate(dir, rot);

            // compute new rotation
            float theta = std::acosf(-new_dir.m128_f32[1]) - mdl_d3d12::PI_OVER_2;
            float phi = std::atan2f(-new_dir.m128_f32[2], -new_dir.m128_f32[0]) + mdl_d3d12::PI_OVER_2;
            theta = std::max(-mdl_d3d12::PI_OVER_2 * 0.99f, std::min(theta, mdl_d3d12::PI_OVER_2 * 0.99f));
            trafo.rotation = DirectX::XMQuaternionRotationRollPitchYaw(-theta, -phi, 0.0f);
            trafo.rotation = DirectX::XMQuaternionNormalize(trafo.rotation);

            // rotate around right
            if ((theta + delta_theta > -mdl_d3d12::PI_OVER_2 * 0.99f) &&
                 (theta + delta_theta < mdl_d3d12::PI_OVER_2 * 0.99f))
            {
                DirectX::XMVECTOR right = DirectX::XMVector3Rotate({ 1.0f, 0.0f, 0.0f, 0.0f }, trafo.rotation);
                rot = DirectX::XMQuaternionRotationAxis(right, -delta_theta);
                new_dir = DirectX::XMVector3Rotate(new_dir, rot);

                // compute new rotation
                theta += delta_theta;
                trafo.rotation = DirectX::XMQuaternionRotationRollPitchYaw(-theta, -phi, 0.0f);
                trafo.rotation = DirectX::XMQuaternionNormalize(trafo.rotation);
            }

            // set position
            DirectX::XMVECTOR new_pos = DirectX::XMVectorMultiplyAdd(new_dir, distance, target);
            DirectX::XMStoreFloat3(&trafo.translation, new_pos);
        }
        else
        {
            // orientation
            // get axis
            DirectX::XMVECTOR right = DirectX::XMVector3Rotate({ 1.0f, 0.0f, 0.0f, 0.0f }, trafo.rotation);
            DirectX::XMVECTOR up = DirectX::XMVector3Rotate({ 0.0f, 1.0f, 0.0f, 0.0f }, trafo.rotation);
            DirectX::XMVECTOR forward = DirectX::XMVector3Rotate({ 0.0f, 0.0f, -1.0f, 0.0f }, trafo.rotation);

            // get sphere coordinates
            float theta = std::acosf(forward.m128_f32[1]) - mdl_d3d12::PI_OVER_2;
            float phi = std::atan2f(forward.m128_f32[2], forward.m128_f32[0]) + mdl_d3d12::PI_OVER_2;

            // apply rotation
            theta = std::max(-mdl_d3d12::PI_OVER_2 * 0.99f,
                std::min(theta + delta_theta, mdl_d3d12::PI_OVER_2 * 0.99f));
            phi += delta_phi;

            trafo.rotation = DirectX::XMQuaternionRotationRollPitchYaw(-theta, -phi, 0.0f);
            trafo.rotation = DirectX::XMQuaternionNormalize(trafo.rotation);

            // apply translation
            DirectX::XMVECTOR pos = DirectX::XMLoadFloat3(&trafo.translation);
            pos = DirectX::XMVectorAdd(pos, DirectX::XMVectorScale(right, delta_right));
            pos = DirectX::XMVectorAdd(pos, DirectX::XMVectorScale(up, delta_up));
            pos = DirectX::XMVectorAdd(pos, DirectX::XMVectorScale(forward, delta_forward));

            DirectX::XMStoreFloat3(&trafo.translation, pos);
        }

        // re-compute the local node transformation
        DirectX::XMMATRIX local_trafo = trafo.get_matrix();
        if(m_target->get_parent())
            local_trafo *= inverse(m_target->get_parent()->get_global_transformation());
        if (!Transform::try_from_matrix(local_trafo, m_target->get_local_transformation()))
            return camera_changed;
    }

    return camera_changed;
}

// ------------------------------------------------------------------------------------------------

void Camera_controls::set_target(mdl_d3d12::Scene_node* node)
{
    if (node && m_target == node)
        return;

    m_target = node;
    m_left_mouse_button_held = false;
    m_middle_mouse_button_held = false;
    m_right_mouse_button_held = false;
    m_mouse_move_start_x = 0;
    m_mouse_move_start_y = 0;
    m_has_focus = false;
}

// ------------------------------------------------------------------------------------------------

void Camera_controls::fit_into_view(const Bounding_box& worldspace_aabb)
{
    if (!worldspace_aabb.is_valid())
        return;

    // fit sphere instead of box
    DirectX::XMFLOAT3 center = worldspace_aabb.center();
    float radius = length(worldspace_aabb.size()) * 0.5f;

    // scale movement speed with scene size
    movement_speed = length(worldspace_aabb.size()) / mdl_d3d12::SQRT_3 * 0.5f;

    Transform trafo;
    if (!Transform::try_from_matrix(m_target->get_global_transformation(), trafo) ||
        length(center - trafo.translation) < 0.00001f)
    {
        // camera transform is degenerated
        trafo = Transform::look_at(
            center + DirectX::XMFLOAT3{ 0.0f, 0.0f, -1.0f },
            center,
            { 0.0f, 1.0f, 0.0f });
    }

    DirectX::XMFLOAT3 view_dir = normalize(center - trafo.translation);

    // get field of view
    float min_fov = DirectX::XM_PI * 0.25f;
    if (m_target->get_kind() == mdl_d3d12::IScene_loader::Node::Kind::Camera)
    {
        auto cam = m_target->get_camera();
        min_fov = cam->get_aspect_ratio() > 1.0f
            ? cam->get_field_of_view()
            : cam->get_field_of_view() * cam->get_aspect_ratio();
    }

    // compute new pose
    float distance = radius / tanf(0.5f * min_fov);
    trafo = Transform::look_at(center - view_dir * distance, center, { 0.0f, 1.0f, 0.0f });

    // re-compute the local node transformation
    DirectX::XMMATRIX local_trafo = trafo.get_matrix();
    if (m_target->get_parent())
        local_trafo *= inverse(m_target->get_parent()->get_global_transformation());
    Transform::try_from_matrix(local_trafo, m_target->get_local_transformation());
}

}}} // mi::examples::mdl_d3d12
