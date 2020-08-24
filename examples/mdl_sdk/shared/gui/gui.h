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

 // examples/mdl_sdk/shared/gui.h

#ifndef EXAMPLE_SHARED_GUI_H
#define EXAMPLE_SHARED_GUI_H

#include <unordered_map>
#include <queue>
#include <mi/mdl_sdk.h>
#include "imgui.h"
#include "imgui_internal.h"
#include <memory>
#include <functional>
#include "../utils/enums.h"
#include <cmath>

namespace mi { namespace examples { namespace gui
{
    class Base_element;
    class Panel;
    class Menu_item;

    enum ImGuiColExt
    {
        ImGuiColExt_Warning,
        ImGuiColExt_COUNT
    };

    enum class Gui_direction
    {
        Left = 0,
        Right = 1,
        Top = 2,
        Bottom = 3,
        Center = 4
    };

    /// The platform dependent API interface handles the binding to the low-level rendering
    /// framework, i.e., OpenGL, Direct3D, ...
    /// An implementation for the selected platform has to be provided to the main GUI instance.
    class Api_interface
    {
    public:
        /// Platform dependent graphics API context.
        struct Render_context
        {
            virtual ~Render_context() = default;
        };

        /// Constructor.
        explicit Api_interface();

        // Destructor.
        virtual ~Api_interface();

        /// called at the beginning of a frame to indicate that a new frame starts.
        virtual void new_frame() = 0;

        /// called at the end of frame to actually draw the controls.
        virtual void render(Render_context* context) = 0;

        /// get the size of the main application window.
        virtual void get_window_size(
            size_t& out_width,
            size_t& out_height) const = 0;
    };

    /// Can be created by Gui elements and added to the Gui instance.
    /// The Application needs to pull the events every frame.
    struct Event final
    {
        /// An event with an invalid type id.
        static const Event Invalid;

        /// Constructor.
        explicit Event(
            Base_element* sender,
            uint64_t event_type_id,
            const void* data);

        /// Check if the event is valid, i.e., if the type id is not -1.
        bool is_valid() const;

        /// Get a pointer to the GUI object that invoked this event.
        Base_element* const get_sender() const;

        /// Get the event type id of this event. IDs are defined bz the application.
        uint64_t get_event_type_id() const;

        // Get the context data passed when creating the event. Or NULL if none was provided.
        template<typename T>
        const T* get_data() const { return static_cast<const T*>(m_data); }

    private:
        Base_element* m_sender;
        uint64_t m_event_type_id;
        const void* m_data;
    };

    /// Main Gui that is created only once for a window.
    class Root
    {
        friend class Base_element;
    public:
        static ImVec4 Colors_ext[ImGuiColExt_COUNT];

        /// Constructor. Creates a new main GUI instance for a certain window and takes
        /// ownership of the binding to the low-level graphics API bindings.
        explicit Root(std::unique_ptr<Api_interface> api_interface);
        ~Root();

        /// Load fonts and apply the standard style.
        bool initialize();

        /// called at the beginning of a frame to indicate that a new frame starts.
        void new_frame();

        /// update and draw all the controls for this window.
        void update(mi::neuraylib::ITransaction* transaction);

        /// let the application react on gui events that occurred within the last update pass.
        Event process_event();

        /// called at the end of frame to actually draw the controls.
        void render(Api_interface::Render_context* context);

        /// Get a panel by name.
        Panel* get_panel(const std::string& name);

        /// add an item to the top menu bar.
        void add(Menu_item item);

        /// Get the platform depended graphics API interface for this GUI element.
        const Api_interface& get_api_interface() const;

    private:
        void apply_style();
        void apply_fonts();
        void add_gui_event(
            Base_element* sender,
            uint64_t event_type_id,
            const void* data);

        std::queue<Event> m_events;
        std::unique_ptr<Api_interface> m_api_interface;
        std::unordered_map<std::string, Panel*> m_panels;
        std::vector<Menu_item> m_menu_items;
        bool m_visible;
    };

    /// Base class for all GUI elements.
    class Base_element
    {
    public:
        enum class Kind
        {
            Unknown = 0,
            Panel,
            Section
        };

        /// Constructor.
        explicit Base_element(Root* gui, Kind kind = Kind::Unknown);

        /// Destructor.
        virtual ~Base_element() = default;

        /// Get the type of GUI element.
        Kind get_kind() const;

        /// Update and show the GUI controls of this element.
        virtual void update(mi::neuraylib::ITransaction* transaction) = 0;

    protected:
        /// Get the parenting GUI root.
        Root& get_gui();

        /// Create an event to be processed by the application.
        void create_event(uint64_t event_type_id);

        /// Create an event to be processed by the application.
        template<typename T>
        void create_event(
            uint64_t event_type_id,
            const T* event_data)
        {
            m_gui->add_gui_event(this, event_type_id, event_data);
        }

    private:
        Root* m_gui;
        Kind m_kind;
    };

    /// A collapsible group of settings for one aspect of an application.
    /// Usually added to a panel.
    class Section : public Base_element
    {
        friend class Panel;
    public:
        /// Constructor.
        explicit Section(
            Root* gui,
            const std::string& name,
            bool collapsed_by_default = false);

        /// Destructor.
        virtual ~Section() = default;

        /// Get the name of the section.
        const std::string& get_name() const;

    protected:
        /// Update and show the GUI controls of this section.
        void update_internal(mi::neuraylib::ITransaction* transaction);

    private:
        const std::string m_name;
        bool m_collapsed;
    };

    /// Gui panel to hold one or multiple sections.
    /// The panel is docked to the left or right border of the window.
    class Panel : public Base_element
    {
    public:
        /// Constructor.
        explicit Panel(
            Root* gui,
            Gui_direction docking_direction);

        /// Destructor.
        virtual ~Panel();

        /// Add GUI elements, usually Gui_sections, but also other ImGui elements.
        void add(std::string name, Base_element* element);

        /// Get a GUI element of this panel by name.
        Base_element* get(const std::string& name);

        /// Add additional spacing to compensate for top or bottom ribbons.
        void set_margin(Gui_direction dir, float value);

        /// Update hierarchy below, called once per frame by the main GUI instance.
        void update(mi::neuraylib::ITransaction* transaction) override;

    private:
        static void on_resized(ImGuiSizeCallbackData* data);

        Root* m_gui;
        float m_width;
        float m_min_width;
        float m_margins[4];
        Gui_direction m_docking_direction;
        std::unordered_map<std::string, Base_element*> m_elements;
    };

    class Menu_item final
    {
    public:
        /// Kind of menu item.
        enum class Kind
        {
            Group,
            Separator,
            Button,
        };

        /// Constructor.
        explicit Menu_item(
            const std::string& name,
            uint64_t event_type_id,
            const void* event_data = nullptr);

        /// Constructor.
        explicit Menu_item(
            const std::string& name,
            std::initializer_list<Menu_item> sub_items);

        /// Constructor.
        explicit Menu_item(
            const std::string& name,
            Kind kind = Kind::Group);

        /// Add a sub menu item to a group.
        void add(Menu_item sub_item);

        /// Get a child item by name.
        Menu_item* operator[] (const std::string& name);

        /// Get a child item by name.
        const Menu_item* operator[] (const std::string& name) const;

        /// Get the name (and label) of the menu item.
        const std::string& get_name() const;

        /// Get the kind of this item.
        const Kind get_kind() const;

        /// Get the number of child items.
        const size_t get_child_count() const;

        /// Get the i`th child item.
        const Menu_item* get_child(size_t index) const;

        /// Get the event type ID of the event that is triggered by this item.
        uint64_t get_event_type_id() const;

        /// Get the event data of the event that is triggered by this item.
        const void* get_event_data() const;

    private:
        std::string m_name;
        Kind m_kind;
        uint64_t m_event_type_id;
        const void* m_event_data;
        std::vector<Menu_item> m_children;
    };

    /// Renders GUI elements to edit individual values. Used in Sections.
    class Control /*static*/
    {
    public:
        /// Influences the rendering of elements.
        enum class Flags
        {
            None = 0,
            Disabled = 1 << 0,
        };

    private:
        template<typename TTooltip>
        static void show_tooltip(TTooltip description);

        template<typename TRet>
        static void show_tooltip(const std::function<TRet()>& action)
        {
            action();
        }

        template<typename TTooltip>
        static void show_property_label(const std::string& text, TTooltip description, Flags flags)
        {
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            if (text.back() == '\n')
                ImGui::TextWrapped("%s", text.c_str());
            else
            {
                float current_indent = ImGui::GetCursorPos().x;
                const ImGuiStyle& style = ImGui::GetStyle();
                const ImGuiWindow* window = ImGui::GetCurrentWindow();

                float control_width =
                    std::min((ImGui::GetWindowWidth() - style.IndentSpacing) * 0.5f, 140.f);
                control_width -= window->ScrollbarSizes.x;
                control_width = std::max(control_width, 50.0f);

                float available_width = ImGui::GetContentRegionAvail().x;

                float avaiable_text_width =
                    available_width - control_width - style.ItemInnerSpacing.x;

                ImVec2 text_size = ImGui::CalcTextSize(
                    text.c_str(), text.c_str() + text.size(), false, avaiable_text_width);

                float indent = current_indent +
                    available_width - control_width - text_size.x - style.ItemInnerSpacing.x;

                ImGui::AlignTextToFramePadding();
                ImGui::NewLine();
                ImGui::SameLine(indent);
                ImGui::PushTextWrapPos(indent + avaiable_text_width);
                ImGui::TextWrapped("%s", text.c_str());
                ImGui::PopTextWrapPos();
                ImGui::SameLine();
            }

            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
                ImGui::PopStyleColor();

            if (ImGui::IsItemHovered())
                show_tooltip(description);
        }

        template<typename TValue>
        static bool show_slider_control(
            TValue* value,
            TValue& min,
            TValue& max,
            const char* format);

        template<typename TValue>
        static bool show_drag_control(
            TValue* value,
            float speed,
            TValue& min,
            TValue& max,
            const char* format);

        template<typename TValue, typename TTooltip>
        static bool show_numeric_control(
            const std::string& label,
            TTooltip description,
            TValue* value,
            const TValue* default_value,
            Flags flags,
            std::function<bool(void)> show_numeric_control)
        {
            ImGui::PushID(value);
            show_property_label(label, description, flags);
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            bool changed = show_numeric_control();

            if (default_value && ImGui::BeginPopupContextItem("item context menu"))
            {
                if (ImGui::Selectable("set default"))
                {
                    changed = *value != *default_value;
                    *value = *default_value;
                }
                ImGui::EndPopup();
            }

            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }

            ImGui::PopID();
            return changed;
        }

    public:
        // slider for values with minimum and maximum.
        template<typename TValue, typename TTooltip>
        static bool slider(
            const std::string& label,
            TTooltip description,
            TValue* value,
            const TValue* default_value,
            Flags flags,
            TValue min = TValue(0),
            TValue max = TValue(1))
        {
            return show_numeric_control(label, description, value, default_value, flags, [&] {
                return show_slider_control<TValue>(value, min, max, nullptr);
                });
        }

        // used for unbound float values. Min and max (or only one) can be specified.
        template<typename TValue, typename TTooltip>
        static bool drag(
            const std::string& label,
            TTooltip description,
            TValue* value,
            const TValue* default_value,
            Flags flags,
            TValue min = std::numeric_limits<TValue>::lowest(),
            TValue max = std::numeric_limits<TValue>::max(),
            float speed = std::is_integral<TValue>::value ? 0.25f : 0.01f,
            const char* format = nullptr)
        {
            return show_numeric_control(label, description, value, default_value, flags, [&] {
                return show_drag_control<TValue>(value, speed, min, max, format);
                });
        }

        // Check-box to toggle options on and off.
        template<typename TTooltip>
        static bool checkbox(
            const std::string& label,
            TTooltip description,
            bool* value,
            const bool* default_value,
            Flags flags)
        {
            ImGui::PushID(value);
            show_property_label(label, description, flags);
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            bool changed = ImGui::Checkbox("##hidden", value);
            if (default_value && ImGui::BeginPopupContextItem("item context menu"))
            {
                if (ImGui::Selectable("set default"))
                {
                    changed = *value != *default_value;
                    *value = *default_value;
                }
                ImGui::EndPopup();
            }
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
            ImGui::PopID();
            return changed;
        }

        // Combo-box to select between options.
        template<typename Tinteger, typename TTooltip>
        static bool selection(
            const std::string& label,
            TTooltip description,
            Tinteger* index,
            const Tinteger* default_index,
            Flags flags,
            std::function<const char* (Tinteger)> get_value)
        {
            return show_numeric_control(label, description, index, default_index, flags, [&] {
                bool valid = false;
                bool changed = false;
                if (ImGui::BeginCombo("##hidden", get_value(*index)))
                {
                    Tinteger i = 0;
                    while (true)
                    {
                        const char* option = get_value(i);
                        if (!option)
                            break;

                        valid |= (i == *index); // check if current selection is a valid option
                        if (ImGui::Selectable(option, i == *index))
                        {
                            *index = i;
                            changed = true;
                            valid = true;
                        }
                        i++;
                    }
                    ImGui::EndCombo();

                    if (!valid && default_index)
                    {
                        *index = *default_index;
                        changed = true;
                    }
                }
                return changed;
                });
        }

        // Combo-box to select between options.
        template<typename Tinteger, typename TTooltip>
        static bool selection(
            const std::string& label,
            TTooltip description,
            Tinteger* index,
            const Tinteger* default_index,
            Flags flags,
            std::vector<std::string> values)
        {
            return selection<Tinteger>(label, description, index, default_index, flags,
                [&](Tinteger i) {
                    return i < values.size() ? values[i].c_str() : nullptr;
                });
        }

        template<typename TTooltip>
        static bool text(
            const std::string& label,
            TTooltip description,
            char* value_buffer, size_t value_buffer_size,
            const char* default_value,
            Flags flags)
        {
            //assuming that this is unique, but constant from frame to frame
            int id = *static_cast<int*>((void*)&label);
            id ^= *static_cast<int*>((void*)&description);
            id ^= *static_cast<int*>((void*)&default_value);

            ImGui::PushID(id);
            show_property_label(label, description, flags);
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            bool changed = ImGui::InputText("##hidden",
                value_buffer, value_buffer_size, ImGuiInputTextFlags_EnterReturnsTrue);
            if (default_value && ImGui::BeginPopupContextItem("item context menu"))
            {
                if (ImGui::Selectable("set default"))
                {
                    changed = strcmp(value_buffer, default_value) != 0;
                    strncpy(value_buffer, default_value, value_buffer_size - 1);
                    value_buffer[value_buffer_size] = '\0';
                }
                ImGui::EndPopup();
            }
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
            ImGui::PopID();
            return changed;
        }

        // Color picker.
        template<typename TTooltip>
        static bool pick(
            const std::string& label,
            TTooltip description,
            float* value_float3,
            const float* default_value_float3,
            Flags flags)
        {
            ImGui::PushID(value_float3);
            show_property_label(label, description, flags);
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            bool changed = ImGui::ColorEdit3("##hidden", &value_float3[0],
                ImGuiColorEditFlags_NoInputs |
                ImGuiColorEditFlags_Float |
                ImGuiColorEditFlags_NoAlpha);

            if (default_value_float3 && ImGui::BeginPopupContextItem("item context menu"))
            {
                if (ImGui::Selectable("set default"))
                {
                    changed = value_float3[0] != default_value_float3[0] ||
                        value_float3[1] != default_value_float3[1] ||
                        value_float3[2] != default_value_float3[2];

                    value_float3[0] = default_value_float3[0];
                    value_float3[1] = default_value_float3[1];
                    value_float3[2] = default_value_float3[2];
                }
                ImGui::EndPopup();
            }
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
            ImGui::PopID();
            return changed;
        }

        /// Button to start an action or for navigation.
        template<typename TTooltip>
        static bool button(
            const std::string& label,
            const std::string& button_text,
            TTooltip description,
            Flags flags)
        {
            ImGui::PushID(&description);

            // show the left side label with tool tips
            if (!label.empty())
                show_property_label(label, description, flags);

            bool label_fits =
                (ImGui::GetContentRegionAvail().x -
                    ImGui::CalcTextSize(button_text.c_str()).x -
                    ImGui::GetStyle().ItemInnerSpacing.x * 2.0f) > 0.0f;

            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            bool pressed = ImGui::Button(
                (label.empty() || label_fits ? button_text.c_str() : "..."),
                ImVec2(ImGui::GetContentRegionAvail().x, 0.0f));

            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }

            // show the tool tip when hovering over the button in case there is no left side label
            if (label.empty() && ImGui::IsItemHovered())
                show_tooltip(description);

            ImGui::PopID();
            return pressed;
        }

        /// Create a collapsible group that can nest other elements.
        template<typename TReturn>
        static TReturn group(
            const std::string& name,
            bool expanded_by_default,
            std::function<TReturn(void)> show_content)
        {
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec4 color = ImGui::GetStyle().Colors[ImGuiCol_Text];
            color.x *= 0.6f;
            color.y *= 0.6f;
            color.z *= 0.6f;
            const ImU32 draw_color = ImColor(color);
            const float draw_line_width = 1.0f;

            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

            TReturn ret = static_cast<TReturn>(0);
            const ImVec2 start_top = ImGui::GetCursorScreenPos();
            ImGui::PushID(&show_content);
            if (ImGui::TreeNodeEx(name.c_str(),
                expanded_by_default ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None))
            {
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();

                const ImVec2 start = ImGui::GetCursorScreenPos();
                ret = show_content();
                const ImVec2 end = ImGui::GetCursorScreenPos();

                draw_list->AddLine(
                    ImVec2(start.x - draw_line_width * 0.5f, (start.y + start_top.y) * 0.5f),
                    ImVec2(end.x - draw_line_width * 0.5f, end.y),
                    draw_color, draw_line_width);

                ImGui::TreePop();
            }
            else
            {
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
            }
            ImGui::PopID();
            ImGui::Spacing();
            return ret;
        }

        /// Shows an text label only without interaction possibility.
        template<typename TTooltip>
        static void info(
            const std::string& label,
            TTooltip description,
            const char* info_text,
            Flags flags)
        {
            ImGui::PushID(&info_text);
            show_property_label(label, description, flags);
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            ImGui::TextWrapped(info_text);
            if (mi::examples::enums::has_flag(flags, Flags::Disabled))
            {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
            ImGui::PopID();
        }
    };
}}}
#endif
