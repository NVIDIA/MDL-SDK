#include "gui.h"

#include <vector>
#include <functional>
#include <iostream>
#include <limits>

#include "../utils/io.h"

namespace ImGuiExt
{
    void Spacing(size_t steps)
    {
        for (size_t i = 0; i < steps; ++i)
            ImGui::Spacing();
    }
}

namespace mi { namespace examples { namespace gui
{

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Api_interface::Api_interface()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO();
}

// ------------------------------------------------------------------------------------------------

Api_interface::~Api_interface()
{
    ImGui::DestroyContext();
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

const Event Event::Invalid = Event(nullptr, static_cast<uint64_t>(-1), nullptr);

// ------------------------------------------------------------------------------------------------

Event::Event(Base_element* sender, uint64_t event_type_id, const void* data)
    : m_sender(sender)
    , m_event_type_id(event_type_id)
    , m_data(data)
{}

// ------------------------------------------------------------------------------------------------

bool Event::is_valid() const
{
    return m_event_type_id != static_cast<uint64_t>(-1);
}

// ------------------------------------------------------------------------------------------------

Base_element* const Event::get_sender() const
{
    return m_sender;
}

// ------------------------------------------------------------------------------------------------

uint64_t Event::get_event_type_id() const
{
    return m_event_type_id;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Panel::Panel(
    Root* gui,
    Gui_direction docking_direction)
    : Base_element(gui, Base_element::Kind::Panel)
    , m_gui(gui)
    , m_width(300)
    , m_min_width(150)
    , m_docking_direction(docking_direction)
{
    m_margins[static_cast<size_t>(Gui_direction::Top)] = 0;
    m_margins[static_cast<size_t>(Gui_direction::Bottom)] = 0;
    m_margins[static_cast<size_t>(Gui_direction::Left)] = 0;
    m_margins[static_cast<size_t>(Gui_direction::Right)] = 0;
}

// ------------------------------------------------------------------------------------------------

Panel::~Panel()
{
    for (auto& e : m_elements)
        delete e.second;
    m_elements.clear();
}

// ------------------------------------------------------------------------------------------------

void Panel::update(mi::neuraylib::ITransaction* transaction)
{
    if (m_elements.empty())
        return;

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoNav;

    float& margin_top = m_margins[static_cast<size_t>(Gui_direction::Top)];
    float& margin_bottom = m_margins[static_cast<size_t>(Gui_direction::Bottom)];

    size_t window_width, window_height;
    m_gui->get_api_interface().get_window_size(window_width, window_height);

    ImGui::SetNextWindowPos(
        ImVec2(float(window_width - m_width), margin_top), ImGuiCond_Always);
    ImGui::SetNextWindowSize(
        ImVec2(float(m_width), float(window_height) - margin_top - margin_bottom),
        ImGuiCond_Always);

    ImGui::SetNextWindowSizeConstraints(
        ImVec2(m_min_width, -1.0f), ImVec2(FLT_MAX, -1.0f), on_resized, this);

    ImGui::Begin("##", NULL, window_flags);
    {
        ImGuiWindow* window = ImGui::GetCurrentWindow();

        // added this to the ImGui library
        window->ResizeHandles[ImGuiWindowResizeHandle_Corner] = false;
        window->ResizeHandles[ImGuiWindowResizeHandle_Right] = false;
        window->ResizeHandles[ImGuiWindowResizeHandle_Top] = false;
        window->ResizeHandles[ImGuiWindowResizeHandle_Bottom] = false;
        // ... currently only right side panel implemented

        for (auto& pair : m_elements)
            if (pair.second->get_kind() == Base_element::Kind::Section)
                (static_cast<Section*>(pair.second))->update_internal(transaction);
            else
                pair.second->update(transaction);
    }
    ImGui::End();
}

// ------------------------------------------------------------------------------------------------

void Panel::add(std::string name, Base_element* element)
{
    m_elements[name] = element;
}

// ------------------------------------------------------------------------------------------------

Base_element* Panel::get(const std::string& name)
{
    auto found = m_elements.find(name);
    return found == m_elements.end() ? nullptr : found->second;
}

// ------------------------------------------------------------------------------------------------

void Panel::set_margin(Gui_direction dir, float value)
{
    size_t index = static_cast<size_t>(dir);
    m_margins[index] = value;
}

// ------------------------------------------------------------------------------------------------

void Panel::on_resized(ImGuiSizeCallbackData* data)
{
    // todo, remove this when docking gets available in ImGui

    Panel* _this = static_cast<Panel*>(data->UserData);
    size_t width, height;
    _this->m_gui->get_api_interface().get_window_size(width, height);
    _this->m_width = std::min(data->DesiredSize.x, float(width) * 0.9f);

    data->DesiredSize.x = data->CurrentSize.x;
    data->DesiredSize.y = data->CurrentSize.y;
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

ImVec4 Root::Colors_ext[ImGuiColExt_COUNT];

// ------------------------------------------------------------------------------------------------

Root::Root(std::unique_ptr<Api_interface> api_interface)
    : m_api_interface(std::move(api_interface))
    , m_visible(true)
{
    m_panels["right"] = new Panel(this, Gui_direction::Right);
}

// ------------------------------------------------------------------------------------------------

Root::~Root()
{
    for (auto& p : m_panels)
        delete p.second;
    m_panels.clear();
}

// ------------------------------------------------------------------------------------------------

bool Root::initialize()
{
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().LogFilename = nullptr;
    apply_style();
    apply_fonts();

    return true;
}

// ------------------------------------------------------------------------------------------------

void Root::apply_style()
{
    ImGui::GetIO();
    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 0.0f;
    style.ColorButtonPosition = ImGuiDir_Left;
    style.FrameRounding = 2.0f;
    style.FrameBorderSize = 1.0f;
    style.GrabRounding = 4.0f;
    style.IndentSpacing = 12.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(0.135f, 0.135f, 0.135f, 1.0f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.4f, 0.4f, 0.4f, 0.5f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.5f);

    std::vector<ImGuiCol> to_change;
    to_change.push_back(ImGuiCol_Header);
    to_change.push_back(ImGuiCol_HeaderActive);
    to_change.push_back(ImGuiCol_HeaderHovered);
    to_change.push_back(ImGuiCol_SliderGrab);
    to_change.push_back(ImGuiCol_SliderGrabActive);
    to_change.push_back(ImGuiCol_Button);
    to_change.push_back(ImGuiCol_ButtonActive);
    to_change.push_back(ImGuiCol_ButtonHovered);
    to_change.push_back(ImGuiCol_FrameBgActive);
    to_change.push_back(ImGuiCol_FrameBgHovered);
    to_change.push_back(ImGuiCol_CheckMark);
    to_change.push_back(ImGuiCol_ResizeGrip);
    to_change.push_back(ImGuiCol_ResizeGripActive);
    to_change.push_back(ImGuiCol_ResizeGripHovered);
    to_change.push_back(ImGuiCol_TextSelectedBg);
    to_change.push_back(ImGuiCol_Separator);
    to_change.push_back(ImGuiCol_SeparatorHovered);
    to_change.push_back(ImGuiCol_SeparatorActive);
    for (auto c : to_change)
    {
        style.Colors[c].x = 0.465f;
        style.Colors[c].y = 0.495f;
        style.Colors[c].z = 0.525f;
    }

    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.465f, 0.465f, 0.465f, 1.0f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.125f, 0.125f, 0.125f, 1.0f);
    style.Colors[ImGuiCol_Tab] = ImVec4(0.05f, 0.05f, 0.05f, 0.5f);
    style.Colors[ImGuiCol_TabHovered] = ImVec4(0.465f, 0.495f, 0.525f, 1.0f);
    style.Colors[ImGuiCol_TabActive] = ImVec4(0.282f, 0.290f, 0.302f, 1.0f);
    style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.465f, 0.465f, 0.465f, 0.350f);

    Colors_ext[ImGuiColExt_Warning] = ImVec4(1.0f, 0.43f, 0.35f, 1.0f);

    ImGui::SetColorEditOptions(
        ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel);
}

// ------------------------------------------------------------------------------------------------

void Root::apply_fonts()
{
    float float_size = 16.0f;
    std::string font_dir = mi::examples::io::get_executable_folder() + "/content/fonts/";
    std::string font_path;

    auto font_atlas = ImGui::GetIO().Fonts;

    // load font for Latin Cyrillic, and Greek characters
    ImFontConfig config;
    auto builder = ImFontGlyphRangesBuilder();
    builder.AddRanges(font_atlas->GetGlyphRangesDefault());
    builder.AddRanges(font_atlas->GetGlyphRangesCyrillic());
    builder.AddRanges(font_atlas->GetGlyphRangesVietnamese());
    static const ImWchar more[] =
    {
        0x0020, 0x00FF, // Basic Latin + Latin Supplement
        0x0100, 0x017F, // Latin Extended - A
        0x0180, 0x024F, // Latin Extended - B
        0x1E00, 0x1EFF, // Latin Extended Additional
        0x0370, 0x03FF, // Greek/Coptic
        0x1F00, 0x1FFF, // Greek Extended
        0,
    };
    builder.AddRanges(more);
    ImVector<ImWchar> ranges1;
    builder.BuildRanges(&ranges1);
    font_path = font_dir + "NotoSans-Medium.ttf";
    font_atlas->AddFontFromFileTTF(font_path.c_str(), float_size, &config, ranges1.Data);

    // load fonts for other characters and merge them
    config.MergeMode = true;
    builder = ImFontGlyphRangesBuilder();
    builder.AddRanges(font_atlas->GetGlyphRangesJapanese());
    builder.AddRanges(font_atlas->GetGlyphRangesKorean());
    builder.AddRanges(font_atlas->GetGlyphRangesChineseSimplifiedCommon());
    builder.AddRanges(font_atlas->GetGlyphRangesChineseFull());
    ImVector<ImWchar> cjk;
    builder.BuildRanges(&cjk);
    font_path = font_dir + "NotoSansCJK-Medium.ttc";
    font_atlas->AddFontFromFileTTF(font_path.c_str(), float_size, &config, cjk.Data);

    font_path = font_dir + "NotoSansThai-Medium.ttf";
    font_atlas->AddFontFromFileTTF(font_path.c_str(), float_size, &config, font_atlas->GetGlyphRangesThai());

    static const ImWchar arabic[] = { 0x0600, 0x06FF, 0 };
    font_path = font_dir + "NotoSansArabic-Medium.ttf";
    font_atlas->AddFontFromFileTTF(font_path.c_str(), float_size, &config, arabic);

    static const ImWchar hebrew[] = { 0x0590, 0x05FF, 0 };
    font_path = font_dir + "NotoSansHebrew-Medium.ttf";
    font_atlas->AddFontFromFileTTF(font_path.c_str(), float_size, &config, hebrew);

    // add more fonts or ranges if required, this set is not complete at all but covers at least a few

    // build the bitmap fonts
    font_atlas->TexDesiredWidth = 1024;
    font_atlas->Build();
}

// ------------------------------------------------------------------------------------------------

void Root::add_gui_event(
    Base_element* sender,
    uint64_t event_type_id,
    const void* data)
{
    m_events.emplace(Event(sender, event_type_id, data));
}

// ------------------------------------------------------------------------------------------------

void Root::new_frame()
{
    m_api_interface->new_frame();
}

// ------------------------------------------------------------------------------------------------

void Root::update(mi::neuraylib::ITransaction* transaction)
{
    if (!m_visible)
        return;

    std::function<void(const Menu_item&)> update_submenu =
        [&](const Menu_item& item)
    {
        switch (item.get_kind())
        {
            case Menu_item::Kind::Group:
                if (ImGui::BeginMenu(item.get_name().c_str()))
                {
                    size_t child_count = item.get_child_count();
                    for (size_t c = 0; c < child_count; ++c)
                        update_submenu(*item.get_child(c));
                    ImGui::EndMenu();
                }
                break;

            case Menu_item::Kind::Button:
                if (ImGui::MenuItem(item.get_name().c_str()))
                    add_gui_event(
                        nullptr, item.get_event_type_id(), item.get_event_data());
                break;

            case Menu_item::Kind::Separator:
                ImGui::Separator();
                break;
        }
    };

    // Menu Bar at the top
    static bool s_show_demo_window = false;
    static bool s_show_style_editor = false;
    bool show_menu_bar = true;
    float menu_bar_height = 0.0f;
    if (show_menu_bar && ImGui::BeginMainMenuBar())
    {
        for (const auto& item : m_menu_items)
            update_submenu(item);

        if (ImGui::BeginMenu("ImGui"))
        {
            if (ImGui::MenuItem("Demo Window.."))
                s_show_demo_window = !s_show_demo_window;

            if (ImGui::MenuItem("Style Editor.."))
                s_show_style_editor = !s_show_style_editor;

            ImGui::TextColored(
                ImGui::GetStyle().Colors[ImGuiCol_TextDisabled],
                "Dear ImGui v.%s", IMGUI_VERSION);
            ImGui::EndMenu();
        }

        menu_bar_height = ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    if (s_show_demo_window)
        ImGui::ShowDemoWindow(&s_show_demo_window);

    if (s_show_style_editor)
        ImGui::ShowStyleEditor();

    // simple layout
    m_panels["right"]->set_margin(Gui_direction::Top, show_menu_bar ? menu_bar_height : 0.0f);


    for (auto& p : m_panels)
        p.second->update(transaction);
}

// ------------------------------------------------------------------------------------------------

Event Root::process_event()
{
    if (m_events.empty())
        return Event(nullptr, static_cast<mi::Size>(-1), nullptr);

    Event e = m_events.front();
    m_events.pop();
    return e;
}

// ------------------------------------------------------------------------------------------------

void Root::render(Api_interface::Render_context* context)
{
    m_api_interface->render(context);
}
// ------------------------------------------------------------------------------------------------

Panel* Root::get_panel(const std::string& name)
{
    auto found = m_panels.find(name);
    return found == m_panels.end() ? nullptr : found->second;
}

// ------------------------------------------------------------------------------------------------

void Root::add(Menu_item item)
{
    m_menu_items.push_back(item);
}

// ------------------------------------------------------------------------------------------------

const Api_interface& Root::get_api_interface() const
{
    return *m_api_interface;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Base_element::Base_element(
    Root* gui,
    Kind kind)
    : m_gui(gui)
    , m_kind(kind)
{
}

// ------------------------------------------------------------------------------------------------

Base_element::Kind Base_element::get_kind() const
{
    return m_kind;
}

// ------------------------------------------------------------------------------------------------

Root& Base_element::get_gui()
{
    return *m_gui;
}

// ------------------------------------------------------------------------------------------------

void Base_element::create_event(uint64_t event_type_id)
{
    m_gui->add_gui_event(this, event_type_id, nullptr);
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Section::Section(
    Root* gui,
    const std::string& name,
    bool collapsed_by_default)
    : Base_element(gui, Base_element::Kind::Section)
    , m_name(name)
    , m_collapsed(collapsed_by_default)
{
}

// ------------------------------------------------------------------------------------------------

void Section::update_internal(mi::neuraylib::ITransaction* transaction)
{
    ImGui::PushID(this);
    if (ImGui::CollapsingHeader(
        m_name.c_str(), m_collapsed ? ImGuiTreeNodeFlags_None : ImGuiTreeNodeFlags_DefaultOpen))
    {
        m_collapsed = false;
        update(transaction);
        ImGuiExt::Spacing(2);
    }
    else
        m_collapsed = true;

    ImGui::PopID();
}

// ------------------------------------------------------------------------------------------------

const std::string& Section::get_name() const
{
    return m_name;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Menu_item::Menu_item(
    const std::string& name,
    uint64_t event_type_id,
    const void* event_data)
    : m_name(name)
    , m_kind(Kind::Button)
    , m_event_type_id(event_type_id)
    , m_event_data(event_data)
    , m_children()
{
}

// ------------------------------------------------------------------------------------------------

Menu_item::Menu_item(
    const std::string & name,
    std::initializer_list<Menu_item> sub_items)
    : m_name(name)
    , m_kind(Kind::Group)
    , m_event_type_id(static_cast<uint64_t>(-1))
    , m_event_data(nullptr)
    , m_children(sub_items)
{
}

// ------------------------------------------------------------------------------------------------

Menu_item::Menu_item(
    const std::string & name, Kind kind)
    : m_name(name)
    , m_kind(kind)
    , m_event_type_id(static_cast<uint64_t>(-1))
    , m_event_data(nullptr)
    , m_children()
{
}

// ------------------------------------------------------------------------------------------------

void Menu_item::add(Menu_item sub_item)
{
    m_children.push_back(std::move(sub_item));
}

// ------------------------------------------------------------------------------------------------

Menu_item* Menu_item::operator[] (const std::string& name)
{
    for (size_t i = 0; i < m_children.size(); ++i)
        if (m_children[i].m_name == name)
            return &m_children[i];

    return nullptr;
}

// ------------------------------------------------------------------------------------------------

const Menu_item* Menu_item::operator[] (const std::string& name) const
{
    for (size_t i = 0; i < m_children.size(); ++i)
        if (m_children[i].m_name == name)
            return &m_children[i];

    return nullptr;
}

// ------------------------------------------------------------------------------------------------

const std::string& Menu_item::get_name() const
{
    return m_name;
}

// ------------------------------------------------------------------------------------------------

const Menu_item::Kind Menu_item::get_kind() const
{
    return m_kind;
}

// ------------------------------------------------------------------------------------------------

const size_t Menu_item::get_child_count() const
{
    return m_children.size();
}

// ------------------------------------------------------------------------------------------------

const Menu_item* Menu_item::get_child(size_t index) const
{
    return index < m_children.size() ? &m_children[index] : nullptr;
}

// ------------------------------------------------------------------------------------------------

uint64_t Menu_item::get_event_type_id() const
{
    return m_event_type_id;
}

// ------------------------------------------------------------------------------------------------

const void* Menu_item::get_event_data() const
{
    return m_event_data;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

template<>
void Control::show_tooltip(const char* description)
{
    if (!description || strlen(description) == 0)
        return;

    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(description);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

// ------------------------------------------------------------------------------------------------

template<>
void Control::show_tooltip(const std::string& description)
{
    if (description.empty()) return;
    show_tooltip<const char*>(description.c_str());
}

// ------------------------------------------------------------------------------------------------

namespace
{
    static const char* visible_labels[] = { "x:", "y:", "z:", "w:" };
    template<typename TScalar, ImGuiDataType type, uint8_t dim>
    bool show_slider_control_scalar(
        TScalar* value,
        TScalar* min,
        TScalar* max,
        const char* format)
    {
        if (dim == 1)
            return ImGui::SliderScalar("##hidden", type, &value[0], &min[0], &max[0], format);

        float indent = ImGui::GetCursorPos().x;
        bool changed = false;
        for (uint8_t c = 0; c < dim; ++c)
        {
            ImGui::PushID(c);
            if (c > 0)
            {
                ImGui::NewLine();
                ImGui::SameLine(indent);
            }
            ImGui::Text("%s", visible_labels[c]);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            changed |= ImGui::SliderScalar("##hidden", type, &value[c], &min[c], &max[c], format);
            ImGui::PopID();
        }
        return changed;
    }

    template<typename TScalar, ImGuiDataType type, uint8_t dim>
    bool show_drag_control_scalar(
        TScalar* value,
        float speed,
        TScalar* min,
        TScalar* max,
        const char* format)
    {
        if (dim == 1)
            return ImGui::DragScalar("##hidden", type, &value[0], speed, &min[0], &max[0], format);

        float indent = ImGui::GetCursorPos().x;
        bool changed = false;

        for (uint8_t c = 0; c < dim; ++c)
        {
            ImGui::PushID(c);
            if (c > 0)
            {
                ImGui::NewLine();
                ImGui::SameLine(indent);
            }
            ImGui::Text("%s", visible_labels[c]);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            changed |= ImGui::DragScalar("##hidden", type, &value[c], speed, &min[c], &max[c], format);
            ImGui::PopID();
        }
        return changed;
    }

} // anonymous

template<>
bool Control::show_slider_control<mi::Float32>(mi::Float32* value, mi::Float32& min, mi::Float32& max, const char* format) {
    return show_slider_control_scalar<mi::Float32, ImGuiDataType_Float, 1>(value, &min, &max, format ? format : "%.3f");
}

template<>
bool Control::show_slider_control<mi::Float32_2>(mi::Float32_2* value, mi::Float32_2& min, mi::Float32_2& max, const char* format) {
    return show_slider_control_scalar<mi::Float32, ImGuiDataType_Float, 2>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_slider_control<mi::Float32_3>(mi::Float32_3* value, mi::Float32_3& min, mi::Float32_3& max, const char* format) {
    return show_slider_control_scalar<mi::Float32, ImGuiDataType_Float, 3>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_slider_control<mi::Float32_4>(mi::Float32_4* value, mi::Float32_4& min, mi::Float32_4& max, const char* format) {
    return show_slider_control_scalar<mi::Float32, ImGuiDataType_Float, 4>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_drag_control<mi::Float32>(mi::Float32* value, float speed, mi::Float32& min, mi::Float32& max, const char* format) {
    return show_drag_control_scalar<mi::Float32, ImGuiDataType_Float, 1>(value, speed, &min, &max, format ? format : "%.3f");
}

template<>
bool Control::show_drag_control<mi::Float32_2>(mi::Float32_2* value, float speed, mi::Float32_2& min, mi::Float32_2& max, const char* format) {
    return show_drag_control_scalar<mi::Float32, ImGuiDataType_Float, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_drag_control<mi::Float32_3>(mi::Float32_3* value, float speed, mi::Float32_3& min, mi::Float32_3& max, const char* format) {
    return show_drag_control_scalar<mi::Float32, ImGuiDataType_Float, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_drag_control<mi::Float32_4>(mi::Float32_4* value, float speed, mi::Float32_4& min, mi::Float32_4& max, const char* format) {
    return show_drag_control_scalar<mi::Float32, ImGuiDataType_Float, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}



template<>
bool Control::show_slider_control<mi::Float64>(mi::Float64* value, mi::Float64& min, mi::Float64& max, const char* format) {
    return show_slider_control_scalar<mi::Float64, ImGuiDataType_Double, 1>(value, &min, &max, format ? format : "%.3f");
}

template<>
bool Control::show_slider_control<mi::Float64_2>(mi::Float64_2* value, mi::Float64_2& min, mi::Float64_2& max, const char* format) {
    return show_slider_control_scalar<mi::Float64, ImGuiDataType_Double, 2>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_slider_control<mi::Float64_3>(mi::Float64_3* value, mi::Float64_3& min, mi::Float64_3& max, const char* format) {
    return show_slider_control_scalar<mi::Float64, ImGuiDataType_Double, 3>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_slider_control<mi::Float64_4>(mi::Float64_4* value, mi::Float64_4& min, mi::Float64_4& max, const char* format) {
    return show_slider_control_scalar<mi::Float64, ImGuiDataType_Double, 4>(&value->x, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_drag_control<mi::Float64>(mi::Float64* value, float speed, mi::Float64& min, mi::Float64& max, const char* format) {
    return show_drag_control_scalar<mi::Float64, ImGuiDataType_Double, 1>(value, speed, &min, &max, format ? format : "%.3f");
}

template<>
bool Control::show_drag_control<mi::Float64_2>(mi::Float64_2* value, float speed, mi::Float64_2& min, mi::Float64_2& max, const char* format) {
    return show_drag_control_scalar<mi::Float64, ImGuiDataType_Double, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_drag_control<mi::Float64_3>(mi::Float64_3* value, float speed, mi::Float64_3& min, mi::Float64_3& max, const char* format) {
    return show_drag_control_scalar<mi::Float64, ImGuiDataType_Double, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}

template<>
bool Control::show_drag_control<mi::Float64_4>(mi::Float64_4* value, float speed, mi::Float64_4& min, mi::Float64_4& max, const char* format) {
    return show_drag_control_scalar<mi::Float64, ImGuiDataType_Double, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%.3f");
}



template<>
bool Control::show_slider_control<mi::Sint32>(mi::Sint32* value, mi::Sint32& min, mi::Sint32& max, const char* format) {
    return show_slider_control_scalar<mi::Sint32, ImGuiDataType_S32, 1>(value, &min, &max, format ? format : "%d");
}

template<>
bool Control::show_slider_control<mi::Sint32_2>(mi::Sint32_2* value, mi::Sint32_2& min, mi::Sint32_2& max, const char* format) {
    return show_slider_control_scalar<mi::Sint32, ImGuiDataType_S32, 2>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_slider_control<mi::Sint32_3>(mi::Sint32_3* value, mi::Sint32_3& min, mi::Sint32_3& max, const char* format) {
    return show_slider_control_scalar<mi::Sint32, ImGuiDataType_S32, 3>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_slider_control<mi::Sint32_4>(mi::Sint32_4* value, mi::Sint32_4& min, mi::Sint32_4& max, const char* format) {
    return show_slider_control_scalar<mi::Sint32, ImGuiDataType_S32, 4>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_drag_control<mi::Sint32>(mi::Sint32* value, float speed, mi::Sint32& min, mi::Sint32& max, const char* format) {
    return show_drag_control_scalar<mi::Sint32, ImGuiDataType_S32, 1>(value, speed, &min, &max, format ? format : "%d");
}

template<>
bool Control::show_drag_control<mi::Sint32_2>(mi::Sint32_2* value, float speed, mi::Sint32_2& min, mi::Sint32_2& max, const char* format) {
    return show_drag_control_scalar<mi::Sint32, ImGuiDataType_S32, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_drag_control<mi::Sint32_3>(mi::Sint32_3* value, float speed, mi::Sint32_3& min, mi::Sint32_3& max, const char* format) {
    return show_drag_control_scalar<mi::Sint32, ImGuiDataType_S32, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_drag_control<mi::Sint32_4>(mi::Sint32_4* value, float speed, mi::Sint32_4& min, mi::Sint32_4& max, const char* format) {
    return show_drag_control_scalar<mi::Sint32, ImGuiDataType_S32, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}



template<>
bool Control::show_slider_control<mi::Uint32>(mi::Uint32* value, mi::Uint32& min, mi::Uint32& max, const char* format) {
    return show_slider_control_scalar<mi::Uint32, ImGuiDataType_U32, 1>(value, &min, &max, format ? format : "%d");
}

template<>
bool Control::show_slider_control<mi::Uint32_2>(mi::Uint32_2* value, mi::Uint32_2& min, mi::Uint32_2& max, const char* format) {
    return show_slider_control_scalar<mi::Uint32, ImGuiDataType_U32, 2>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_slider_control<mi::Uint32_3>(mi::Uint32_3* value, mi::Uint32_3& min, mi::Uint32_3& max, const char* format) {
    return show_slider_control_scalar<mi::Uint32, ImGuiDataType_U32, 3>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_slider_control<mi::Uint32_4>(mi::Uint32_4* value, mi::Uint32_4& min, mi::Uint32_4& max, const char* format) {
    return show_slider_control_scalar<mi::Uint32, ImGuiDataType_U32, 4>(&value->x, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_drag_control<mi::Uint32>(mi::Uint32* value, float speed, mi::Uint32& min, mi::Uint32& max, const char* format) {
    return show_drag_control_scalar<mi::Uint32, ImGuiDataType_U32, 1>(value, speed, &min, &max, format ? format : "%d");
}

template<>
bool Control::show_drag_control<mi::Uint32_2>(mi::Uint32_2* value, float speed, mi::Uint32_2& min, mi::Uint32_2& max, const char* format) {
    return show_drag_control_scalar<mi::Uint32, ImGuiDataType_U32, 2>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_drag_control<mi::Uint32_3>(mi::Uint32_3* value, float speed, mi::Uint32_3& min, mi::Uint32_3& max, const char* format) {
    return show_drag_control_scalar<mi::Uint32, ImGuiDataType_U32, 3>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}

template<>
bool Control::show_drag_control<mi::Uint32_4>(mi::Uint32_4* value, float speed, mi::Uint32_4& min, mi::Uint32_4& max, const char* format) {
    return show_drag_control_scalar<mi::Uint32, ImGuiDataType_U32, 4>(&value->x, speed, &min.x, &max.x, format ? format : "%d");
}



template<>
bool Control::show_slider_control<size_t>(size_t* value, size_t& min, size_t& max, const char* format) {
    return show_slider_control_scalar<size_t, ImGuiDataType_U64, 1>(value, &min, &max, format ? format : "%d");
}

template<>
bool Control::show_drag_control<size_t>(size_t* value, float speed, size_t& min, size_t& max, const char* format) {
    return show_drag_control_scalar<size_t, ImGuiDataType_U64, 1>(value, speed, &min, &max, format ? format : "%d");
}

}}}
