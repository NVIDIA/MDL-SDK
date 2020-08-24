#include "gui_material_properties.h"

namespace mi { namespace examples { namespace gui
{

mi::Size Section_material_resource_handler::get_available_resource_count(
    mi::neuraylib::IValue::Kind kind)
{
    return 1;
}

// ------------------------------------------------------------------------------------------------

mi::Uint32 Section_material_resource_handler::get_available_resource_id(
    mi::neuraylib::IValue::Kind kind,
    mi::Size index)
{
    return 0;
}

// ------------------------------------------------------------------------------------------------

const char* Section_material_resource_handler::get_available_resource_name(
    mi::neuraylib::IValue::Kind kind,
    mi::Size index)
{
    return "<invalid>";
}

// ------------------------------------------------------------------------------------------------

mi::Uint32 Section_material_resource_handler::get_available_resource_id(
    mi::neuraylib::IValue::Kind kind,
    const char* db_name)
{
    return static_cast<mi::Uint32>(-1);
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

namespace /*anonymous*/
{

/// There is an ordered from no change, over lightweight change to recompile.
/// Returns to one with the heaviest impact of two states.
Section_material_update_state max(Section_material_update_state a, Section_material_update_state b)
{
    return mi::examples::enums::from_integer<Section_material_update_state>(std::max(
        mi::examples::enums::to_integer(a),
        mi::examples::enums::to_integer(b)));
}

// ------------------------------------------------------------------------------------------------

/// Information required to bind argument block data to the UI controls
struct Argument_block_field_info
{
    explicit Argument_block_field_info()
        : name("")
        , argument_block(nullptr)
        , argument_layout(nullptr)
        , state()
        , kind(mi::neuraylib::IValue::VK_BOOL)
        , size(0)
        , offset(0)
    {}

    std::string name;
    char* argument_block;
    const mi::neuraylib::ITarget_value_layout* argument_layout;
    mi::neuraylib::Target_value_layout_state state;
    mi::neuraylib::IValue::Kind kind;
    mi::Size size;
    mi::Size offset;
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

class Parameter_context : public Section_material::IParameter_context
{
public:
    /// Constructor.
    explicit Parameter_context(
        bool class_compilation,
        const mi::neuraylib::ITarget_code* target_code,
        const char* material_instance_db_name,
        const char* material_defintion_db_name,
        Section_material_resource_handler* resource_handler)
        : m_class_compilation(class_compilation)
        , m_string_table()
        , m_string_table_inv()
        , m_string_max_value_length(0)
        , m_material_instance_db_name(material_instance_db_name)
        , m_material_defintion_db_name(material_defintion_db_name)
        , m_resource_handler(resource_handler)
    {
        mi::Size string_count = target_code->get_string_constant_count();
        m_string_table.resize(string_count);
        m_string_table_inv.reserve(string_count);
        m_string_max_value_length = 0;
        for (mi::Uint32 i = 0; i < string_count; ++i)
        {
            m_string_table[i] = target_code->get_string_constant(i);
            m_string_max_value_length =
                std::max(m_string_max_value_length, m_string_table[i].length());

            m_string_table_inv[m_string_table[i]] = i;
        }
    }

    // --------------------------------------------------------------------------------------------

    /// Get the number of string present in the target code.
    size_t get_string_count() const
    {
        return m_string_table.size();
    }

    // --------------------------------------------------------------------------------------------

    /// Get the i`th string present in the target code.
    const char* get_string(size_t index) const
    {
        if (index >= m_string_table.size())
            return nullptr;
        return m_string_table[index].c_str();
    }

    // --------------------------------------------------------------------------------------------

    /// Get length of the longest string present in the target code.
    /// Used to limit text input buffers.
    size_t get_max_string_length() const
    {
        return m_string_max_value_length;
    }

    // --------------------------------------------------------------------------------------------

    /// Get the ID for a given string, return 0 if the string does not exist in the table.
    mi::Uint32 get_id_for_string(const std::string& string) const
    {
        auto found = m_string_table_inv.find(string);
        if (found == m_string_table_inv.end())
            return 0;
        return found->second;
    }

    // --------------------------------------------------------------------------------------------

    /// get the database name of the material instance that is currently bound
    const char* get_material_instance_db_name() const
    {
        return m_material_instance_db_name.c_str();
    }

    // --------------------------------------------------------------------------------------------

    /// get the database name of the material definition that is currently bound
    const char* get_material_definition_db_name() const
    {
        return m_material_defintion_db_name.c_str();
    }

    // --------------------------------------------------------------------------------------------

    /// Get the resource handler for this material if available.
    Section_material_resource_handler* get_resource_handler() const
    {
        return m_resource_handler;
    }

    // --------------------------------------------------------------------------------------------

    /// True if the currently bound material is compiled in class compilation mode.
    bool get_class_compilation_mode() const
    {
        return m_class_compilation;
    }

    // --------------------------------------------------------------------------------------------

    /// Add an argument block field information by name.
    void add_argument_block_info(const std::string& name, Argument_block_field_info info)
    {
        auto found = m_argument_block_infos.find(name);
        if (found != m_argument_block_infos.end())
        {
            mi_neuray_assert(false && "argument block info name is not unique");
            return;
        }
        m_argument_block_infos[name] = info;
    }

    // --------------------------------------------------------------------------------------------

    /// Get an argument block field information by name.
    const Argument_block_field_info* get_argument_block_info(const std::string& name) const
    {
        auto found = m_argument_block_infos.find(name);
        return found == m_argument_block_infos.end() ? nullptr : &found->second;
    }

    // --------------------------------------------------------------------------------------------

private:
    bool m_class_compilation;
    std::vector<std::string> m_string_table;
    std::unordered_map<std::string, mi::Uint32> m_string_table_inv;
    size_t m_string_max_value_length;
    std::string m_material_instance_db_name;
    std::string m_material_defintion_db_name;
    Section_material_resource_handler* m_resource_handler;
    std::unordered_map<std::string, Argument_block_field_info> m_argument_block_infos;
};

// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------

class Parameter_node_call;
class Parameter_node_group;
class Parameter_node_stacked;

class Parameter_node_base : public Section_material::IParameter_node
{
public:
    /// Type of node
    enum class Kind
    {
        UNKNOWN = -1,
        Constant_bool = 0,
        // Constant_bool2, // not implemented yet
        // Constant_bool3, // not implemented yet
        // Constant_bool4, // not implemented yet
        Constant_int,
        Constant_int2,
        Constant_int3,
        Constant_int4,
        Constant_enum,
        Constant_float,
        Constant_float2,
        Constant_float3,
        Constant_float4,
        Constant_double,
        Constant_double2,
        Constant_double3,
        Constant_double4,
        Constant_string,
        Constant_color,
        Constant_texture,
        Constant_light_profile,
        Constant_bsdf_measurement,
        LAST_DIRECT = Constant_bsdf_measurement,
        // Matrix_float,  // not implemented yet
        // Matrix_double, // not implemented yet
        Struct,
        Array,
        LAST_COMPOUND = Array,
        Call,
        LAST_STACKED = Call,
        Group,
    };

    // --------------------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------------------

    /// Constructor.
    explicit Parameter_node_base(
        Kind kind,
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : m_node_kind(kind)
        , m_description("")
        , m_context(context)
        , m_groups()
        , m_usages()
        , m_marked_unused(false)
        , m_marked_unused_description("")
        , m_marked_deprecated(false)
        , m_marked_deprecated_description("")
        , m_hidden(false)
        , m_ui_order(0)
        , m_name(std::string(name))
        , m_display_name("")
        , m_enable_if_enabled(true)
        , m_parameters()
        , m_parameters_to_index()
        , m_parent_call(parenting_call)
        , m_child_nodes()
        , m_parent_node(nullptr)
    {
    }

    // --------------------------------------------------------------------------------------------

    /// Destructor.
    virtual ~Parameter_node_base()
    {
        m_parent_call = nullptr;
        m_parameters.clear();

        for (auto child : m_child_nodes)
            delete child;
    }

    // --------------------------------------------------------------------------------------------

protected:
    /// Processes common parameter data, mainly annotations.
    /// Deriving class are extending this functionality.
    void initialize(
        const mi::neuraylib::IAnnotation_block* annos)
    {
        // read annotations that are type independent
        if (annos)
        {
            mi::neuraylib::Annotation_wrapper anno_wrapper(annos);

            // display name
            mi::Size idx = anno_wrapper.get_annotation_index("::anno::display_name(string)");
            if (idx != mi::Size(-1))
            {
                char const* anno_value = nullptr;
                anno_wrapper.get_annotation_param_value(idx, 0, anno_value);
                m_display_name = std::string(anno_value);
            }

            // description
            idx = anno_wrapper.get_annotation_index("::anno::description(string)");
            if (idx != mi::Size(-1))
            {
                char const* anno_value = nullptr;
                anno_wrapper.get_annotation_param_value(idx, 0, anno_value);
                m_description = std::string(anno_value);
            }

            // usages // TODO more than one
            idx = anno_wrapper.get_annotation_index("::anno::usage(string)");
            while (idx != mi::Size(-1))
            {
                char const* anno_value = nullptr;
                anno_wrapper.get_annotation_param_value(idx, 0, anno_value);
                m_usages.push_back(std::string(anno_value));
                idx = anno_wrapper.get_annotation_index("::anno::usage(string)", idx + 1);
            }

            // groups
            mi::Size levels = 3;
            idx = anno_wrapper.get_annotation_index("::anno::in_group(string,string,string)");
            if (idx == mi::Size(-1))
            {
                levels = 2;
                idx = anno_wrapper.get_annotation_index("::anno::in_group(string,string)");
            }
            if (idx == mi::Size(-1))
            {
                levels = 1;
                idx = anno_wrapper.get_annotation_index("::anno::in_group(string)");
            }
            if (idx != mi::Size(-1))
            {
                m_groups.resize(levels);
                char const* group_name = nullptr;
                for (mi::Size i = 0; i < levels; ++i)
                {
                    anno_wrapper.get_annotation_param_value(idx, i, group_name);
                    m_groups[i] = std::string(group_name);
                }
                for (mi::Size i = 0; i < levels; ++i)
                    if (m_groups.back().empty())
                        m_groups.pop_back();
            }

            // unused
            idx = anno_wrapper.get_annotation_index("::anno::unused(string)");
            if (idx != mi::Size(-1))
            {
                m_marked_unused = true;
                char const* anno_value = nullptr;
                anno_wrapper.get_annotation_param_value(idx, 0, anno_value);
                m_marked_unused_description = std::string(anno_value);
            }
            else
            {
                idx = anno_wrapper.get_annotation_index("::anno::unused()");
                if (idx != mi::Size(-1))
                    m_marked_unused = true;
            }

            // deprecated
            idx = anno_wrapper.get_annotation_index("::anno::deprecated(string)");
            if (idx != mi::Size(-1))
            {
                m_marked_deprecated = true;
                char const* anno_value = nullptr;
                anno_wrapper.get_annotation_param_value(idx, 0, anno_value);
                m_marked_deprecated_description = std::string(anno_value);
            }
            else
            {
                idx = anno_wrapper.get_annotation_index("::anno::deprecated()");
                if (idx != mi::Size(-1))
                    m_marked_deprecated = true;
            }

            // hidden
            idx = anno_wrapper.get_annotation_index("::anno::hidden()");
            if (idx != mi::Size(-1))
                m_hidden = true;

            idx = anno_wrapper.get_annotation_index("::anno::ui_order(int)");
            if (idx != mi::Size(-1))
            {
                mi::Sint32 anno_value = 0;
                anno_wrapper.get_annotation_param_value(idx, 0, anno_value);
                m_ui_order = anno_value;
            }
        }
    }

    // --------------------------------------------------------------------------------------------

public:
    /// Add child elements of groups, calls, structures, arrays, ...
    void add_element(Parameter_node_base* element); /*defined after groups are defined*/

    // --------------------------------------------------------------------------------------------

    /// Get the direct children of this node of the hierarchy.
    /// The list can be different from the parameter list due to grouping.
    /// This list is also sorted and used for displaying.
    const std::vector<Parameter_node_base*>& get_childeren() const
    {
        return m_child_nodes;
    }

    // --------------------------------------------------------------------------------------------

    /// Parameters in order of the argument list. This list is used for editing along
    /// with the parenting call and the index in the parenting calls argument list.
    Parameter_node_base* get_parameter(mi::Size index)
    {
        return index >= m_parameters.size() ? nullptr : m_parameters[index];
    }

    // --------------------------------------------------------------------------------------------

    /// returns the order number of this control.
    virtual int get_order() const
    {
        return m_ui_order;
    }

    // --------------------------------------------------------------------------------------------

    /// Sort the children based on the order annotation values.
    void sort()
    {
        // sort children recursively
        for (auto& e : m_child_nodes)
            e->sort();

        // sort direct children
        std::sort(m_child_nodes.begin(), m_child_nodes.end(),
            [&](const Parameter_node_base* a, const Parameter_node_base* b)
            {
                return a->get_order() < b->get_order();
            });
    }

    // --------------------------------------------------------------------------------------------

    /// get the name of the property shown during rendering
    const std::string& get_name()
    {
        return m_display_name.empty() ? m_name : m_display_name;
    }

    // --------------------------------------------------------------------------------------------

    /// get the display name, can be used to check if there was an display name annotation before
    /// calling set_display_name
    const std::string& get_display_name()
    {
        return m_display_name;
    }

    // --------------------------------------------------------------------------------------------

    /// allows to override the display name. E.g., for array elements or struct fields
    const void set_display_name(const std::string& value)
    {
        m_display_name = value;
    }

    // --------------------------------------------------------------------------------------------

    /// get the node kind.
    Parameter_node_base::Kind get_kind() const
    {
        return m_node_kind;
    }

    // --------------------------------------------------------------------------------------------

    /// get the node (or parameter) name.
    const std::string& get_name() const
    {
        return m_name;
    }

    // --------------------------------------------------------------------------------------------

    /// The call this parameter is an argument of.
    /// This is not automatically the parent in the hierarchy due to grouping,
    /// structures, and arrays.
    Parameter_node_call* get_parent_call()
    {
        return m_parent_call;
    }

    // --------------------------------------------------------------------------------------------

    /// The parent in the node hierarchy, this is not necessarily the same as the parent call.
    Parameter_node_base* get_parent_node()
    {
        return m_parent_node;
    }

    // --------------------------------------------------------------------------------------------

    /// enables or disables a parameter control.
    /// set by the enable_if logic.
    void set_enabled(bool value)
    {
        m_enable_if_enabled = value;
    }

    // --------------------------------------------------------------------------------------------

    /// check if the parameter control should be enabled or disabled.
    bool get_enabled() const
    {
        return m_enable_if_enabled;
    }

    // --------------------------------------------------------------------------------------------

    /// If the current node is a call with parameters, this maps a from a
    /// node representing this parameter to the index in the calls argument list.
    /// Otherwise -1 is returned.
    mi::Size get_parameter_index(
        Parameter_node_base* child) const
    {
        const auto& found = m_parameters_to_index.find(child);
        if (found == m_parameters_to_index.end())
            return static_cast<mi::Size>(-1);
        return found->second;
    }

    // --------------------------------------------------------------------------------------------

protected:
    const Kind m_node_kind;
    std::string m_description;
    const Parameter_context* m_context;
    std::vector<std::string> m_groups;
    std::vector<std::string> m_usages;
    bool m_marked_unused;
    std::string m_marked_unused_description;
    bool m_marked_deprecated;
    std::string m_marked_deprecated_description;
    bool m_hidden;
    int m_ui_order;

private:
    const std::string m_name;
    std::string m_display_name;
    bool m_enable_if_enabled;

    std::vector<Parameter_node_base*> m_parameters;
    std::unordered_map<Parameter_node_base*, mi::Size> m_parameters_to_index;
    Parameter_node_call* m_parent_call;

    std::vector<Parameter_node_base*> m_child_nodes;
    Parameter_node_base* m_parent_node;
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

std::string to_string(Parameter_node_base::Kind kind)
{
    switch (kind)
    {
    case Parameter_node_base::Kind::Constant_bool: return "bool";
    // case Parameter_node_base::Kind::Constant_bool2: return "bool2";  // not implemented yet
    // case Parameter_node_base::Kind::Constant_bool3: return "bool3";  // not implemented yet
    // case Parameter_node_base::Kind::Constant_bool4: return "bool4";  // not implemented yet
    case Parameter_node_base::Kind::Constant_int: return "int";
    case Parameter_node_base::Kind::Constant_int2: return "int2";
    case Parameter_node_base::Kind::Constant_int3: return "int3";
    case Parameter_node_base::Kind::Constant_int4: return "int4";
    case Parameter_node_base::Kind::Constant_enum: return "enum";
    case Parameter_node_base::Kind::Constant_float: return "float";
    case Parameter_node_base::Kind::Constant_float2: return "float2";
    case Parameter_node_base::Kind::Constant_float3: return "float3";
    case Parameter_node_base::Kind::Constant_float4: return "float4";
    case Parameter_node_base::Kind::Constant_double: return "double";
    case Parameter_node_base::Kind::Constant_double2: return "double2";
    case Parameter_node_base::Kind::Constant_double3: return "double3";
    case Parameter_node_base::Kind::Constant_double4: return "double4";
    case Parameter_node_base::Kind::Constant_string: return "string";
    case Parameter_node_base::Kind::Constant_color: return "color";
    case Parameter_node_base::Kind::Constant_texture: return "texture";
    case Parameter_node_base::Kind::Constant_light_profile: return "light_profile";
    case Parameter_node_base::Kind::Constant_bsdf_measurement: return "bsdf_measurement";
    case Parameter_node_base::Kind::Struct: return "struct";
    // case Parameter_node_base::Kind::Array:                           // not implemented yet
    // case Parameter_node_base::Kind::Matrix_float:                    // not implemented yet
    // case Parameter_node_base::Kind::Matrix_double:                   // not implemented yet
    case Parameter_node_base::Kind::Call: return "call";
    case Parameter_node_base::Kind::Group: return "group";
    default: return "";
    }
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

/// Base node for all parameters that are constant expressions.
class Parameter_node_constant : public Parameter_node_base
{
public:
    explicit Parameter_node_constant(
        Kind kind,
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_base(kind, name, parenting_call, context)
        , m_light_weight_parameter(false)
        , m_constant_value(nullptr)
    {
    }

    // --------------------------------------------------------------------------------------------

    virtual ~Parameter_node_constant()
    {
        m_constant_value = nullptr;
    }

    // --------------------------------------------------------------------------------------------

protected:
    void initialize(
        mi::neuraylib::IValue* value,
        const mi::neuraylib::IAnnotation_block* annos,
        const Argument_block_field_info* block_info)
    {
        m_constant_value = mi::base::make_handle_dup(value);
        Parameter_node_base::initialize(annos);
        m_light_weight_parameter = block_info != nullptr;

        m_tooltip = "";
        if (!m_description.empty())
            m_tooltip += "Description\n" + m_description;

        if (!m_usages.empty())
        {
            m_tooltip += std::string(m_tooltip.empty() ? "" : "\n\n") + "Usage";
            for (auto& u : m_usages)
                m_tooltip += "\n" + u;
        }

        if (m_marked_unused && !m_marked_unused_description.empty())
        {
            m_tooltip += std::string(m_tooltip.empty() ? "" : "\n\n") + "Info";
            m_tooltip += "\nThis parameter is marked as unused. ";
            m_tooltip += m_marked_unused_description;
        }
        else if (m_marked_unused && m_marked_unused_description.empty())
        {
            m_tooltip += std::string(m_tooltip.empty() ? "" : "\n\n") + "Info";
            m_tooltip += "\nThis parameter is marked as unused.";
        }

        m_tooltip_warning = "";
        if (m_marked_deprecated)
        {
            m_tooltip_warning += std::string(m_tooltip_warning.empty() ? "" : "\n");
            m_tooltip_warning += "This parameter is marked as deprecated.";
            if (m_marked_deprecated_description.empty())
                m_tooltip_warning += " " + m_marked_deprecated_description;
        }

        m_show_tooltip = std::function<void()>([&]() {

            bool enable_if_enabled = get_enabled();
            if (m_tooltip.empty() && m_tooltip_warning.empty() &&
                enable_if_enabled && m_light_weight_parameter)
                    return;

            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            if (!m_tooltip.empty())
            {
                ImGui::TextUnformatted(m_tooltip.c_str());
                if (!m_tooltip_warning.empty() || !enable_if_enabled)
                    ImGui::NewLine();
            }
            if (!enable_if_enabled)
            {
                ImGui::TextUnformatted(
                    "This parameter is disabled because of other parameter values.");
                if (!m_tooltip_warning.empty())
                    ImGui::NewLine();
            }
            else if (!m_light_weight_parameter)
            {
                ImGui::TextUnformatted(
                    "Changing this parameter triggers recompilation.");
                if (!m_tooltip_warning.empty())
                    ImGui::NewLine();
            }

            if (!m_tooltip_warning.empty())
            {
                ImGui::PushStyleColor(
                    ImGuiCol_Text,
                    mi::examples::gui::Root::Colors_ext[mi::examples::gui::ImGuiColExt_Warning]);
                ImGui::TextUnformatted(m_tooltip_warning.c_str());
                ImGui::PopStyleColor();
            }
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
            });
    }

    // --------------------------------------------------------------------------------------------

public:
    /// Show the control for the actual data manipulation.
    virtual bool update_value() = 0;

    // --------------------------------------------------------------------------------------------

    /// Get the neuray representation of the current value of this parameter.
    mi::neuraylib::IValue* get_value()
    {
        m_constant_value->retain();
        return m_constant_value.get();
    }

    // --------------------------------------------------------------------------------------------

    /// If true, the parameter is a class compilation parameter that is exposed in argument block.
    /// In that case changing the parameter will not required recompilation of the material.
    bool get_is_light_weight() const
    {
        return m_light_weight_parameter;
    }

    // --------------------------------------------------------------------------------------------

protected:
    const std::function<void()>& get_tooltip_function() const
    {
        return m_show_tooltip;
    }

    // --------------------------------------------------------------------------------------------

    bool get_unused() const
    {
        return false;
        //return m_context->get_class_compilation_mode() && m_unused;
    }

    // --------------------------------------------------------------------------------------------

private:
    bool m_light_weight_parameter;
    std::string m_tooltip;
    std::string m_tooltip_warning;
    std::function<void()> m_show_tooltip;
    mi::base::Handle<mi::neuraylib::IValue> m_constant_value;
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

/// Nodes that require a navigation down the hierarchy.
/// All nodes with children expect groups.
class Parameter_node_stacked : public Parameter_node_base
{
public:
    explicit Parameter_node_stacked(
        Parameter_node_base::Kind kind,
        const char* name,
        const std::string& button_text,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_base(kind, name, parenting_call, context)
        , m_button_text(button_text)
    {
    }

    // --------------------------------------------------------------------------------------------

    // show the UI control for this parameter, which is a button to navigate down the hierarchy.
    bool show()
    {
        return Control::button(
            get_name(), m_button_text,
            "Show options of child expressions.", Control::Flags::None);
    }

    // --------------------------------------------------------------------------------------------

private:
    const std::string m_button_text;

};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

/// Generic handling for parameters that can get big, so they are stacked
class Parameter_node_stacked_compound : public Parameter_node_stacked
{
public:
    explicit Parameter_node_stacked_compound(
        Parameter_node_base::Kind kind,
        const char* name,
        const std::string& button_text,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_stacked(kind, name, button_text, parenting_call, context)
    {
    }

    // --------------------------------------------------------------------------------------------

    virtual ~Parameter_node_stacked_compound()
    {
        m_compound_value = nullptr;
    }

    // --------------------------------------------------------------------------------------------

    void initialize(
        mi::neuraylib::IValue_compound* value,
        const mi::neuraylib::IAnnotation_block* annos,
        const Argument_block_field_info* block_info)
    {
        m_compound_value = mi::base::make_handle_dup(value);
        Parameter_node_base::initialize(annos);
    }

    // --------------------------------------------------------------------------------------------

    /// Stores the value of a changed children in the neuray representation of this parameter.
    /// TODO it's enough to do this once after setup
    Section_material_update_state store_updated_value()
    {
        Section_material_update_state change = Section_material_update_state::No_change;
        for (mi::Size i = 0, n = m_compound_value->get_size(); i < n; ++i)
        {
            Parameter_node_constant* child =
                static_cast<Parameter_node_constant*>(get_childeren()[i]);

            change = max(change, child->get_is_light_weight()
                ? Section_material_update_state::Argument_block_change
                : Section_material_update_state::Unknown_change);

            mi::base::Handle<mi::neuraylib::IValue> child_value(child->get_value());
            m_compound_value->set_value(i, child_value.get());
        }
        return change;
    }

    // --------------------------------------------------------------------------------------------

    /// Get the neuray representation of the current value of this parameter.
    mi::neuraylib::IValue_compound* get_value()
    {
        m_compound_value->retain();
        return m_compound_value.get();
    }

    // --------------------------------------------------------------------------------------------

private:
    mi::base::Handle<mi::neuraylib::IValue_compound> m_compound_value;
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

/// Specialization for parameters that are structures.
class Parameter_node_struct : public Parameter_node_stacked_compound
{
public:
    explicit Parameter_node_struct(
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_stacked_compound(
            Parameter_node_base::Kind::Struct, name, "struct", parenting_call, context)
    {
    }
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

/// Specialization for parameters that are structures.
class Parameter_node_array : public Parameter_node_stacked_compound
{
public:
    explicit Parameter_node_array(
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_stacked_compound(
            Parameter_node_base::Kind::Array, name, "array", parenting_call, context)
    {
    }
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

/// Specialization for groups which can be collapsed and expanded but down require a
/// a navigation down the hierarchy.
class Parameter_node_group : public Parameter_node_base
{
public:
    explicit Parameter_node_group(
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_base(Kind::Group, name, parenting_call, context)
        , m_expanded(true)
    {
    }

    // --------------------------------------------------------------------------------------------

    /// The order of a group is the minimum order if its children.
    int get_order() const final
    {
        int c = std::numeric_limits<int>::max();
        for (auto& e : get_childeren())
            c = std::min(c, e->get_order());
        return c;
    }

    // --------------------------------------------------------------------------------------------

    /// Flag that is bound to the ImGui element to store the state (collapsed or not)
    bool is_expanded() const
    {
        return m_expanded;
    }

    // --------------------------------------------------------------------------------------------

private:
    bool m_expanded;
};

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

/// Generic node for simple numeric parameters.
template<typename TIValue, typename TTCTValue, Parameter_node_base::Kind TType>
class Parameter_node_constant_generic : public Parameter_node_constant
{
public:
    explicit Parameter_node_constant_generic(
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_constant(TType, name, parenting_call, context)
        , m_value_ptr(nullptr)
        , m_value{ 0 }
        , m_value_default{ 0 }
        , m_has_soft_range(false)
        , m_has_hard_range(false)
        , m_range_min{ 0 }
        , m_range_max{ 0 }
    {
    }

    // --------------------------------------------------------------------------------------------

    virtual void initialize(
        mi::neuraylib::IValue* value,
        const mi::neuraylib::IValue* default_value,
        const mi::neuraylib::IAnnotation_block* annos,
        const Argument_block_field_info* block_info)
    {
        Parameter_node_constant::initialize(value, annos, block_info);

        // get the value from the material instance
        mi::base::Handle<const TIValue> value_generic(value->get_interface<const TIValue>());
        read_value(value_generic.get(), m_value);

        // get the default value (if there is one)
        if (default_value)
        {
            mi::base::Handle<const TIValue> default_generic(
                default_value->get_interface<const TIValue>());
            read_value(default_generic.get(), m_value_default);
        }
        else
            m_value_default = TTCTValue{ 0 };

        // read annotations
        mi::neuraylib::Annotation_wrapper anno_wrapper(annos);
        const std::string type_name = to_string(TType);
        std::string anno_signature = "::anno::hard_range(" + type_name + "," + type_name + ")";
        mi::Size idx = anno_wrapper.get_annotation_index(anno_signature.c_str());
        if (idx != mi::Size(-1))
        {
            m_has_hard_range = true;
            mi::base::Handle<const mi::neuraylib::IValue> anno_value_min(
                anno_wrapper.get_annotation_param_value(idx, 0));

            mi::base::Handle<const TIValue> anno_value_min_generic(
                anno_value_min->get_interface<const TIValue>());

            read_value(anno_value_min_generic.get(), m_range_min);

            mi::base::Handle<const mi::neuraylib::IValue> anno_value_max(
                anno_wrapper.get_annotation_param_value(idx, 1));

            mi::base::Handle<const TIValue> anno_value_max_generic(
                anno_value_max->get_interface<const TIValue>());

            read_value(anno_value_max_generic.get(), m_range_max);
        }

        anno_signature = "::anno::soft_range(" + type_name + "," + type_name + ")";
        idx = anno_wrapper.get_annotation_index(anno_signature.c_str());
        if (idx != mi::Size(-1))
        {

            TTCTValue v;
            m_has_soft_range = true;
            mi::base::Handle<const mi::neuraylib::IValue> anno_value_min(
                anno_wrapper.get_annotation_param_value(idx, 0));

            mi::base::Handle<const TIValue> anno_value_min_generic(
                anno_value_min->get_interface<const TIValue>());

            read_value(anno_value_min_generic.get(), v);
            m_range_min = m_has_hard_range ? min(m_range_min, v) : v;

            mi::base::Handle<const mi::neuraylib::IValue> anno_value_max(
                anno_wrapper.get_annotation_param_value(idx, 1));

            mi::base::Handle<const TIValue> anno_value_max_generic(
                anno_value_max->get_interface<const TIValue>());

            read_value(anno_value_max_generic.get(), v);
            m_range_max = m_has_hard_range ? max(m_range_max, v) : v;
        }

        // get a pointer directly into the argument block
        if (block_info)
        {
            void* ptr = block_info->argument_block + block_info->offset;
            m_value_ptr = static_cast<TTCTValue*>(ptr);
        }
    }

    // --------------------------------------------------------------------------------------------

    bool update_value() override
    {
        if (m_hidden)
            return false;

        TTCTValue* value = m_value_ptr ? m_value_ptr : &(m_value);
        TTCTValue current = *value;
        bool changed = false;
        if (m_has_soft_range || m_has_hard_range)
        {
            if (Control::slider(
                get_name(), get_tooltip_function(),
                value, &get_default(),
                (get_unused() || !get_enabled())
                ? Control::Flags::Disabled : Control::Flags::None,
                m_range_min, m_range_max))
            {
                changed = true;
                // update local or argument block
                changed = !m_has_hard_range;
                if (m_has_hard_range)
                {
                    *value = std::max(*value, m_range_min); // clamp in case of hard range
                    *value = std::min(*value, m_range_max);
                    changed = current != *value;
                }
            }
        }
        else
        {
            changed = Control::drag(
                get_name(), get_tooltip_function(),
                value, &get_default(),
                (get_unused() || !get_enabled())
                ? Control::Flags::Disabled : Control::Flags::None);
        }

        if (changed)
        {
            // update neuray value
            mi::base::Handle<mi::neuraylib::IValue> ivalue(get_value());
            mi::base::Handle<TIValue> tivalue(ivalue->get_interface<TIValue>());
            write_value(*value, tivalue.get());
        }
        return changed;
    }

    // --------------------------------------------------------------------------------------------

protected:
    const TTCTValue& get_default() const
    {
        return m_value_default;
    }

    // --------------------------------------------------------------------------------------------

    virtual void read_value(
        const TIValue* source,
        TTCTValue& target)
    {
        // see explicit specializations and overrides
        mi_neuray_assert(false || "Missing specialization or override");
    }

    // --------------------------------------------------------------------------------------------

    virtual void write_value(
        const TTCTValue& source,
        TIValue* target)
    {
        // see explicit specializations and overrides
        mi_neuray_assert(false || "Missing specialization or override");
    }

    // --------------------------------------------------------------------------------------------

    TTCTValue min(const TTCTValue& a, const TTCTValue& b)
    {
        return std::min(a, b);
    }

    // --------------------------------------------------------------------------------------------

    TTCTValue max(const TTCTValue& a, const TTCTValue& b)
    {
        return std::max(a, b);
    }

    // --------------------------------------------------------------------------------------------

    TTCTValue* m_value_ptr;    // a pointer to the data in the argument block
    TTCTValue m_value;         // used if the pointer is not set
    TTCTValue m_value_default; // stored to be able to reset

    bool m_has_soft_range;
    bool m_has_hard_range;
    TTCTValue m_range_min;
    TTCTValue m_range_max;
};

// special handing for unknown (unsupported)
typedef Parameter_node_constant_generic<
    mi::neuraylib::IValue, bool, Parameter_node_base::Kind::UNKNOWN>
    Parameter_node_constant_unknown;

// special handing for booleans
typedef Parameter_node_constant_generic<
    mi::neuraylib::IValue_bool, bool, Parameter_node_base::Kind::Constant_bool>
    Parameter_node_constant_bool;

// special handing for colors
typedef Parameter_node_constant_generic<
    mi::neuraylib::IValue_color, mi::Float32_3, Parameter_node_base::Kind::Constant_color>
    Parameter_node_constant_color;

// special handing for strings in class compilation mode
typedef Parameter_node_constant_generic<
    mi::neuraylib::IValue_string, mi::Uint32,Parameter_node_base::Kind::Constant_string>
    Parameter_node_constant_string_cc;

// special handing for strings in instance compilation mode
typedef Parameter_node_constant_generic<
    mi::neuraylib::IValue_string, std::string, Parameter_node_base::Kind::Constant_string>
    Parameter_node_constant_string_ic;

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

// special handing for resources
template<Parameter_node_base::Kind TType>
class Parameter_node_constant_generic_resource
    : public Parameter_node_constant_generic<mi::neuraylib::IValue_resource, mi::Uint32, TType>
{
    typedef Parameter_node_constant_generic<mi::neuraylib::IValue_resource, mi::Uint32, TType>
        Base;
public:
    explicit Parameter_node_constant_generic_resource(
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Base(name, parenting_call, context)
        , m_resource_kind(mi::neuraylib::IValue::VK_TEXTURE)
        , m_current_db_name("")
        , m_current_index(0)
        , m_default_index(0)
    {
    }

    // --------------------------------------------------------------------------------------------

    void initialize(
        mi::neuraylib::IValue* value,
        const mi::neuraylib::IValue* default_value,
        const mi::neuraylib::IAnnotation_block* annos,
        const Argument_block_field_info* block_info) final
    {
        // get current value infos
        mi::base::Handle<mi::neuraylib::IValue_resource> resource(
            value->get_interface<mi::neuraylib::IValue_resource>());
        mi_neuray_assert(resource && "value is not a resource");

        m_resource_kind = resource->get_kind();

        // after getting resource kind
        Base::initialize(value, default_value, annos, block_info);

        if (resource->get_value())
            m_current_db_name = resource->get_value();

        // get default value infos
        const char* default_db_name = nullptr;
        if (default_value)
        {
            mi::base::Handle<const mi::neuraylib::IValue_resource> default_resource(
                default_value->get_interface<const mi::neuraylib::IValue_resource>());
            default_db_name = default_resource->get_value();
        }

        // get drop down indices
        Section_material_resource_handler* handler =
            Base::m_context->get_resource_handler();
        if (handler)
        {
            mi::Size avail_count = handler->get_available_resource_count(m_resource_kind);
            for (mi::Size i = 0; i < avail_count; ++i)
            {
                const char* db_name = handler->get_available_resource_name(m_resource_kind, i);

                if (strcmp(db_name, m_current_db_name.c_str()) == 0)
                    m_current_index = i;

                if (default_db_name && strcmp(db_name, default_db_name) == 0)
                    m_default_index = i;
            }
        }
    }

    // --------------------------------------------------------------------------------------------

protected:
    void read_value(const mi::neuraylib::IValue_resource* source, mi::Uint32& target) final
    {
        Section_material_resource_handler* handler =
            Base::m_context->get_resource_handler();

        if (handler)
        {
            mi::Size avail_count = handler->get_available_resource_count(m_resource_kind);
            for (mi::Size i = 0; i < avail_count; ++i)
            {
                const char* db_name = handler->get_available_resource_name(m_resource_kind, i);
                if (strcmp(db_name, m_current_db_name.c_str()) == 0)
                {
                    target = handler->get_available_resource_id(m_resource_kind, i);
                    return;
                }
            }
        }
        target = 0;
    }

    // --------------------------------------------------------------------------------------------

    bool update_value() final
    {
        if (Base::m_hidden)
            return false;

        Section_material_resource_handler* handler =
            Base::m_context->get_resource_handler();

        if (handler)
        {
            mi::Size avail_count = handler->get_available_resource_count(m_resource_kind);

            if (Control::selection<mi::Size>(
                Base::get_name(), Base::get_tooltip_function(),
                &m_current_index, &m_default_index,
                (Base::get_unused() || !Base::get_enabled())
                ? Control::Flags::Disabled : Control::Flags::None,
                [&](mi::Size i)
                {
                    if (i >= avail_count)
                        return (const char*) nullptr;
                    return handler->get_available_resource_name(m_resource_kind, i);
                }))
            {
                // update local or argument block
                mi::Uint32* value = Base::m_value_ptr ? Base::m_value_ptr : &(Base::m_value);
                *value = handler->get_available_resource_id(m_resource_kind, m_current_index);
                m_current_db_name =
                    handler->get_available_resource_name(m_resource_kind, m_current_index);

                // update neuray
                mi::base::Handle<mi::neuraylib::IValue> ivalue(Base::get_value());
                mi::base::Handle<mi::neuraylib::IValue_resource> ivalue_texture(
                    ivalue->get_interface<mi::neuraylib::IValue_resource>());
                if (*value == 0)
                    ivalue_texture->set_value(nullptr);
                else
                    ivalue_texture->set_value(m_current_db_name.c_str());
                return true;
            }
        }
        return false;
    }

    // --------------------------------------------------------------------------------------------

private:
    mi::neuraylib::IValue::Kind m_resource_kind;
    std::string m_current_db_name;
    mi::Size m_current_index;
    mi::Size m_default_index;
};


// special handing for textures
typedef Parameter_node_constant_generic_resource
    <Parameter_node_base::Kind::Constant_texture>
    Parameter_node_constant_texture;

// special handing for light profiles
typedef Parameter_node_constant_generic_resource
    <Parameter_node_base::Kind::Constant_light_profile>
    Parameter_node_constant_light_profile;

// special handing for measured BSDFs
typedef Parameter_node_constant_generic_resource
    <Parameter_node_base::Kind::Constant_bsdf_measurement>
    Parameter_node_constant_bsdf_measurement;

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

// abstraction for integer and floating point types
template<typename TIValue, typename TTCTValue, Parameter_node_base::Kind TType>
class Parameter_node_constant_generic_atomic
    : public Parameter_node_constant_generic<TIValue, TTCTValue, TType>
{
    typedef Parameter_node_constant_generic<TIValue, TTCTValue, TType> Base;
public:
    explicit Parameter_node_constant_generic_atomic(
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Base(name, parenting_call, context)
    {
    }

    // --------------------------------------------------------------------------------------------

protected:
    void read_value(const TIValue* source, TTCTValue& target) final
    {
        target = source->get_value();
    }

    // --------------------------------------------------------------------------------------------

    void write_value(const TTCTValue& source, TIValue* target) final
    {
        target->set_value(source);
    }

    // --------------------------------------------------------------------------------------------

    bool update_value() final // see explicit specializations
    {
        return Base::update_value();
    }
};

// special handing for floats
typedef Parameter_node_constant_generic_atomic<
    mi::neuraylib::IValue_float, mi::Float32, Parameter_node_base::Kind::Constant_float>
    Parameter_node_constant_float;

// special handing for doubles
typedef Parameter_node_constant_generic_atomic<
    mi::neuraylib::IValue_double, mi::Float64, Parameter_node_base::Kind::Constant_double>
    Parameter_node_constant_double;

// special handing for ints
typedef Parameter_node_constant_generic_atomic<
    mi::neuraylib::IValue_int, mi::Sint32, Parameter_node_base::Kind::Constant_int>
    Parameter_node_constant_int;

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

// abstraction for integer and floating point vector types
template<typename TIValueElement, typename TTCTValue, Parameter_node_base::Kind TType>
class Parameter_node_constant_generic_vector
    : public Parameter_node_constant_generic<mi::neuraylib::IValue_vector, TTCTValue, TType>
{
    typedef Parameter_node_constant_generic<mi::neuraylib::IValue_vector, TTCTValue, TType>
        Base;
public:
    explicit Parameter_node_constant_generic_vector(
        const char* name,
        Parameter_node_call* paranting_call,
        const Parameter_context* context)
        : Base(name, paranting_call, context)
    {
    }

    // --------------------------------------------------------------------------------------------

protected:
    void read_value(
        const mi::neuraylib::IValue_vector* source,
        TTCTValue& target) override
    {
        for (mi::Size c = 0, n = source->get_size(); c < n; ++c)
        {
            mi::base::Handle<const mi::neuraylib::IValue_atomic> c_atomic(source->get_value(c));
            mi::base::Handle<const TIValueElement> c_value(
                c_atomic->get_interface<const TIValueElement>());
            target[c] = c_value->get_value();
        }
    }

    // --------------------------------------------------------------------------------------------

    void write_value(
        const TTCTValue& source,
        mi::neuraylib::IValue_vector* target) final
    {
        for (mi::Size c = 0, n = target->get_size(); c < n; ++c)
        {
            mi::base::Handle<mi::neuraylib::IValue_atomic> c_atomic(target->get_value(c));
            mi::base::Handle<TIValueElement> c_value(
                c_atomic->get_interface<TIValueElement>());
            c_value->set_value(source[c]);
        }
    }
};

// special handing for float2
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_float, mi::Float32_2, Parameter_node_base::Kind::Constant_float2>
    Parameter_node_constant_float2;

// special handing for float3
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_float, mi::Float32_3, Parameter_node_base::Kind::Constant_float3>
    Parameter_node_constant_float3;

// special handing for float4
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_float, mi::Float32_4, Parameter_node_base::Kind::Constant_float4>
    Parameter_node_constant_float4;

// special handing for double2
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_double, mi::Float64_2, Parameter_node_base::Kind::Constant_double2>
    Parameter_node_constant_double2;

// special handing for double3
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_double, mi::Float64_3, Parameter_node_base::Kind::Constant_double3>
    Parameter_node_constant_double3;

// special handing for double4
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_double, mi::Float64_4, Parameter_node_base::Kind::Constant_double4>
    Parameter_node_constant_double4;

// special handing for int2
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_int, mi::Sint32_2, Parameter_node_base::Kind::Constant_int2>
    Parameter_node_constant_int2;

// special handing for int3
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_int, mi::Sint32_3, Parameter_node_base::Kind::Constant_int3>
    Parameter_node_constant_int3;

// special handing for int4
typedef Parameter_node_constant_generic_vector<
    mi::neuraylib::IValue_int, mi::Sint32_4, Parameter_node_base::Kind::Constant_int4>
    Parameter_node_constant_int4;

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

/// Specialization for enumeration parameters.
class Parameter_node_constant_enum : public Parameter_node_constant
{
public:
    // Constructor.
    explicit Parameter_node_constant_enum(
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_constant(Kind::Constant_enum, name, parenting_call, context)
        , m_value_ptr(nullptr)
        , m_value(0)
        , m_value_index(0)
        , m_default_index(0)
    {
    }

    // --------------------------------------------------------------------------------------------

    void initialize(
        mi::neuraylib::IValue* value,
        const mi::neuraylib::IValue* default_value,
        const mi::neuraylib::IAnnotation_block* annos,
        const Argument_block_field_info* block_info)
    {
        Parameter_node_constant::initialize(value, annos, block_info);

        // get the value from the material instance
        mi::base::Handle<const mi::neuraylib::IValue_enum> value_enum(
            value->get_interface<const mi::neuraylib::IValue_enum>());
        m_value = value_enum->get_value();
        m_value_index = value_enum->get_index();

        mi::base::Handle<const mi::neuraylib::IType_enum> type_enum(
            value_enum->get_type());
        for (mi::Size i = 0, n = type_enum->get_size(); i < n; ++i)
            m_options.push_back(
                { type_enum->get_value_code(i), type_enum->get_value_name(i) });

        // get the default value (if there is one)
        m_default_index = 0;
        if (default_value)
        {
            mi::base::Handle<const mi::neuraylib::IValue_enum> default_generic(
                default_value->get_interface<const mi::neuraylib::IValue_enum>());
            m_default_index = default_generic->get_index();
        }

        // get a pointer directly into the argument block
        if (block_info)
        {
            void* ptr = block_info->argument_block + block_info->offset;
            m_value_ptr = static_cast<mi::Sint32*>(ptr);
        }
    }

    // --------------------------------------------------------------------------------------------

    bool update_value() override
    {
        if (m_hidden)
            return false;

        if (mi::examples::gui::Control::selection<mi::Size>(
            get_name(), get_tooltip_function(), &m_value_index, &m_default_index,
            (get_unused() || !get_enabled())
                ? mi::examples::gui::Control::Flags::Disabled
                : mi::examples::gui::Control::Flags::None,
            [&](mi::Size i)
            {
                if (i >= m_options.size())
                    return (const char*) nullptr;
                return m_options[i].second.c_str();
            }))
        {
            // update local or argument block
            mi::Sint32* value = m_value_ptr ? m_value_ptr : &m_value;
            *value = m_options[m_value_index].first;

            // update neuray
            mi::base::Handle<mi::neuraylib::IValue> ivalue(get_value());
            mi::base::Handle<mi::neuraylib::IValue_enum> ivalue_enum(
                ivalue->get_interface<mi::neuraylib::IValue_enum>());
            ivalue_enum->set_value(*value);
            return true;
        }
            return false;
    }

    // --------------------------------------------------------------------------------------------

private:
    std::vector<std::pair<mi::Sint32, std::string>> m_options;

    mi::Sint32* m_value_ptr;    // a pointer to the data in the argument block
    mi::Sint32 m_value;         // used if the pointer is not set
    mi::Size m_value_index;     // index of the current value into the options list
    mi::Size m_default_index;   // index of the default value into the options list
};


// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

/// Specialization for parameters that are function calls or material instances.
class Parameter_node_call : public Parameter_node_stacked
{
public:
    explicit Parameter_node_call(
        const char* name,
        Parameter_node_call* parenting_call,
        const Parameter_context* context)
        : Parameter_node_stacked(Kind::Call, name, "call", parenting_call, context)
        , m_db_name("")
    {
    }

    // --------------------------------------------------------------------------------------------

    void initialize(const mi::neuraylib::IAnnotation_block* annos, const char* db_name)
    {
        Parameter_node_base::initialize(annos);
        m_db_name = db_name;
    }

    // --------------------------------------------------------------------------------------------

    // stores the value of a changed child in the material instance or function call
    // returns true, if that change comes with a structural change which requires a
    // recompilation of the material
    Section_material_update_state store_updated_value(
        mi::neuraylib::ITransaction* transaction,
        mi::neuraylib::IMdl_factory* factory,
        Parameter_node_base* child)
    {
        Section_material_update_state change = Section_material_update_state::No_change;

        Parameter_node_base::Kind child_kind = child->get_kind();
        if (child_kind == Parameter_node_base::Kind::UNKNOWN)
            return change; // not supported type

        mi::Size param_index = get_parameter_index(child);
        mi::base::Handle<mi::neuraylib::IValue> child_value;

        mi::base::Handle<mi::neuraylib::IScene_element> access(
            transaction->edit<mi::neuraylib::IScene_element>(get_db_name()));

        // handle constants
        if (child_kind <= Parameter_node_base::Kind::LAST_DIRECT)
        {
            Parameter_node_constant* const_param_child =
                static_cast<Parameter_node_constant*>(child);
            child_value =
                mi::base::Handle<mi::neuraylib::IValue>(const_param_child->get_value());

            change = const_param_child->get_is_light_weight()
                ? Section_material_update_state::Argument_block_change
                : Section_material_update_state::Unknown_change;
        }
        // handle compounds
        else if (child_kind <= Parameter_node_base::Kind::LAST_COMPOUND)
        {
            Parameter_node_stacked_compound* compound_child =
                static_cast<Parameter_node_stacked_compound*>(child);
            child_value =
                mi::base::Handle<mi::neuraylib::IValue>(compound_child->get_value());
        }

        // update instance/call
        if (child_value)
        {
            // handle material instances and functions calls
            base::Handle<mi::neuraylib::IMaterial_instance> mati(
                access->get_interface<mi::neuraylib::IMaterial_instance>());
            base::Handle<mi::neuraylib::IFunction_call> fc(
                access->get_interface<mi::neuraylib::IFunction_call>());

            mi_neuray_assert((mati || fc) &&
                "Database element to update is not a function call nor a material instance");

            base::Handle<const mi::neuraylib::IExpression_list> arguments(
                mati ? mati->get_arguments() : fc->get_arguments());

            base::Handle<const mi::neuraylib::IExpression> argument(
                arguments->get_expression(param_index));
            mi_neuray_assert(argument && "Parameter_index is out of range");

            base::Handle<const mi::neuraylib::IType> argument_type(argument->get_type());
            base::Handle<const mi::neuraylib::IType> value_type(child_value->get_type());
            mi_neuray_assert(argument_type->get_kind() == value_type->get_kind() &&
                "The type of the argument and the value do not match");

            mi::base::Handle<mi::neuraylib::IValue_factory> vf(
                factory->create_value_factory(transaction));

            mi::base::Handle<mi::neuraylib::IExpression_factory> ef(
                factory->create_expression_factory(transaction));

            base::Handle<mi::neuraylib::IValue> new_value(vf->clone(child_value.get()));
            base::Handle<mi::neuraylib::IExpression> new_expression(
                ef->create_constant(new_value.get()));
            Sint32 result = mati
                ? mati->set_argument(param_index, new_expression.get())
                : fc->set_argument(param_index, new_expression.get());

            if (result != 0)
                mi_neuray_assert(false && "Setting value failed");
        }

        return change;
    }

    // --------------------------------------------------------------------------------------------

    /// Get the database name of the function call or material instance.
    const char* get_db_name() const
    {
        return m_db_name.c_str();
    }

    // --------------------------------------------------------------------------------------------

private:
    std::string m_db_name;
};

// ------------------------------------------------------------------------------------------------
// template specializations to handle individual behaviors
// ------------------------------------------------------------------------------------------------

template<>
void Parameter_node_constant_bool::read_value(
    const mi::neuraylib::IValue_bool* source,
    bool& target)
{
    target = source->get_value();
}

// ------------------------------------------------------------------------------------------------

template<>
bool Parameter_node_constant_bool::update_value()
{
    // update local or argument block
    bool* value = m_value_ptr ? m_value_ptr : &m_value;
    if (mi::examples::gui::Control::checkbox(get_name(), get_tooltip_function(),
        value, &get_default(),
        (get_unused() || !get_enabled())
            ? mi::examples::gui::Control::Flags::Disabled
            : mi::examples::gui::Control::Flags::None))
    {
        // update neuray value
        mi::base::Handle<mi::neuraylib::IValue> ivalue(get_value());
        mi::base::Handle<mi::neuraylib::IValue_bool> ivalue_bool(
            ivalue->get_interface<mi::neuraylib::IValue_bool>());
        ivalue_bool->set_value(*value);
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

template<>
void Parameter_node_constant_unknown::read_value(
    const mi::neuraylib::IValue* source,
    bool& target)
{
    target = false;
}

// ------------------------------------------------------------------------------------------------

template<>
bool Parameter_node_constant_unknown::update_value()
{
    return false;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

template<>
void Parameter_node_constant_color::read_value(
    const mi::neuraylib::IValue_color* source,
    mi::Float32_3& target)
{
    mi::base::Handle<const mi::neuraylib::IValue_float> x(source->get_value(0));
    target.x = x->get_value();

    mi::base::Handle<const mi::neuraylib::IValue_float> y(source->get_value(1));
    target.y = y->get_value();

    mi::base::Handle<const mi::neuraylib::IValue_float> z(source->get_value(2));
    target.z = z->get_value();
}

// ------------------------------------------------------------------------------------------------

template<>
mi::Float32_3 Parameter_node_constant_color::max(
    const mi::Float32_3& a,
    const mi::Float32_3& b)
{
    mi::Float32_3 max;
    max.x = std::max(a.x, b.x);
    max.y = std::max(a.y, b.y);
    max.z = std::max(a.z, b.z);
    return max;
}

// ------------------------------------------------------------------------------------------------

template<>
mi::Float32_3 Parameter_node_constant_color::min(
    const mi::Float32_3& a,
    const mi::Float32_3& b)
{
    mi::Float32_3 min;
    min.x = std::min(a.x, b.x);
    min.y = std::min(a.y, b.y);
    min.z = std::min(a.z, b.z);
    return min;
}

// ------------------------------------------------------------------------------------------------

template<>
bool Parameter_node_constant_color::update_value()
{
    if (m_hidden)
        return false;

    // update local or argument block
    mi::Float32_3* value = m_value_ptr ? m_value_ptr : &m_value;
    mi::Float32_3 current = *value;
    if (mi::examples::gui::Control::pick(get_name(), get_tooltip_function(),
        &value->x, &get_default().x,
        (get_unused() || !get_enabled())
            ? mi::examples::gui::Control::Flags::Disabled
            : mi::examples::gui::Control::Flags::None))
    {
        bool changed = !m_has_hard_range;
        if (m_has_hard_range)
        {
            *value = max(*value, m_range_min); // clamp in case of hard range
            *value = min(*value, m_range_max);
            changed |= current.x != value->x;
            changed |= current.y != value->y;
            changed |= current.z != value->z;
        }

        // update neuray value
        if (changed)
        {
            mi::base::Handle<mi::neuraylib::IValue> ivalue(get_value());
            mi::base::Handle<mi::neuraylib::IValue_color> ivalue_color(
                ivalue->get_interface<mi::neuraylib::IValue_color>());

            for (mi::Size c = 0; c < 3; ++c)
            {
                mi::base::Handle<mi::neuraylib::IValue> ivalue_c(ivalue_color->get_value(c));
                mi::base::Handle<mi::neuraylib::IValue_float> ivalue_float_c(
                    ivalue_c->get_interface<mi::neuraylib::IValue_float>());
                ivalue_float_c->set_value((*value)[c]);
            }
        }
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

template<>
void Parameter_node_constant_string_cc::read_value(
    const mi::neuraylib::IValue_string* source,
    mi::Uint32& target)
{
    target = m_context->get_id_for_string(source->get_value());
}

// ------------------------------------------------------------------------------------------------

template<>
bool Parameter_node_constant_string_cc::update_value()
{
    if (m_hidden)
        return false;

    mi::Uint32* value = m_value_ptr ? m_value_ptr : &m_value;
    static std::vector<char> buf;

    size_t max_len = m_context->get_max_string_length();
    max_len = max_len > 63 ? max_len + 1 : 64;
    buf.resize(max_len);

    // fill the current value
    const char* opt = m_context->get_string(*value);
    const char* default_opt = m_context->get_string(get_default());
    strcpy(buf.data(), opt != nullptr ? opt : "");

    if (mi::examples::gui::Control::text(get_name(), get_tooltip_function(),
        buf.data(), max_len, default_opt,
        (get_unused() || !get_enabled())
            ? mi::examples::gui::Control::Flags::Disabled
            : mi::examples::gui::Control::Flags::None))
    {
        mi::Uint32 index = m_context->get_id_for_string(buf.data());
        if (*value == index)
            return false;

        // update local or argument block
        *value = index;

        // update neuray value
        mi::base::Handle<mi::neuraylib::IValue> ivalue(get_value());
        mi::base::Handle<mi::neuraylib::IValue_string> ivalue_string(
            ivalue->get_interface<mi::neuraylib::IValue_string>());
        ivalue_string->set_value(m_context->get_string(*value));
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

template<>
void Parameter_node_constant_string_ic::read_value(
    const mi::neuraylib::IValue_string* source,
    std::string& target)
{
    target = source->get_value();
}

// ------------------------------------------------------------------------------------------------

template<>
bool Parameter_node_constant_string_ic::update_value()
{
    if (m_hidden)
        return false;

    const static size_t max_len = 255;
    static std::vector<char> buf(max_len + 1, '\0');

    // fill the current value
    const char* opt = m_value.c_str();
    const char* default_opt = m_value_default.c_str();
    strcpy(buf.data(), opt != nullptr ? opt : "");

    if (mi::examples::gui::Control::text(get_name(), get_tooltip_function(),
        buf.data(), max_len, default_opt,
        (get_unused() || !get_enabled())
            ? mi::examples::gui::Control::Flags::Disabled
            : mi::examples::gui::Control::Flags::None))
    {
        const std::string value = buf.data();
        if (value == m_value)
            return false;

        // update local or argument block
        m_value = value;

        // update neuray value
        mi::base::Handle<mi::neuraylib::IValue> ivalue(get_value());
        mi::base::Handle<mi::neuraylib::IValue_string> ivalue_string(
            ivalue->get_interface<mi::neuraylib::IValue_string>());
        ivalue_string->set_value(m_value.c_str());
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
// define forward declared methods
// ------------------------------------------------------------------------------------------------

void Parameter_node_base::add_element(Parameter_node_base* element)
{
    // add element to the parameter list
    m_parameters_to_index[element] = m_parameters.size();
    m_parameters.push_back(element);

    // create the group hierarchy
    Parameter_node_base* parent = this;
    for (auto& g : element->m_groups)
    {
        auto found = std::find_if(parent->m_child_nodes.begin(), parent->m_child_nodes.end(),
            [&](const Parameter_node_base* a)
            {
                return (a->m_node_kind == Parameter_node_base::Kind::Group) &&
                    (a->m_name == g);
            });

        if (found == parent->m_child_nodes.end())
        {
            Parameter_node_group* created =
                new Parameter_node_group(
                    g.c_str(), element->get_parent_call(), element->m_context);
            parent->m_child_nodes.emplace_back(created);
            created->m_parent_node = parent;
            parent = created;
        }
        else
            parent = *found;
    }

    // add element to its parent
    element->m_parent_node = parent;
    parent->m_child_nodes.emplace_back(element);
}

} // anonymous

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

Section_material::Section_material(
    Root* gui,
    const std::string& name,
    bool collapsed_by_default,
    mi::neuraylib::IMdl_evaluator_api* evaluator,
    mi::neuraylib::IMdl_factory* factory)
    : Section(gui, name, collapsed_by_default)
    , m_material_parameters(nullptr)
    , m_material_parameters_context(nullptr)
    , m_evaluator(mi::base::make_handle_dup(evaluator))
    , m_factory(mi::base::make_handle_dup(factory))
    , m_update_state(Section_material_update_state::No_change)
{
}

// ------------------------------------------------------------------------------------------------

Section_material::~Section_material()
{
    m_evaluator = nullptr;
    m_factory = nullptr;
}

// ------------------------------------------------------------------------------------------------

void Section_material::unbind_material()
{
    if (m_material_parameters)
    {
        delete m_material_parameters;
        m_material_parameters = nullptr;
    }

    if (m_material_parameters_context)
    {
        delete m_material_parameters_context;
        m_material_parameters_context = nullptr;
    }

    while (!m_call_history.empty())
        m_call_history.pop();
}

// ------------------------------------------------------------------------------------------------

void Section_material::bind_material(
    mi::neuraylib::ITransaction* transaction,
    const char* material_instance_db_name,
    const mi::neuraylib::ICompiled_material* compiled_material,
    const mi::neuraylib::ITarget_code* target_code,
    const mi::neuraylib::ITarget_value_layout* argument_block_layout,
    char* argument_block,
    Section_material_resource_handler* resource_handler)
{
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance(
        transaction->access<const mi::neuraylib::IMaterial_instance>(
            material_instance_db_name));

    Parameter_context* parameter_context = new Parameter_context(
        argument_block_layout != nullptr && argument_block != nullptr,
        target_code,
        material_instance_db_name,
        material_instance->get_material_definition(),
        resource_handler);

    // iterate over all parameters exposed in the compiled material
    for (mi::Size j = 0, n = compiled_material->get_parameter_count(); j < n; ++j)
    {
        mi::base::Handle<mi::neuraylib::IValue const> arg(compiled_material->get_argument(j));

        Argument_block_field_info info;
        info.name = compiled_material->get_parameter_name(j);
        info.argument_block = argument_block;
        info.argument_layout = argument_block_layout;

        // get the offset of the argument within the argument block
        if (info.argument_layout && info.argument_block)
        {
            info.state = argument_block_layout->get_nested_state(j);
            info.offset = argument_block_layout->get_layout(info.kind, info.size, info.state);
        }
        parameter_context->add_argument_block_info(info.name, info);
    }

    mi::base::Handle<const mi::neuraylib::IMaterial_definition> mat_definition(
        transaction->access<const mi::neuraylib::IMaterial_definition>(
            parameter_context->get_material_definition_db_name()));

    mi::base::Handle<const mi::neuraylib::IAnnotation_block> mat_annotations(
        mat_definition->get_annotations());

    mi::base::Handle<const mi::neuraylib::IExpression_list> param_values(
        material_instance->get_arguments());

    mi::base::Handle<mi::neuraylib::IAnnotation_list const> param_annotations(
        mat_definition->get_parameter_annotations());

    Parameter_node_call* new_material_parameters =
        new Parameter_node_call("", nullptr, parameter_context);
    new_material_parameters->initialize(mat_annotations.get(), material_instance_db_name);

    mi::base::Handle<const mi::neuraylib::IExpression_list> param_defaults(
        mat_definition->get_defaults());

    mi::neuraylib::Argument_editor ae(
        transaction,
        material_instance_db_name,
        m_factory.get());

    // iterate over all parameters of the material instance. These are usually different from
    // the parameters above! However, for displaying a UI to the user, we want to be as close
    // as possible to the parameters of the instance rather than displaying an unstructured
    // list of all parameters the compiler left.
    std::vector<Parameter_node_base*> root_parameters(
        material_instance->get_parameter_count(), nullptr);

    for (mi::Size j = 0, n = material_instance->get_parameter_count(); j < n; ++j)
    {
        const char* name = material_instance->get_parameter_name(j);

        mi::base::Handle<const mi::neuraylib::IExpression> param_value(
            param_values->get_expression(name));

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> param_anno(
            param_annotations->get_annotation_block(name));

        mi::base::Handle<const mi::neuraylib::IExpression> param_default(
            param_defaults->get_expression(name));

        mi::Size root_parameter_index = mat_definition->get_parameter_index(name);
        mi_neuray_assert(root_parameter_index == j);

        Parameter_node_base* param = static_cast<Parameter_node_base*>(create(
            transaction,
            m_factory.get(),
            name,
            new_material_parameters,
            param_value.get(),
            param_default.get(),
            param_anno.get(),
            parameter_context));

        if (!param)
        {
            mi_neuray_assert(param && "Parameter node creation failed.");
            continue;
        }

        // initialize enable_if state
        if (root_parameter_index != static_cast<mi::Size>(-1))
        {
            bool enabled = ae.is_parameter_enabled(root_parameter_index, m_evaluator.get());
            param->set_enabled(enabled);
        }
        new_material_parameters->add_element(param);
    }

    // sort parameters based on the order annotation
    new_material_parameters->sort();

    // replace the old hierarchy with the new one
    unbind_material();
    m_material_parameters = new_material_parameters;
    m_material_parameters_context = parameter_context;
}

// ------------------------------------------------------------------------------------------------

void Section_material::update(mi::neuraylib::ITransaction* transaction)
{
    if (!m_material_parameters)
    {
        ImGui::Text("no material selected");
        return;
    }

    if (m_call_history.empty())
    {
        ImGui::Text("Material Parameters:");
        ImGui::Separator();
    }
    else
    {
        auto top = static_cast<Parameter_node_stacked*>(m_call_history.top());
        ImGui::Text("%s-Attachment Parameters:", top->get_name().c_str());
        ImGui::Separator();
        if (ImGui::Button("back", ImVec2(ImGui::GetContentRegionAvail().x, 0)))
            m_call_history.pop();
        ImGui::Spacing();
    }

    auto new_state = update_material_parameters(transaction, m_call_history.empty()
        ? m_material_parameters
        : m_call_history.top());

    // keep the most disrupting change until the next reset
    m_update_state = static_cast<Section_material_update_state>(std::max(
        static_cast<size_t>(m_update_state), static_cast<size_t>(new_state)));
}

// ------------------------------------------------------------------------------------------------

Section_material_update_state Section_material::get_update_state() const
{
    return m_update_state;
}

// ------------------------------------------------------------------------------------------------

void Section_material::reset_update_state()
{
    m_update_state = Section_material_update_state::No_change;
}

// ------------------------------------------------------------------------------------------------

Section_material::IParameter_node* Section_material::create(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* factory,
    const char* name,
    IParameter_node* parenting_call,
    const mi::neuraylib::IExpression* expression,
    const mi::neuraylib::IExpression* default_value,
    const mi::neuraylib::IAnnotation_block* annos,
    IParameter_context* context)
{
    // dots are used as separator in the parameter hierarchy
    // child elements will have the current name as prefix
    std::string name_prefix = std::string(name) + ".";

    // call
    mi::neuraylib::IExpression::Kind expression_kind = expression->get_kind();
    if (expression_kind == mi::neuraylib::IExpression::EK_CALL)
    {
        mi::base::Handle<const mi::neuraylib::IExpression_call> call(
            expression->get_interface<const mi::neuraylib::IExpression_call>());

        const char* call_db_name = call->get_call();

        Parameter_node_call* created = new Parameter_node_call(
            name,
            static_cast<Parameter_node_call*>(parenting_call),
            static_cast<const Parameter_context*>(context));
        created->initialize(annos, call_db_name);

        mi::base::Handle<const mi::neuraylib::IExpression_list> params;
        mi::base::Handle<const mi::neuraylib::IExpression_list> param_defaults;
        mi::base::Handle<mi::neuraylib::IAnnotation_list const> param_annotations;

        mi::base::Handle<const mi::neuraylib::IFunction_call> func_call(
            transaction->access<const mi::neuraylib::IFunction_call>(call_db_name));
        if (func_call)
        {
            mi::base::Handle<const mi::neuraylib::IFunction_definition> func_definition(
                transaction->access<const mi::neuraylib::IFunction_definition>(
                    func_call->get_function_definition()));

            params = func_call->get_arguments();
            param_defaults = func_definition->get_defaults();
            param_annotations = func_definition->get_parameter_annotations();
        }
        else
        {
            mi::base::Handle<const mi::neuraylib::IMaterial_instance> mat_inst(
                transaction->access<const mi::neuraylib::IMaterial_instance>(call_db_name));

            mi::base::Handle<const mi::neuraylib::IMaterial_definition> mat_definition(
                transaction->access<const mi::neuraylib::IMaterial_definition>(
                    mat_inst->get_material_definition()));

            params = mat_inst->get_arguments();
            param_defaults = mat_definition->get_defaults();
            param_annotations = mat_definition->get_parameter_annotations();
        }

        for (mi::Size j = 0, n = params->get_size(); j < n; ++j)
        {
            const char* arg_name = params->get_name(j);

            mi::base::Handle<const mi::neuraylib::IExpression> param_value(
                params->get_expression(arg_name));

            mi::base::Handle<const mi::neuraylib::IExpression> param_default(
                param_defaults->get_expression(name));

            mi::base::Handle<const mi::neuraylib::IAnnotation_block> param_anno(
                param_annotations->get_annotation_block(name));

            std::string child_name = name_prefix + arg_name;
            Parameter_node_base* child = static_cast<Parameter_node_base*>(create(
                transaction,
                factory,
                child_name.c_str(),
                created,
                param_value.get(),
                param_default.get(),
                param_anno.get(),
                context));

            created->add_element(child);
        }

        return created;
    }

    // constant
    if (expression_kind == mi::neuraylib::IExpression::EK_CONSTANT)
    {
        mi::base::Handle<const mi::neuraylib::IExpression_constant> constant(
            expression->get_interface<const mi::neuraylib::IExpression_constant>());

        mi::base::Handle<const mi::neuraylib::IValue> constant_value(
            constant->get_value());

        mi::base::Handle<const mi::neuraylib::IValue> default_constant_value(nullptr);

        if (default_value)
        {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> default_constant(
                default_value->get_interface<const mi::neuraylib::IExpression_constant>());

            if (default_constant)
                default_constant_value = mi::base::Handle<const mi::neuraylib::IValue>(
                    default_constant->get_value());
        }

        return create(
            transaction,
            factory,
            name,
            parenting_call,
            constant_value.get(),
            default_constant_value.get(),
            annos,
            context);
    }
    return nullptr;
}

// ------------------------------------------------------------------------------------------------

Section_material::IParameter_node* Section_material::create(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_factory* factory,
    const char* name,
    IParameter_node* parenting_call,
    const mi::neuraylib::IValue* value,
    const mi::neuraylib::IValue* default_value,
    const mi::neuraylib::IAnnotation_block* annos,
    IParameter_context* context)
{
    Parameter_node_base* created = nullptr;

    auto ctx = static_cast<Parameter_context*>(context);
    auto p_call = static_cast<Parameter_node_call*>(parenting_call);

    mi::neuraylib::IValue::Kind value_kind = value->get_kind();
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(factory->create_value_factory(transaction));
    mi::base::Handle<mi::neuraylib::IValue> value_copy(vf->clone(value));


    // dots are used as separator in the parameter hierarchy
    // child elements will have the current name as prefix
    std::string name_prefix = std::string(name) + ".";
    const Argument_block_field_info* arg_info = ctx->get_argument_block_info(name);

    // keep the simple name of the parameter as default name
    std::string param_name = name;
    size_t p = param_name.rfind('.');
    if (p != std::string::npos && p < param_name.length() - 1)
        param_name = param_name.substr(p + 1);
    name = param_name.c_str();

    switch (value_kind)
    {
    case mi::neuraylib::IValue::VK_BOOL:
    {
        auto constant = new Parameter_node_constant_bool(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_INT:
    {
        auto constant = new Parameter_node_constant_int(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_ENUM:
    {
        auto constant = new Parameter_node_constant_enum(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_FLOAT:
    {
        auto constant = new Parameter_node_constant_float(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_DOUBLE:
    {
        auto constant = new Parameter_node_constant_double(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_COLOR:
    {
        auto constant = new Parameter_node_constant_color(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_STRING:
    {
        if (ctx->get_class_compilation_mode())
        {
            auto c = new Parameter_node_constant_string_cc(name, p_call, ctx);
            c->initialize(value_copy.get(), default_value, annos, arg_info);
            created = c;
        }
        else
        {
            auto c = new Parameter_node_constant_string_ic(name, p_call, ctx);
            c->initialize(value_copy.get(), default_value, annos, arg_info);
            created = c;
        }
        break;
    }
    case mi::neuraylib::IValue::VK_TEXTURE:
    {
        auto constant = new Parameter_node_constant_texture(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
    {
        auto constant = new Parameter_node_constant_light_profile(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
    {
        auto constant = new Parameter_node_constant_bsdf_measurement(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
        break;
    }
    case mi::neuraylib::IValue::VK_VECTOR:
    {
        mi::base::Handle<mi::neuraylib::IValue_vector const> val(
            value->get_interface<mi::neuraylib::IValue_vector const>());
        mi::base::Handle<mi::neuraylib::IType_vector const> val_type(
            val->get_type());
        mi::base::Handle<mi::neuraylib::IType_atomic const> elem_type(
            val_type->get_element_type());

        if (elem_type->get_kind() == mi::neuraylib::IType::TK_FLOAT)
        {
            switch (val_type->get_size())
            {
            case 2:
            {
                auto constant = new Parameter_node_constant_float2(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            case 3:
            {
                auto constant = new Parameter_node_constant_float3(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            case 4:
            {
                auto constant = new Parameter_node_constant_float4(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            }
        }

        if (elem_type->get_kind() == mi::neuraylib::IType::TK_DOUBLE)
        {
            switch (val_type->get_size())
            {
            case 2:
            {
                auto constant = new Parameter_node_constant_double2(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            case 3:
            {
                auto constant = new Parameter_node_constant_double3(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            case 4:
            {
                auto constant = new Parameter_node_constant_double4(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            }
        }

        if (elem_type->get_kind() == mi::neuraylib::IType::TK_INT)
        {
            switch (val_type->get_size())
            {
            case 2:
            {
                auto constant = new Parameter_node_constant_int2(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            case 3:
            {
                auto constant = new Parameter_node_constant_int3(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            case 4:
            {
                auto constant = new Parameter_node_constant_int4(name, p_call, ctx);
                constant->initialize(value_copy.get(), default_value, annos, arg_info);
                created = constant;
                break;
            }
            }
        }
        break;
    }
    case mi::neuraylib::IValue::VK_STRUCT:
    {
        auto s = new Parameter_node_struct(name, p_call, ctx);

        mi::base::Handle<mi::neuraylib::IValue_struct> struct_value(
            value_copy->get_interface<mi::neuraylib::IValue_struct>());

        mi::base::Handle<const mi::neuraylib::IValue_struct> struct_default_value(default_value
            ? default_value->get_interface<const mi::neuraylib::IValue_struct>()
            : nullptr);

        s->initialize(struct_value.get(), annos, arg_info);
        created = s;

        mi::base::Handle<const mi::neuraylib::IType> type(struct_value->get_type());
        mi::base::Handle<const mi::neuraylib::IType_struct> struct_type(
            type.get_interface<const mi::neuraylib::IType_struct>());

        for (mi::Size i = 0, n = struct_type->get_size(); i < n; ++i)
        {
            mi::base::Handle<const mi::neuraylib::IValue> field_value(
                struct_value->get_value(i));

            mi::base::Handle<const mi::neuraylib::IValue> field_default_value(struct_default_value
                ? struct_default_value->get_value(i)
                : nullptr);

            mi::base::Handle<const mi::neuraylib::IAnnotation_block> param_anno(
                struct_type->get_field_annotations(i));

            // get name from the structure type first
            const char* field_name = struct_type->get_field_name(i);
            std::string block_name = name_prefix + field_name;

            // fill remaining details from with infos from the layout
            if (arg_info)
            {
                Argument_block_field_info info;
                info.name = field_name;
                info.argument_layout = arg_info->argument_layout;
                info.argument_block = arg_info->argument_block;

                // get the offset of the argument within the argument block
                if (info.argument_layout && info.argument_block)
                {
                    info.state = info.argument_layout->get_nested_state(
                        i, arg_info->state);

                    info.offset = info.argument_layout->get_layout(
                        info.kind, info.size, info.state);
                }

                ctx->add_argument_block_info(block_name, info);
            }

            Parameter_node_base* child = static_cast<Parameter_node_base*>(create(
                transaction,
                factory,
                block_name.c_str(),
                parenting_call,
                field_value.get(),
                field_default_value.get(),
                param_anno.get(),
                context));

            if (child)
            {
                // use the field name as display name if there wasn't a annotation
                if (child->get_display_name().empty())
                    child->set_display_name(field_name);

                created->add_element(child);
            }
        }
        break;
    }
    case mi::neuraylib::IValue::VK_ARRAY:
    {
        auto s = new Parameter_node_array(name, p_call, ctx);

        mi::base::Handle<mi::neuraylib::IValue_array> array_value(
            value_copy->get_interface<mi::neuraylib::IValue_array>());

        mi::base::Handle<const mi::neuraylib::IValue_array> array_default_value(default_value
            ? default_value->get_interface<const mi::neuraylib::IValue_array>()
            : nullptr);

        s->initialize(array_value.get(), annos, arg_info);
        created = s;

        // possible extension: check if the IType_array size. If thats a deferred size
        // array, adding and removing items could be possible
        for (mi::Size i = 0; i < array_value->get_size(); ++i)
        {
            mi::base::Handle<const mi::neuraylib::IValue> field_value(
                array_value->get_value(i));

            mi::base::Handle<const mi::neuraylib::IValue> field_default_value(array_default_value
                ? array_default_value->get_value(i)
                : nullptr);

            // get name from the structure type first
            std::string field_name = "[" + std::to_string(i) + "]";
            std::string block_name = name_prefix + field_name;

            // fill remaining details from with infos from the layout
            if (arg_info)
            {
                Argument_block_field_info info;
                info.name = field_name;
                info.argument_layout = arg_info->argument_layout;
                info.argument_block = arg_info->argument_block;

                // get the offset of the argument within the argument block
                if (info.argument_layout && info.argument_block)
                {
                    info.state = info.argument_layout->get_nested_state(
                        i, arg_info->state);

                    info.offset = info.argument_layout->get_layout(
                        info.kind, info.size, info.state);
                }

                ctx->add_argument_block_info(block_name, info);
            }

            Parameter_node_base* child = static_cast<Parameter_node_base*>(create(
                transaction,
                factory,
                block_name.c_str(),
                parenting_call,
                field_value.get(),
                field_default_value.get(),
                nullptr,
                context));

            if (child)
            {
                if (child->get_display_name().empty())
                    child->set_display_name(created->get_name() + " [" + std::to_string(i) + "]");

                created->add_element(child);
            }
        }
        break;
    }


    // TODO
    case mi::neuraylib::IValue::VK_MATRIX:
    case mi::neuraylib::IValue::VK_INVALID_DF:
    default:
        auto constant = new Parameter_node_constant_unknown(name, p_call, ctx);
        constant->initialize(value_copy.get(), default_value, annos, arg_info);
        created = constant;
    }

    return created;
}

// ------------------------------------------------------------------------------------------------

Section_material_update_state Section_material::update_material_parameters(
    mi::neuraylib::ITransaction* transaction,
    IParameter_node* call)
{
    std::function<Section_material_update_state(Parameter_node_base*, const std::vector<Parameter_node_base*>)>
        show_children = [&](
            Parameter_node_base* parent,
            const std::vector<Parameter_node_base*> children)
    {
        Section_material_update_state change = Section_material_update_state::No_change;
        for (auto param : children)
        {
            if (!param) // skip unsupported parameter types
                continue;

            if (param->get_kind() == Parameter_node_base::Kind::UNKNOWN)
                continue;

            // simple constant values create their own controls
            if (param->get_kind() <= Parameter_node_base::Kind::LAST_DIRECT)
            {
                Parameter_node_constant* child = dynamic_cast<Parameter_node_constant*>(param);
                if (!child)
                    continue;

                if (child->update_value())
                {
                    // get the parent call and in order to update the material instance or
                    // the function call
                    Parameter_node_call* call = child->get_parent_call();
                    Parameter_node_base* current = child;
                    mi::Size invalid_index = static_cast<mi::Size>(-1);
                    mi::Size call_index = call->get_parameter_index(current);
                    while (call_index == invalid_index)
                    {
                        current = current->get_parent_node();
                        if (current == nullptr)
                            break;

                        // store values in compounds first
                        if (current->get_kind() > Parameter_node_base::Kind::LAST_DIRECT&&
                            current->get_kind() <= Parameter_node_base::Kind::LAST_COMPOUND)
                        {
                            Parameter_node_stacked_compound* compound =
                                static_cast<Parameter_node_stacked_compound*>(current);

                            // store changes in the neuray instance
                            change = max(change, compound->store_updated_value());
                        }

                        call_index = call->get_parameter_index(current);
                    }
                    if (call_index == invalid_index)
                        continue;

                    // store changes in the neuray instance
                    change = max(change, call->store_updated_value(
                        transaction, m_factory.get(), current));

                    // check if this parameter is part of any enable_if condition and
                    // re-evaluate the enabled conditions for those parameters
                    mi::neuraylib::Argument_editor ae(
                        transaction, call->get_db_name(), m_factory.get());

                    mi::neuraylib::Definition_wrapper definition(
                        transaction, ae.get_definition(), m_factory.get());

                    mi::Size users = definition.get_enable_if_users(call_index);
                    if (users > 0)
                    {
                        // iterate over all parameters that depend on the changed one
                        for (mi::Size i = 0; i < users; ++i)
                        {
                            mi::Size dep_par_idx = definition.get_enable_if_user(call_index, i);
                            bool enabled =
                                ae.is_parameter_enabled(dep_par_idx, m_evaluator.get());
                            call->get_parameter(dep_par_idx)->set_enabled(enabled);
                        }
                    }
                }
                continue;
            }

            // calls, structures, arrays, ... create their own controls
            if (param->get_kind() <= Parameter_node_base::Kind::LAST_STACKED)
            {
                Parameter_node_stacked* child = static_cast<Parameter_node_stacked*>(param);
                if (child->show())
                    m_call_history.push(child);
                continue;
            }

            // groups can be collapsed and expanded. The show a label and a bar and render
            // their children next directly
            if (param->get_kind() == Parameter_node_base::Kind::Group)
            {
                Parameter_node_group* child_group = static_cast<Parameter_node_group*>(param);
                change = max(change, Control::group<Section_material_update_state>(
                    child_group->get_name().c_str(), child_group->is_expanded(), [&]()
                    {
                        return show_children(parent, child_group->get_childeren());
                    }));
                continue;
            }
        }
        return change;
    };

    // start the updates
    auto c = static_cast<Parameter_node_call*>(call);
    Section_material_update_state change = show_children(c, c->get_childeren());
    return change;
}

}}}
