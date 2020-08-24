/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MATERIAL_INFO_HELPER_H
#define MATERIAL_INFO_HELPER_H

#include <list>
#include <map>
#include <memory>
#include <vector>

#include "example_shared.h"

namespace mdl_d3d12
{

    // Possible enum values if any.
    struct Enum_value
    {
        std::string name;
        int         value;

        Enum_value(const std::string &name, int value)
            : name(name), value(value)
        {
        }
    };


    // Info for an enum type.
    struct Enum_type_info
    {
        std::vector<Enum_value> values;

        // Adds a enum value and its integer value to the enum type info.
        void add(const std::string &name, int value)
        {
            values.push_back(Enum_value(name, value));
        }
    };


    // Material parameter information structure.
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
            PK_ARRAY,
            PK_BOOL,
            PK_INT,
            PK_ENUM,
            PK_STRING,
            PK_TEXTURE,
            PK_LIGHT_PROFILE,
            PK_BSDF_MEASUREMENT
        };

        Param_info(
            size_t index,
            char const *name,
            char const *display_name,
            char const *group_name,
            Param_kind kind,
            Param_kind array_elem_kind,
            mi::Size   array_size,
            mi::Size   array_pitch,
            char *data_ptr,
            const Enum_type_info *enum_info = nullptr)
            : m_index(index)
            , m_name(name)
            , m_display_name(display_name)
            , m_group_name(group_name)
            , m_kind(kind)
            , m_array_elem_kind(array_elem_kind)
            , m_array_size(array_size)
            , m_array_pitch(array_pitch)
            , m_data_ptr(data_ptr)
            , m_range_min(0), m_range_max(1)
            , m_enum_info(enum_info)
        {
        }

        // Get data as T&.
        template<typename T>
        T &data() { return *reinterpret_cast<T *>(m_data_ptr); }

        // Get data as const T&.
        template<typename T>
        const T &data() const { return *reinterpret_cast<const T *>(m_data_ptr); }

        const char *display_name() const { return m_display_name.c_str(); }
        void set_display_name(const char *display_name)
        {
            m_display_name = display_name;
        }

        const char *group_name() const { return m_group_name.c_str(); }
        void set_group_name(const char *group_name)
        {
            m_group_name = group_name;
        }

        Param_kind kind() const { return m_kind; }

        Param_kind array_elem_kind() const { return m_array_elem_kind; }
        mi::Size array_size() const { return m_array_size; }
        mi::Size array_pitch() const { return m_array_pitch; }

        float &range_min() { return m_range_min; }
        float range_min() const { return m_range_min; }
        float &range_max() { return m_range_max; }
        float range_max() const { return m_range_max; }

        template<typename T, int N = 1>
        void update_range()
        {
            T *val_ptr = &data<T>();
            for (int i = 0; i < N; ++i)
            {
                float val = float(val_ptr[i]);
                if (val < m_range_min)
                    m_range_min = val;
                if (m_range_max < val)
                    m_range_max = val;
            }
        }

        const Enum_type_info *enum_info() const { return m_enum_info; }

    private:
        size_t               m_index;
        std::string          m_name;
        std::string          m_display_name;
        std::string          m_group_name;
        Param_kind           m_kind;
        Param_kind           m_array_elem_kind;
        mi::Size             m_array_size;
        mi::Size             m_array_pitch;   // the distance between two array elements
        char                 *m_data_ptr;
        float                m_range_min, m_range_max;
        const Enum_type_info *m_enum_info;
    };


    // Material information structure.
    class Mdl_material_info
    {
    public:
        Mdl_material_info(
            mi::neuraylib::ICompiled_material const *comp_mat,
            mi::neuraylib::IMaterial_definition const *mat_def,
            mi::neuraylib::ITarget_value_layout const *arg_block_layout,
            mi::neuraylib::ITarget_argument_block const *arg_block)
            : m_name(mat_def->get_mdl_name())
        {
            char *arg_block_data = nullptr;
            if (arg_block != nullptr)
            {
                m_arg_block = mi::base::Handle<mi::neuraylib::ITarget_argument_block>(
                    arg_block->clone());
                arg_block_data = m_arg_block->get_data();
            }

            mi::base::Handle<mi::neuraylib::IAnnotation_list const> anno_list(
                mat_def->get_parameter_annotations());

            for (mi::Size j = 0, num_params = comp_mat->get_parameter_count(); j < num_params; ++j)
            {
                const char *name = comp_mat->get_parameter_name(j);
                if (name == nullptr) continue;

                // Determine the type of the argument
                mi::base::Handle<mi::neuraylib::IValue const> arg(comp_mat->get_argument(j));
                mi::neuraylib::IValue::Kind kind = arg->get_kind();

                Param_info::Param_kind param_kind = Param_info::PK_UNKNOWN;
                Param_info::Param_kind param_array_elem_kind = Param_info::PK_UNKNOWN;
                mi::Size               param_array_size = 0;
                mi::Size               param_array_pitch = 0;
                const Enum_type_info   *enum_type = nullptr;

                switch (kind)
                {
                case mi::neuraylib::IValue::VK_FLOAT:
                    param_kind = Param_info::PK_FLOAT;
                    break;
                case mi::neuraylib::IValue::VK_COLOR:
                    param_kind = Param_info::PK_COLOR;
                    break;
                case mi::neuraylib::IValue::VK_BOOL:
                    param_kind = Param_info::PK_BOOL;
                    break;
                case mi::neuraylib::IValue::VK_INT:
                    param_kind = Param_info::PK_INT;
                    break;
                case mi::neuraylib::IValue::VK_VECTOR:
                {
                    mi::base::Handle<mi::neuraylib::IValue_vector const> val(
                        arg.get_interface<mi::neuraylib::IValue_vector const>());
                    mi::base::Handle<mi::neuraylib::IType_vector const> val_type(
                        val->get_type());
                    mi::base::Handle<mi::neuraylib::IType_atomic const> elem_type(
                        val_type->get_element_type());
                    if (elem_type->get_kind() == mi::neuraylib::IType::TK_FLOAT)
                    {
                        switch (val_type->get_size())
                        {
                        case 2: param_kind = Param_info::PK_FLOAT2; break;
                        case 3: param_kind = Param_info::PK_FLOAT3; break;
                        }
                    }
                }
                break;
                case mi::neuraylib::IValue::VK_ARRAY:
                {
                    mi::base::Handle<mi::neuraylib::IValue_array const> val(
                        arg.get_interface<mi::neuraylib::IValue_array const>());
                    mi::base::Handle<mi::neuraylib::IType_array const> val_type(
                        val->get_type());
                    mi::base::Handle<mi::neuraylib::IType const> elem_type(
                        val_type->get_element_type());

                    // we currently only support arrays of some values
                    switch (elem_type->get_kind())
                    {
                    case mi::neuraylib::IType::TK_FLOAT:
                        param_array_elem_kind = Param_info::PK_FLOAT;
                        break;
                    case mi::neuraylib::IType::TK_COLOR:
                        param_array_elem_kind = Param_info::PK_COLOR;
                        break;
                    case mi::neuraylib::IType::TK_BOOL:
                        param_array_elem_kind = Param_info::PK_BOOL;
                        break;
                    case mi::neuraylib::IType::TK_INT:
                        param_array_elem_kind = Param_info::PK_INT;
                        break;
                    case mi::neuraylib::IType::TK_VECTOR:
                    {
                        mi::base::Handle<mi::neuraylib::IType_vector const> val_type(
                            elem_type.get_interface<
                            mi::neuraylib::IType_vector const>());
                        mi::base::Handle<mi::neuraylib::IType_atomic const> velem_type(
                            val_type->get_element_type());
                        if (velem_type->get_kind() == mi::neuraylib::IType::TK_FLOAT)
                        {
                            switch (val_type->get_size())
                            {
                            case 2:
                                param_array_elem_kind = Param_info::PK_FLOAT2;
                                break;
                            case 3:
                                param_array_elem_kind = Param_info::PK_FLOAT3;
                                break;
                            }
                        }
                    }
                    break;
                    }
                    if (param_array_elem_kind != Param_info::PK_UNKNOWN)
                    {
                        param_kind = Param_info::PK_ARRAY;
                        param_array_size = val_type->get_size();

                        // determine pitch of array if there are at least two elements
                        if (param_array_size > 1)
                        {
                            mi::neuraylib::Target_value_layout_state array_state(
                                arg_block_layout->get_nested_state(j));
                            mi::neuraylib::Target_value_layout_state next_elem_state(
                                arg_block_layout->get_nested_state(1, array_state));

                            mi::neuraylib::IValue::Kind kind;
                            mi::Size param_size;
                            mi::Size start_offset = arg_block_layout->get_layout(
                                kind, param_size, array_state);
                            mi::Size next_offset = arg_block_layout->get_layout(
                                kind, param_size, next_elem_state);
                            param_array_pitch = next_offset - start_offset;
                        }
                    }
                }
                break;
                case mi::neuraylib::IValue::VK_ENUM:
                {
                    mi::base::Handle<mi::neuraylib::IValue_enum const> val(
                        arg.get_interface<mi::neuraylib::IValue_enum const>());
                    mi::base::Handle<mi::neuraylib::IType_enum const> val_type(
                        val->get_type());

                    // prepare info for this enum type if not seen so far
                    const Enum_type_info *info = get_enum_type(val_type->get_symbol());
                    if (info == nullptr)
                    {
                        std::shared_ptr<Enum_type_info> p(new Enum_type_info());

                        for (mi::Size i = 0, n = val_type->get_size(); i < n; ++i)
                        {
                            p->add(val_type->get_value_name(i), val_type->get_value_code(i));
                        }
                        add_enum_type(val_type->get_symbol(), p);
                        info = p.get();
                    }
                    enum_type = info;

                    param_kind = Param_info::PK_ENUM;
                }
                break;
                case mi::neuraylib::IValue::VK_STRING:
                    param_kind = Param_info::PK_STRING;
                    break;
                case mi::neuraylib::IValue::VK_TEXTURE:
                    param_kind = Param_info::PK_TEXTURE;
                    break;
                case mi::neuraylib::IValue::VK_LIGHT_PROFILE:
                    param_kind = Param_info::PK_LIGHT_PROFILE;
                    break;
                case mi::neuraylib::IValue::VK_BSDF_MEASUREMENT:
                    param_kind = Param_info::PK_BSDF_MEASUREMENT;
                    break;
                default:
                    // Unsupported? -> skip
                    continue;
                }

                // Get the offset of the argument within the target argument block
                mi::neuraylib::Target_value_layout_state state(
                    arg_block_layout->get_nested_state(j));
                mi::neuraylib::IValue::Kind kind2;
                mi::Size param_size;
                mi::Size offset = arg_block_layout->get_layout(kind2, param_size, state);
                if (kind != kind2)
                    continue;  // layout is invalid -> skip

                Param_info param_info(
                    j,
                    name,
                    name,
                    /*group_name=*/ "",
                    param_kind,
                    param_array_elem_kind,
                    param_array_size,
                    param_array_pitch,
                    arg_block_data + offset,
                    enum_type);

                // Check for annotation info
                mi::base::Handle<mi::neuraylib::IAnnotation_block const> anno_block(
                    anno_list->get_annotation_block(name));
                if (anno_block)
                {
                    mi::neuraylib::Annotation_wrapper annos(anno_block.get());
                    mi::Size anno_index =
                        annos.get_annotation_index("::anno::hard_range(float,float)");
                    if (anno_index != mi::Size(-1))
                    {
                        annos.get_annotation_param_value(anno_index, 0, param_info.range_min());
                        annos.get_annotation_param_value(anno_index, 1, param_info.range_max());
                    }
                    else
                    {
                        anno_index = annos.get_annotation_index("::anno::soft_range(float,float)");
                        if (anno_index != mi::Size(-1))
                        {
                            annos.get_annotation_param_value(anno_index, 0, param_info.range_min());
                            annos.get_annotation_param_value(anno_index, 1, param_info.range_max());
                        }
                    }
                    anno_index = annos.get_annotation_index("::anno::display_name(string)");
                    if (anno_index != mi::Size(-1))
                    {
                        char const *display_name = nullptr;
                        annos.get_annotation_param_value(anno_index, 0, display_name);
                        param_info.set_display_name(display_name);
                    }
                    anno_index = annos.get_annotation_index("::anno::in_group(string)");
                    if (anno_index != mi::Size(-1))
                    {
                        char const *group_name = nullptr;
                        annos.get_annotation_param_value(anno_index, 0, group_name);
                        param_info.set_group_name(group_name);
                    }
                }

                add_sorted_by_group(param_info);
            }
        }

        // Add the parameter information as last entry of the corresponding group, or to the
        // end of the list, if no group name is available.
        void add_sorted_by_group(const Param_info &info)
        {
            bool group_found = false;
            if (info.group_name() != nullptr)
            {
                for (std::list<Param_info>::iterator it = params().begin(); it != params().end(); ++it)
                {
                    const bool same_group =
                        it->group_name() != nullptr && strcmp(it->group_name(), info.group_name()) == 0;
                    if (group_found && !same_group)
                    {
                        m_params.insert(it, info);
                        return;
                    }
                    if (same_group)
                        group_found = true;
                }
            }
            m_params.push_back(info);
        }

        // Add a new enum type to the list of used enum types.
        void add_enum_type(const std::string name, std::shared_ptr<Enum_type_info> enum_info)
        {
            enum_types[name] = enum_info;
        }

        // Lookup enum type info for a given enum type absolute MDL name.
        const Enum_type_info *get_enum_type(const std::string name)
        {
            Enum_type_map::const_iterator it = enum_types.find(name);
            if (it != enum_types.end())
                return it->second.get();
            return nullptr;
        }

        // Get the name of the material.
        char const *name() const { return m_name.c_str(); }

        // Get the parameters of this material.
        std::list<Param_info> &params() { return m_params; }

        // Get the modifiable argument block data.
        char *get_argument_block_data()
        {
            return m_arg_block->get_data();
        }

        // Get the modifiable argument block size.
        size_t get_argument_block_size()
        {
            return m_arg_block->get_size();
        }

    private:
        // name of the material
        std::string m_name;

        // modifiable argument block
        mi::base::Handle<mi::neuraylib::ITarget_argument_block> m_arg_block;

        // parameters of the material
        std::list<Param_info> m_params;

        typedef std::map<std::string, std::shared_ptr<Enum_type_info>> Enum_type_map;

        // used enum types of the material
        Enum_type_map enum_types;
    };

}

#endif  // MATERIAL_INFO_HELPER_H
