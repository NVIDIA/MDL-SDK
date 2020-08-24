/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include <mdl/compiler/compilercore/compilercore_errors.h>
#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_printers.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

#include <llvm/IR/Module.h>

#include "generator_jit.h"
#include "generator_jit_code_printer.h"
#include "generator_jit_generated_code.h"
#include "generator_jit_llvm.h"
#include "generator_jit_opt_pass_gate.h"

namespace mi {
namespace mdl {

// Helper class to build the layout structure.
class Layout_builder
{
    struct Build_state
    {
        size_t m_state_offs;
        mi::mdl::IType const *m_mdl_type;
        llvm::Type *m_llvm_type;

        Build_state(
            size_t state_offs,
            mi::mdl::IType const *mdl_type,
            llvm::Type *llvm_type)
        : m_state_offs(state_offs)
        , m_mdl_type(mdl_type)
        , m_llvm_type(llvm_type)
        {}
    };

public:
    /// Constructor.
    ///
    /// \param alloc        the allocator
    /// \param code_gen     the code generator
    /// \param layout_buf   the layout buffer
    /// \param data_layout  the LLVM data layout
    Layout_builder(
        IAllocator                  *alloc,
        LLVM_code_generator         &code_gen,
        vector<unsigned char>::Type &layout_buf,
        llvm::DataLayout const *data_layout)
    : m_code_gen(code_gen)
    , m_layout_buf(layout_buf)
    , m_data_layout(data_layout)
    , m_cur_offs(0)
    , m_total_size(0)
    , m_child_worklist(alloc)
    {
    }

    /// Get the total size ofter the builder creates the layout.
    size_t get_total_size() const
    {
        return m_total_size;
    }

    /// Builds the layout.
    ///
    /// \param mdl_param_types   a vector of the MDL parameter types
    /// \param llvm_type         an LLVM struct type corresponding to ALL parameter types
    void build(
        LLVM_code_generator::Type_vector const &mdl_param_types,
        llvm::StructType                       *llvm_type)
    {
        size_t cur_store_size = size_t(m_data_layout->getTypeStoreSize(llvm_type));
        size_t cur_alloc_size = size_t(m_data_layout->getTypeAllocSize(llvm_type));

        int num_params = llvm_type->getNumContainedTypes();

        MDL_ASSERT(num_params == mdl_param_types.size());

        // write "root" parameter structure layout
        write(mi::Uint8(IValue::VK_STRUCT));
        write(mi::Uint8(cur_alloc_size - cur_store_size));
        write(mi::Uint16(0));                        // unused
        write(mi::Uint32(cur_store_size));
        write(mi::Uint32(0));                        // element offset starts at 0
        write(mi::Uint16(num_params));               // num_children
        write(mi::Uint16(m_layout_buf.size() + 2));  // children layouts start after this entry

        // handle all parameter "root" types, filling the child worklist
        size_t max_align = 1;
        for (int i = 0; i < num_params; ++i) {
            llvm::Type *param_llvm_type = llvm_type->getTypeAtIndex(unsigned(i));
            unsigned param_align = size_t(m_data_layout->getABITypeAlignment(param_llvm_type));
            if (param_align > max_align) max_align = param_align;
            build_one_type_layout(
                mdl_param_types[i],
                param_llvm_type);
        }

        // maximum alignment padding is applied to allow consecutive elements of same type
        m_cur_offs = (m_cur_offs + max_align - 1) & ~(max_align - 1);

        // all root parameter types written and alignment added -> offset is total size
        m_total_size = m_cur_offs;

        // create children layouts
        while (!m_child_worklist.empty()) {
            Build_state cur_build_state = m_child_worklist.back();
            m_child_worklist.pop_back();

            size_t cur_state_offs = m_layout_buf.size();
            Layout_struct *parent_layout =
                reinterpret_cast<Layout_struct *>(&m_layout_buf[cur_build_state.m_state_offs]);
            MDL_ASSERT(cur_state_offs < 65536);
            parent_layout->children_state_offs = mi::Uint16(cur_state_offs);

            // reset current offset to the element offset of the parent
            m_cur_offs = parent_layout->element_offset;

            mi::mdl::IType const *mdl_type = cur_build_state.m_mdl_type;
            IType::Kind kind = mdl_type->get_kind();
            switch (kind) {
                case IType::TK_ALIAS:
                case IType::TK_BOOL:
                case IType::TK_INT:
                case IType::TK_ENUM:
                case IType::TK_FLOAT:
                case IType::TK_DOUBLE:
                case IType::TK_STRING:
                case IType::TK_LIGHT_PROFILE:
                case IType::TK_BSDF:
                case IType::TK_HAIR_BSDF:
                case IType::TK_EDF:
                case IType::TK_VDF:
                case IType::TK_FUNCTION:
                case IType::TK_TEXTURE:
                case IType::TK_BSDF_MEASUREMENT:
                case IType::TK_INCOMPLETE:
                case IType::TK_ERROR:
                    MDL_ASSERT(!"unexpected kind");
                    break;

                case IType::TK_VECTOR:
                case IType::TK_MATRIX:
                case IType::TK_ARRAY:
                case IType::TK_COLOR:
                {
                    IType_compound const *comp_type =
                        static_cast<IType_compound const *>(mdl_type);

                    // all elements have the same type, only generate it for the first element
                    IType const *subcomp_type = comp_type->get_compound_type(0);
                    build_one_type_layout(
                        comp_type->get_compound_type(0),
                        m_code_gen.lookup_type(subcomp_type));
                    break;
                }

                case IType::TK_STRUCT:
                {
                    IType_compound const *comp_type =
                        static_cast<IType_compound const *>(mdl_type);

                    // generate the layouts for all sub-components
                    for (int i = 0, num = comp_type->get_compound_size(); i < num; ++i) {
                        IType const *subcomp_type = comp_type->get_compound_type(i);
                        build_one_type_layout(
                            subcomp_type,
                            m_code_gen.lookup_type(subcomp_type));
                    }
                    break;
                }
            }
        }
    }

private:
    /// Writes a character value into the layout buffer.
    void write(mi::Uint8 val)
    {
        m_layout_buf.push_back(char(val));
    }

    /// Writes a 16 bit value into the layout buffer.
    void write(mi::Uint16 val)
    {
        size_t cur_size = m_layout_buf.size();
        m_layout_buf.resize(cur_size + 2);
        *reinterpret_cast<mi::Uint16 *>(&m_layout_buf[cur_size]) = val;
    }

    /// Writes a 32 bit value into the layout buffer.
    void write(mi::Uint32 val)
    {
        size_t cur_size = m_layout_buf.size();
        m_layout_buf.resize(cur_size + 4);
        *reinterpret_cast<mi::Uint32 *>(&m_layout_buf[cur_size]) = val;
    }

    /// Maps an MDL type kind to a value kind.
    mi::mdl::IValue::Kind to_value_kind(mi::mdl::IType::Kind kind)
    {
        switch (kind) {
            #define MAP_KIND( from, to) \
                case mi::mdl::IType::from: return mi::mdl::IValue::to

            MAP_KIND( TK_ALIAS,            VK_BAD);
            MAP_KIND( TK_BOOL,             VK_BOOL);
            MAP_KIND( TK_INT,              VK_INT);
            MAP_KIND( TK_ENUM,             VK_ENUM);
            MAP_KIND( TK_FLOAT,            VK_FLOAT);
            MAP_KIND( TK_DOUBLE,           VK_DOUBLE);
            MAP_KIND( TK_STRING,           VK_STRING);
            MAP_KIND( TK_LIGHT_PROFILE,    VK_LIGHT_PROFILE);
            MAP_KIND( TK_BSDF,             VK_BAD);
            MAP_KIND( TK_HAIR_BSDF,        VK_BAD);
            MAP_KIND( TK_EDF,              VK_BAD);
            MAP_KIND( TK_VDF,              VK_BAD);
            MAP_KIND( TK_VECTOR,           VK_VECTOR);
            MAP_KIND( TK_MATRIX,           VK_MATRIX);
            MAP_KIND( TK_ARRAY,            VK_ARRAY);
            MAP_KIND( TK_COLOR,            VK_RGB_COLOR);
            MAP_KIND( TK_FUNCTION,         VK_BAD);
            MAP_KIND( TK_STRUCT,           VK_STRUCT);
            MAP_KIND( TK_TEXTURE,          VK_TEXTURE);
            MAP_KIND( TK_BSDF_MEASUREMENT, VK_BSDF_MEASUREMENT);
            MAP_KIND( TK_INCOMPLETE,       VK_BAD);
            MAP_KIND( TK_ERROR,            VK_BAD);

            #undef MAP_KIND
        }
        MDL_ASSERT(!"unexpected kind");
        return mi::mdl::IValue::VK_BAD;
    }

    /// Creates the type layout for one type.
    ///
    /// \param mdl_type   the MDL type
    /// \param llvm_type  the corresponding LLVM type
    void build_one_type_layout(mi::mdl::IType const *mdl_type, llvm::Type *llvm_type)
    {
        mdl_type = mdl_type->skip_type_alias();

        size_t cur_state_offs = m_layout_buf.size();

        size_t cur_store_size = size_t(m_data_layout->getTypeStoreSize(llvm_type));
        size_t cur_align      = size_t(m_data_layout->getABITypeAlignment(llvm_type));
        size_t cur_alloc_size = size_t(m_data_layout->getTypeAllocSize(llvm_type));
        size_t element_offset = (m_cur_offs + cur_align - 1) & ~(cur_align - 1);

        IType::Kind kind = mdl_type->get_kind();
        write(mi::Uint8(to_value_kind(kind)));
        write(mi::Uint8(cur_alloc_size - cur_store_size));
        write(mi::Uint16(0));  // unused
        write(mi::Uint32(cur_store_size));
        write(mi::Uint32(element_offset));

        switch (kind) {
            case IType::TK_BOOL:
            case IType::TK_INT:
            case IType::TK_ENUM:
            case IType::TK_FLOAT:
            case IType::TK_DOUBLE:
            case IType::TK_LIGHT_PROFILE:
            case IType::TK_TEXTURE:
            case IType::TK_BSDF_MEASUREMENT:
            case IType::TK_STRING:
                write(mi::Uint16(0));  // num_children
                write(mi::Uint16(0));  // children_state_offs
                break;

            case IType::TK_VECTOR:
            case IType::TK_MATRIX:
            case IType::TK_ARRAY:
            case IType::TK_COLOR:
            case IType::TK_STRUCT:
            {
                IType_compound const *comp_type = static_cast<IType_compound const *>(mdl_type);
                int num = comp_type->get_compound_size();
                write(mi::Uint16(num));  // num_children
                write(mi::Uint16(0));    // placeholder for children_state_offs

                // append current type to worklist to process it later
                m_child_worklist.push_back(Build_state(
                    cur_state_offs,
                    mdl_type,
                    llvm_type));
                break;
            }

            case IType::TK_ALIAS:
            case IType::TK_BSDF:
            case IType::TK_HAIR_BSDF:
            case IType::TK_EDF:
            case IType::TK_VDF:
            case IType::TK_FUNCTION:
            case IType::TK_INCOMPLETE:
            case IType::TK_ERROR:
                MDL_ASSERT(!"unexpected kind");
                break;
        }

        m_cur_offs = element_offset + cur_alloc_size;
    }

private:
    LLVM_code_generator &m_code_gen;
    vector<unsigned char>::Type &m_layout_buf;
    llvm::DataLayout const *m_data_layout;
    size_t m_cur_offs;
    size_t m_total_size;
    vector<Build_state>::Type m_child_worklist;
};

// ------------------------------- Generated_code_value_layout ------------------------------

// Constructor.
Generated_code_value_layout::Generated_code_value_layout(
    IAllocator          *alloc,
    LLVM_code_generator *code_gen)
: Base(alloc)
, m_layout_data(alloc)
, m_strings_mapped_to_ids(code_gen->strings_mapped_to_ids())
{
    llvm::StructType       *llvm_args_type = code_gen->get_captured_arguments_llvm_type();
    llvm::DataLayout const *data_layout    = code_gen->get_target_layout_data();

    Layout_builder builder(alloc, *code_gen, m_layout_data, data_layout);
    builder.build(code_gen->get_captured_argument_mdl_types(), llvm_args_type);
    MDL_ASSERT(size_t  (builder.get_total_size()) ==
        size_t  (data_layout->getTypeAllocSize(llvm_args_type)) &&
        size_t  (builder.get_total_size()) == get_size());
}

// Constructor from layout data block.
Generated_code_value_layout::Generated_code_value_layout(
    IAllocator *alloc,
    char const *layout_data_block,
    size_t     layout_data_size,
    bool       strings_mapped_to_ids)
: Base(alloc)
, m_layout_data(layout_data_block, layout_data_block + layout_data_size, alloc)
, m_strings_mapped_to_ids(strings_mapped_to_ids)
{
}

// Constructor used for deserialization.
Generated_code_value_layout::Generated_code_value_layout(
    IAllocator* alloc)
    : Base(alloc)
    , m_layout_data(alloc)
    , m_strings_mapped_to_ids(false)
{
}

// Get the size of the target argument block.
size_t Generated_code_value_layout::get_size() const
{
    if (sizeof(Layout_struct) > m_layout_data.size())
        return 0;
    Layout_struct const *layout = reinterpret_cast<Layout_struct const *>(&m_layout_data[0]);

    return size_t(layout->element_size) + size_t(layout->alloc_size_padding);
}

// Returns the number of arguments / elements at the given layout state.
size_t Generated_code_value_layout::get_num_elements(
    IGenerated_code_value_layout::State state) const
{
    mi::Uint32 offs = state.m_state_offs;
    if (offs + sizeof(Layout_struct) > m_layout_data.size())
        return ~size_t(0);

    Layout_struct const *layout = reinterpret_cast<Layout_struct const *>(&m_layout_data[offs]);
    return size_t(layout->num_children);
}

// Get the offset, the size and the kind of the argument / element inside the argument
// block at the given layout state.
size_t Generated_code_value_layout::get_layout(
    mi::mdl::IValue::Kind               &kind,
    size_t                              &arg_size,
    IGenerated_code_value_layout::State state) const
{
    mi::Uint32 offs = state.m_state_offs;
    if (offs + sizeof(Layout_struct) > m_layout_data.size()) {
        arg_size = 0;
        kind = mi::mdl::IValue::VK_BAD;
        return ~size_t(0);
    }

    Layout_struct const *layout = reinterpret_cast<Layout_struct const *>(&m_layout_data[offs]);
    arg_size = layout->element_size;
    kind = mi::mdl::IValue::Kind(layout->kind);
    return layout->element_offset + state.m_data_offs;
}

// Get the layout state for the i'th argument / element inside the argument value block
// at the given layout state.
IGenerated_code_value_layout::State Generated_code_value_layout::get_nested_state(
    size_t                              i,
    IGenerated_code_value_layout::State state) const
{
    mi::Uint32 offs = state.m_state_offs;
    if (offs + sizeof(Layout_struct) > m_layout_data.size())
        return IGenerated_code_value_layout::State(~mi::Uint32(0), ~mi::Uint32(0));

    Layout_struct const *layout = reinterpret_cast<Layout_struct const *>(&m_layout_data[offs]);
    if (i >= layout->num_children)
        return IGenerated_code_value_layout::State(~mi::Uint32(0), ~mi::Uint32(0));

    switch (mi::mdl::IValue::Kind(layout->kind))
    {
        case mi::mdl::IValue::VK_BOOL:
        case mi::mdl::IValue::VK_INT:
        case mi::mdl::IValue::VK_ENUM:
        case mi::mdl::IValue::VK_FLOAT:
        case mi::mdl::IValue::VK_DOUBLE:
        case mi::mdl::IValue::VK_STRING:
        {
            MDL_ASSERT(!"not a compound type");
            return IGenerated_code_value_layout::State(~mi::Uint32(0), ~mi::Uint32(0));
        }

        case mi::mdl::IValue::VK_VECTOR:
        case mi::mdl::IValue::VK_MATRIX:
        case mi::mdl::IValue::VK_ARRAY:
        case mi::mdl::IValue::VK_RGB_COLOR:
        {
            // homogenous types have same state for all of them
            Layout_struct const *child_layout = reinterpret_cast<Layout_struct const *>(
                &m_layout_data[layout->children_state_offs]);

            mi::Uint32 elem_alloc_size =
                child_layout->element_size + child_layout->alloc_size_padding;

            return IGenerated_code_value_layout::State(
                layout->children_state_offs,
                mi::Uint32(
                    state.m_data_offs + i * elem_alloc_size));
        }

        case mi::mdl::IValue::VK_STRUCT:
        {
            mi::Uint32 child_state_offs =
                mi::Uint32(layout->children_state_offs + i * sizeof(Layout_struct));

            return IGenerated_code_value_layout::State(
                child_state_offs,
                state.m_data_offs);
        }

        case mi::mdl::IValue::VK_BAD:
        case mi::mdl::IValue::VK_INVALID_REF:
        case mi::mdl::IValue::VK_TEXTURE:
        case mi::mdl::IValue::VK_LIGHT_PROFILE:
        case mi::mdl::IValue::VK_BSDF_MEASUREMENT:
        {
            MDL_ASSERT(!"unexpected value type");
            return IGenerated_code_value_layout::State(~mi::Uint32(0), ~mi::Uint32(0));
        }
    }
    MDL_ASSERT(!"unsupported value type");
    return IGenerated_code_value_layout::State(~mi::Uint32(0), ~mi::Uint32(0));
}

// Set the value inside the given block at the given layout state.
int Generated_code_value_layout::set_value(
    char                                *block,
    mi::mdl::IValue const               *value,
    IGenerated_code_value_callback      *value_callback,
    IGenerated_code_value_layout::State state) const
{
    if (block == NULL || value == NULL) return -1;

    unsigned layout_offs = state.m_state_offs;
    if (layout_offs + sizeof(Layout_struct) > m_layout_data.size()) return -2;

    Layout_struct const *layout =
        reinterpret_cast<Layout_struct const *>(&m_layout_data[layout_offs]);
    if (value->get_kind() != mi::mdl::IValue::Kind(layout->kind)) return -3;

    unsigned data_offs = state.m_data_offs + layout->element_offset;

    switch (mi::mdl::IValue::Kind(layout->kind)) {
    case mi::mdl::IValue::VK_BOOL:
        *reinterpret_cast<bool *>(block + data_offs) =
            cast<IValue_bool>(value)->get_value();
        return 0;

    case mi::mdl::IValue::VK_INT:
    case mi::mdl::IValue::VK_ENUM:
        *reinterpret_cast<int *>(block + data_offs) =
            cast<IValue_int_valued>(value)->get_value();
        return 0;

    case mi::mdl::IValue::VK_FLOAT:
        *reinterpret_cast<float *>(block + data_offs) =
            cast<IValue_float>(value)->get_value();
        return 0;

    case mi::mdl::IValue::VK_DOUBLE:
        *reinterpret_cast<double *>(block + data_offs) =
            cast<IValue_double>(value)->get_value();
        return 0;

    case mi::mdl::IValue::VK_STRING:
        if (m_strings_mapped_to_ids) {
            mi::Uint32 id = value_callback != NULL ?
                value_callback->get_string_index(cast<IValue_string>(value)) : 0u;
            *reinterpret_cast<unsigned *>(block + data_offs) = id;
        } else {
            // unmapped string are not supported
            *reinterpret_cast<char **>(block + data_offs) = NULL;
        }
        return 0;

    case mi::mdl::IValue::VK_VECTOR:
    case mi::mdl::IValue::VK_MATRIX:
    case mi::mdl::IValue::VK_ARRAY:
    case mi::mdl::IValue::VK_RGB_COLOR:
        {
            // homogeneous types have same state for all of them
            mi::mdl::IValue_compound const *v = cast<mi::mdl::IValue_compound>(value);
            if (v->get_component_count() != layout->num_children) return -4;

            Layout_struct const *child_layout = reinterpret_cast<Layout_struct const *>(
                &m_layout_data[layout->children_state_offs]);

            mi::Uint32 elem_alloc_size =
                child_layout->element_size + child_layout->alloc_size_padding;

            for (mi::Uint32 i = 0; i < layout->num_children; ++i) {
                IGenerated_code_value_layout::State child_state(
                    layout->children_state_offs,
                    state.m_data_offs + i * elem_alloc_size);
                int err = set_value(block, v->get_value(int(i)), value_callback, child_state);
                if (err != 0)
                    return err;
            }
            return 0;
        }

    case mi::mdl::IValue::VK_STRUCT:
        {
            mi::mdl::IValue_compound const *v = cast<mi::mdl::IValue_compound>(value);
            if (v->get_component_count() != layout->num_children) return -4;

            for (mi::Uint32 i = 0; i < layout->num_children; ++i) {
                mi::Uint32 child_state_offs =
                    layout->children_state_offs + i * sizeof(Layout_struct);

                IGenerated_code_value_layout::State child_state(
                    child_state_offs,
                    state.m_data_offs);
                int err = set_value(block, v->get_value(int(i)), value_callback, child_state);
                if (err != 0)
                    return err;
            }
            return 0;
        }

    case mi::mdl::IValue::VK_TEXTURE:
        {
            IValue_texture const *tex = cast<IValue_texture>(value);
            unsigned index = value_callback->get_resource_index(tex);
            *reinterpret_cast<mi::Uint32 *>(block + data_offs) = index;
            return 0;
        }
    case mi::mdl::IValue::VK_LIGHT_PROFILE:
    case mi::mdl::IValue::VK_BSDF_MEASUREMENT:
        {
            IValue_resource const *res = cast<IValue_resource>(value);
            unsigned index = value_callback->get_resource_index(res);
            *reinterpret_cast<mi::Uint32 *>(block + data_offs) = index;
            return 0;
        }

    case mi::mdl::IValue::VK_BAD:
    case mi::mdl::IValue::VK_INVALID_REF:
        MDL_ASSERT(!"unexpected value type");
        return -5;
    }
    MDL_ASSERT(!"unsupported value type");
    return -5;
}

char const *Generated_code_value_layout::get_layout_data(size_t &size) const
{
    size = m_layout_data.size();
    return (char const *)&m_layout_data[0];
}

void Generated_code_value_layout::set_layout_data(char const* data, size_t& size)
{
    m_layout_data.resize(size);
    memcpy(m_layout_data.data(), data, size);
}

// ------------------------------- Generated_code_executable_base -------------------------------

// Get the number of functions in this link unit.
template <class I>
size_t Generated_code_executable_base<I>::get_function_count() const
{
    return m_func_infos.size();
}

// Get the name of the i'th function inside this link unit.
template <class I>
char const *Generated_code_executable_base<I>::get_function_name(size_t i) const
{
    if (i < m_func_infos.size())
        return m_func_infos[i].m_name.c_str();
    return NULL;
}

// Returns the distribution kind of the i'th function inside this link unit.
template <class I>
IGenerated_code_executable::Distribution_kind
Generated_code_executable_base<I>::get_distribution_kind(size_t i) const
{
    if (i < m_func_infos.size())
        return m_func_infos[i].m_dist_kind;
    return IGenerated_code_executable::DK_INVALID;
}

// Returns the function kind of the i'th function inside this link unit.
template <class I>
IGenerated_code_executable::Function_kind
Generated_code_executable_base<I>::get_function_kind(size_t i) const
{
    if (i < m_func_infos.size())
        return m_func_infos[i].m_kind;
    return IGenerated_code_executable::FK_INVALID;
}

// Get the index of the target argument block layout for the i'th function inside this link
// unit if used.
template <class I>
size_t Generated_code_executable_base<I>::get_function_arg_block_layout_index(size_t i) const
{
    if (i < m_func_infos.size())
        return m_func_infos[i].m_arg_block_index;
    return ~0;
}

// Returns the prototype of the i'th function inside this link unit.
template <class I>
char const *Generated_code_executable_base<I>::get_function_prototype(
    size_t index,
    IGenerated_code_executable::Prototype_language lang) const
{
    if (index >= m_func_infos.size() ||
            lang >= m_func_infos[index].m_prototypes.size()) {
        return NULL;
    }
    return m_func_infos[index].m_prototypes[lang].c_str();
}

// Set a function prototype for a function.
template <class I>
void Generated_code_executable_base<I>::set_function_prototype(
    size_t index,
    IGenerated_code_executable::Prototype_language lang,
    char const *prototype)
{
    MDL_ASSERT(index < m_func_infos.size());
    if (index >= m_func_infos.size())
        return;

    if (lang >= m_func_infos[index].m_prototypes.size()) {
        m_func_infos[index].m_prototypes.resize(lang + 1, string(this->get_allocator()));
    }
    m_func_infos[index].m_prototypes[lang] = string(prototype, this->get_allocator());
}

// Add information for a generated function.
template <class I>
size_t Generated_code_executable_base<I>::add_function_info(
    char const *name,
    IGenerated_code_executable::Distribution_kind dist_kind,
    IGenerated_code_executable::Function_kind func_kind,
    size_t arg_block_index,
    IGenerated_code_executable::State_usage state_usage)
{
    m_func_infos.push_back(
        Generated_code_function_info(
            string(name, this->get_allocator()),
            dist_kind,
            func_kind,
            arg_block_index,
            state_usage));

    return m_func_infos.size() - 1;
}

// Get the number of distribution function handles referenced by a function.
template <class I>
size_t Generated_code_executable_base<I>::get_function_df_handle_count(size_t func_index) const
{
    if (func_index >= m_func_infos.size()) {
        return 0;
    }
    return m_func_infos[func_index].m_df_handle_name_table.size();
}

// Get the name of a distribution function handle referenced by a function.
template <class I>
char const *Generated_code_executable_base<I>::get_function_df_handle(
    size_t func_index, size_t handle_index) const
{
    if (func_index >= m_func_infos.size() ||
            handle_index >= m_func_infos[func_index].m_df_handle_name_table.size()) {
        return NULL;
    }
    return m_func_infos[func_index].m_df_handle_name_table[handle_index].c_str();
}

// Add the name of a distribution function handle referenced by a function.
template <class I>
size_t Generated_code_executable_base<I>::add_function_df_handle(
    size_t func_index,
    char const *handle_name)
{
    MDL_ASSERT(func_index < m_func_infos.size());
    if (func_index >= m_func_infos.size())
        return ~0;

    m_func_infos[func_index].m_df_handle_name_table.push_back(
        string(handle_name, this->get_allocator()));

    return m_func_infos[func_index].m_df_handle_name_table.size() - 1;
}

/// Get the state properties used by a function.
template <class I>
IGenerated_code_executable::State_usage Generated_code_executable_base<I>::get_function_state_usage(
    size_t func_index) const
{
    if (func_index >= m_func_infos.size()) {
        return 0;
    }
    return m_func_infos[func_index].m_state_usage;
}

// ----------------------------------- Generated_code_jit -----------------------------------

// Constructor.
Generated_code_jit::Generated_code_jit(
    IAllocator  *alloc,
    Jitted_code *jitted_code,
    char const  *filename)
: Base(alloc)
, m_builder(alloc)
, m_llvm_context()
, m_jitted_code(mi::base::make_handle_dup(jitted_code))
, m_messages(alloc, filename)
, m_source_code(alloc)
, m_render_state_usage(-1)
, m_mapped_strings(alloc)
{
}

// Destructor.
Generated_code_jit::~Generated_code_jit()
{
}

// Get the kind of code generated.
IGenerated_code::Kind Generated_code_jit::get_kind() const
{
    return CK_EXECUTABLE;
}

// Get the target language.
char const *Generated_code_jit::get_target_language() const
{
    return "executable";
}

// Check if the code contents are valid.
bool Generated_code_jit::is_valid() const
{
    return m_messages.get_error_message_count() == 0;
}

// Access messages.
Messages const &Generated_code_jit::access_messages() const
{
    return m_messages;
}

// Acquires a const interface.
mi::base::IInterface const *Generated_code_jit::get_interface(
    mi::base::Uuid const &interface_id) const
{
    if (interface_id == IPrinter_interface::IID()) {
        return m_builder.create<JIT_code_printer>(m_builder.get_allocator());
    }
    return Base::get_interface(interface_id);
}

// Returns the assembler code of the executable code module if available.
char const *Generated_code_jit::get_source_code(size_t &size) const
{
    size = m_source_code.size();
    return m_source_code.c_str();
}

// Get the data for the read-only data segment if available.
char const *Generated_code_jit::get_ro_data_segment(size_t &size) const
{
    size = 0;
    return NULL;
}

// Get the used state properties of  the generated lambda function code.
IGenerated_code_executable::State_usage Generated_code_jit::get_state_usage() const
{
    return m_render_state_usage;
}

// Get the number of captured argument block layouts.
size_t Generated_code_jit::get_captured_argument_layouts_count() const
{
    // FIXME
    return 0;
}

// Get a captured arguments block layout if available.
IGenerated_code_value_layout const *Generated_code_jit::get_captured_arguments_layout(
    size_t i) const
{
    // FIXME
    return NULL;
}

// Get the number of mapped string constants used inside the generated code.
size_t Generated_code_jit::get_string_constant_count() const
{
    return m_mapped_strings.size();
}

// Get the mapped string constant for a given id.
char const *Generated_code_jit::get_string_constant(size_t id) const
{
    if (id < m_mapped_strings.size()) {
        return m_mapped_strings[id].c_str();
    }
    return NULL;
}

// Add a mapped string.
void Generated_code_jit::add_mapped_string(char const *s, size_t id)
{
    if (id >= m_mapped_strings.size()) {
        m_mapped_strings.resize(id + 1, string(get_allocator()));
    }
    m_mapped_strings[id] = string(s, get_allocator());
}

// Compile a whole module into LLVM-IR.
void Generated_code_jit::compile_module_to_llvm(
    IModule const      *module,
    Options_impl const &options)
{
    Module const *mod = impl_cast<Module>(module);
    mi::base::Handle<MDL> compiler(mod->get_compiler());

    llvm::LLVMContext llvm_context;

    LLVM_code_generator llvm_generator(
        m_jitted_code.get(),
        compiler.get(),
        m_messages,
        llvm_context,
        /*target_language=*/LLVM_code_generator::TL_NATIVE,
        Type_mapper::TM_NATIVE_X86,
        /*sm_version=*/0,
        /*has_tex_handler=*/true,
        Type_mapper::SSM_FULL_SET,
        /*num_texture_spaces=*/4,
        /*num_texture_results=*/32,
        options,
        /*incremental=*/false,
        /*state_mapping=*/0,
        /*res_manag=*/NULL,
        /*debug=*/false);

    // for now, mark exported functions as entries, so the module will not be empty
    llvm_generator.mark_exported_funcs_as_entries();

    std::unique_ptr<llvm::Module> llvm_module(llvm_generator.compile_module(module));

    if (llvm_module) {
        llvm_generator.llvm_ir_compile(llvm_module.get(), m_source_code);
    } else {
        size_t file_id = m_messages.register_fname(module->get_filename());
        m_messages.add_error_message(INTERNAL_COMPILER_ERROR, MESSAGE_CLASS, file_id, NULL,
            "Compiling LLVM code failed");
    }

    m_render_state_usage = llvm_generator.get_render_state_usage();
}

// Compile a whole module into PTX.
void Generated_code_jit::compile_module_to_ptx(
    IModule const      *module,
    Options_impl const &options)
{
    Module const *mod = impl_cast<Module>(module);
    mi::base::Handle<MDL> compiler(mod->get_compiler());

    llvm::LLVMContext llvm_context;

    unsigned sm_version = 20;
    LLVM_code_generator llvm_generator(
        m_jitted_code.get(),
        compiler.get(),
        m_messages,
        llvm_context,
        /*target_language=*/LLVM_code_generator::TL_PTX,
        Type_mapper::TM_PTX,
        sm_version,
        /*has_tex_handler=*/false,
        Type_mapper::SSM_FULL_SET,
        /*num_texture_spaces=*/4,
        /*num_texture_results=*/32,
        options,
        /*incremental=*/false,
        /*state_mapping=*/0,
        /*res_manag=*/NULL,
        /*debug=*/false);

    // for now, mark exported functions as entries, so the module will not be empty
    llvm_generator.mark_exported_funcs_as_entries();

    std::unique_ptr<llvm::Module> llvm_module(llvm_generator.compile_module(module));
    if (llvm_module) {
        // FIXME: pass the sm version here. However, this is currently used for debugging
        // only.
        llvm_generator.ptx_compile(llvm_module.get(), m_source_code);
    } else {
        size_t file_id = m_messages.register_fname(module->get_filename());
        m_messages.add_error_message(INTERNAL_COMPILER_ERROR, MESSAGE_CLASS, file_id, NULL,
            "Compiling LLVM code failed");
    }

    m_render_state_usage = llvm_generator.get_render_state_usage();
}

// Compile a whole module into HLSL.
void Generated_code_jit::compile_module_to_hlsl(
    IModule const      *module,
    Options_impl const &options)
{
    Module const *mod = impl_cast<Module>(module);
    mi::base::Handle<MDL> compiler(mod->get_compiler());

    llvm::LLVMContext llvm_context;
    HLSLOptPassGate opt_pass_gate;
    llvm_context.setOptPassGate(opt_pass_gate);

    LLVM_code_generator llvm_generator(
        m_jitted_code.get(),
        compiler.get(),
        m_messages,
        llvm_context,
        /*target_language=*/LLVM_code_generator::TL_HLSL,
        Type_mapper::TM_HLSL,
        /*sm_version=*/0,
        /*has_tex_handler=*/false,
        Type_mapper::SSM_FULL_SET,
        /*num_texture_spaces=*/4,
        /*num_texture_results=*/32,
        options,
        /*incremental=*/false,
        /*state_mapping=*/Type_mapper::SM_INCLUDE_UNIFORM_STATE,  // include uniform state for HLSL
        /*res_manag=*/NULL,
        /*debug=*/false);

    // for now, mark exported functions as entries, so the module will not be empty
    llvm_generator.mark_exported_funcs_as_entries();

    // the HLSL backend expects the RO-data-segment to be used
    llvm_generator.enable_ro_data_segment();

    std::unique_ptr<llvm::Module> llvm_module(llvm_generator.compile_module(module));
    if (llvm_module) {
        llvm_generator.hlsl_compile(llvm_module.get(), m_source_code);
    } else {
        size_t file_id = m_messages.register_fname(module->get_filename());
        m_messages.add_error_message(INTERNAL_COMPILER_ERROR, MESSAGE_CLASS, file_id, NULL,
            "Compiling LLVM code failed");
    }

    m_render_state_usage = llvm_generator.get_render_state_usage();
}

// --------------------------------- Generated_code_source ----------------------------------

// Constructor.
Generated_code_source::Generated_code_source(
    IAllocator            *alloc,
    IGenerated_code::Kind kind)
: Base(alloc)
, m_kind(kind)
, m_render_state_usage(-1)
, m_messages(alloc, "<lambda expression>")
, m_src_code(alloc)
, m_ro_segment(alloc)
, m_captured_arguments_layouts(alloc)
, m_mappend_strings(alloc)
{
}

// Destructor.
Generated_code_source::~Generated_code_source()
{
}

// Get the kind of code generated.
IGenerated_code::Kind Generated_code_source::get_kind() const
{
    return m_kind;
}

// Get the target language.
char const *Generated_code_source::get_target_language() const
{
    return m_kind == CK_PTX ? "PTX" : "LLVM-IR";
}

// Check if the code contents are valid.
bool Generated_code_source::is_valid() const
{
    return m_messages.get_error_message_count() == 0;
}

// Access messages.
Messages const &Generated_code_source::access_messages() const
{
    return m_messages;
}

// Returns the assembler code of the executable code module if available.
char const *Generated_code_source::get_source_code(size_t &size) const
{
    size = m_src_code.size();
    return m_src_code.c_str();
}

// Get the data for the read-only data segment if available.
char const *Generated_code_source::get_ro_data_segment(size_t &size) const
{
    size = m_ro_segment.size();
    return size == 0 ? NULL : &m_ro_segment[0];
}

// Get the used state properties of  the generated lambda function code.
Generated_code_source::State_usage Generated_code_source::get_state_usage() const
{
    return m_render_state_usage;
}

// Get the number of captured argument block layouts.
size_t Generated_code_source::get_captured_argument_layouts_count() const
{
    return m_captured_arguments_layouts.size();
}

// Get a captured arguments block layout if available.
IGenerated_code_value_layout const *Generated_code_source::get_captured_arguments_layout(
    size_t i) const
{
    if (i >= m_captured_arguments_layouts.size())
        return NULL;

    IGenerated_code_value_layout const *layout = m_captured_arguments_layouts[i].get();
    layout->retain();
    return layout;
}

// Get the number of mapped string constants used inside the generated code.
size_t Generated_code_source::get_string_constant_count() const
{
    return m_mappend_strings.size();
}

// Get the mapped string constant for a given id.
char const *Generated_code_source::get_string_constant(size_t id) const
{
    if (id < m_mappend_strings.size()) {
        return m_mappend_strings[id].c_str();
    }
    return NULL;
}

// Add a mapped string.
void Generated_code_source::add_mapped_string(char const *s, size_t id)
{
    if (id >= m_mappend_strings.size()) {
        m_mappend_strings.resize(id + 1, string(get_allocator()));
    }
    m_mappend_strings[id] = string(s, get_allocator());
}


// Constructor.
Generated_code_source::Source_res_manag::Source_res_manag(
    IAllocator              *alloc,
    Resource_attr_map const *resource_attr_map)
: m_alloc(alloc)
, m_resource_attr_map(alloc)
, m_res_indexes(
    0, Tag_index_map::hasher(), Tag_index_map::key_equal(), alloc)
, m_string_indexes(
    0, String_index_map::hasher(), String_index_map::key_equal(), alloc)
, m_curr_res_idx(0)
, m_curr_string_idx(0)
{
    if (resource_attr_map != NULL) {
        // import
        m_resource_attr_map.insert(resource_attr_map->begin(), resource_attr_map->end());
    }
}

// Register the given resource value and return its 1-based index in the resource table.
// Index 0 represents an invalid resource reference.
size_t Generated_code_source::Source_res_manag::get_resource_index(
    Resource_tag_tuple::Kind   kind,
    char const                 *url,
    int                        tag,
    IType_texture::Shape,
    IValue_texture::gamma_mode)
{
    if (!m_resource_attr_map.empty()) {
        Resource_tag_tuple key(kind, url, tag);

        Resource_attr_map::const_iterator it(m_resource_attr_map.find(key));
        if (it != m_resource_attr_map.end()) {
            mi::mdl::Resource_attr_entry const &e = it->second;
            return e.index;
        }
        // Bad: we have a resource map, but could not find the requested resource.
        // This means the integration was not able to retrieve it from the material
        // and has not loaded it. return 0 (invalid) here, the resource *is* missing.
        return 0;
    }

    switch (kind) {
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
    case Resource_tag_tuple::RK_LIGHT_PROFILE:
    case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
    case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
    case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
    case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
    case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
    case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
    case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
    case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
    case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
        // we support textures, light profiles, bsdf_measurements, and bsdf_data textures
        {
            Tag_index_map::const_iterator it = m_res_indexes.find(tag);
            if (it != m_res_indexes.end())
                return it->second;

            size_t idx = ++m_curr_res_idx;
            m_res_indexes[tag] = idx;
            return idx;
        }

    default:
        // those should never occur in functions
        MDL_ASSERT(!"Unexpected resource type");
        return tag;
    }
}

// Register a string constant and return its 1 based index in the string table.
size_t Generated_code_source::Source_res_manag::get_string_index(IValue_string const *s)
{
    string str(s->get_value(), m_alloc);

    String_index_map::const_iterator it = m_string_indexes.find(str);
    if (it != m_string_indexes.end())
        return it->second;

    if (m_curr_res_idx == 0) {
        // zero is reserved for "Not-a-known-String"
        m_string_indexes[string("<NULL>", m_alloc)] = 0;
    }

    size_t idx = ++m_curr_string_idx;
    m_string_indexes[str] = idx;
    return idx;
}

// Imports a new resource attribute map.
void Generated_code_source::Source_res_manag::import_resource_attribute_map(
    Resource_attr_map const *resource_attr_map)
{
    if (resource_attr_map != NULL) {
        m_resource_attr_map.insert(resource_attr_map->begin(), resource_attr_map->end());
    }
}

// --------------------------- Generated_code_lambda_function ----------------------------

// Constructor.
Generated_code_lambda_function::Generated_code_lambda_function(
    Jitted_code *jitted_code)
: Base(jitted_code->get_allocator())
, m_jitted_code(mi::base::make_handle_dup(jitted_code))
, m_context()
, m_module(NULL)
, m_module_key(0)
, m_jitted_funcs(get_allocator())
, m_res_entries(get_allocator())
, m_string_entries(get_allocator())
, m_messages(get_allocator(), "<lambda expression>")
, m_res_data()
, m_exc_handler(NULL)
, m_aborted(0)
, m_ro_segment(NULL)
, m_ro_length(0)
, m_render_state_usage(
    IGenerated_code_executable::SU_ALL_VARYING_MASK |
    IGenerated_code_executable::SU_ALL_UNIFORM_MASK)
, m_captured_arguments_layouts(get_allocator())
, m_mappend_strings(get_allocator())
{
}

// Destructor.
Generated_code_lambda_function::~Generated_code_lambda_function()
{
    if (m_ro_segment) {
        IAllocator *alloc = m_jitted_code->get_allocator();

        alloc->free((void *)m_ro_segment);
    }

    if (m_module != NULL) {
        m_module = NULL;  // avoid dangling pointer, module will be deleted by delete_llvm_module
        m_jitted_code->delete_llvm_module(m_module_key);
    }
}

// Get the kind of code generated.
IGenerated_code::Kind Generated_code_lambda_function::get_kind() const
{
    return CK_EXECUTABLE;
}

// Get the target language.
char const *Generated_code_lambda_function::get_target_language() const
{
    return "executable";
}

// Check if the code contents are valid.
bool Generated_code_lambda_function::is_valid() const
{
    return m_messages.get_error_message_count() == 0;
}

// Access messages.
Messages const &Generated_code_lambda_function::access_messages() const
{
    return m_messages;
}

// Returns the assembler code of the executable code module if available.
char const *Generated_code_lambda_function::get_source_code(size_t &size) const
{
    // disassembling native code is not yet supported
    size = 0;
    return NULL;
}

// Get the data for the read-only data segment if available.
char const *Generated_code_lambda_function::get_ro_data_segment(size_t &size) const
{
    size = m_ro_length;

    // return the address aligned by 16, this is, were the data is stored
    uintptr_t adr = ((uintptr_t)(m_ro_segment) + 15) & ~15;
    return (char const *)adr;
}

// Initialize the resource handling of this compiled lambda function.
void Generated_code_lambda_function::init(
    void                   *ctx,
    IMDL_exception_handler *exc_handler,
    IResource_handler      *res_handler)
{
    m_exc_handler = exc_handler;
    m_aborted     = 0;

    if (res_handler == NULL)
        return;

    size_t n_res_entries = m_res_entries.size();
    if (n_res_entries == 0)
        return;

    m_res_data.m_obj_size = (res_handler->get_data_size() + 15) & ~size_t(15);
    if (m_res_data.m_obj_size == 0)
        return;

    m_res_data.m_resource_handler = res_handler;

    mi::mdl::IAllocator *alloc = m_jitted_code->get_allocator();

    char *p = m_res_data.m_res_arr =
        reinterpret_cast<char *>(alloc->malloc(m_res_data.m_obj_size * n_res_entries));
    for (size_t i = 0; i < n_res_entries; ++i, p += m_res_data.m_obj_size) {
        Resource_entry const     &entry = m_res_entries[i];
        Resource_tag_tuple::Kind kind   = entry.get_kind();

        switch (kind) {
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
            res_handler->tex_init(
                (void *)p,
                entry.get_shape(),
                entry.get_tag(),
                entry.get_gamma_mode(),
                ctx);
            break;
        case Resource_tag_tuple::RK_LIGHT_PROFILE:
            res_handler->lp_init((void *)p, entry.get_tag(), ctx);
            break;
        case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
            res_handler->bm_init((void *)p, entry.get_tag(), ctx);
            break;

        case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
        case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
            res_handler->tex_init(
                (void *)p,
                IType_texture::TS_BSDF_DATA,
                entry.get_tag(),
                IValue_texture::gamma_linear,
                ctx);
            break;

        default:
            MDL_ASSERT(!"unexpected resource kind");
        }
    }
}

// Terminates the resource handling.
void Generated_code_lambda_function::term()
{
    m_exc_handler = NULL;

    if (m_res_data.m_obj_size == 0)
        return;

    size_t n_res_entries = m_res_entries.size();
    if (n_res_entries > 0) {
        char *p = m_res_data.m_res_arr;
        for (size_t i = 0; i < n_res_entries; ++i, p += m_res_data.m_obj_size) {
            Resource_entry const     &entry = m_res_entries[i];
            Resource_tag_tuple::Kind kind   = entry.get_kind();

            switch (kind) {
            case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
            case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
            case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
                // real texture
                m_res_data.m_resource_handler->tex_term((void *)p, entry.get_shape());
                break;
            case Resource_tag_tuple::RK_LIGHT_PROFILE:
                // light profile
                m_res_data.m_resource_handler->lp_term((void *)p);
                break;
            case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
                // bsdf measurement
                m_res_data.m_resource_handler->bm_term((void *)p);
                break;
            case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
            case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
            case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
            case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
            case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
            case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
            case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
            case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
                // bsdf data texture
                m_res_data.m_resource_handler->tex_term((void *)p, IType_texture::TS_BSDF_DATA);
                break;
            default:
                MDL_ASSERT(!"unexpected value kind");
            }
        }

        mi::mdl::IAllocator *alloc = m_jitted_code->get_allocator();
        alloc->free(m_res_data.m_res_arr);
        m_res_data.clear();
    }
}

// Run the environment function on the current transaction.
bool Generated_code_lambda_function::run_environment(
    size_t                          index,
    RGB_color                       *result,
    Shading_state_environment const *state,
    void                            *tex_data)
{
    if (!m_aborted && index < m_jitted_funcs.size()) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, tex_data);

        if (setjmp(exc.env) == 0) {
            Env_func *env_func = reinterpret_cast<Env_func *>(m_jitted_funcs[index]);
            env_func(result, state, pair, exc, NULL);
            return true;
        }
    }

    // black for now
    result->r = result->g = result->b = 0.0f;
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(bool &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_bool *bool_func = reinterpret_cast<Lambda_func_bool *>(m_jitted_funcs[0]);
            bool_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(int &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_int *int_func = reinterpret_cast<Lambda_func_int *>(m_jitted_funcs[0]);
            int_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(unsigned &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_unsigned *unsigned_func =
                reinterpret_cast<Lambda_func_unsigned *>(m_jitted_funcs[0]);
            unsigned_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(float &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_float *float_func =
                reinterpret_cast<Lambda_func_float *>(m_jitted_funcs[0]);
            float_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(Float2_struct &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_float2 *float2_func =
                reinterpret_cast<Lambda_func_float2 *>(m_jitted_funcs[0]);
            float2_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(Float3_struct &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_float3 *float3_func =
                reinterpret_cast<Lambda_func_float3 *>(m_jitted_funcs[0]);
            float3_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(Float4_struct &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_float4 *float4_func =
                reinterpret_cast<Lambda_func_float4 *>(m_jitted_funcs[0]);
            float4_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(Matrix3x3_struct &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_float3x3 *float3x3_func =
                reinterpret_cast<Lambda_func_float3x3 *>(m_jitted_funcs[0]);
            float3x3_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(Matrix4x4_struct &result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_float4x4 *float4x4_func =
                reinterpret_cast<Lambda_func_float4x4 *>(m_jitted_funcs[0]);
            float4x4_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run(char const *&result)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, NULL);

        if (setjmp(exc.env) == 0) {
            Lambda_func_string *string_func =
                reinterpret_cast<Lambda_func_string *>(m_jitted_funcs[0]);
            string_func(result, pair, exc, NULL);
            return true;
        }
    }
    return false;
}

// Run a switch function on the current transaction.
bool Generated_code_lambda_function::run_core(
    unsigned                     proj,
    Float3_struct                &result,
    Shading_state_material const *state,
    void                         *tex_data,
    void const                   *cap_args)
{
    if (!m_aborted && m_jitted_funcs.size() > 0) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, tex_data);

        if (setjmp(exc.env) == 0) {
            Core_func *core_func = reinterpret_cast<Core_func *>(m_jitted_funcs[0]);
            return core_func(state, pair, exc, cap_args, result, proj);
        }
    }
    return false;
}

// Run the function on the current transaction.
bool Generated_code_lambda_function::run_generic(
    size_t                       index,
    void                         *result,
    Shading_state_material const *state,
    void                         *tex_data,
    void const                   *cap_args)
{
    if (!m_aborted && index < m_jitted_funcs.size()) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, tex_data);

        if (setjmp(exc.env) == 0) {
            Gen_func *gen_func = reinterpret_cast<Gen_func *>(m_jitted_funcs[index]);
            gen_func(result, state, pair, exc, cap_args);
            return true;
        }
    }
    return false;
}

// Run the init function on the current transaction.
bool Generated_code_lambda_function::run_init(
    size_t                 index,
    Shading_state_material *state,
    void                   *tex_data,
    void const             *cap_args)
{
    if (!m_aborted && index < m_jitted_funcs.size()) {
        Exc_state     exc(m_exc_handler, m_aborted);
        Res_data_pair pair(m_res_data, tex_data);

        if (setjmp(exc.env) == 0) {
            Init_func *init_func = reinterpret_cast<Init_func *>(m_jitted_funcs[index]);
            init_func(state, pair, exc, cap_args);
            return true;
        }
    }
    return false;
}

// Get the used state properties of  the generated lambda function code.
IGenerated_code_lambda_function::State_usage
    Generated_code_lambda_function::get_state_usage() const
{
    return m_render_state_usage;
}

// Get the number of captured argument block layouts.
size_t Generated_code_lambda_function::get_captured_argument_layouts_count() const
{
    return m_captured_arguments_layouts.size();
}

// Get a captured arguments block layout if available.
IGenerated_code_value_layout const *Generated_code_lambda_function::get_captured_arguments_layout(
    size_t i) const
{
    if (i >= m_captured_arguments_layouts.size())
        return NULL;

    IGenerated_code_value_layout const *layout = m_captured_arguments_layouts[i].get();
    layout->retain();
    return layout;
}

// Get the number of mapped string constants used inside the generated code.
size_t Generated_code_lambda_function::get_string_constant_count() const
{
    return m_mappend_strings.size();
}

// Get the mapped string constant for a given id.
char const *Generated_code_lambda_function::get_string_constant(size_t id) const
{
    if (id < m_mappend_strings.size())
        return m_mappend_strings[id].c_str();
    return NULL;
}

// Add a mapped string.
void Generated_code_lambda_function::add_mapped_string(char const *s, size_t id)
{
    if (id >= m_mappend_strings.size()) {
        m_mappend_strings.resize(id + 1, string(get_allocator()));
    }
    m_mappend_strings[id] = string(s, get_allocator());
}

// Set the entry point the the JIT compiled function.
void Generated_code_lambda_function::add_entry_point(void *address)
{
    m_jitted_funcs.push_back(reinterpret_cast<Jitted_func *>(address));
}

// Set the Read-Only data segment.
void Generated_code_lambda_function::set_ro_segment(char const *data, size_t size)
{
    IAllocator *alloc = m_jitted_code->get_allocator();

    if (m_ro_segment != NULL)
        alloc->free((void *)m_ro_segment);
    m_ro_segment = NULL;

    m_ro_length = size;
    if (size > 0) {
        m_ro_segment = (char *)alloc->malloc(size + 15);

        // the data will be directly accessible from the JITed code, for that
        // it must be aligned by 16
        uintptr_t adr = ((uintptr_t)m_ro_segment + 15) & ~15;
        char *p = (char *)adr;
        memcpy(p, data, size);
    }
}

// Returns the index of the given resource for use as an parameter to a resource-related
// function in the generated CPU code.
mi::Uint32 Generated_code_lambda_function::get_known_resource_index(mi::Uint32 tag) const
{
    for (size_t i = 0, n = m_res_entries.size(); i < n; ++i) {
        if (m_res_entries[i].get_tag() == tag)
            return i + 1;
    }
    return 0;  // invalid resource reference
}

// Register a new non-texture resource tag.
size_t Generated_code_lambda_function::register_resource_tag(
    unsigned                 tag,
    Resource_tag_tuple::Kind kind)
{
    m_res_entries.push_back(Resource_entry(tag, kind));
    return m_res_entries.size();
}

// Register a new texture resource tag.
size_t Generated_code_lambda_function::register_texture_tag(
    unsigned                   tag,
    IType_texture::Shape       tex_shape,
    IValue_texture::gamma_mode gamma_mode)
{
    Resource_tag_tuple::Kind kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT;
    switch (gamma_mode) {
    case IValue_texture::gamma_default:
        kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT;
        break;
    case IValue_texture::gamma_linear:
        kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR;
        break;
    case IValue_texture::gamma_srgb:
        kind = Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT;
        break;
    }
    m_res_entries.push_back(
        Resource_entry(tag, kind, tex_shape));
    return m_res_entries.size();
}

// Register a new string.
size_t Generated_code_lambda_function::register_string(
    char const *s)
{
    m_string_entries.push_back(string(s, get_allocator()));
    return m_string_entries.size();
}

// Constructor.
Generated_code_lambda_function::Lambda_res_manag::Lambda_res_manag(
    Generated_code_lambda_function &lambda,
    Resource_attr_map const       *resource_map)
: m_lambda(lambda)
, m_resource_map(resource_map)
, m_res_indexes(
    0, Tag_index_map::hasher(), Tag_index_map::key_equal(), lambda.get_allocator())
, m_string_indexes(
    0, String_index_map::hasher(), String_index_map::key_equal(), lambda.get_allocator())
{
}

// Returns the resource index for the given resource usable by the target code resource
// handler for the corresponding resource type.
// Index 0 represents an invalid resource reference.
size_t Generated_code_lambda_function::Lambda_res_manag::get_resource_index(
    Resource_tag_tuple::Kind   kind,
    char const                 *url,
    int                        tag,
    IType_texture::Shape       shape,
    IValue_texture::gamma_mode gamma_mode)
{
    if (m_resource_map != NULL) {
        Resource_tag_tuple key(kind, url, tag);
        Resource_attr_map::const_iterator it(m_resource_map->find(key));
        if (it != m_resource_map->end()) {
            mi::mdl::Resource_attr_entry const &e = it->second;
            return e.index;
        }
    }

    switch (kind) {
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
    case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
    case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
    case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
    case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
    case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
    case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
    case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
    case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
        // we support textures, ...
        {
            Tag_index_map::const_iterator it = m_res_indexes.find(tag);
            if (it != m_res_indexes.end())
                return it->second;

            size_t idx = m_lambda.register_texture_tag(tag, shape, gamma_mode);
            m_res_indexes[tag] = idx;
            return idx;
        }
    case Resource_tag_tuple::RK_LIGHT_PROFILE:
    case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
        // ... light profiles, and bsdf_measurements
        {
            Tag_index_map::const_iterator it = m_res_indexes.find(tag);
            if (it != m_res_indexes.end())
                return it->second;

            size_t idx = m_lambda.register_resource_tag(tag, kind);
            m_res_indexes[tag] = idx;
            return idx;
        }

    default:
        // those should never occur in functions
        MDL_ASSERT(!"Unexpected resource type");
        return tag;
    }
}

// Register a string constant and return its 1 based index in the string table.
size_t Generated_code_lambda_function::Lambda_res_manag::get_string_index(
    IValue_string const *s)
{
    if (m_resource_map != NULL) {
        Resource_tag_tuple k(kind_from_value(s), s->get_value(), /*tag=*/0);
        Resource_attr_map::const_iterator it(m_resource_map->find(k));
        if (it != m_resource_map->end()) {
            mi::mdl::Resource_attr_entry const &e = it->second;
            return e.index;
        }
    }

    string str(s->get_value(), m_lambda.get_allocator());
    String_index_map::const_iterator it = m_string_indexes.find(str);
    if (it != m_string_indexes.end())
        return it->second;

    size_t idx = m_lambda.register_string(s->get_value());
    m_string_indexes[str] = idx;
    return idx;
}

/// Helper struct for sorting IValue_resources.
struct Resource_index_pair
{
    Resource_index_pair(
        Resource_tag_tuple const& val,
        size_t index, IType_texture::Shape shape,
        IValue_texture::gamma_mode gamma_mode)
    : val(val), index(index), shape(shape), gamma_mode(gamma_mode)
    {}

    Resource_tag_tuple val;
    size_t index;
    IType_texture::Shape shape;
    IValue_texture::gamma_mode gamma_mode;
};

/// Helper struct for comparing Resource_index_pair objects.
struct Resource_index_pair_compare
{
    /// Returns true if 'a' should be placed before 'b'.
    bool operator()(Resource_index_pair const &a, Resource_index_pair const &b) const
    {
        return a.index < b.index;
    }
};

/// Registers all resources in the given resource map in the order of the associated indices.
void Generated_code_lambda_function::Lambda_res_manag::import_from_resource_attribute_map(
    Resource_attr_map const *resource_map)
{
    if (resource_map->size() == 0)
        return;
    // Sort by index to avoid indeterministic behavior due to pointer hash map.
    mi::mdl::vector<Resource_index_pair>::Type sorted_resources(m_lambda.get_allocator());
    sorted_resources.reserve(resource_map->size());
    for (Resource_attr_map::const_iterator it = resource_map->begin(), end = resource_map->end();
        it != end; ++it)
    {
        Resource_tag_tuple const& val = it->first;
        Resource_attr_entry const& e = it->second;
        if (val.m_kind == Resource_tag_tuple::RK_BAD ||
                val.m_kind == Resource_tag_tuple::RK_INVALID_REF)
            continue;
        IType_texture::Shape shape = IType_texture::TS_2D;
        IValue_texture::gamma_mode gamma_mode = IValue_texture::gamma_default;

        switch (val.m_kind) {
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
            gamma_mode = IValue_texture::gamma_default;
            shape = e.u.tex.shape;
            break;
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
            gamma_mode = IValue_texture::gamma_srgb;
            shape = e.u.tex.shape;
            break;
        case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
        case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
        case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
        case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
        case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
        case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
            gamma_mode = IValue_texture::gamma_linear;
            shape = e.u.tex.shape;
            break;
        case Resource_tag_tuple::RK_LIGHT_PROFILE:
        case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
            // gamma_mode and shape will be ignored
            break;
        default:
            MDL_ASSERT(!"unexpected kind");
            break;
        }

        sorted_resources.push_back(Resource_index_pair(val, it->second.index, shape, gamma_mode));
    }
    std::sort(sorted_resources.begin(), sorted_resources.end(), Resource_index_pair_compare());

    for (size_t i = 0, n = sorted_resources.size(); i < n; ++i) {
        Resource_tag_tuple const& val = sorted_resources[i].val;
        get_resource_index(
            val.m_kind,
            val.m_url,
            val.m_tag,
            sorted_resources[i].shape,
            sorted_resources[i].gamma_mode);
    }
}

}  // mdl
}  // mi
