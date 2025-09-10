/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "generator_code_executable_base.h"

namespace mi {
namespace mdl {


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
    if (i < m_func_infos.size()) {
        return m_func_infos[i].m_name.c_str();
    }
    return NULL;
}

// Returns the distribution kind of the i'th function inside this link unit.
template <class I>
IGenerated_code_executable::Distribution_kind
Generated_code_executable_base<I>::get_distribution_kind(size_t i) const
{
    if (i < m_func_infos.size()) {
        return m_func_infos[i].m_dist_kind;
    }
    return IGenerated_code_executable::DK_INVALID;
}

// Returns the function kind of the i'th function inside this link unit.
template <class I>
IGenerated_code_executable::Function_kind
Generated_code_executable_base<I>::get_function_kind(size_t i) const
{
    if (i < m_func_infos.size()) {
        return m_func_infos[i].m_kind;
    }
    return IGenerated_code_executable::FK_INVALID;
}

// Get the index of the target argument block layout for the i'th function inside this link
// unit if used.
template <class I>
size_t Generated_code_executable_base<I>::get_function_arg_block_layout_index(size_t i) const
{
    if (i < m_func_infos.size()) {
        return m_func_infos[i].m_arg_block_index;
    }
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
    if (index >= m_func_infos.size()) {
        return;
    }

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
    if (func_index >= m_func_infos.size()) {
        return ~0;
    }

    m_func_infos[func_index].m_df_handle_name_table.push_back(
        string(handle_name, this->get_allocator()));

    return m_func_infos[func_index].m_df_handle_name_table.size() - 1;
}

// Get the state properties used by a function.
template <class I>
IGenerated_code_executable::State_usage Generated_code_executable_base<I>::get_function_state_usage(
    size_t func_index) const
{
    if (func_index >= m_func_infos.size()) {
        return 0;
    }
    return m_func_infos[func_index].m_state_usage;
}

// Get the number of captured argument block layouts.
template <class I>
size_t Generated_code_executable_base<I>::get_captured_argument_layouts_count() const
{
    return m_captured_arguments_layouts.size();
}

// Get a captured arguments block layout if available.
template <class I>
IGenerated_code_value_layout const *
Generated_code_executable_base<I>::get_captured_arguments_layout(
    size_t i) const
{
    if (i >= m_captured_arguments_layouts.size()) {
        return NULL;
    }
    IGenerated_code_value_layout const *layout = m_captured_arguments_layouts[i].get();
    layout->retain();
    return layout;
}

// Get the number of mapped string constants used inside the generated code.
template <class I>
size_t Generated_code_executable_base<I>::get_string_constant_count() const
{
    return m_mappend_strings.size();
}

// Get the mapped string constant for a given id.
template <class I>
char const *Generated_code_executable_base<I>::get_string_constant(size_t id) const
{
    if (id < m_mappend_strings.size()) {
        return m_mappend_strings[id].c_str();
    }
    return NULL;
}

// instantiate
template class Generated_code_executable_base<IGenerated_code_executable>;
template class Generated_code_executable_base<IGenerated_code_lambda_function>;

} // mdl
} // mi
