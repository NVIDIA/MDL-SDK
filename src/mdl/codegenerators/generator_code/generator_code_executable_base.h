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

#ifndef MDL_GENERATOR_CODE_EXECUTABLE_BASE_H
#define MDL_GENERATOR_CODE_EXECUTABLE_BASE_H 1

#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_generated_executable.h>
#include <mi/base/handle.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"

namespace mi {
namespace mdl {

/// Structure containing information about a function in a generated executable code object.
struct Generated_code_function_info
{
    /// Constructor.
    ///
    /// \param name         The name of the function.
    /// \param dist_kind    The kind of distribution function, if it is a distribution function.
    /// \param kind         The kind of the function.
    /// \param index        The index of the target argument block associated with this function,
    ///                     or ~0 if not used.
    /// \param state_usage  The state usage of the function.
    Generated_code_function_info(
        string const                                  &name,
        IGenerated_code_executable::Distribution_kind dist_kind,
        IGenerated_code_executable::Function_kind     kind,
        size_t                                        index,
        IGenerated_code_executable::State_usage       state_usage)
    : m_name(name)
    , m_dist_kind(dist_kind)
    , m_kind(kind)
    , m_prototypes(name.get_allocator())
    , m_arg_block_index(index)
    , m_df_handle_name_table(name.get_allocator())
    , m_state_usage(state_usage)
    {}

    /// The name of the function.
    string m_name;

    /// The kind of distribution function, if it is a distribution function.
    IGenerated_code_executable::Distribution_kind m_dist_kind;

    /// The kind of the function.
    IGenerated_code_executable::Function_kind m_kind;

    /// The prototypes for the different languages according to
    /// #mi::mdl::ILink_unit::Prototype_language.
    vector<string>::Type m_prototypes;

    /// The index of the target argument block associated with this function, or ~0 if not used.
    size_t m_arg_block_index;

    /// The DF handle name table.
    vector<string>::Type m_df_handle_name_table;

    /// The state usage of the function.
    IGenerated_code_executable::State_usage m_state_usage;
};

///
/// Base class for classes implementing IGenerated_code_executable.
///
template <class Interface>
class Generated_code_executable_base : public Allocator_interface_implement<Interface>
{
    typedef Allocator_interface_implement<Interface> Base;
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    Generated_code_executable_base(IAllocator *alloc)
    : Base(alloc)
    , m_func_infos(alloc)
    , m_captured_arguments_layouts(alloc)
    , m_mappend_strings(alloc)
    {}

    // ------------------- from IGenerated_code_executable -------------------

    /// Get the number of functions in this link unit.
    size_t get_function_count() const MDL_FINAL;

    /// Get the name of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return the name of the i'th function or NULL if the index is out of bounds
    char const *get_function_name(size_t i) const MDL_FINAL;

    /// Returns the distribution kind of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return The distribution kind of the i'th function or \c FK_INVALID if \p i was invalid.
    IGenerated_code_executable::Distribution_kind get_distribution_kind(size_t i) const MDL_FINAL;

    /// Returns the function kind of the i'th function inside this link unit.
    ///
    /// \param i  the index of the function
    ///
    /// \return The function kind of the i'th function or \c FK_INVALID if \p i was invalid.
    IGenerated_code_executable::Function_kind get_function_kind(size_t i) const MDL_FINAL;

    /// Get the index of the target argument block layout for the i'th function inside this link
    /// unit if used.
    ///
    /// \param i  the index of the function
    ///
    /// \return The index of the target argument block layout or ~0 if not used or \p i is invalid.
    size_t get_function_arg_block_layout_index(size_t i) const MDL_FINAL;

    /// Returns the prototype of the i'th function inside this link unit.
    ///
    /// \param index   the index of the function.
    /// \param lang    the language to use for the prototype.
    ///
    /// \return The prototype or NULL if \p index is out of bounds or \p lang cannot be used
    ///         for this target code.
    char const *get_function_prototype(
        size_t                                         index,
        IGenerated_code_executable::Prototype_language lang) const MDL_FINAL;

    /// Set a function prototype for a function.
    ///
    /// \param index  the index of the function
    /// \param lang   the language of the prototype being set
    /// \param proto  the function prototype
    void set_function_prototype(
        size_t                                         index,
        IGenerated_code_executable::Prototype_language lang,
        char const *prototype) MDL_FINAL;

    /// Add a function to the given target code, also registering the function prototypes
    /// applicable for the used backend.
    ///
    /// \param name             the name of the function to add
    /// \param dist_kind        the kind of distribution to add
    /// \param func_kind        the kind of the function to add
    /// \param arg_block_index  the argument block index for this function or ~0 if not used
    /// \param state_usage      the state usage of the function to add
    ///
    /// \returns the function index of the added function
    size_t add_function_info(
        char const *name,
        IGenerated_code_executable::Distribution_kind dist_kind,
        IGenerated_code_executable::Function_kind     func_kind,
        size_t                                        arg_block_index,
        IGenerated_code_executable::State_usage       state_usage) MDL_FINAL;

    /// Get the number of distribution function handles referenced by a function.
    ///
    /// \param func_index   the index of the function
    ///
    /// \return The number of distribution function handles referenced or \c 0, if the
    ///         function is not a distribution function.
    size_t get_function_df_handle_count(size_t func_index) const MDL_FINAL;

    /// Get the name of a distribution function handle referenced by a function.
    ///
    /// \param func_index     The index of the function.
    /// \param handle_index   The index of the handle.
    ///
    /// \return The name of the distribution function handle or \c NULL, if the
    ///         function is not a distribution function or \p index is invalid.
    char const *get_function_df_handle(
        size_t func_index,
        size_t handle_index) const MDL_FINAL;

    /// Add the name of a distribution function handle referenced by a function.
    ///
    /// \param func_index     The index of the function.
    /// \param handle_name    The name of the handle.
    ///
    /// \return The index of the added handle.
    size_t add_function_df_handle(
        size_t     func_index,
        char const *handle_name) MDL_FINAL;

    /// Get the state properties used by a function.
    ///
    /// \param func_index     The index of the function.
    ///
    /// \return The state usage or 0, if the \p func_index was invalid.
    IGenerated_code_executable::State_usage get_function_state_usage(
        size_t func_index) const MDL_FINAL;

    /// Get the number of captured argument block layouts.
    size_t get_captured_argument_layouts_count() const MDL_FINAL;

    /// Get a captured arguments block layout if available.
    ///
    /// \param i   the index of the block layout
    ///
    /// \returns the layout or NULL if no arguments were captured or the index is invalid.
    IGenerated_code_value_layout const *get_captured_arguments_layout(
        size_t i) const MDL_FINAL;

    /// Get the number of mapped string constants used inside the generated code.
    size_t get_string_constant_count() const MDL_FINAL;

    /// Get the mapped string constant for a given id.
    ///
    /// \param id  the string id (as used in the generated code)
    ///
    /// \return the MDL string constant that corresponds to the given id or NULL
    ///         if id is out of range
    ///
    /// \note that the id 0 is ALWAYS mapped to the empty string ""
    char const *get_string_constant(size_t id) const MDL_FINAL;

public:
    /// Add a captured arguments layout.
    void add_captured_arguments_layout(
        IGenerated_code_value_layout const *layout)
    {
        m_captured_arguments_layouts.push_back(mi::base::make_handle_dup(layout));
    }

    /// Add a mapped string.
    ///
    /// \param s   the string constant
    /// \param id  the assigned id for this constant
    void add_mapped_string(
        char const *s,
        size_t     id)
    {
        if (id >= m_mappend_strings.size()) {
            m_mappend_strings.resize(id + 1, string(this->get_allocator()));
        }
        m_mappend_strings[id] = string(s, this->get_allocator());
    }


private:
    typedef vector<Generated_code_function_info>::Type Func_info_vec;

    /// Function infos of all externally visible functions inside this generated code object.
    Func_info_vec m_func_infos;

    typedef vector<mi::base::Handle<IGenerated_code_value_layout const> >::Type Layout_vec;

    /// The list of captured arguments block layouts.
    Layout_vec m_captured_arguments_layouts;

    typedef vector<string>::Type Mappend_string_vector;

    /// The mapped strings.
    Mappend_string_vector m_mappend_strings;
};


}  // mdl
}  // mi

#endif // MDL_GENERATOR_CODE_EXECUTABLE_BASE_H
