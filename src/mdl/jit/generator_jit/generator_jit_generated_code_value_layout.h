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

#ifndef MDL_GENERATOR_JIT_GENERATED_CODE_VALUE_LAYOUT
#define MDL_GENERATOR_JIT_GENERATED_CODE_VALUE_LAYOUT 1

#include <mi/mdl/mdl_generated_executable.h>

#include <mdl/compiler/compilercore/compilercore_cc_conf.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_messages.h>
#include <mdl/compiler/compilercore/compilercore_tools.h>

namespace mi {
namespace mdl {

class LLVM_code_generator;

/// Represents the layout of an argument value block with support for nested elements.
/// Implementation of #mi::mdl::IGenerated_code_value_layout.
class Generated_code_value_layout
    : public Allocator_interface_implement<IGenerated_code_value_layout>
{
    typedef Allocator_interface_implement<IGenerated_code_value_layout> Base;

public:
    /// Constructor.
    ///
    /// \param alloc     The allocator.
    /// \param code_gen  The LLVM code generator providing the information about the captured
    ///                  arguments and the type mapper.
    Generated_code_value_layout(
        IAllocator          *alloc,
        LLVM_code_generator *code_gen);

    /// Constructor from layout data block.
    ///
    /// \param alloc                  The allocator.
    /// \param layout_data_block      The data block containing the layout data.
    /// \param layout_data_size       The size of the layout data block.
    /// \param strings_mapped_to_ids  True, if strings are mapped to IDs.
    Generated_code_value_layout(
        IAllocator *alloc,
        char const *layout_data_block,
        size_t     layout_data_size,
        bool       strings_mapped_to_ids);

    /// Constructor used for deserialization.
    ///
    /// \param alloc     The allocator.
    Generated_code_value_layout(
        IAllocator* alloc);

    /// Returns the size of the target argument block.
    size_t get_size() const MDL_FINAL;

    /// Returns the number of arguments / elements at the given layout state.
    ///
    /// \param state  The layout state representing the current nesting within the
    ///               argument value block. The default value is used for the top-level.
    size_t get_num_elements(State state = State()) const MDL_FINAL;

    /// Get the offset, the size and the kind of the argument / element inside the argument
    /// block at the given layout state.
    ///
    /// \param[out]  kind      Receives the kind of the argument.
    /// \param[out]  arg_size  Receives the size of the argument.
    /// \param       state     The layout state representing the current nesting within the
    ///                        argument value block. The default value is used for the top-level.
    ///
    /// \returns the offset of the requested argument / element or ~size_t  (0) if the state
    ///          is invalid.
    size_t get_layout(
        mi::mdl::IValue::Kind &kind,
        size_t                &arg_size,
        State                 state = State()) const MDL_FINAL;

    /// Get the layout state for the i'th argument / element inside the argument value block
    /// at the given layout state.
    ///
    /// \param i      The index of the argument / element.
    /// \param state  The layout state representing the current nesting within the argument
    ///               value block. The default value is used for the top-level.
    ///
    /// \returns the layout state for the nested element or a state with m_state_offs set to
    ///          ~mi::Uint32(0) if the element is atomic.
    IGenerated_code_value_layout::State get_nested_state(
        size_t i,
        State  state = State()) const MDL_FINAL;

    /// Set the value inside the given block at the given layout state.
    ///
    /// \param[inout] block           The argument value block buffer to be modified.
    /// \param[in]    value           The value to be set. It has to match the expected kind.
    /// \param[in]    value_callback  Callback for retrieving resource indices for resource values.
    /// \param[in]    state           The layout state representing the current nesting within the
    ///                               argument value block.
    ///                               The default value is used for the top-level.
    ///
    /// \return
    ///                      -  0: Success.
    ///                      - -1: Invalid parameters, block or value is a \c NULL pointer.
    ///                      - -2: Invalid state provided.
    ///                      - -3: Value kind does not match expected kind.
    ///                      - -4: Size of compound value does not match expected size.
    ///                      - -5: Unsupported value type.
    int set_value(
        char                           *block,
        mi::mdl::IValue const          *value,
        IGenerated_code_value_callback *value_callback,
        State                          state = State()) const MDL_FINAL;

    // Non-API methods

    /// Get the layout data buffer and its size.
    char const* get_layout_data(size_t& size) const;

    /// Set the layout data buffer and its size.
    /// Setter is used for deserialization.
    void set_layout_data(char const* data, size_t &size);

    /// If true, string argument values are mapped to string identifiers.
    bool get_strings_mapped_to_ids() const { return m_strings_mapped_to_ids; }

    /// If true, string argument values are mapped to string identifiers.
    /// Setter is used for deserialization.
    void set_strings_mapped_to_ids(bool value) { m_strings_mapped_to_ids = value; }

private:
    /// The layout data buffer.
    vector<unsigned char>::Type m_layout_data;

    /// If true, string argument values are mapped to string identifiers.
    bool m_strings_mapped_to_ids;
};

// Allow impl_cast on Generated_code_value_layout
template<>
inline Generated_code_value_layout const *impl_cast(IGenerated_code_value_layout const *t) {
    return static_cast<Generated_code_value_layout const *>(t);
}

// Allow impl_cast on Generated_code_value_layout
template<>
inline Generated_code_value_layout* impl_cast(IGenerated_code_value_layout* t) {
    return static_cast<Generated_code_value_layout*>(t);
}

}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_GENERATED_CODE
