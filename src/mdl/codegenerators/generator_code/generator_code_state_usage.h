/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_CODE_STATE_USAGE_H
#define MDL_GENERATOR_CODE_STATE_USAGE_H 1

#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_generated_code.h>
#include <mi/mdl/mdl_generated_executable.h>
#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

namespace mi {
namespace mdl {

/// Helper class storing information about state usage per function and module.
///
/// \tparam Function  the function object type
/// \tparam Code_gen  the code generator type
template<typename Function, typename Code_gen>
class State_usage_analysis
{
public:
    typedef IGenerated_code_lambda_function::State_usage State_usage;

    /// Constructor.
    ///
    /// \param code_gen  the code generator
    State_usage_analysis(Code_gen &code_gen);

    /// Register a function to take part in the analysis.
    ///
    /// \param func  the function
    void register_function(
        Function *func);

    /// Register a function to take part in the analysis, which has been cloned from an
    /// already registered function. The state usage is initialized with the usage of the
    /// original function.
    ///
    /// \param cloned_func  the cloned function which shall be registered
    /// \param orig_func    the original function
    void register_cloned_function(
        Function *cloned_func,
        Function *orig_func);

    /// Register a mapped function to set the "expected" usage.
    ///
    /// \param func  the function
    /// \param sema  the MDL semantics
    void register_mapped_function(
        Function                    *func,
        mdl::IDefinition::Semantics sema);

    /// Add a state usage flag to the given function.
    ///
    /// \param func         the function
    /// \param flag_to_add  the state usage to add
    void add_state_usage(
        Function    *func,
        State_usage flag_to_add);

    /// Add a call edge from a caller to a callee to the call graph.
    ///
    /// \param caller  the caller function
    /// \param callee  the callee function
    void add_call(
        Function *caller,
        Function *callee);

    /// Updates the state usage of the exported functions of the code generator.
    void update_exported_functions_state_usage();

    /// Returns the state usage for the whole module.
    State_usage get_module_state_usage() const
    {
        return m_module_state_usage;
    }

    void clear()
    {
        m_module_state_usage = 0;
        m_func_state_usage_info_map.clear();
    }

private:
    /// The code generator whose exported functions will be updated in finalize state usage.
    Code_gen &m_code_gen;

    /// The memory arena used to allocate usage information.
    mi::mdl::Memory_arena m_arena;

    /// The builder for objects on the memory arena.
    mi::mdl::Arena_builder m_arena_builder;

    /// The state usage of the whole module.
    State_usage m_module_state_usage;

    class Function_state_usage_info
    {
    public:
        State_usage state_usage;
        typename mi::mdl::Arena_ptr_hash_set<Function>::Type called_funcs;

        Function_state_usage_info(Memory_arena *arena)
            : state_usage(0)
            , called_funcs(arena)
        {}
    };

    typedef typename mi::mdl::ptr_hash_map<Function, Function_state_usage_info *>::Type
        Function_state_usage_info_map;

    /// Map from functions to per-function state-usage information.
    Function_state_usage_info_map m_func_state_usage_info_map;
};

// -------------------------------- Implementation --------------------------------

// Constructor.
template<typename Function, typename Code_gen>
State_usage_analysis<Function, Code_gen>::State_usage_analysis(Code_gen &code_gen)
: m_code_gen(code_gen)
, m_arena(code_gen.get_allocator())
, m_arena_builder(m_arena)
, m_module_state_usage(0)
, m_func_state_usage_info_map(code_gen.get_allocator())
{
}

// Register a function to take part in the analysis.
template<typename Function, typename Code_gen>
void State_usage_analysis<Function, Code_gen>::register_function(Function *func)
{
    Function_state_usage_info *info =
        m_arena_builder.create<Function_state_usage_info>(&m_arena);
    m_func_state_usage_info_map[func] = info;
}

// Register a function to take part in the analysis, which has been cloned from an
// already registered function. The state usage is initialized with the usage of the
// original function.
template<typename Function, typename Code_gen>
void State_usage_analysis<Function, Code_gen>::register_cloned_function(
    Function *cloned_func,
    Function *orig_func)
{
    Function_state_usage_info *info =
        m_arena_builder.create<Function_state_usage_info>(&m_arena);
    m_func_state_usage_info_map[cloned_func] = info;

    typename Function_state_usage_info_map::iterator it = m_func_state_usage_info_map.find(orig_func);
    if (it == m_func_state_usage_info_map.end()) {
        MDL_ASSERT(!"Function not registered for state usage info");
        return;
    }

    Function_state_usage_info *orig_info = it->second;
    info->state_usage = orig_info->state_usage;
    info->called_funcs.insert(orig_info->called_funcs.cbegin(), orig_info->called_funcs.cend());
}

// Register a mapped function to set the "expected" usage.
template<typename Function, typename Code_gen>
void State_usage_analysis<Function, Code_gen>::register_mapped_function(
    Function                    *func,
    mdl::IDefinition::Semantics sema)
{
    Function_state_usage_info *info =
        m_arena_builder.create<Function_state_usage_info>(&m_arena);
    m_func_state_usage_info_map[func] = info;

    State_usage flag_to_add;

    // If a state function is mapped, assume its state is accessed. This might be
    // not enough, but probably an educated guess.
    switch (sema) {
    default:
    case mi::mdl::IDefinition::DS_UNKNOWN:
        return;

#define CASE(state) \
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_##state: \
        flag_to_add = mi::mdl::IGenerated_code_executable::SU_##state; \
        break;

#define CASE_TEXTURE_TANGENTS(state) \
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_##state: \
        flag_to_add = mi::mdl::IGenerated_code_executable::SU_TEXTURE_TANGENTS; \
        break;

#define CASE_GEOMETRY_TANGENTS(state) \
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_##state: \
        flag_to_add = mi::mdl::IGenerated_code_executable::SU_GEOMETRY_TANGENTS; \
        break;

#define CASE_TRANSFORMS(state) \
    case mi::mdl::IDefinition::DS_INTRINSIC_STATE_##state: \
        flag_to_add = mi::mdl::IGenerated_code_executable::SU_TRANSFORMS; \
        break;

    CASE(POSITION)
    CASE(NORMAL)
    CASE(GEOMETRY_NORMAL)
    CASE(MOTION)
    CASE(TEXTURE_COORDINATE)
    CASE_TEXTURE_TANGENTS(TEXTURE_TANGENT_U)
    CASE_TEXTURE_TANGENTS(TEXTURE_TANGENT_V)
    CASE(TANGENT_SPACE)
    CASE_GEOMETRY_TANGENTS(GEOMETRY_TANGENT_U)
    CASE_GEOMETRY_TANGENTS(GEOMETRY_TANGENT_V)
    CASE(DIRECTION)
    CASE(ANIMATION_TIME)
    CASE_TRANSFORMS(TRANSFORM)
    CASE_TRANSFORMS(TRANSFORM_POINT)
    CASE_TRANSFORMS(TRANSFORM_VECTOR)
    CASE_TRANSFORMS(TRANSFORM_NORMAL)
    CASE_TRANSFORMS(TRANSFORM_SCALE)
    CASE(ROUNDED_CORNER_NORMAL)
    CASE(OBJECT_ID)

#undef CASE_TRANSFORMS
#undef CASE_GEOMETRY_TANGENTS
#undef CASE_TEXTURE_TANGENTS
#undef CASE
    }

    info->state_usage    |= flag_to_add;
    m_module_state_usage |= flag_to_add;
}

// Add a state usage flag to the currently compiled function.
template<typename Function, typename Code_gen>
void State_usage_analysis<Function, Code_gen>::add_state_usage(Function *func, State_usage flag_to_add)
{
    m_module_state_usage |= flag_to_add;

    typename Function_state_usage_info_map::iterator it = m_func_state_usage_info_map.find(func);
    if (it == m_func_state_usage_info_map.end()) {
        MDL_ASSERT(!"Function not registered for state usage info");
        return;
    }

    it->second->state_usage |= flag_to_add;
}

// Add a call to the call graph.
template<typename Function, typename Code_gen>
void State_usage_analysis<Function, Code_gen>::add_call(Function *caller, Function *callee)
{
    typename Function_state_usage_info_map::iterator it = m_func_state_usage_info_map.find(caller);
    if (it == m_func_state_usage_info_map.end()) {
        MDL_ASSERT(!"Function not registered for state usage info");
        return;
    }

    it->second->called_funcs.insert(callee);
}

// Updates the state usage of the exported functions of the code generator.
template<typename Function, typename Code_gen>
void State_usage_analysis<Function, Code_gen>::update_exported_functions_state_usage()
{
    // Note: This implementation requires Code_gen to have an m_exported_func_list member
    // with a state_usage field and func field.
    for (auto &exported_func : m_code_gen.m_exported_func_list) {
        // Create a visited set
        typename mi::mdl::ptr_hash_set<Function>::Type visited(m_arena.get_allocator());
        typename mi::mdl::vector<Function *>::Type worklist(m_arena.get_allocator());
        worklist.push_back(exported_func.func);

        while (!worklist.empty()) {
            Function *cur = worklist.back();
            worklist.pop_back();
            if (visited.find(cur) != visited.end()) {
                continue;
            }
            visited.insert(cur);
            Function_state_usage_info const *info = m_func_state_usage_info_map[cur];
            exported_func.state_usage |= info->state_usage;
            for (typename mi::mdl::Arena_ptr_hash_set<Function>::Type::const_iterator
                     it = info->called_funcs.begin(), end = info->called_funcs.end();
                 it != end;
                 ++it) {
                worklist.push_back(*it);
            }
        }
    }
}

}  // mdl
}  // mi

#endif // MDL_GENERATOR_CODE_STATE_USAGE_H
