/******************************************************************************
 * Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>

#include <mdl/compiler/compilercore/compilercore_allocator.h>

#include "generator_jit_sl_utils.h"

namespace mi {
namespace mdl {
namespace sl {

// Constructor.
DebugTypeHelper::DebugTypeHelper(
    IAllocator *alloc)
: m_alloc(alloc)
, m_arena(alloc)
, m_api_type_dbg_info(0, API_type_map::hasher(), API_type_map::key_equal(), alloc)
, m_struct_dbg_info(0, Struct_info_map::hasher(), Struct_info_map::key_equal(), alloc)
{
}

// Fill type debug info from a module.
void DebugTypeHelper::enter_type_debug_info(
    llvm::Module const &llvm_module)
{
    llvm::DebugInfoFinder finder;

    finder.processModule(llvm_module);

    for (llvm::DIType *dt : finder.types()) {
        if (dt->getTag() == llvm::dwarf::DW_TAG_structure_type) {
            llvm::DICompositeType *st = llvm::cast<llvm::DICompositeType>(dt);
            llvm::StringRef       name = st->getName();

            string s_name(name.begin(), name.end(), m_alloc);

            m_struct_dbg_info[s_name] = st;
        }
    }
}

// Add API type info.
void DebugTypeHelper::add_api_type_info(
    char const              *llvm_type_name,
    char const              *api_type_name,
    Array_ref<char const *> fields)
{
    size_t n = fields.size();

    API_type_info *entry = (API_type_info *)m_arena.allocate(
        sizeof(API_type_info) + n * sizeof(entry->fields[0]) - sizeof(entry->fields[0]));

    entry->llvm_name = Arena_strdup(m_arena, llvm_type_name);
    entry->api_name  = Arena_strdup(m_arena, api_type_name);
    entry->n_fields  = n;

    for (size_t i = 0; i < n; ++i) {
        entry->fields[i] = Arena_strdup(m_arena, fields[i]);
    }

    MDL_ASSERT(
        m_api_type_dbg_info.find(llvm::StringRef(llvm_type_name)) == m_api_type_dbg_info.end());
    m_api_type_dbg_info[llvm::StringRef(llvm_type_name)] = entry;
}

// Find the debug type info for a given type name.
llvm::DICompositeType *DebugTypeHelper::find_composite_type_info(
    llvm::StringRef const &name) const
{
    auto it = m_struct_dbg_info.find(string(name.begin(), name.end(), m_alloc));
    if (it != m_struct_dbg_info.end()) {
        return it->second;
    }
    return nullptr;
}

// Find the subelement name if exists.
llvm::DIType *DebugTypeHelper::find_subelement_type_info(
    llvm::DICompositeType *di_type,
    unsigned              field_index)
{
    llvm::DINodeArray elements = di_type->getElements();

    if (field_index < elements.size()) {
        return llvm::cast<llvm::DIType>(elements[field_index]);
    }
    return nullptr;
}

// Find the API debug type info for a given (composite) type name.
DebugTypeHelper::API_type_info const *DebugTypeHelper::find_api_type_info(
    llvm::StringRef const &name) const
{
    auto it = m_api_type_dbg_info.find(name);
    if (it != m_api_type_dbg_info.end()) {
        return it->second;
    }
    return nullptr;
}

}  // sl
}  // mdl
}  // mi
