/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_JIT_SL_UTILS_H
#define MDL_GENERATOR_JIT_SL_UTILS_H 1

#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/compiler/compilercore/compilercore_array_ref.h>

#include <llvm/ADT/StringRef.h>

namespace llvm {
class DICompositeType;
class DIType;
class Module;
}

namespace mi {
namespace mdl {
namespace sl {

/// A functor for hashing llvm::StringRef.
struct StringRef_hash {
    size_t operator()(llvm::StringRef const &s) const {
        return llvm::hash_value(s);
    }
};

/// A functor for comparing llvm::StringRef.
struct StringRef_equal {
    bool operator()(llvm::StringRef const &s, llvm::StringRef const &t) const {
        return s.equals(t);
    }
};

/// Helper class to retrieve struct type field names from debug info (if exists).
class DebugTypeHelper {
public:
    /// API type entry.
    struct API_type_info {
        char const *llvm_name; ///< Name of the type in the LLVM-IR.
        char const *api_name;  ///< Name of the type in the API.
        size_t     n_fields;   ///< Number of fields.
        char const *fields[1]; ///< Name of the fields.
    };

public:
    /// Constructor.
    ///
    /// \param alloc  the allocator to be used
    DebugTypeHelper(
        IAllocator *alloc);

    /// Fill type debug info from a module.
    void enter_type_debug_info(
        llvm::Module const &module);

    /// Add API type info.
    ///
    /// \param llvm_type_name  name of the API type in the LLVM-IR
    /// \param sl_type_name    name of the API type in the target language
    /// \param fields          field names
    void add_api_type_info(
        char const              *llvm_type_name,
        char const              *sl_type_name,
        Array_ref<char const *> fields);

    /// Find the debug type info for a given (composite) type name.
    ///
    /// \param name  an LLVM type name
    llvm::DICompositeType *find_composite_type_info(
        llvm::StringRef const &name) const;

    /// Find the subelement name if exists.
    llvm::DIType *find_subelement_type_info(
        llvm::DICompositeType *type_info,
        unsigned              field_index);

    /// Find the API debug type info for a given (composite) type name.
    ///
    /// \param name  an LLVM type name
    API_type_info const *find_api_type_info(
        llvm::StringRef const &name) const;

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// Used memory arena.
    Memory_arena m_arena;

    typedef hash_map<
        llvm::StringRef,
        API_type_info *,
        StringRef_hash,
        StringRef_equal>::Type API_type_map;

    API_type_map m_api_type_dbg_info;

    typedef hash_map<
        string,
        llvm::DICompositeType *,
        string_hash<string> >::Type Struct_info_map;

    /// Debug info regarding struct types.
    Struct_info_map  m_struct_dbg_info;
};

}  // sl
}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_SL_UTILS_H
