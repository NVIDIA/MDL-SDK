/******************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_FUNCTION_INSTANCE_H
#define MDL_COMPILERCORE_FUNCTION_INSTANCE_H 1

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_definitions.h>

#include "compilercore_allocator.h"
#include "compilercore_tools.h"
#include "compilercore_array_ref.h"

namespace mi {
namespace mdl {

class ILambda_function;

/// The storage modifier for a variable or parameter.
enum Storage_modifier {
    SM_NORMAL,      ///< Stored in normal memory
    SM_PARAMETER,   ///< Stored in the argument block
    SM_RODATA       ///< Stored in the read-only data segment
};

///
/// Helper class to map deferred array types to immediate array types.
///
class Array_instance {
public:
    /// Constructor.
    ///
    /// \param deferred_size   the deferred size symbol
    /// \param immediate_size  the target immediate size
    Array_instance(
        mi::mdl::IType_array_size const *deferred_size,
        size_t const                    immediate_size)
    : m_deferred_size(deferred_size)
    , m_immediate_size(immediate_size)
    {
    }

    /// Get the deferred sized array type.
    mi::mdl::IType_array_size const *get_deferred_size() const { return m_deferred_size; }

    /// Get the immediate sized array type.
    int get_immediate_size() const { return m_immediate_size; }

private:
    mi::mdl::IType_array_size const *m_deferred_size;
    int                             m_immediate_size;
};

///
/// Helper class to express a function instance
///
class Function_instance
{
public:
    typedef void const                              *Key_t;
    typedef mi::mdl::vector<Array_instance>::Type   Array_instances;
    typedef mi::mdl::vector<Storage_modifier>::Type Parameter_storage_modifiers;

    /// The kind object which was used to construct the function instance.
    enum Kind {
        KI_DEFINITION,          ///< This function instance is based on a function definition.
        KI_LAMBDA,              ///< This function instance is based on a lambda function.
        KI_PROTOTYPE_CODE       ///< This function instance is based on a prototype code.
    };

    /// Constructor from a function definition and a set of array instance.
    ///
    /// \param def                 the function definition
    /// \param arr_instances       type instance for this function instance
    /// \param param_mods          the parameter storage modifiers for this function instance
    /// \param return_derivs       if true, derivatives will be generated for the return value
    /// \param has_storage_spaces  if true, target supports storage spaces
    ///
    /// \note only this constructor creates a real (template) instance
    explicit Function_instance(
        IDefinition const                 *def,
        Array_instances const             &arr_instances,
        Parameter_storage_modifiers const &param_mods,
        bool                              return_derivs,
        bool                              has_storage_spaces);

    /// Constructor from a function definition.
    ///
    /// \param alloc               the allocator
    /// \param def                 the function definition
    /// \param return_derivs       if true, derivatives will be generated for the return value
    /// \param has_storage_spaces  if true, target supports storage spaces
    explicit Function_instance(
        IAllocator        *alloc,
        IDefinition const *def,
        bool               return_derivs,
        bool               has_storage_spaces);

    /// Constructor from a lambda function.
    ///
    /// \param alloc               the allocator
    /// \param lambda              the lambda function
    /// \param has_storage_spaces  if true, target supports storage spaces
    explicit Function_instance(
        IAllocator             *alloc,
        ILambda_function const *lambda,
        bool                    has_storage_spaces);

    /// Constructor from a common prototype code.
    ///
    /// \param alloc               the allocator
    /// \param code                the common prototype code
    /// \param has_storage_spaces  if true, target supports storage spaces
    explicit Function_instance(
        IAllocator *alloc,
        size_t     code,
        bool       has_storage_spaces);

    /// Get the key.
    Key_t get_key() const { return m_key; }

    /// Get the function definition if any.
    IDefinition const *get_def() const {
        if (m_kind != KI_DEFINITION)
            return NULL;
        return static_cast<IDefinition const *>(m_key);
    }

    /// Get the lambda function if any.
    ILambda_function const *get_lambda() const {
        if (m_kind != KI_LAMBDA)
            return NULL;
        return static_cast<ILambda_function const *>(m_key);
    }

    /// Get the common prototype code if any.
     size_t get_common_prototype_code() const {
        if (m_kind != KI_PROTOTYPE_CODE)
            return 0u;
        return size_t(m_key);
    }

    /// Get the array instances.
    Array_instances const &get_array_instances() const { return m_array_instances; }

    /// Get the storage modifiers of the parameters.
    Parameter_storage_modifiers const &get_parameter_storage_modifiers() const {
        return m_parameter_storage_mods;
    }

    /// Get the storage modifiers of the parameters.
    Parameter_storage_modifiers &get_parameter_storage_modifiers() {
        return m_parameter_storage_mods;
    }

    /// Get whether derivatives for the return value will be generated.
    bool get_return_derivs() const { return m_return_derivs; }

    /// Map types due to function instancing.
    ///
    /// \param type  an MDL type
    ///
    /// \return if type is a deferred size array type and instancing is enabled, the immediate
    /// size instance of this type, else -1
    MDL_CHECK_RESULT int instantiate_type_size(IType const *type) const;

    /// Get the storage modifier of the i'th parameter.
    Storage_modifier get_parameter_storage_modifier(size_t i) const {
        if (!m_has_storage_spaces) {
            return SM_NORMAL;
        }
        if (m_kind == KI_LAMBDA) {
            // all arguments of a lambda function are in parameter space
            return SM_PARAMETER;
        }
        if (i >= m_parameter_storage_mods.size()) {
            MDL_ASSERT(m_parameter_storage_mods.empty() && "Invalid parameter index");
            return SM_NORMAL;
        }
        return m_parameter_storage_mods[i];
    }

    /// Check if this is an instantiated function (in the sense of a instantiated template).
    bool is_instantiated() const { return !m_array_instances.empty(); }

    /// Equal functor.
    template<bool ignore_array_instance = false>
    class Equal {
    public:
        bool operator()(Function_instance const &a, Function_instance const &b) const {
            if (a.get_key() != b.get_key())
                return false;

            Array_instances const &a_ais = a.get_array_instances();
            Array_instances const &b_ais = b.get_array_instances();

            size_t n = a_ais.size();
            if (n != b_ais.size())
                return false;

            if (a.m_return_derivs != b.m_return_derivs)
                return false;

            if (!ignore_array_instance) {
                for (size_t i = 0; i < n; ++i) {
                    Array_instance const &ai = a_ais[i];
                    Array_instance const &bi = b_ais[i];

                    if (ai.get_deferred_size()  != bi.get_deferred_size() ||
                        // Note: the immediate values might be owned by different modules, so
                        // check there sizes. There element types are assumed to by equal,
                        // otherwise the deferred types should be different
                        ai.get_immediate_size() != bi.get_immediate_size())
                    {
                        return false;
                    }
                }

                Parameter_storage_modifiers const &a_mods = a.get_parameter_storage_modifiers();
                Parameter_storage_modifiers const &b_mods = b.get_parameter_storage_modifiers();

                size_t mods_n = a_mods.size();
                if (mods_n != b_mods.size())
                    return false;

                for (size_t i = 0; i < mods_n; ++i) {
                    if (a_mods[i] != b_mods[i]) {
                        return false;
                    }
                }
            }

            return true;
        }
    };

    /// Hash function.
    template<bool ignore_array_instance = false>
    class Hash {
    public:
        size_t operator()(Function_instance const &a) const {
            Hash_ptr<void const>             key_hasher;
            Hash_ptr<IType_array_size const> type_size_hasher;

            size_t res = key_hasher(a.get_key());

            if (!ignore_array_instance) {
                Array_instances const &ais = a.get_array_instances();
                for (size_t i = 0, n = ais.size(); i < n; ++i) {
                    Array_instance const &ai = ais[i];
                    res = res * 33 + 3 * type_size_hasher(ai.get_deferred_size())
                        + 7u * ai.get_immediate_size();
                }

                Parameter_storage_modifiers const &mods = a.get_parameter_storage_modifiers();
                for (size_t i = 0, n = mods.size(); i < n; ++i) {
                    res = res * 33 + 19 * size_t(mods[i]);
                }
            }

            if (a.m_return_derivs) {
                res = (res * 33) + 196613;
            }

            return res;
        }
    };

    bool operator==(Function_instance const &o) const {
        Equal<> eq;
        return eq(*this, o);
    }

    bool operator!=(Function_instance const &o) const {
        return !operator==(o);
    }

private:
    /// This function's key.
    Key_t m_key;

    /// The (array) type instances for the return type and the argument types.
    Array_instances m_array_instances;

    /// The storage modifiers of the parameters.
    Parameter_storage_modifiers m_parameter_storage_mods;

    /// If true, derivatives will be generated for the return value.
    bool m_return_derivs;

    /// If true, storage spaces are supported by the target.
    bool m_has_storage_spaces;

    /// The kind of this function instance.
    Kind m_kind;

};

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_FUNCTION_INSTANCE_H
