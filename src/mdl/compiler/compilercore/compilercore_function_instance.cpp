/******************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "compilercore_assert.h"
#include "compilercore_function_instance.h"

namespace mi {
namespace mdl {

// Constructor from a function definition.
Function_instance::Function_instance(
    IDefinition const     *def,
    Array_instances const &arg_instances,
    bool                   return_derivs)
: m_key(def)
, m_array_instances(arg_instances)
, m_return_derivs(return_derivs)
, m_kind(KI_DEFINITION)
{
    MDL_ASSERT(def->get_kind() == IDefinition::DK_FUNCTION);
}

// Constructor from a function definition.
Function_instance::Function_instance(
    IAllocator        *alloc,
    IDefinition const *def,
    bool               return_derivs)
: m_key(def)
, m_array_instances(alloc)
, m_return_derivs(
    return_derivs ||
    (def != NULL ? impl_cast<Definition>(def)->has_flag(Definition::DEF_IS_DERIVABLE) : false))
, m_kind(KI_DEFINITION)
{
    // def might be NULL for intrinsic functions
    MDL_ASSERT(
        def == NULL ||
        def->get_kind() == IDefinition::DK_FUNCTION ||
        def->get_kind() == IDefinition::DK_CONSTRUCTOR);
}

// Constructor from a lambda function.
Function_instance::Function_instance(
    IAllocator             *alloc,
    ILambda_function const *lambda)
: m_key(lambda)
, m_array_instances(alloc)
, m_return_derivs(false)
, m_kind(KI_LAMBDA)
{
}

// Constructor from a common prototype code.
Function_instance::Function_instance(
    IAllocator        *alloc,
    size_t           code)
: m_key(Key_t(code))
, m_array_instances(alloc)
, m_return_derivs(false)
, m_kind(KI_PROTOTYPE_CODE)
{
    // zero is reserved
    MDL_ASSERT(code != 0);
}

// Map types due to function instancing.
int Function_instance::instantiate_type_size(
    IType const *type) const
{
    IType_array const *a_type = as<IType_array>(type);
    if (a_type != NULL && !a_type->is_immediate_sized()) {
        IType_array_size const *deferred_size = a_type->get_deferred_size();
        for (size_t i = 0, n = m_array_instances.size(); i < n; ++i) {
            Array_instance const &ai = m_array_instances[i];

            if (ai.get_deferred_size() == deferred_size)
                return ai.get_immediate_size();
        }
    }
    return -1;
}

}  // mdl
}  // mi
