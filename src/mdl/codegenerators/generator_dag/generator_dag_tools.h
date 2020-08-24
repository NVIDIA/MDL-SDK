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

#ifndef MDL_GENERATOR_DAG_TOOLS_H
#define MDL_GENERATOR_DAG_TOOLS_H

#include "mdl/compiler/compilercore/compilercore_tools.h"
#include "generator_dag_generated_dag.h"
#include "generator_dag_lambda_function.h"

namespace mi {
namespace mdl {

template<typename T>
inline T *cast(DAG_node *node)
{
    MDL_ASSERT(node == NULL || is<T>(node));
    return static_cast<T *>(node);
}

template<typename T>
inline T const *cast(DAG_node const *node)
{
    MDL_ASSERT(node == NULL || is<T>(node));
    return static_cast<T const *>(node);
}

// An impl_cast allows casting from an Interface pointer to its (only) Implementation class.
template<>
inline Generated_code_dag const *impl_cast(IGenerated_code_dag const *t) {
    return static_cast<Generated_code_dag const *>(t);
}

// An impl_cast allows casting from an Interface pointer to its (only) Implementation class.
template<>
inline Generated_code_dag::Material_instance const *impl_cast(
    IGenerated_code_dag::IMaterial_instance const *t)
{
    return static_cast<Generated_code_dag::Material_instance const *>(t);
}

// An impl_cast allows casting from an Interface pointer to its (only) Implementation class.
template<>
inline Lambda_function const *impl_cast(ILambda_function const *t) {
    return static_cast<Lambda_function const *>(t);
}

template<>
inline Lambda_function *impl_cast(ILambda_function *t) {
    return static_cast<Lambda_function *>(t);
}

// An impl_cast allows casting from an Interface pointer to its (only) Implementation class.
template<>
inline Distribution_function const *impl_cast(IDistribution_function const *t) {
    return static_cast<Distribution_function const *>(t);
}

template<>
inline Distribution_function *impl_cast(IDistribution_function *t) {
    return static_cast<Distribution_function *>(t);
}

/// Get the DAG signature of the array constructor.
extern inline char const *get_array_constructor_signature() { return "T[](...)"; }

/// Get the DAG signature of the ternary operator.
extern inline char const *get_ternary_operator_signature() { return "operator?(bool,<0>,<0>)"; }

/// Get the DAG signature of the array constructor without the suffix.
extern inline char const *get_array_constructor_signature_without_suffix() { return "T[]"; }

/// Get the DAG signature of the ternary operator without the suffix.
extern inline char const *get_ternary_operator_signature_without_suffix() { return "operator?"; }

} // mdl
} // mi

#endif
