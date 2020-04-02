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

#ifndef MDL_GENERATOR_DAG_H
#define MDL_GENERATOR_DAG_H 1

#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mdl/compiler/compilercore/compilercore_cc_conf.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>
#include <mdl/codegenerators/generator_code/generator_code.h>

namespace mi {
namespace mdl {

class MDL;
class IModule;
class IType_factory;

///
/// Implementation of the code generator for DAGs.
///
class Code_generator_dag : public Code_generator<ICode_generator_dag>
{
    typedef Code_generator<ICode_generator_dag> Base;
    friend class Allocator_builder;
public:
    /// Get the name of the target language.
    char const *get_target_language() const MDL_FINAL;

    /// Compile a module.
    /// \param      module  The module to compile.
    /// \returns            The generated code.
    IGenerated_code_dag *compile(IModule const *module) MDL_FINAL;

    /// Create a new MDL lambda function.
    ///
    /// \param context  the execution context for this lambda function.
    ///
    /// \returns  a new lambda function.
    ILambda_function *create_lambda_function(
        ILambda_function::Lambda_execution_context context) MDL_FINAL;

    /// Create a new MDL distribution function.
    ///
    /// \returns  a new distribution function.
    IDistribution_function *create_distribution_function() MDL_FINAL;

    /// Serialize a lambda function to the given serializer.
    ///
    /// \param lambda                the lambda function to serialize
    /// \param is                    the serializer data is written to
    void serialize_lambda(
        ILambda_function const *lambda,
        ISerializer            *is) const MDL_FINAL;

    /// Deserialize a lambda function from a given de-serializer.
    ///
    /// \param ds  the deserializer data is read from
    ///
    /// \return the lambda function
    ILambda_function *deserialize_lambda(IDeserializer *ds) MDL_FINAL;

private:
    /// Constructor.
    ///
    /// \param alloc        The allocator.
    /// \param mdl          The compiler interface.
    explicit Code_generator_dag(
        IAllocator *alloc,
        MDL        *mdl);

private:
    /// The builder for code DAGs.
    Allocator_builder m_builder;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_DAG_H
