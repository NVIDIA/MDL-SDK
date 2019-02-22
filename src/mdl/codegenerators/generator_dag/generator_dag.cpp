/******************************************************************************
 * Copyright (c) 2012-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/handle.h>

#include <mdl/compiler/compilercore/compilercore_checker.h>

#include "generator_dag.h"
#include "generator_dag_generated_dag.h"
#include "generator_dag_tools.h"
#include "generator_dag_lambda_function.h"
#include "generator_dag_tools.h"

namespace mi {
namespace mdl {

/// A checker for DAGs.
class DAG_code_checker : protected Code_checker {
    typedef Code_checker Base;

public:
    /// Check a DAG.
    ///
    /// \param compiler  the MDL compiler (that owns the module)
    /// \param dag       the DAG to check
    /// \param name      the name of the DAG
    /// \param verbose   if true, write a verbose output the stderr
    ///
    /// \return true on success
    static bool check(
        IMDL               *compiler,
        Generated_code_dag *dag,
        char const         *name,
        bool               verbose);

private:
    /// Constructor.
    ///
    /// \param dag      the dag to check
    /// \param verbose  if true, write a verbose output to stderr
    /// \param printer  the printer for writing error messages, takes ownership
    explicit DAG_code_checker(
        Generated_code_dag const *dag,
        bool                     verbose,
        IPrinter                 *printer);

    /// Check all materials
    void check_materials(Generated_code_dag const *dag);

    /// Check a DAG IR node.
    void check_dag_node(DAG_node const *node);

private:
    /// The current dag that is checked.
    Generated_code_dag const * const m_dag;

    /// The current material index.
    int m_current_mat_index;

};

// ------------------------ DAG code checker ------------------------

// Constructor.
DAG_code_checker::DAG_code_checker(
    Generated_code_dag const *dag,
    bool                     verbose,
    IPrinter                 *printer)
: Base(verbose, printer)
, m_dag(dag)
, m_current_mat_index(-1)
{
}

// Check a DAG.
bool DAG_code_checker::check(
    IMDL               *compiler,
    Generated_code_dag *dag,
    char const         *name,
    bool               verbose)
{
    mi::base::Handle<IOutput_stream> os_stderr(compiler->create_std_stream(IMDL::OS_STDERR));
    IPrinter *printer = compiler->create_printer(os_stderr.get());
    printer->enable_color(true);

    DAG_code_checker checker(dag, verbose, printer);

    if (verbose) {
        printer->print("Checking DAG ");
        printer->print(name);
        printer->print("\n");
    }

    // check all values
    checker.check_factory(dag->get_value_factory());

    checker.check_materials(dag);

    if (checker.get_error_count() != 0) {
        if (verbose) {
            printer->print("Checking DAG ");
            printer->print(name);
            printer->print(" FAILED!\n\n");
        }
        abort();
        return false;
    }

    if (verbose) {
        printer->print("Checking DAG ");
        printer->print(name);
        printer->print(" OK!\n\n");
    }
    return true;
}

// Check all materials
void DAG_code_checker::check_materials(
    Generated_code_dag const *dag)
{
    for (int i = 0, n = dag->get_material_count(); i < n; ++i) {
        DAG_node const *expr = dag->get_material_value(i);

        m_current_mat_index = i;
        check_dag_node(expr);
    }
}

// Check a DAG IR node.
void DAG_code_checker::check_dag_node(DAG_node const *node)
{
    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c = cast<DAG_constant>(node);
            IValue const *v = c->get_value();
            check_value(v);
        }
        break;
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *temp = cast<DAG_temporary>(node);
            int tmp_index = temp->get_index();

            DAG_node const *init = m_dag->get_material_temporary(m_current_mat_index, tmp_index);

            check_dag_node(init);
        }
        break;
    case DAG_node::EK_CALL:
        {
            DAG_call const *c = cast<DAG_call>(node);

            if (m_verbose) {
                m_printer->print("Checking call ");
                m_printer->print(c->get_name());
                m_printer->print("\n");
            }

            IType const *ret_type = c->get_type();
            check_type(ret_type);

            for (int i = 0, n = c->get_argument_count(); i < n; ++i) {
                DAG_node const *arg = c->get_argument(i);

                check_dag_node(arg);
            }
        }
        break;
    case DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *p = cast<DAG_parameter>(node);
            int param_index = p->get_index();

            IType const *type = m_dag->get_material_parameter_type(
                m_current_mat_index, param_index);
            check_type(type);
        }
        break;
    default:
        report("Wrong DAG Expression kind");
        break;
    }
}

Code_generator_dag::Code_generator_dag(
    IAllocator *alloc,
    MDL        *mdl)
: Base(alloc,mdl)
, m_builder(alloc)
{
    m_options.add_option(
        MDL_CG_DAG_OPTION_DUMP_MATERIAL_DAG,
        "false",
        "Dump the material expression DAGs for every compiled module");
    m_options.add_option(
        MDL_CG_DAG_OPTION_NO_LOCAL_FUNC_CALLS,
        "false",
        "Forbid calls to local functions inside MDL materials");
    m_options.add_option(
        MDL_CG_DAG_OPTION_INCLUDE_LOCAL_ENTITIES,
        "false",
        "Include local entities that are used inside material bodies");
    m_options.add_option(
        MDL_CG_DAG_OPTION_CONTEXT_NAME,
        "renderer",
        "The name of the context for error messages");
    m_options.add_option(
        MDL_CG_DAG_OPTION_MARK_DAG_GENERATED,
        "true",
        "Mark all DAG backend generated entities");
}

char const *Code_generator_dag::get_target_language() const
{
    return "dag";
}


IGenerated_code_dag *Code_generator_dag::compile(IModule const *module)
{
    Generated_code_dag::Compile_options options = 0;
    if (m_options.get_bool_option(MDL_CG_DAG_OPTION_NO_LOCAL_FUNC_CALLS))
        options |= Generated_code_dag::FORBID_LOCAL_FUNC_CALLS;
    else if (m_options.get_bool_option(MDL_CG_DAG_OPTION_INCLUDE_LOCAL_ENTITIES))
        options |= Generated_code_dag::INCLUDE_LOCAL_ENTITIES;

    if (m_options.get_bool_option(MDL_CG_DAG_OPTION_MARK_DAG_GENERATED))
        options |= Generated_code_dag::MARK_GENERATED_ENTITIES;


    Generated_code_dag *result = m_builder.create<Generated_code_dag>(
        m_builder.get_allocator(),
        m_compiler.get(),
        module,
        m_options.get_string_option(MDL_CG_OPTION_INTERNAL_SPACE),
        options,
        m_options.get_string_option(MDL_CG_DAG_OPTION_CONTEXT_NAME));

    result->compile(module);

    if (m_options.get_bool_option(MDL_CG_DAG_OPTION_DUMP_MATERIAL_DAG)) {
        for (int i = 0, n = result->get_material_count(); i < n; ++i) {
            result->dump_material_dag(i, NULL);
        }
    }
    return result;
}

// Create a new MDL lambda function.
ILambda_function *Code_generator_dag::create_lambda_function(
    ILambda_function::Lambda_execution_context context)
{
    return m_builder.create<Lambda_function>(
        m_builder.get_allocator(),
        m_compiler.get(),
        context);
}

// Create a new MDL distribution function.
IDistribution_function *Code_generator_dag::create_distribution_function()
{
    return m_builder.create<Distribution_function>(
        m_builder.get_allocator(),
        m_compiler.get());
}

// Serialize a lambda function to the given serializer.
void Code_generator_dag::serialize_lambda(
    ILambda_function const *lambda,
    ISerializer            *is) const
{
    Lambda_function const *func = impl_cast<Lambda_function>(lambda);
    func->serialize(is);
}

// Deserialize a lambda function from a given de-serializer.
ILambda_function *Code_generator_dag::deserialize_lambda(IDeserializer *ds)
{
    return Lambda_function::deserialize(get_allocator(), m_compiler.get(), ds);
}

} // mdl
} // mi

