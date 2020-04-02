/******************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_IR_CHECKER_H
#define MDL_GENERATOR_DAG_IR_CHECKER_H

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"

#include <mi/mdl/mdl_generated_dag.h>

#include "generator_dag_walker.h"

namespace mi {
namespace mdl {

#if !defined(NDEBUG) || defined(DEBUG)

/// Checks a DAG-IR.
class DAG_ir_checker : private IDAG_ir_visitor {
public:
    enum Error_code {
        EC_OK                    = 0,
        EC_PARAMETER_NOT_ALLOWED,
        EC_TEMPORARY_NOT_ALLOWED,
        EC_UNRESOLVED_MODULE_NAME,
        EC_UNDECLARED_NAME,
        EC_PARAM_COUNT,
        EC_PARAM_NAME_MISMATCH,
        EC_NULL_ARG,
        EC_ARG_TYPE_MISMATCH,
        EC_ARG_NON_UNIFORM,
        EC_WRONG_RET_TYPE,
        EC_RET_TYPE_NON_UNIFORM,
        EC_NULL_TEMP,
        EC_SEMANTIC_MISMATCH,
        EC_NULL_DAG,
        EC_NULL_TYPE,
        EC_NULL_VALUE,
        EC_VALUE_NOT_OWNED,
        EC_TYPE_NOT_OWNED,
        EC_DAG_NOT_OWNED,
        EC_TEMP_INDEX_USED_TWICE,
        EC_WRONG_TEMP,
        EC_TEMP_INDEX_TOO_HIGH,
    };

public:
    /// Check the given instance.
    ///
    /// \param inst   the instance to check
    bool check_instance(Generated_code_dag::Material_instance const *inst);

    /// Check a DAG node.
    size_t check_node(DAG_node const *node);

    /// Check a constant.
    size_t check_const(DAG_constant const *c);

    /// Check a call.
    size_t check_call(DAG_call const *call);

    /// Check a parameter.
    size_t check_parameter(DAG_parameter const *param);

    /// Check a temporary.
    size_t check_temp(DAG_temporary const *temp);

    /// Set the node owner for following check operations.
    DAG_node_factory_impl const *set_owner(DAG_node_factory_impl const *owner);

    /// Enable/disable temporaries.
    void enable_temporaries(bool flag) { m_allow_temporary = flag; }

    /// Enable/disable parameter.
    void enable_parameters(bool flag) { m_allow_parameters = flag; }

private:

    /// Report an error detected on given node.
    void error(DAG_node const *node, Error_code code);

private:
    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit(DAG_constant *cnst) MDL_FINAL;

    /// Post-visit a Temporary.
    ///
    /// \param tmp  the temporary that is visited
    void visit(DAG_temporary *tmp) MDL_FINAL;

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    void visit(DAG_call *call) MDL_FINAL;

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL;

    /// Post-visit a temporary initializer.
    ///
    /// \param index  the index of the temporary
    /// \param init   the initializer expression of this temporary
    void visit(int index, DAG_node *init) MDL_FINAL;

    /// Check that two symbols (potentially from different modules) are equal.
    bool equal_symbols(ISymbol const *t1, ISymbol const *t2) const;

    /// Check that two types (potentially from different modules) are equal.
    bool equal_types(IType const *t1, IType const *t2) const;

public:
    /// Constructor.
    ///
    /// \param alloc          the allocator
    /// \param call_resolver  an MDL call name resolver
    DAG_ir_checker(
        IAllocator          *alloc,
        ICall_name_resolver *call_resolver);

private:
    /// The allocator.
    IAllocator *m_alloc;

    typedef vector<DAG_node const *>::Type Node_vec;

    /// For checking temporaries: the list of seen temporaries.
    Node_vec m_temporaries;

    /// The call resolver.
    ICall_name_resolver *m_call_resolver;

    /// The node factory that owns all nodes.
    DAG_node_factory_impl const *m_node_fact;

    /// The type factory that owns all types.
    Type_factory const *m_tf;

    /// The value factory that owns all values.
    Value_factory const *m_vf;

    /// Number of detected errors.
    size_t m_errors;

    /// If true, temporaries are allowed.
    bool m_allow_temporary;

    /// If true, parameters are allowed.
    bool m_allow_parameters;

    /// If true, collect temporaries.
    bool m_collect_temporary;
};

#else

/// Checks a DAG-IR.
class DAG_ir_checker {
public:
    enum Error_code {
        EC_OK = 0,
        EC_PARAMETER_NOT_ALLOWED,
        EC_TEMPORARY_NOT_ALLOWED,
        EC_UNRESOLVED_MODULE_NAME,
        EC_UNDECLARED_NAME,
        EC_PARAM_COUNT,
        EC_PARAM_NAME_MISMATCH,
        EC_NULL_ARG,
        EC_ARG_TYPE_MISMATCH,
        EC_ARG_NON_UNIFORM,
        EC_WRONG_RET_TYPE,
        EC_RET_TYPE_NON_UNIFORM,
        EC_NULL_TEMP,
        EC_SEMANTIC_MISMATCH,
        EC_NULL_DAG,
        EC_NULL_TYPE,
        EC_NULL_VALUE,
        EC_VALUE_NOT_OWNED,
        EC_TYPE_NOT_OWNED,
        EC_DAG_NOT_OWNED,
        EC_TEMP_INDEX_USED_TWICE,
        EC_WRONG_TEMP,
        EC_TEMP_INDEX_TOO_HIGH,
    };

public:
    /// Check the given instance.
    ///
    /// \param inst   the instance to check
    bool check_instance(Generated_code_dag::Material_instance const *inst) {
        return true;
    }

    /// Check a DAG node.
    size_t check_node(DAG_node const *node) {
        return 0;
    }

    /// Check a constant.
    size_t check_const(DAG_constant const *c) {
        return 0;
    }

    /// Check a call.
    size_t check_call(DAG_call const *call) {
        return 0;
    }

    /// Check a parameter.
    size_t check_parameter(DAG_parameter const *param) {
        return 0;
    }

    /// Check a temporary.
    size_t check_temp(DAG_temporary const *temp) {
        return 0;
    }

    /// Set the node owner for following check operations.
    DAG_node_factory_impl const *set_owner(DAG_node_factory_impl const *) {
        return NULL;
    }

    /// Enable/disable temporaries.
    void enable_temporaries(bool) {}

    /// Enable/disable parameter.
    void enable_parameters(bool) {}

public:
    /// Constructor.
    DAG_ir_checker(
        IAllocator          *,
        ICall_name_resolver *)
    {
    }
};

#endif


} // mdl
} // mi

#endif
