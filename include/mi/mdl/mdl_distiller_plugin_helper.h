/******************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

/// \file mi/mdl/mdl_distiller_plugin_helper.h
/// \brief MDL distiller plugin helper classes

#ifndef MDL_DISTILLER_PLUGIN_HELPER_H
#define MDL_DISTILLER_PLUGIN_HELPER_H

#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_distiller_plugin_api.h>

#include <mi/mdl/mdl_distiller_node_types.h>

namespace mi {
namespace mdl {

template <int DIM>
struct Args_wrapper {
    mi::mdl::DAG_call::Call_argument args[DIM];

    void init( mi::mdl::IDistiller_plugin_api &e, size_t dim, Node_type const *node_type) {
        if ( node_type == 0)
            return;
        /* MDL_ASSERT( node_type->parameters.size() == DIM); */
        /* MDL_ASSERT( DIM >= dim); */
        /* MDL_ASSERT( dim >= node_type->min_parameters); */

        for ( size_t i = 0; i < DIM; ++i) {
            const Node_param& param( node_type->parameters[i]);
            args[i].param_name = param.param_name;
            if ( i >= dim) { // note: glossy BSDFs have at least three positional args
                args[i].arg = e.mk_default( param.param_type, param.param_default);
            }
        }
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e, mi::mdl::Node_types const *nt, int node_idx)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 0, node_type);
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 1, node_type);
        w.args[0].arg = arg0;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 2, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 3, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        w.args[2].arg = arg2;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 4, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        w.args[2].arg = arg2;
        w.args[3].arg = arg3;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 5, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        w.args[2].arg = arg2;
        w.args[3].arg = arg3;
        w.args[4].arg = arg4;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 6, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        w.args[2].arg = arg2;
        w.args[3].arg = arg3;
        w.args[4].arg = arg4;
        w.args[5].arg = arg5;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5,
        mi::mdl::DAG_node const *arg6)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 7, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        w.args[2].arg = arg2;
        w.args[3].arg = arg3;
        w.args[4].arg = arg4;
        w.args[5].arg = arg5;
        w.args[6].arg = arg6;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5,
        mi::mdl::DAG_node const *arg6,
        mi::mdl::DAG_node const *arg7)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 8, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        w.args[2].arg = arg2;
        w.args[3].arg = arg3;
        w.args[4].arg = arg4;
        w.args[5].arg = arg5;
        w.args[6].arg = arg6;
        w.args[7].arg = arg7;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5,
        mi::mdl::DAG_node const *arg6,
        mi::mdl::DAG_node const *arg7,
        mi::mdl::DAG_node const *arg8)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 9, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        w.args[2].arg = arg2;
        w.args[3].arg = arg3;
        w.args[4].arg = arg4;
        w.args[5].arg = arg5;
        w.args[6].arg = arg6;
        w.args[7].arg = arg7;
        w.args[8].arg = arg8;
        return w;
    }

    static Args_wrapper<DIM> mk_args(
        mi::mdl::IDistiller_plugin_api &e,
        mi::mdl::Node_types const *nt,
        int node_idx,
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5,
        mi::mdl::DAG_node const *arg6,
        mi::mdl::DAG_node const *arg7,
        mi::mdl::DAG_node const *arg8,
        mi::mdl::DAG_node const *arg9)
    {
        Args_wrapper<DIM> w;
        Node_type const *node_type =
            (node_idx == node_null) ? nullptr : nt->type_from_idx( node_idx);
        w.init( e, 10, node_type);
        w.args[0].arg = arg0;
        w.args[1].arg = arg1;
        w.args[2].arg = arg2;
        w.args[3].arg = arg3;
        w.args[4].arg = arg4;
        w.args[5].arg = arg5;
        w.args[6].arg = arg6;
        w.args[7].arg = arg7;
        w.args[8].arg = arg8;
        w.args[9].arg = arg9;
        return w;
    }

    static Args_wrapper<DIM> mk_named_args(
        char const *name0, mi::mdl::DAG_node const *arg0)
    {
        /* MDL_ASSERT( DIM == 1); */
        Args_wrapper<DIM> w;
        w.args[0].param_name = name0;
        w.args[0].arg        = arg0;
        return w;
    }

    static Args_wrapper<DIM> mk_named_args(
        char const *name0, mi::mdl::DAG_node const *arg0,
        char const *name1, mi::mdl::DAG_node const *arg1)
    {
        /* MDL_ASSERT( DIM == 2); */
        Args_wrapper<DIM> w;
        w.args[0].param_name = name0;
        w.args[0].arg        = arg0;
        w.args[1].param_name = name1;
        w.args[1].arg        = arg1;
        return w;
    }

    static Args_wrapper<DIM> mk_named_args(
        char const *name0, mi::mdl::DAG_node const *arg0,
        char const *name1, mi::mdl::DAG_node const *arg1,
        char const *name2, mi::mdl::DAG_node const *arg2)
    {
        /* MDL_ASSERT( DIM == 3); */
        Args_wrapper<DIM> w;
        w.args[0].param_name = name0;
        w.args[0].arg        = arg0;
        w.args[1].param_name = name1;
        w.args[1].arg        = arg1;
        w.args[2].param_name = name2;
        w.args[2].arg        = arg2;
        return w;
    }
};


template <size_t DIM>
class Nodes_wrapper {
private:
    Nodes_wrapper<DIM>() /*MDL_DELETED_FUNCTION*/ = delete;
};

template <>
class Nodes_wrapper<0> {
public:
    explicit Nodes_wrapper() {}

    mi::mdl::DAG_node const * const *data() const { return NULL; };
};

template <>
struct Nodes_wrapper<1> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0)
    {
        nodes[0] = arg0;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[1];
};

template <>
struct Nodes_wrapper<2> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[2];
};

template <>
struct Nodes_wrapper<3> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
        nodes[2] = arg2;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[3];
};

template <>
struct Nodes_wrapper<4> {
public:
    explicit Nodes_wrapper<4>(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
        nodes[2] = arg2;
        nodes[3] = arg3;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[4];
};

template <>
struct Nodes_wrapper<5> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
        nodes[2] = arg2;
        nodes[3] = arg3;
        nodes[4] = arg4;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[5];
};

template <>
struct Nodes_wrapper<6> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
        nodes[2] = arg2;
        nodes[3] = arg3;
        nodes[4] = arg4;
        nodes[5] = arg5;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[6];
};

template <>
struct Nodes_wrapper<7> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5,
        mi::mdl::DAG_node const *arg6)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
        nodes[2] = arg2;
        nodes[3] = arg3;
        nodes[4] = arg4;
        nodes[5] = arg5;
        nodes[6] = arg6;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[7];
};

template <>
struct Nodes_wrapper<8> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5,
        mi::mdl::DAG_node const *arg6,
        mi::mdl::DAG_node const *arg7)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
        nodes[2] = arg2;
        nodes[3] = arg3;
        nodes[4] = arg4;
        nodes[5] = arg5;
        nodes[6] = arg6;
        nodes[7] = arg7;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[8];
};

template <>
struct Nodes_wrapper<9> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5,
        mi::mdl::DAG_node const *arg6,
        mi::mdl::DAG_node const *arg7,
        mi::mdl::DAG_node const *arg8)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
        nodes[2] = arg2;
        nodes[3] = arg3;
        nodes[4] = arg4;
        nodes[5] = arg5;
        nodes[6] = arg6;
        nodes[7] = arg7;
        nodes[8] = arg8;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[9];
};

template <>
struct Nodes_wrapper<10> {
public:
    explicit Nodes_wrapper(
        mi::mdl::DAG_node const *arg0,
        mi::mdl::DAG_node const *arg1,
        mi::mdl::DAG_node const *arg2,
        mi::mdl::DAG_node const *arg3,
        mi::mdl::DAG_node const *arg4,
        mi::mdl::DAG_node const *arg5,
        mi::mdl::DAG_node const *arg6,
        mi::mdl::DAG_node const *arg7,
        mi::mdl::DAG_node const *arg8,
        mi::mdl::DAG_node const *arg9)
    {
        nodes[0] = arg0;
        nodes[1] = arg1;
        nodes[2] = arg2;
        nodes[3] = arg3;
        nodes[4] = arg4;
        nodes[5] = arg5;
        nodes[6] = arg6;
        nodes[7] = arg7;
        nodes[8] = arg8;
        nodes[9] = arg9;
    }

    mi::mdl::DAG_node const * const *data() const { return nodes; };

private:
    mi::mdl::DAG_node const *nodes[10];
};


} // mdl
} // mi

#endif // MDL_DISTILLER_PLUGIN_HELPER_H
