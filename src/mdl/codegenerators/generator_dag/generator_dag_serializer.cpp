/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_assert.h"

#include "generator_dag_walker.h"
#include "generator_dag_generated_dag.h"
#include "generator_dag_ir.h"
#include "generator_dag_tools.h"
#include "generator_dag_serializer.h"

namespace mi {
namespace mdl {

namespace {

/// Base class for DAG IR node serialization tools; creates a walker over a root set.
class Abstract_dag_ir_walker : public IDAG_ir_visitor {
    typedef IDAG_ir_visitor Base;
public:
    /// Constructor.
    ///
    /// \param alloc  an allocator for temporary memory
    explicit Abstract_dag_ir_walker(
        IAllocator *alloc)
    : Base()
    , m_marker(
        0, Visited_node_set::hasher(), Visited_node_set::key_equal(), alloc)
    {
    }

    /// Walk a DAG IR node.
    ///
    /// \param root  the root DAG IR node
    void walk_node(DAG_node const *root)
    {
        DAG_node *node = const_cast<DAG_node *>(root);

        if (m_marker.find(node) != m_marker.end())
            return;
        m_marker.insert(node);

        switch (node->get_kind()) {
        case DAG_node::EK_CONSTANT:
            {
                DAG_constant *c = cast<DAG_constant>(node);
                visit(c);
                return;
            }
        case DAG_node::EK_TEMPORARY:
            {
                DAG_temporary *t    = cast<DAG_temporary>(node);
                DAG_node      *expr = const_cast<DAG_node *>(t->get_expr());

                walk_node(expr);
                visit(t);
                return;
            }
        case DAG_node::EK_CALL:
            {
                DAG_call *c = cast<DAG_call>(node);

                for (int i = 0, n = c->get_argument_count(); i < n; ++i) {
                    DAG_node *arg = const_cast<DAG_node *>(c->get_argument(i));

                    walk_node(arg);
                }
                visit(c);
                return;
            }
        case DAG_node::EK_PARAMETER:
            {
                DAG_parameter *p = cast<DAG_parameter>(node);
                visit(p);
                return;
            }
        }
        MDL_ASSERT(!"Unsupported DAG node kind");
    }

    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    virtual void visit(DAG_constant *cnst) = 0;

    /// Post-visit a Temporary.
    ///
    /// \param tmp  the temporary that is visited
    virtual void visit(DAG_temporary *tmp) = 0;

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    virtual void visit(DAG_call *call) = 0;

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    virtual void visit(DAG_parameter *param) = 0;

    /// Post-visit a temporary initializer.
    ///
    /// \param index  the index of the temporary
    /// \param init   the initializer expression of this temporary
    void visit(int index, DAG_node *init) MDL_FINAL {
        // UNUSED in this walker style
        MDL_ASSERT(!"should not be called");
    }

private:
    typedef ptr_hash_set<DAG_node>::Type Visited_node_set;

    /// The marker set.
    Visited_node_set m_marker;
};

/// A walker for DAG IR nodes serialization.
/// Counts the number of reachable nodes.
class DAG_ir_counter : public Abstract_dag_ir_walker {
    typedef Abstract_dag_ir_walker Base;
public:
    /// Constructor.
    ///
    /// \param alloc  an allocator for temporary memory
    explicit DAG_ir_counter(IAllocator *alloc)
    : Base(alloc)
    , m_count(0)
    {
    }

    /// Get the number of reachable DAG IR nodes.
    size_t get_count() const { return m_count; }

    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit(DAG_constant *cnst) MDL_FINAL {
        ++m_count;
    }

    /// Post-visit a Temporary.
    ///
    /// \param tmp  the temporary that is visited
    void visit(DAG_temporary *tmp) MDL_FINAL {
        ++m_count;
    }

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    void visit(DAG_call *call) MDL_FINAL {
        ++m_count;
    }

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL {
        ++m_count;
    }

private:
    /// Number of reachable IR nodes.
    size_t m_count;
};

/// A walker for DAG IR nodes serialization.
/// Writes all DAG IR nodes reachable from given roots into the serializer
/// and assigns tags to them.
class DAG_ir_serializer : public Abstract_dag_ir_walker {
    typedef Abstract_dag_ir_walker Base;
public:
    /// Constructor.
    ///
    /// \param alloc  an allocator for temporary memory
    explicit DAG_ir_serializer(
        DAG_serializer &serializer)
    : Base(serializer.get_allocator())
    , m_serializer(serializer)
    {
    }

    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit(DAG_constant *cnst) MDL_FINAL {
        // register this node, so it will receive a tag
        (void)m_serializer.register_ir_node(cnst);
        encode(DAG_constant::s_kind);

        IValue const *value = cnst->get_value();
        encode(value);
    }

    /// Post-visit a Temporary.
    ///
    /// \param tmp  the temporary that is visited
    void visit(DAG_temporary *tmp) MDL_FINAL {
        // register this node, so it will receive a tag
        (void)m_serializer.register_ir_node(tmp);
        encode(DAG_temporary::s_kind);

        unsigned idx = tmp->get_index();
        encode(idx);
        DAG_node const *expr = tmp->get_expr();
        encode(expr);
    }

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    void visit(DAG_call *call) MDL_FINAL {
        // register this node, so it will receive a tag
        (void)m_serializer.register_ir_node(call);
        encode(DAG_call::s_kind);

        IType const *type = call->get_type();
        encode(type);

        char const *name = call->get_name();
        encode(name);

        IDefinition::Semantics sema = call->get_semantic();
        encode(sema);

        unsigned n_args = call->get_argument_count();
        encode(n_args);
        for (unsigned i = 0; i < n_args; ++i) {
            char const *param_name = call->get_parameter_name(i);
            MDL_ASSERT(param_name != NULL);
            encode(param_name);

            DAG_node const *arg = call->get_argument(i);
            encode(arg);
        }
    }

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL {
        // register this node, so it will receive a tag
        (void)m_serializer.register_ir_node(param);
        encode(DAG_parameter::s_kind);

        IType const *type = param->get_type();
        encode(type);

        unsigned idx = param->get_index();
        encode(idx);
    }

private:
    /// Encode (and write) a type.
    void encode(IType const *type) {
        Tag_t t = type == NULL ? Tag_t(0) : m_serializer.get_type_tag(type);

        m_serializer.write_encoded_tag(t);
    }

    /// Encode (and write) a value.
    void encode(IValue const *value) {
        Tag_t t = m_serializer.get_value_tag(value);

        m_serializer.write_encoded_tag(t);
    }

    /// Encode (and write) an unsigned integer.
    void encode(unsigned n) {
        m_serializer.write_encoded(n);
    }

    /// Encode (and write) a C-string.
    void encode(char const *s) {
        m_serializer.write_encoded(s);
    }

    /// Encode (and write) a semantic code.
    void encode(IDefinition::Semantics sema) {
        m_serializer.write_encoded(sema);
    }

    /// Encode (and write) an DAG IR node.
    void encode(DAG_node const *node) {
        Tag_t t = m_serializer.get_ir_node_tag(node);

        m_serializer.write_encoded_tag(t);
    }

    /// Encode (and write) an DAG IR node kind.
    void encode(DAG_node::Kind kind) {
        m_serializer.write_encoded(kind);
    }

private:
    /// The DAG serializer.
    DAG_serializer &m_serializer;
};

}  // anonymous

// ----------------------- DAG serializer -----------------------

// Constructor.
DAG_serializer::DAG_serializer(
    IAllocator            *alloc,
    ISerializer           *serializer,
    MDL_binary_serializer *bin_serializer)
: Factory_serializer(alloc, serializer, bin_serializer)
, m_ir_nodes(alloc)
{
}

// Serialize a vector of DAG_vector.
void DAG_serializer::serialize(Dag_vector const &vec)
{
    size_t l = vec.size();

    write_encoded_tag(l);
    for (size_t i = 0; i < l; ++i) {
        DAG_node const *node = vec[i];

        // we must support NULL pointer here
        if (node == NULL)
            write_encoded_tag(Tag_t(0));
        else
            write_encoded(node);
    }
}

// Write all DAGs given by a root set.
void DAG_serializer::write_dags(Dag_vector const * const roots[], size_t n)
{
    // first step: count all nodes
    {
        DAG_ir_counter counter(get_allocator());

        for (size_t i = 0; i < n; ++i) {
            Dag_vector const &vec = *roots[i];
            for (size_t j = 0, l = vec.size(); j < l; ++j) {
                if (DAG_node const *node = vec[j])
                    counter.walk_node(node);
            }
        }

        // write the number of all nodes
        size_t n_nodes = counter.get_count();

        DOUT(("#IR nodes: %u\n", unsigned(n_nodes)));
        write_encoded_tag(n_nodes);
    }

    // now serialize them
    {
        DAG_ir_serializer ir_node_serializer(*this);

        for (size_t i = 0; i < n; ++i) {
            Dag_vector const &vec = *roots[i];
            for (size_t j = 0, l = vec.size(); j < l; ++j) {
                if (DAG_node const *node = vec[j])
                    ir_node_serializer.walk_node(node);
            }
        }
    }
}

// ---------------------- DAG deserializer ----------------------

// Creates a new (empty) code DAG for deserialization.
Generated_code_dag *DAG_deserializer::create_code_dag(
    IAllocator    *alloc,
    MDL           *mdl,
    IModule const *module,
    char const    *internal_space,
    char const    *context_name)
{
    Generated_code_dag *result = m_builder.create<Generated_code_dag>(
        m_builder.get_allocator(),
        mdl,
        module,
        internal_space,
        /*options=*/0,
        // FIXME: we do not serialize the context name here because we do not want to break
        // compatibility with the beta release.
        // However, this IS safe, because only compiled entities are serialized/deserialized
        // which are error free, so the context name is not needed.
        "renderer");
    return result;
}

// Constructor.
DAG_deserializer::DAG_deserializer(
    IDeserializer           *deserializer,
    MDL_binary_deserializer *bin_deserializer)
: Factory_deserializer(bin_deserializer->get_allocator(), deserializer, bin_deserializer)
, m_builder(bin_deserializer->get_allocator())
, m_ir_nodes(bin_deserializer->get_allocator())
{
}

// Deserialize a DAG_vector.
void DAG_deserializer::deserialize(Dag_vector &vec)
{
    size_t l = read_encoded_tag();
    vec.reserve(l);

    for (size_t i = 0; i < l; ++i) {
        Tag_t t = read_encoded_tag();
        DAG_node const *node = t == Tag_t(0) ? NULL : get_ir_node(t);
        vec.push_back(node);
    }
}

// Read all DAGs from a serializer.
void DAG_deserializer::read_dags(DAG_node_factory_impl &node_factory)
{
    size_t n_nodes = read_encoded_tag();
    DOUT(("#IR nodes: %u\n", unsigned(n_nodes)));

    Tag_t t(0);
    for (size_t i = 0; i < n_nodes; ++i) {
        DAG_node::Kind kind = read_encoded<DAG_node::Kind>();

        ++t;
        switch (kind) {
        case DAG_node::EK_CONSTANT:
            {
                IValue const *value = read_encoded<IValue const *>();

                DAG_node const *n = node_factory.create_constant(value);
                register_ir_node(t, n);
            }
            break;
        case DAG_node::EK_TEMPORARY:
            {
                unsigned       idx   = read_encoded<unsigned>();
                DAG_node const *expr = read_encoded<DAG_node const *>();

                DAG_node const *n = node_factory.create_temporary(expr, idx);
                register_ir_node(t, n);
            }
            break;
        case DAG_node::EK_CALL:
            {
                IType const *type = read_encoded<IType const *>();

                string name(read_encoded<string>());

                IDefinition::Semantics sema = read_encoded<IDefinition::Semantics>();

                unsigned n_args = read_encoded<unsigned>();
                VLA<DAG_call::Call_argument> args(get_allocator(), n_args);

                vector<string>::Type param_names(get_allocator());
                param_names.reserve(n_args);

                for (unsigned i = 0; i < n_args; ++i) {
                    param_names.push_back(read_encoded<string>());

                    args[i].arg        = read_encoded<DAG_node const *>();
                    args[i].param_name = param_names[i].c_str();
                }

                DAG_node const *n = node_factory.create_call(
                    name.c_str(), sema, args.data(), n_args, type);
                register_ir_node(t, n);
            }
            break;
        case DAG_node::EK_PARAMETER:
            {
                IType const *type = read_encoded<IType const *>();
                unsigned    idx   = read_encoded<unsigned>();

                DAG_node const *n = node_factory.create_parameter(type, idx);
                register_ir_node(t, n);
            }
            break;
        default:
            MDL_ASSERT(!"unsupported DAG_node kind");
            break;
        }
    }
}

} // mdl
} // mi
