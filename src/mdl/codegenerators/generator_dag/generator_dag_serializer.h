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

#ifndef MDL_GENERATOR_DAG_SERIALIZER_H
#define MDL_GENERATOR_DAG_SERIALIZER_H

#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_generated_dag.h>

#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_serializer.h"

namespace mi {
namespace mdl {

class MDL;
class Generated_code_dag;
class DAG_node_factory_impl;

/// The MDL code DAG serializer.
class DAG_serializer : public Factory_serializer {
public:
    typedef vector<DAG_node const *>::Type Dag_vector;

    /// Constructor.
    ///
    /// \param alloc           the allocator
    /// \param serializer      the serializer used to write the low level data
    /// \param bin_serializer  the serializer used for serializing "the binary"
    DAG_serializer(
        IAllocator            *alloc,
        ISerializer           *serializer,
        MDL_binary_serializer *bin_serializer);

    /// Serialize a vector of T.
    template<typename T, typename A>
    void serialize(std::vector<T, A> const &vec)
    {
        size_t l = vec.size();

        write_encoded_tag(l);
        for (size_t i = 0; i < l; ++i) {
            write_encoded(vec[i]);
        }
    }

    /// Serialize a vector of DAG_vector.
    void serialize(Dag_vector const &vec);

    /// Write all DAGs given by a root set.
    ///
    /// \param roots  the root set, given as array of Dag_vectors
    /// \param n      the length of the roots array
    void write_dags(Dag_vector const * const roots[], size_t n);

    /// Register a DAG IR node.
    ///
    /// \param node  the IR node
    Tag_t register_ir_node(DAG_node const *node) {
        return m_ir_nodes.create_tag(node);
    }

    /// Get the tag for a known DAG IR node.
    ///
    /// \param node  the node
    Tag_t get_ir_node_tag(DAG_node const *node)
    {
        Tag_t tag = m_ir_nodes.get_tag(node);
        MDL_ASSERT(tag != 0);
        return tag;
    }

    /// Write a semantics.
    void write_encoded(IDefinition::Semantics sema) {
        write_unsigned(sema);
    }

    /// Write a type.
    void write_encoded(IType const *tp) {
        if (tp == NULL) {
            // the 0 tag is used for the "void" type
            write_encoded_tag(Tag_t(0));
        } else {
            Tag_t t = get_type_tag(tp);
            write_encoded_tag(t);
        }
    }

    /// Write a string.
    void write_encoded(string const &s) {
        write_cstring(s.c_str());
    }

    /// Write an int.
    void write_encoded(int i) {
        write_int(i);
    }

    /// Write an unsigned.
    void write_encoded(size_t i) {
        write_encoded_tag(i);
    }

    /// Write an unsigned.
    void write_encoded(unsigned i) {
        write_unsigned(i);
    }

    /// Write a bool.
    void write_encoded(bool b) {
        write_bool(b);
    }

    /// Write an unsigned char.
    void write_encoded(unsigned char b) {
        write_byte(b);
    }

    /// Write a C-string.
    void write_encoded(const char *s) {
        write_cstring(s);
    }

    /// Write a DAG node.
    void write_encoded(DAG_node const *node) {
        Tag_t t = get_ir_node_tag(node);
        write_encoded_tag(t);
    }

    /// Write a DAG kind.
    void write_encoded(DAG_node::Kind kind) {
        write_unsigned(kind);
    }

    /// Write an IValue kind.
    void write_encoded(IValue::Kind kind) {
        write_unsigned(kind);
    }

    /// Write a Resource_tag_tuple kind.
    void write_encoded(Resource_tag_tuple::Kind kind) {
        write_unsigned(kind);
    }

private:
    /// pointer serializer for DAG IR nodes.
    Pointer_serializer<DAG_node> m_ir_nodes;
};

/// The MDL code DAG deserializer.
class DAG_deserializer : public Factory_deserializer {
public:
    typedef vector<DAG_node const *>::Type Dag_vector;

    /// Creates a new (empty) code DAG for deserialization.
    ///
    /// \param alloc            The allocator.
    /// \param mdl              The mdl compiler.
    /// \param module           The module from which this code was generated.
    /// \param internal_space   The internal space for which to compile.
    /// \param context_name     The name of the context (used for error messages).
    ///
    /// \return a new empty code DAG
    Generated_code_dag *create_code_dag(
        IAllocator    *alloc,
        MDL           *mdl,
        IModule const *module,
        char const    *internal_space,
        char const    *context_name);

    /// Constructor.
    ///
    /// \param deserializer      the deserializer used to read the low level data
    /// \param bin_deserializer  the serializer used for deserializing "the binary"
    DAG_deserializer(
        IDeserializer           *deserializer,
        MDL_binary_deserializer *bin_deserializer);

    /// Deserialize a vector of T.
    template<typename T, typename A>
    void deserialize(std::vector<T, A> &vec)
    {
        size_t l = read_encoded_tag();
        vec.reserve(l);

        for (size_t i = 0; i < l; ++i) {
            vec.push_back(read_encoded<T>());
        }
    }

    /// Deserialize a DAG_vector.
    void deserialize(Dag_vector &vec);

    /// Read all DAGs from a serializer.
    ///
    /// \param node_factory  the IR node factory used to create new nodes
    void read_dags(DAG_node_factory_impl &node_factory);

    /// Register an IR node.
    ///
    /// \param tag   the node tag
    /// \param node  the IR node
    void register_ir_node(Tag_t tag, DAG_node const *node) {
        m_ir_nodes.register_obj(tag, node);
    }

    /// Get the IR node for for a known tag.
    ///
    /// \param tag     the node tag
    DAG_node const *get_ir_node(Tag_t tag)
    {
        return m_ir_nodes.get_obj(tag);
    }

    /// Read an encoded entity of type T.
    template<typename T>
    T read_encoded();

private:
    /// The builder for code DAGs.
    Allocator_builder m_builder;

    /// pointer deserializer for DAG IR nodes.
    Pointer_deserializer<DAG_node const> m_ir_nodes;
};

/// Read a semantic code.
template<>
inline IDefinition::Semantics DAG_deserializer::read_encoded() {
    return IDefinition::Semantics(read_unsigned());
}

/// Read a type.
template<>
inline IType const *DAG_deserializer::read_encoded() {
    Tag_t t = read_encoded_tag();
    // The NULL tag is used as the "void" type
    if (t == Tag_t(0))
        return NULL;
    return get_type(t);
}

/// Read a value.
template<>
inline IValue const *DAG_deserializer::read_encoded() {
    Tag_t t = read_encoded_tag();
    return get_value(t);
}

/// Read a string.
template<>
inline string DAG_deserializer::read_encoded() {
    char const *s = read_cstring();
    return string(s, get_allocator());
}

/// Read an int.
template<>
inline int DAG_deserializer::read_encoded() {
    return read_int();
}

/// Read an size_t.
template<>
inline size_t DAG_deserializer::read_encoded() {
    return read_size_t();
}

/// Read an unsigned.
template<>
inline unsigned DAG_deserializer::read_encoded() {
    return read_unsigned();
}

/// Read a bool.
template<>
inline bool DAG_deserializer::read_encoded() {
    return read_bool();
}

/// Read an unsigned char.
template<>
inline unsigned char DAG_deserializer::read_encoded() {
    return read_byte();
}

/// Read a DAG node.
template<>
inline DAG_node const *DAG_deserializer::read_encoded() {
    Tag_t t = read_encoded_tag();
    return get_ir_node(t);
}

/// Read a DAG kind.
template<>
inline DAG_node::Kind DAG_deserializer::read_encoded() {
    unsigned k = read_unsigned();
    return DAG_node::Kind(k);
}

/// Read a DAG kind.
template<>
inline IValue::Kind DAG_deserializer::read_encoded() {
    unsigned k = read_unsigned();
    return IValue::Kind(k);
}

/// Read a Resource_tag_tuple kind.
template<>
inline Resource_tag_tuple::Kind DAG_deserializer::read_encoded() {
    unsigned k = read_unsigned();
    return Resource_tag_tuple::Kind(k);
}

} // mdl
} // mi

#endif
