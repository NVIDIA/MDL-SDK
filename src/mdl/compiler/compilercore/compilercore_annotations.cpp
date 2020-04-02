/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_annotations.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/base/iallocator.h>

#include "compilercore_cc_conf.h"

#include "compilercore_memory_arena.h"
#include "compilercore_factories.h"
#include "compilercore_positions.h"

namespace mi {
namespace mdl {

/// A mixin for all base annotation methods.
template <typename Interface>
class Annotation_base : public Interface
{
    typedef Interface Base;
public:
    /// Get the kind of expression.
    typename Interface::Kind get_kind() const MDL_FINAL { return Interface::s_kind; }

    /// Get the annotation name.
    IQualified_name const *get_name() const MDL_FINAL { return m_name; }

    /// Get the argument count.
    int get_argument_count() const MDL_FINAL { return m_args.size(); }

    /// Get the argument at index.
    IArgument const *get_argument(int index) const MDL_FINAL { return m_args.at(index); }

    /// Add an argument.
    void add_argument(IArgument const *arg) MDL_FINAL { m_args.push_back(arg); }

    /// Replace an argument.
    void replace_argument(int index, IArgument const *arg) MDL_FINAL {
        m_args.at(index) = arg;
    }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

    /// Set the annotation name.
    void set_name(IQualified_name const *name) { m_name = name; }

protected:
    explicit Annotation_base(Memory_arena *arena, IQualified_name const *name)
    : Base()
    , m_name(name)
    , m_args(arena)
    , m_pos(0, 0, 0, 0)
    {
    }

    /// The name of this annotation.
    IQualified_name const *m_name;

    /// Arguments of this annotation.
    Arena_vector<const IArgument *>::Type m_args;

    /// The position of this annotation.
    Position_impl m_pos;
};

// Implementation of IAnnotation
class Annotation : public Annotation_base<IAnnotation>
{
    typedef Annotation_base<IAnnotation> Base;
    friend class Arena_builder;
private:

    explicit Annotation(Memory_arena *arena, IQualified_name const *name)
        : Base(arena, name)
    {
    }
};

/// Implementation of IAnnotation_enable_if,
class Annotation_enable_if : public Annotation_base<IAnnotation_enable_if>
{
    typedef Annotation_base<IAnnotation_enable_if> Base;
    friend class Arena_builder;
public:

    /// Get the expression parsed from the parameter of the enable_if annotation.
    IExpression const *get_expression() const MDL_FINAL { return m_expr; }

    /// Set the expression parsed from the parameter of the enable_if annotation.
    void set_expression(IExpression const *expr) MDL_FINAL { m_expr = expr; }

private:
    explicit Annotation_enable_if(Memory_arena *arena, IQualified_name const *name)
        : Base(arena, name)
        , m_expr(NULL)
    {
    }

    /// The expression parsed from the parameter.
    /// Will be set during semantic analysis.
    IExpression const *m_expr;
};

/// Implementation of an annotation block.
class Annotation_block : public IAnnotation_block
{
    typedef IAnnotation_block Base;
    friend class Arena_builder;
public:

    /// Get the number of annotations.
    int get_annotation_count() const MDL_FINAL { return m_annos.size(); }

    /// Get the annotation at index.
    IAnnotation const *get_annotation(int index) const MDL_FINAL {
        return m_annos.at(index);
    }

    /// Add an annotation.
    void add_annotation(IAnnotation const *anno) MDL_FINAL { m_annos.push_back(anno); }

    /// Access the position.
    Position &access_position() MDL_FINAL { return m_pos; }

    /// Access the position.
    Position const &access_position() const MDL_FINAL { return m_pos; }

    /// Delete an annotation.
    void delete_annotation(int index) MDL_FINAL {
        if (0 <= index && size_t(index) < m_annos.size())
            m_annos.erase(m_annos.begin() + index);
    }

private:
    explicit Annotation_block(Memory_arena *arena)
    : Base()
    , m_annos(arena)
    , m_pos(0, 0, 0, 0)
    {
    }

    /// Annotations of this annotation block.
    Arena_vector<IAnnotation const *>::Type m_annos;

    /// The position of this annotation block.
    Position_impl m_pos;
};

// ------------------------------------ Annotation factory ------------------------------------

Annotation_factory::Annotation_factory(Memory_arena &arena)
: Base()
, m_builder(arena)
{
}

/// Create a new annotation.
IAnnotation *Annotation_factory::create_annotation(
    IQualified_name const *qname,
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IAnnotation *result;
    // The qname definition is not set, yet, so check the components directly
    if (qname->get_component_count() == 2 &&
            strcmp(qname->get_component(1)->get_symbol()->get_name(), "enable_if") == 0 &&
            strcmp(qname->get_component(0)->get_symbol()->get_name(), "anno") == 0)
        result = m_builder.create<Annotation_enable_if>(m_builder.get_arena(), qname);
    else
        result = m_builder.create<Annotation>(m_builder.get_arena(), qname);
    Position &pos = result->access_position();
    pos.set_start_line(start_line);
    pos.set_start_column(start_column);
    pos.set_end_line(end_line);
    pos.set_end_column(end_column);
    return result;
}

/// Create a new annotation block.
IAnnotation_block *Annotation_factory::create_annotation_block(
    int start_line,
    int start_column,
    int end_line,
    int end_column)
{
    IAnnotation_block *result = m_builder.create<Annotation_block>(m_builder.get_arena());
    Position &pos = result->access_position();
    pos.set_start_line(start_line);
    pos.set_start_column(start_column);
    pos.set_end_line(end_line);
    pos.set_end_column(end_column);
    return result;
}

}  // mdl
}  // mi
