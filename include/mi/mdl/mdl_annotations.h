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
/// \file mi/mdl/mdl_annotations.h
/// \brief Interfaces for MDL annotations in the AST
#ifndef MDL_ANNOTATIONS_H
#define MDL_ANNOTATIONS_H 1

#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_positions.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_expressions.h>

namespace mi {
namespace mdl {

/// An MDL annotation.
class IAnnotation : public Interface_owned
{
public:

    /// The possible kinds of annotations.
    enum Kind {
        AK_NORMAL,               ///< An annotation without any special sub-type
        AK_ENABLE_IF,            ///< An enable_if annotation
    };

    /// The kind of this subclass.
    static Kind const s_kind = AK_NORMAL;

    /// Get the annotation kind.
    virtual Kind get_kind() const = 0;

    /// Get the annotation name of this annotation.
    virtual IQualified_name const *get_name() const = 0;

    /// Get the argument count of this annotation.
    virtual int get_argument_count() const = 0;

    /// Get the argument at index.
    ///
    /// \param index  the index of the requested argument
    virtual IArgument const *get_argument(int index) const = 0;

    /// Add an argument.
    ///
    /// \param arg  the argument to add
    virtual void add_argument(IArgument const *arg) = 0;

    /// Replace an argument by another argument.
    ///
    /// \param index  the index of the argument to replace
    /// \param arg    the new argument
    virtual void replace_argument(int index, IArgument const *arg) = 0;

    /// Access the position of this annotation.
    virtual Position &access_position() = 0;

    /// Access the position of this annotation.
    virtual Position const &access_position() const = 0;
};

/// An enable_if annotation.
class IAnnotation_enable_if : public IAnnotation
{
public:
    /// The kind of this subclass.
    static Kind const s_kind = AK_ENABLE_IF;

    /// Get the expression parsed from the parameter of the enable_if annotation.
    virtual IExpression const *get_expression() const = 0;

    /// Set the expression parsed from the parameter of the enable_if annotation.
    virtual void set_expression(IExpression const *expr) = 0;
};

/// An annotation block in the MDL AST.
class IAnnotation_block : public Interface_owned
{
public:
    /// Get the number of annotations in this block.
    virtual int get_annotation_count() const = 0;

    /// Get the annotation at index.
    ///
    /// \param index  the index of the requested annotation
    virtual IAnnotation const *get_annotation(int index) const = 0;

    /// Add an annotation to this block.
    ///
    /// \param anno  the annotation to add
    virtual void add_annotation(IAnnotation const *anno) = 0;

    /// Access the position of this block.
    virtual Position &access_position() = 0;

    /// Access the position of this block.
    virtual Position const &access_position() const = 0;

    /// Delete the annotation at given index.
    ///
    /// \param index  the index of the annotation to remove
    virtual void delete_annotation(int index) = 0;
};

/// The interface for creating annotations in the MDL AST.
/// An IAnnotation_factory interface can be obtained by calling
/// the method create_annotation_factory() on the interface IModule.
class IAnnotation_factory : public Interface_owned
{
public:
    /// Create a new annotation.
    ///
    /// \param qname         the qualified name of the annotation
    /// \param start_line    start line in the input
    /// \param start_column  start column in the input
    /// \param end_line      end line in the input
    /// \param end_column    end column in the input
    ///
    /// \return the newly created annotation
    virtual IAnnotation *create_annotation(
        IQualified_name const *qname,
        int                   start_line = 0,
        int                   start_column = 0,
        int                   end_line = 0,
        int                   end_column = 0) = 0;

    /// Create a new annotation block.
    ///
    /// \param start_line    start line in the input
    /// \param start_column  start column in the input
    /// \param end_line      end line in the input
    /// \param end_column    end column in the input
    ///
    /// \return the newly created annotation block
    virtual IAnnotation_block *create_annotation_block(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) = 0;
};

/// Cast to subtype or return NULL if annotation type do not match.
template<typename T>
T const *as(IAnnotation const *anno) {
    return (anno->get_kind() == T::s_kind) ? static_cast<T const *>(anno) : NULL;
}

/// Check if an annotation is of a certain type.
template<typename T>
bool is(IAnnotation const *anno) {
    return as<T>(anno) != NULL;
}

}  // mdl
}  // mi

#endif
