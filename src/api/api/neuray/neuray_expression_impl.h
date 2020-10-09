/***************************************************************************************************
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
 **************************************************************************************************/

/** \file
 ** \brief Header for the IExpression implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_EXPRESSION_IMPL_H
#define API_API_NEURAY_NEURAY_EXPRESSION_IMPL_H

#include <mi/neuraylib/iexpression.h>

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/itransaction.h>

#include <base/lib/log/i_log_assert.h>
#include <io/scene/mdl_elements/i_mdl_elements_expression.h>

#include "neuray_type_impl.h" // for external IType_factory implementation

namespace MI {

namespace NEURAY {

class Expression_factory;

/// All implementations of mi::neuraylib::IExpression also implement this interface which allows
/// access to the wrapped MI::MDL::IExpression instance.
class IExpression_wrapper : public
    mi::base::Interface_declare<0x33769c59,0xc832,0x4334,0x9f,0xe6,0xb7,0x7a,0x13,0x3f,0x61,0x17>
{
public:
    /// Returns the internal expression wrapped by this wrapper.
    virtual MDL::IExpression* get_internal_expression() = 0;

    /// Returns the internal expression wrapped by this wrapper.
    virtual const MDL::IExpression* get_internal_expression() const = 0;
};

/// Returns the internal expression wrapped by \p expr (or \c NULL if expr is \c NULL or not a
/// wrapper).
MDL::IExpression* get_internal_expression( mi::neuraylib::IExpression* expr);

/// Returns the internal expression wrapped by \p expr (or \c NULL if expr is \c NULL or not a
/// wrapper).
template <class T>
T* get_internal_expression( mi::neuraylib::IExpression* expr)
{
    mi::base::Handle<MDL::IExpression> ptr_expr( get_internal_expression( expr));
    if( !ptr_expr)
        return 0;
    return static_cast<T*>( ptr_expr->get_interface( typename T::IID()));
}

/// Returns the internal expression wrapped by \p expr (or \c NULL if expr is \c NULL or not a
/// wrapper).
const MDL::IExpression* get_internal_expression( const mi::neuraylib::IExpression* expr);

/// Returns the internal expression wrapped by \p expr (or \c NULL if expr is \c NULL or not a
/// wrapper).
template <class T>
const T* get_internal_expression( const mi::neuraylib::IExpression* expr)
{
    mi::base::Handle<const MDL::IExpression> ptr_expr( get_internal_expression( expr));
    if( !ptr_expr)
        return 0;
    return static_cast<const T*>( ptr_expr->get_interface( typename T::IID()));
}

/// The implementation of mi::neuraylib::IExpression_list also implement this interface which allows
/// access to the wrapped MI::MDL::IExpression_list instance.
class IExpression_list_wrapper : public
    mi::base::Interface_declare<0x1c1085ae,0x8a8b,0x488e,0x84,0xaf,0xce,0xc6,0xc2,0xc2,0x62,0xe3>
{
public:
    /// Returns the internal expression list wrapped by this wrapper.
    virtual MDL::IExpression_list* get_internal_expression_list() = 0;

    /// Returns the internal expression list wrapped by this wrapper.
    virtual const MDL::IExpression_list* get_internal_expression_list() const = 0;
};

/// Returns the internal expression list wrapped by \p expr_list (or \c NULL if expr_list is \c NULL
/// or not a wrapper).
MDL::IExpression_list* get_internal_expression_list( mi::neuraylib::IExpression_list* expr_list);

/// Returns the internal expression list wrapped by \p expr_list (or \c NULL if expr_list is \c NULL
/// or not a wrapper).
const MDL::IExpression_list* get_internal_expression_list(
    const mi::neuraylib::IExpression_list* expr_list);

/// Returns the internal annotation wrapped by \p anno (or \c NULL if anno is \c NULL or
/// not a wrapper).
MDL::IAnnotation* get_internal_annotation( mi::neuraylib::IAnnotation* anno);

/// Returns the internal annotation wrapped by \p anno (or \c NULL if anno is \c NULL or
/// not a wrapper).
const MDL::IAnnotation* get_internal_annotation( const mi::neuraylib::IAnnotation* anno);

/// Returns the internal annotation block wrapped by \p block (or \c NULL if block is \c NULL or
/// not a wrapper).
MDL::IAnnotation_block* get_internal_annotation_block( mi::neuraylib::IAnnotation_block* block);

/// Returns the internal annotation block wrapped by \p block (or \c NULL if block is \c NULL or
/// not a wrapper).
const MDL::IAnnotation_block* get_internal_annotation_block(
    const mi::neuraylib::IAnnotation_block* block);


/// Wrapper that implements an external expression interface by wrapping an internal one.
///
/// \tparam E   The external expression interface implemented by this class.
/// \tparam I   The internal expression interface wrapped by this class.
template <class E, class I>
class Expression_base : public mi::base::Interface_implement_2<E, IExpression_wrapper>
{
public:
    typedef E External_expr;
    typedef I Internal_expr;
    typedef Expression_base<E, I> Base;

    Expression_base( const Expression_factory* ef, I* expr, const mi::base::IInterface* owner)
      : m_ef( ef, mi::base::DUP_INTERFACE),
        m_expr( expr, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE)
    {
        ASSERT( M_NEURAY_API, ef);
        ASSERT( M_NEURAY_API, expr);
    }

    ~Expression_base(); // trivial, just because Expression_factory is forward declared

    // public API methods

    mi::neuraylib::IExpression::Kind get_kind() const { return E::s_kind; };

    const mi::neuraylib::IType* get_type() const;

    // internal methods (IExpression_wrapper)

    I* get_internal_expression() { m_expr->retain(); return m_expr.get(); }

    const I* get_internal_expression() const { m_expr->retain(); return m_expr.get(); }

protected:
    const mi::base::Handle<const Expression_factory> m_ef;
    const mi::base::Handle<I> m_expr;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Expression_constant
  : public Expression_base<mi::neuraylib::IExpression_constant, MDL::IExpression_constant>
{
public:
    Expression_constant(
        const Expression_factory* ef, Internal_expr* expr, const mi::base::IInterface* owner)
      : Base( ef, expr, owner) { }

    const mi::neuraylib::IValue* get_value() const;

    mi::neuraylib::IValue* get_value();

    mi::Sint32 set_value( mi::neuraylib::IValue* value);
};

class Expression_call
  : public Expression_base<mi::neuraylib::IExpression_call, MDL::IExpression_call>
{
public:
    Expression_call(
        const Expression_factory* ef,
        mi::neuraylib::ITransaction* transaction,
        Internal_expr* expr,
        const mi::base::IInterface* owner)
      : Base( ef, expr, owner), m_transaction( transaction, mi::base::DUP_INTERFACE) { }

    const char* get_call() const;

    mi::Sint32 set_call( const char* name);

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
};

class Expression_parameter
  : public Expression_base<mi::neuraylib::IExpression_parameter, MDL::IExpression_parameter>
{
public:
    Expression_parameter(
        const Expression_factory* ef, Internal_expr* expr, const mi::base::IInterface* owner)
      : Base( ef, expr, owner) { }

    mi::Size get_index() const { return m_expr->get_index(); }

    void set_index( mi::Size index) { m_expr->set_index( index); }
};

class Expression_direct_call
  : public Expression_base<mi::neuraylib::IExpression_direct_call, MDL::IExpression_direct_call>
{
public:
    Expression_direct_call(
        const Expression_factory* ef,
        mi::neuraylib::ITransaction* transaction,
        Internal_expr* expr,
        const mi::base::IInterface* owner)
      : Base( ef, expr, owner), m_transaction( transaction, mi::base::DUP_INTERFACE) { }

    const char* get_definition() const;

    const mi::neuraylib::IExpression_list* get_arguments() const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
};

class Expression_temporary
  : public Expression_base<mi::neuraylib::IExpression_temporary, MDL::IExpression_temporary>
{
public:
    Expression_temporary(
        const Expression_factory* ef, Internal_expr* expr, const mi::base::IInterface* owner)
      : Base( ef, expr, owner) { }

    mi::Size get_index() const { return m_expr->get_index(); }

    void set_index( mi::Size index) { m_expr->set_index( index); }
};

class Expression_list
  : public mi::base::Interface_implement_2<mi::neuraylib::IExpression_list,IExpression_list_wrapper>
{
public:
    Expression_list(
        const Expression_factory* ef,
        MDL::IExpression_list* expr_list,
        const mi::base::IInterface* owner)
      : m_ef( ef, mi::base::DUP_INTERFACE),
        m_expression_list( expr_list, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE)
    {
        ASSERT( M_NEURAY_API, m_ef);
        ASSERT( M_NEURAY_API, m_expression_list);
    }

    // public API methods

    mi::Size get_size() const { return m_expression_list->get_size(); }

    mi::Size get_index( const char* name) const { return m_expression_list->get_index( name); }

    const char* get_name( mi::Size index) const { return m_expression_list->get_name( index); }

    const mi::neuraylib::IExpression* get_expression( mi::Size index) const;

    const mi::neuraylib::IExpression* get_expression( const char* name) const;

    mi::Sint32 set_expression( mi::Size index, const mi::neuraylib::IExpression* expr);

    mi::Sint32 set_expression( const char* name, const mi::neuraylib::IExpression* expr);

    mi::Sint32 add_expression( const char* name, const mi::neuraylib::IExpression* expr);

    // internal methods (IExpression_list_wrapper)

    MDL::IExpression_list* get_internal_expression_list();

    const MDL::IExpression_list* get_internal_expression_list() const;

private:
    const mi::base::Handle<const Expression_factory> m_ef;
    const mi::base::Handle<MDL::IExpression_list> m_expression_list;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Annotation_definition : public mi::base::Interface_implement<mi::neuraylib::IAnnotation_definition>
{
public:
    Annotation_definition(
        const Expression_factory* ef,
        const Type_factory* tf,
        const MDL::IAnnotation_definition* anno_def,
        const mi::base::IInterface* owner)
        : m_ef(ef, mi::base::DUP_INTERFACE)
        , m_tf(tf, mi::base::DUP_INTERFACE)
        , m_anno_def(anno_def, mi::base::DUP_INTERFACE)
        , m_owner(owner, mi::base::DUP_INTERFACE) { }

    // public API methods

    const char* get_module() const;

    const char* get_name() const;

    const char* get_mdl_module_name() const;

    const char* get_mdl_simple_name() const;

    const char* get_mdl_parameter_type_name( Size index) const;

    mi::neuraylib::IAnnotation_definition::Semantics get_semantic() const;

    mi::Size get_parameter_count() const;

    const char* get_parameter_name(mi::Size index) const;

    mi::Size get_parameter_index(const char* name) const;

    const mi::neuraylib::IType_list* get_parameter_types() const;

    const mi::neuraylib::IExpression_list* get_defaults() const;

    bool is_exported() const;

    const mi::neuraylib::IAnnotation_block* get_annotations() const;

    const mi::neuraylib::IAnnotation* create_annotation(const mi::neuraylib::IExpression_list* arguments) const;

private:
    const mi::base::Handle<const Expression_factory> m_ef;
    const mi::base::Handle<const Type_factory> m_tf;
    const mi::base::Handle<const MDL::IAnnotation_definition> m_anno_def;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Annotation : public mi::base::Interface_implement<mi::neuraylib::IAnnotation>
{
public:
    Annotation(
        const Expression_factory* ef,
        MDL::IAnnotation* anno,
        const mi::base::IInterface* owner)
      : m_ef( ef, mi::base::DUP_INTERFACE),
        m_annotation( anno, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE) { }

    // public API methods

    const char* get_name() const { return m_annotation->get_name(); }

    void set_name( const char* name) { m_annotation->set_name( name); }

    const mi::neuraylib::IExpression_list* get_arguments() const;

    const mi::neuraylib::IAnnotation_definition* get_definition() const;

    // internal methods

    MDL::IAnnotation* get_internal_annotation();

    const MDL::IAnnotation* get_internal_annotation() const;

private:
    const mi::base::Handle<const Expression_factory> m_ef;
    const mi::base::Handle<MDL::IAnnotation> m_annotation;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Annotation_block : public mi::base::Interface_implement<mi::neuraylib::IAnnotation_block>
{
public:
    Annotation_block(
        const Expression_factory* ef,
        MDL::IAnnotation_block* block,
        const mi::base::IInterface* owner)
      : m_ef( ef, mi::base::DUP_INTERFACE),
        m_annotation_block( block, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE) { }

    // public API methods

    mi::Size get_size() const { return m_annotation_block->get_size(); }

    const mi::neuraylib::IAnnotation* get_annotation( mi::Size index) const;

    mi::Sint32 set_annotation( mi::Size index, const mi::neuraylib::IAnnotation* annotation);

    mi::Sint32 add_annotation( mi::neuraylib::IAnnotation* annotation);

    // internal methods

    MDL::IAnnotation_block* get_internal_annotation_block();

    const MDL::IAnnotation_block* get_internal_annotation_block() const;

private:
    const mi::base::Handle<const Expression_factory> m_ef;
    const mi::base::Handle<MDL::IAnnotation_block> m_annotation_block;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Annotation_list : public mi::base::Interface_implement<mi::neuraylib::IAnnotation_list>
{
public:
    Annotation_list(
        const Expression_factory* ef,
        MDL::IAnnotation_list* anno_list,
        const mi::base::IInterface* owner)
      : m_ef( ef, mi::base::DUP_INTERFACE),
        m_annotation_list( anno_list, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE)
    {
        ASSERT( M_NEURAY_API, m_ef);
        ASSERT( M_NEURAY_API, m_annotation_list);
    }

    // public API methods

    mi::Size get_size() const { return m_annotation_list->get_size(); }

    mi::Size get_index( const char* name) const { return m_annotation_list->get_index( name); }

    const char* get_name( mi::Size index) const { return m_annotation_list->get_name( index); }

    const mi::neuraylib::IAnnotation_block* get_annotation_block( mi::Size index) const;

    const mi::neuraylib::IAnnotation_block* get_annotation_block( const char* name) const;

    mi::Sint32 set_annotation_block(
        mi::Size index, const mi::neuraylib::IAnnotation_block* expr);

    mi::Sint32 set_annotation_block(
        const char* name, const mi::neuraylib::IAnnotation_block* expr);

    mi::Sint32 add_annotation_block(
        const char* name, const mi::neuraylib::IAnnotation_block* expr);

    // internal methods

    MDL::IAnnotation_list* get_internal_annotation_list();

    const MDL::IAnnotation_list* get_internal_annotation_list() const;

private:
    const mi::base::Handle<const Expression_factory> m_ef;
    const mi::base::Handle<MDL::IAnnotation_list> m_annotation_list;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Expression_factory : public mi::base::Interface_implement<mi::neuraylib::IExpression_factory>
{
public:
    /// Constructor
    ///
    /// Most methods require a valid transaction, but #dump() works without transaction (but
    /// providing a transaction will produce better output).
    Expression_factory( mi::neuraylib::ITransaction* transaction, mi::neuraylib::IValue_factory* vf)
      : m_transaction( transaction, mi::base::DUP_INTERFACE),
        m_ef( MDL::get_expression_factory()),
        m_vf( vf, mi::base::DUP_INTERFACE) { }

    // public API methods

    mi::neuraylib::IValue_factory* get_value_factory() const { m_vf->retain(); return m_vf.get(); }

    mi::neuraylib::IExpression_constant* create_constant( mi::neuraylib::IValue* value) const;

    const mi::neuraylib::IExpression_constant* create_constant(
        const mi::neuraylib::IValue* value) const;

    mi::neuraylib::IExpression_call* create_call( const char* name) const;

    mi::neuraylib::IExpression_parameter* create_parameter(
        const mi::neuraylib::IType* type, mi::Size index) const;

    mi::neuraylib::IExpression_list* create_expression_list() const;

    mi::neuraylib::IAnnotation* create_annotation(
        const char* name, const mi::neuraylib::IExpression_list* arguments) const;

    mi::neuraylib::IAnnotation_block* create_annotation_block() const;

    mi::neuraylib::IAnnotation_list* create_annotation_list() const;

    mi::neuraylib::IExpression* clone( const mi::neuraylib::IExpression* expr) const;

    using mi::neuraylib::IExpression_factory::clone;

    mi::neuraylib::IExpression_list* clone( const mi::neuraylib::IExpression_list* expr_list) const;

    mi::Sint32 compare(
        const mi::neuraylib::IExpression* lhs, const mi::neuraylib::IExpression* rhs) const;

    mi::Sint32 compare(
        const mi::neuraylib::IExpression_list* lhs,
        const mi::neuraylib::IExpression_list* rhs) const;

    const mi::IString* dump(
        const mi::neuraylib::IExpression* expr, const char* name, mi::Size depth) const;

    const mi::IString* dump(
        const mi::neuraylib::IExpression_list* list, const char* name, mi::Size depth) const;

    const mi::IString* dump(
        const mi::neuraylib::IAnnotation* annotation, const char* name, mi::Size depth) const;

    const mi::IString* dump(
        const mi::neuraylib::IAnnotation_block* block, const char* name, mi::Size depth) const;

    const mi::IString* dump(
        const mi::neuraylib::IAnnotation_list* list, const char* name, mi::Size depth) const;

    mi::neuraylib::IExpression* create_cast(
        mi::neuraylib::IExpression* src_expr,
        const mi::neuraylib::IType* target_type,
        const char* cast_db_name,
        bool force_cast,
        mi::Sint32* errors) const;

    // internal methods

    /// Creates an direct call.
    ///
    /// \param name         The corresponding function or material definition.
    /// \param arguments    The arguments of the direct call.
    /// \return             The created call.
    mi::neuraylib::IExpression_direct_call* create_direct_call(
        const char* name, mi::neuraylib::IExpression_list* arguments) const;

    /// Creates a temporary reference.
    ///
    /// \param type         The type of the temporary.
    /// \param index        The index of the temporary.
    /// \return             The created temporary reference.
    mi::neuraylib::IExpression_temporary* create_temporary(
        const mi::neuraylib::IType* type, mi::Size index) const;

    /// Returns the wrapped internal expression factory.
    MDL::IExpression_factory* get_internal_expression_factory() const
    { m_ef->retain(); return m_ef.get(); }

    /// Creates a wrapper for the internal expression.
    mi::neuraylib::IExpression* create(
        MDL::IExpression* expr, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal expression.
    template <class T>
    T* create( MDL::IExpression* expr, const mi::base::IInterface* owner) const
    {
        mi::base::Handle<mi::neuraylib::IExpression> ptr_expr( create( expr, owner));
        if( !ptr_expr)
            return 0;
        return static_cast<T*>( ptr_expr->get_interface( typename T::IID()));
    }

    /// Creates a wrapper for the internal expression.
    const mi::neuraylib::IExpression* create(
        const MDL::IExpression* expr, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal expression.
    template <class T>
    const T* create( const MDL::IExpression* expr, const mi::base::IInterface* owner) const
    {
        mi::base::Handle<const mi::neuraylib::IExpression> ptr_expr( create( expr, owner));
        if( !ptr_expr)
            return 0;
        return static_cast<const T*>( ptr_expr->get_interface( typename T::IID()));
    }

    /// Creates a wrapper for the internal expression list.
    mi::neuraylib::IExpression_list* create_expression_list(
        MDL::IExpression_list* expr_list, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal expression list.
    const mi::neuraylib::IExpression_list* create_expression_list(
        const MDL::IExpression_list* expr_list, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal annotation.
    mi::neuraylib::IAnnotation* create_annotation(
        MDL::IAnnotation* anno, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal annotation.
    const mi::neuraylib::IAnnotation* create_annotation(
        const MDL::IAnnotation* anno, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal annotation block.
    mi::neuraylib::IAnnotation_block* create_annotation_block(
        MDL::IAnnotation_block* block, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal annotation block.
    const mi::neuraylib::IAnnotation_block* create_annotation_block(
        const MDL::IAnnotation_block* block, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal annotation list.
    mi::neuraylib::IAnnotation_list* create_annotation_list(
        MDL::IAnnotation_list* list, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal annotation list.
    const mi::neuraylib::IAnnotation_list* create_annotation_list(
        const MDL::IAnnotation_list* list, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal annotation definition.
    const mi::neuraylib::IAnnotation_definition* create_annotation_definition(
        const MDL::IAnnotation_definition* anno_def, const mi::base::IInterface* owner) const;

    /// Returns the DB transaction corresponding to m_transaction.
    DB::Transaction* get_db_transaction() const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    const mi::base::Handle<MDL::IExpression_factory> m_ef;
    const mi::base::Handle<mi::neuraylib::IValue_factory> m_vf;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_EXPRESSION_IMPL_H
