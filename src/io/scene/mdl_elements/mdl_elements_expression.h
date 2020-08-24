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
/// \file
/// \brief      Header for the IExpression hierarchy and IExpression_factory implementation.

#ifndef IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_EXPRESSION_IMPL_H
#define IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_EXPRESSION_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

#include "i_mdl_elements_expression.h"

#include <map>
#include <vector>
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace MDL {

class IValue_factory;

template <class E>
class Expression_base : public mi::base::Interface_implement<E>
{
public:
    typedef Expression_base<E> Base;

    Expression_base( const IType* type) : m_type( type, mi::base::DUP_INTERFACE)
    { ASSERT( M_SCENE, type); }

    IExpression::Kind get_kind() const { return E::s_kind; };

    const IType* get_type() const { m_type->retain(); return m_type.get(); }

    using E::get_type;

protected:
    mi::base::Handle<const IType> m_type;
};


class Expression_constant : public Expression_base<IExpression_constant>
{
public:
    Expression_constant( IValue* value)
      : Base( make_handle( value->get_type()).get()), m_value( value, mi::base::DUP_INTERFACE)
    { ASSERT( M_SCENE, value); }

    const IValue* get_value() const { m_value->retain(); return m_value.get(); }

    IValue* get_value() { m_value->retain(); return m_value.get(); }

    using IExpression_constant::get_value;

    mi::Sint32 set_value( IValue* value);

    mi::Size get_memory_consumption() const;

private:
    mi::base::Handle<IValue> m_value;
};


class Expression_call : public Expression_base<IExpression_call>
{
public:
    Expression_call( const IType* type, DB::Tag tag) : Base( type), m_tag( tag)
    { ASSERT( M_SCENE, tag); }

    DB::Tag get_call() const { return m_tag; }

    mi::Sint32 set_call( DB::Tag tag) { if( !tag) return -1; m_tag = tag; return 0; }

    mi::Size get_memory_consumption() const;

private:
    DB::Tag m_tag;
};


class Expression_parameter : public Expression_base<IExpression_parameter>
{
public:
    Expression_parameter( const IType* type, mi::Size index) : Base( type), m_index( index) { }

    mi::Size get_index() const { return m_index; }

    void set_index( mi::Size index) { m_index = index; }

    mi::Size get_memory_consumption() const;

private:
    mi::Size m_index;
};


class Expression_direct_call : public Expression_base<IExpression_direct_call>
{
public:
    Expression_direct_call(
        const IType* type,
        DB::Tag module_tag,
        const Mdl_tag_ident& definition_ident,
        const std::string& definition_db_name,
        IExpression_list* arguments)
        : Base(type)
        , m_module_tag(module_tag)
        , m_definition_ident(definition_ident)
        , m_definition_db_name(definition_db_name)
        , m_arguments(arguments, mi::base::DUP_INTERFACE)
    {
        ASSERT(M_SCENE, definition_ident.first); ASSERT(M_SCENE, module_tag); ASSERT(M_SCENE, arguments);
    }

    DB::Tag get_definition(DB::Transaction *transaction) const;

    DB::Tag get_module() const { return m_module_tag; }

    Mdl_ident get_definition_ident() const { return m_definition_ident.second; }

    const char* get_definition_db_name() const { return m_definition_db_name.c_str(); }

    mi::Sint32 set_definition( const Mdl_tag_ident& definition_ident)
    {
        if( !definition_ident.first)
            return -1;
        m_definition_ident = definition_ident;
        return 0;
    }

    const IExpression_list* get_arguments() const;

    mi::Size get_memory_consumption() const;

private:
    DB::Tag m_module_tag;
    Mdl_tag_ident m_definition_ident;
    std::string m_definition_db_name;

    mi::base::Handle<IExpression_list> m_arguments;
};


class Expression_temporary : public Expression_base<IExpression_temporary>
{
public:
    Expression_temporary( const IType* type, mi::Size index) : Base( type), m_index( index) { }

    mi::Size get_index() const { return m_index; }

    void set_index( mi::Size index) { m_index = index; }

    mi::Size get_memory_consumption() const;

private:
    mi::Size m_index;
};


class Expression_list : public mi::base::Interface_implement<IExpression_list>
{
public:
    // public API methods

    mi::Size get_size() const;

    mi::Size get_index( const char* name) const;

    const char* get_name( mi::Size index) const;

    const IExpression* get_expression( mi::Size index) const;

    const IExpression* get_expression( const char* name) const;

    mi::Sint32 set_expression( mi::Size index, const IExpression* expression);

    mi::Sint32 set_expression( const char* name, const IExpression* expression);

    mi::Sint32 add_expression( const char* name, const IExpression* expression);

    mi::Size get_memory_consumption() const;

    friend class Expression_factory; // for serialization/deserialization

private:

    typedef std::map<std::string, mi::Size> Name_index_map;
    Name_index_map m_name_index;

    typedef std::vector<std::string> Index_name_vector;
    Index_name_vector m_index_name;

    typedef std::vector<mi::base::Handle<const IExpression> > Expressions_vector;
    Expressions_vector m_expressions;
};


class Annotation_definition : public mi::base::Interface_implement<IAnnotation_definition>
{
public:
    Annotation_definition(
        const char* name,
        const char* module_name,
        const char* simple_name,
        const std::vector<std::string>& parameter_type_names,
        mi::neuraylib::IAnnotation_definition::Semantics semantic,
        bool is_exported,
        const IType_list* parameter_types,
        const IExpression_list* parameter_defaults,
        const IAnnotation_block* annotations)
        : m_name(name)
        , m_module_name(module_name)
        , m_module_db_name(get_db_name(module_name))
        , m_simple_name(simple_name)
        , m_parameter_type_names(parameter_type_names)
        , m_semantic(semantic)
        , m_is_exported(is_exported)
        , m_parameter_types(parameter_types, mi::base::DUP_INTERFACE)
        , m_parameter_defaults(parameter_defaults, mi::base::DUP_INTERFACE)
        , m_annotations(annotations, mi::base::DUP_INTERFACE) { }

    const char* get_module() const { return m_module_db_name.c_str(); }

    const char* get_name() const { return m_name.c_str(); }

    const char* get_mdl_module_name() const { return m_module_name.c_str(); }

    const char* get_mdl_simple_name() const { return m_simple_name.c_str(); }

    const char* get_mdl_parameter_type_name( Size index) const;

    mi::neuraylib::IAnnotation_definition::Semantics get_semantic() const;

    mi::Size get_parameter_count() const;

    const char* get_parameter_name(mi::Size index) const;

    mi::Size get_parameter_index(const char* name) const;

    const IType_list* get_parameter_types() const;

    const IExpression_list* get_defaults() const;

    bool is_exported() const;

    const IAnnotation_block* get_annotations() const;

    const IAnnotation* create_annotation(const IExpression_list* arguments) const;

    mi::Size get_memory_consumption() const;

    std::string get_mdl_name_without_parameter_types() const;

private:
    std::string m_name;
    std::string m_module_name;
    std::string m_module_db_name;
    std::string m_simple_name;
    std::vector<std::string> m_parameter_type_names;
    mi::neuraylib::IAnnotation_definition::Semantics m_semantic;
    bool m_is_exported;
    mi::base::Handle<const IType_list> m_parameter_types;
    mi::base::Handle<const IExpression_list> m_parameter_defaults;
    mi::base::Handle<const IAnnotation_block> m_annotations;
};


class Annotation : public mi::base::Interface_implement<IAnnotation>
{
public:
    Annotation( const char* name, const IExpression_list* arguments)
      : m_name( name), m_arguments( arguments, mi::base::DUP_INTERFACE) { }

    const char* get_name() const { return m_name.c_str(); }

    void set_name( const char* name) { m_name = name ? name : ""; }

    const IExpression_list* get_arguments() const;

    const IAnnotation_definition* get_definition(DB::Transaction* transaction) const;

    mi::Size get_memory_consumption() const;

private:
    std::string m_name;
    mi::base::Handle<const IExpression_list> m_arguments;
};


class Annotation_block : public mi::base::Interface_implement<IAnnotation_block>
{
public:
    // public API methods

    mi::Size get_size() const;

    const IAnnotation* get_annotation( mi::Size index) const;

    mi::Sint32 set_annotation( mi::Size index, const IAnnotation* annotation);

    mi::Sint32 add_annotation( const IAnnotation* annotation);

    mi::Size get_memory_consumption() const;

private:
    std::vector<mi::base::Handle<const IAnnotation> > m_annotations;
};


class Annotation_list : public mi::base::Interface_implement<IAnnotation_list>
{
public:
    // public API methods

    mi::Size get_size() const;

    mi::Size get_index( const char* name) const;

    const char* get_name( mi::Size index) const;

    const IAnnotation_block* get_annotation_block( mi::Size index) const;

    const IAnnotation_block* get_annotation_block( const char* name) const;

    mi::Sint32 set_annotation_block( mi::Size index, const IAnnotation_block* block);

    mi::Sint32 set_annotation_block( const char* name, const IAnnotation_block* block);

    mi::Sint32 add_annotation_block( const char* name, const IAnnotation_block* block);

    mi::Size get_memory_consumption() const;

    friend class Expression_factory; // for serialization/deserialization

private:

    typedef std::map<std::string, mi::Size> Name_index_map;
    Name_index_map m_name_index;

    typedef std::vector<std::string> Index_name_vector;
    Index_name_vector m_index_name;

    typedef std::vector<mi::base::Handle<const IAnnotation_block> > Annotation_blocks_vector;
    Annotation_blocks_vector m_annotation_blocks;
};


class Expression_factory : public mi::base::Interface_implement<IExpression_factory>
{
public:
    Expression_factory( IValue_factory* value_factory);

    ~Expression_factory();

    // public API methods

    IValue_factory* get_value_factory() const;

    IExpression_constant* create_constant( IValue* value) const;

    IExpression_call* create_call( const IType* type, DB::Tag tag) const;

    IExpression_parameter* create_parameter( const IType* type, mi::Size index) const;

    IExpression_direct_call* create_direct_call(
        const IType* type,
        DB::Tag module_tag,
        const Mdl_tag_ident& definition_ident,
        const std::string& definition_db_name,
        IExpression_list* arguments) const;

    IExpression_temporary* create_temporary( const IType* type, mi::Size index) const;

    IExpression_list* create_expression_list() const;

    IAnnotation* create_annotation(
        const char* name, const IExpression_list* arguments) const;

    IAnnotation_definition* create_annotation_definition(
        const char* name,
        const char* module_name,
        const char* simple_name,
        const std::vector<std::string>& parameter_type_names,
        mi::neuraylib::IAnnotation_definition::Semantics sema,
        bool is_exported,
        const IType_list* parameter_types,
        const IExpression_list* parameter_defaults,
        const IAnnotation_block* annotations) const;

    IAnnotation_block* create_annotation_block() const;

    IAnnotation_list* create_annotation_list() const;

    IAnnotation_definition_list* create_annotation_definition_list() const;

    IExpression* clone(
        const IExpression* expr,
        DB::Transaction* transaction,
        bool copy_immutable_calls) const;

    using IExpression_factory::clone;

    IExpression_list* clone(
        const IExpression_list* expr,
        DB::Transaction* transaction,
        bool copy_immutable_calls) const;

    mi::Sint32 compare( const IExpression* lhs, const IExpression* rhs) const
    { return compare_static( lhs, rhs); }

    mi::Sint32 compare( const IExpression_list* lhs, const IExpression_list* rhs) const
    { return compare_static( lhs, rhs); }

    const mi::IString* dump(
        DB::Transaction* transaction,
        const IExpression* expr,
        const char* name,
        mi::Size depth = 0) const;

    const mi::IString* dump(
        DB::Transaction* transaction,
        const IExpression_list* list,
        const char* name,
        mi::Size depth = 0) const;

    const mi::IString* dump(
        DB::Transaction* transaction,
        const IAnnotation* anno,
        const char* name,
        mi::Size depth = 0) const;

    const mi::IString* dump(
        DB::Transaction* transaction,
        const IAnnotation_block* block,
        const char* name,
        mi::Size depth = 0) const;

    const mi::IString* dump(
        DB::Transaction* transaction,
        const IAnnotation_list* list,
        const char* name,
        mi::Size depth = 0) const;

    IExpression* create_cast(
        DB::Transaction* transaction,
        IExpression* src_expr,
        const IType* target_type,
        const char* cast_db_name,
        bool force_cast,
        bool direct_call,
        mi::Sint32* errors) const;

    void serialize( SERIAL::Serializer* serializer, const IExpression* expr) const;

    IExpression* deserialize( SERIAL::Deserializer* deserializer) const;

    using IExpression_factory::deserialize;

    void serialize_list( SERIAL::Serializer* serializer, const IExpression_list* list) const;

    IExpression_list* deserialize_list( SERIAL::Deserializer* deserializer) const;

    void serialize_annotation( SERIAL::Serializer* serializer, const IAnnotation* annotation) const;

    IAnnotation* deserialize_annotation( SERIAL::Deserializer* deserializer) const;

    void serialize_annotation_definition(
        SERIAL::Serializer* serializer, const IAnnotation_definition* anno_def) const;

    IAnnotation_definition* deserialize_annotation_definition(SERIAL::Deserializer* deserializer) const;

    void serialize_annotation_block(
        SERIAL::Serializer* serializer, const IAnnotation_block* block) const;

    IAnnotation_block* deserialize_annotation_block( SERIAL::Deserializer* deserializer) const;

    void serialize_annotation_list(
        SERIAL::Serializer* serializer, const IAnnotation_list* list) const;

    IAnnotation_list* deserialize_annotation_list( SERIAL::Deserializer* deserializer) const;

    void serialize_annotation_definition_list(
        SERIAL::Serializer* serializer, const IAnnotation_definition_list* anno_def_list) const;

    IAnnotation_definition_list* deserialize_annotation_definition_list(
        SERIAL::Deserializer* deserializer) const;

    // internal methods

    static mi::Sint32 compare_static( const IExpression* lhs, const IExpression* rhs);

    static mi::Sint32 compare_static( const IExpression_list* lhs, const IExpression_list* rhs);

private:
    static void dump_static(
        DB::Transaction* transaction,
        IType_factory* tf,
        const IExpression* expr,
        const char* name,
        mi::Size depth,
        std::ostringstream& s);

    static void dump_static(
        DB::Transaction* transaction,
        IType_factory* tf,
        const IExpression_list* expr_list,
        const char* name,
        mi::Size depth,
        std::ostringstream& s);

    static void dump_static(
        DB::Transaction* transaction,
        IType_factory* tf,
        const IAnnotation* anno,
        const char* name,
        mi::Size depth,
        std::ostringstream& s);

    static void dump_static(
        DB::Transaction* transaction,
        IType_factory* tf,
        const IAnnotation_block* block,
        const char* name,
        mi::Size depth,
        std::ostringstream& s);

    static void dump_static(
        DB::Transaction* transaction,
        IType_factory* tf,
        const IAnnotation_list* list,
        const char* name,
        mi::Size depth,
        std::ostringstream& s);

    mi::base::Handle<IValue_factory> m_value_factory;
};

/// Helper class to store annotation definitions.
class Annotation_definition_list : public mi::base::Interface_implement<IAnnotation_definition_list>
{
public:
    mi::Sint32 add_definition(const IAnnotation_definition* anno_def);

    const IAnnotation_definition* get_definition(mi::Size index) const;

    const IAnnotation_definition* get_definition(const char* name) const;

    mi::Size get_size() const { return m_anno_definitions.size(); }

    mi::Size get_memory_consumption() const;

    friend class Expression_factory; // for serialization/deserialization

private:
    std::vector<mi::base::Handle<const IAnnotation_definition>> m_anno_definitions;
    std::map<std::string, mi::Size> m_name_to_index;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_EXPRESSION_IMPL_H
