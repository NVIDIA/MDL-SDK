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
/// \brief      Expressions of the MDL type system
///
///             For documentation, see the counterparts in the public API
///             in <mi/neuraylib/iexpression.h>.

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_EXPRESSION_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_EXPRESSION_H

#include <mi/neuraylib/iexpression.h>

#include "i_mdl_elements_value.h"
#include "i_mdl_elements_module.h"

namespace mi { class IString; }

namespace MI {

namespace DB { class Transaction; }

namespace MDL {

class IExpression_list;
class IAnnotation;
class IAnnotation_block;

class IExpression : public
    mi::base::Interface_declare<0xdb6bbe22,0xc36b,0x4786,0x9f,0x6e,0xda,0x45,0x12,0x0f,0x0c,0x31>
{
public:
    enum Kind {
        EK_CONSTANT,
        EK_CALL,
        EK_PARAMETER,
        EK_DIRECT_CALL,
        EK_TEMPORARY,
        EK_FORCE_32_BIT = 0xffffffffU
    };

    virtual Kind get_kind() const = 0;

    virtual const IType* get_type() const = 0;

    template <class T>
    const T* get_type() const
    {
        mi::base::Handle<const IType> ptr_type( get_type());
        if( !ptr_type)
            return nullptr;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    virtual mi::Size get_memory_consumption() const = 0;
};

mi_static_assert( sizeof( IExpression::Kind) == sizeof( mi::Uint32));

class IExpression_constant : public
    mi::base::Interface_declare<0x7befdc92,0x0be6,0x4cf6,0xbf,0x7f,0xf1,0x75,0xe1,0xbb,0x5f,0xbd,
                                IExpression>
{
public:
    static const Kind s_kind = EK_CONSTANT;

    virtual const IValue* get_value() const = 0;

    template <class T>
    const T* get_value() const
    {
        mi::base::Handle<const IValue> ptr_value( get_value());
        if( !ptr_value)
            return nullptr;
        return static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual IValue* get_value() = 0;

    template <class T>
    T* get_value()
    {
        mi::base::Handle<IValue> ptr_value( get_value());
        if( !ptr_value)
            return nullptr;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }

    virtual mi::Sint32 set_value( IValue* value) = 0;
};

class IExpression_call : public
    mi::base::Interface_declare<0x02384935,0x254c,0x4d69,0xb4,0x44,0xdf,0xe6,0x5f,0x1b,0x1c,0x28,
                                IExpression>
{
public:
    static const Kind s_kind = EK_CALL;

    virtual DB::Tag get_call() const = 0;

    virtual mi::Sint32 set_call( DB::Tag tag) = 0;
};

class IExpression_parameter : public
    mi::base::Interface_declare<0x3320969b,0xf5e1,0x4213,0xaf,0x9e,0x64,0xc2,0x11,0x82,0x01,0x89,
                               IExpression>
{
public:
    static const Kind s_kind = EK_PARAMETER;

    virtual mi::Size get_index() const = 0;

    virtual void set_index( mi::Size index) = 0;
};

class IExpression_direct_call : public
    mi::base::Interface_declare<0xae432486,0x36be,0x4e61,0xb5,0x4d,0xd9,0x71,0xd3,0xcc,0x6f,0x00,
                                IExpression>
{
public:
    static const Kind s_kind = EK_DIRECT_CALL;

    virtual DB::Tag get_definition(DB::Transaction *transaction) const = 0;

    virtual DB::Tag get_module() const = 0;

    virtual Mdl_ident get_definition_ident() const = 0;

    virtual const char* get_definition_db_name() const = 0;

    virtual mi::Sint32 set_definition( const Mdl_tag_ident& definition_ident) = 0;

    virtual const IExpression_list* get_arguments() const = 0;
};

class IExpression_temporary : public
    mi::base::Interface_declare<0x19e30339,0x5148,0x4647,0x88,0xfd,0x73,0x2d,0xe8,0x89,0x58,0xfa,
                                IExpression>
{
public:
    static const Kind s_kind = EK_TEMPORARY;

    virtual mi::Size get_index() const = 0;

    virtual void set_index( mi::Size index) = 0;
};

class IExpression_list : public
    mi::base::Interface_declare<0x6f649c76,0xe019,0x47fb,0xac,0xc2,0x0f,0xc1,0x1c,0x6a,0xd1,0xfb>
{
public:
    virtual mi::Size get_size() const = 0;

    virtual mi::Size get_index( const char* name) const = 0;

    virtual const char* get_name( mi::Size index) const = 0;

    virtual const IExpression* get_expression( mi::Size index) const = 0;

    template <class T>
    const T* get_expression( mi::Size index) const
    {
        mi::base::Handle<const IExpression> ptr_expr( get_expression( index));
        if( !ptr_expr)
            return nullptr;
        return static_cast<const T*>( ptr_expr->get_interface( typename T::IID()));
    }

    virtual const IExpression* get_expression( const char* name) const = 0;

    template <class T>
    const T* get_expression( const char* name) const
    {
        mi::base::Handle<const IExpression> ptr_expr( get_expression( name));
        if( !ptr_expr)
            return nullptr;
        return static_cast<const T*>( ptr_expr->get_interface( typename T::IID()));
    }

    virtual mi::Sint32 set_expression( const char* name, const IExpression* expression) = 0;

    virtual mi::Sint32 set_expression( mi::Size index, const IExpression* expression) = 0;

    virtual mi::Sint32 add_expression( const char* name, const IExpression* expression) = 0;

    virtual mi::Size get_memory_consumption() const = 0;
};

class IAnnotation_definition : public
    mi::base::Interface_declare<0x6ca3757d,0x995f,0x49e5,0xbc,0x10,0x70,0xec,0xdd,0x82,0x5e,0x1b>
{
public:
    // Returns "const char*" instead of the usual DB::Tag since the API wrapper does not hold a
    // transaction to do the conversion.
    virtual const char* get_module() const = 0;

    virtual const char* get_name() const = 0;

    virtual const char* get_mdl_module_name() const = 0;

    virtual const char* get_mdl_simple_name() const = 0;

    virtual const char* get_mdl_parameter_type_name( Size index) const = 0;

    virtual mi::neuraylib::IAnnotation_definition::Semantics get_semantic() const = 0;

    virtual mi::Size get_parameter_count() const = 0;

    virtual const char* get_parameter_name(mi::Size index) const = 0;

    virtual mi::Size get_parameter_index(const char* name) const = 0;

    virtual const IType_list* get_parameter_types() const = 0;

    virtual const IExpression_list* get_defaults() const = 0;

    virtual bool is_exported() const = 0;

    virtual const IAnnotation_block* get_annotations() const = 0;

    virtual const IAnnotation* create_annotation(const IExpression_list* arguments) const = 0;

    virtual mi::Size get_memory_consumption() const = 0;

    virtual std::string get_mdl_name_without_parameter_types() const = 0;
};

class IAnnotation : public
    mi::base::Interface_declare<0xa9c652e7,0x952e,0x4887,0x93,0xb4,0x55,0xc8,0x66,0xd0,0x1a,0x1f>
{
public:
    virtual const char* get_name() const = 0;

    virtual void set_name( const char* name) = 0;

    virtual const IExpression_list* get_arguments() const = 0;

    virtual const IAnnotation_definition* get_definition(DB::Transaction* transaction) const = 0;

    virtual mi::Size get_memory_consumption() const = 0;
};

class IAnnotation_block : public
    mi::base::Interface_declare<0x4e85fed6,0x3583,0x4d9f,0xb7,0xef,0x2d,0x2f,0xab,0xed,0x20,0x51>
{
public:
    virtual mi::Size get_size() const = 0;

    virtual const IAnnotation* get_annotation( mi::Size index) const = 0;

    virtual mi::Sint32 set_annotation( mi::Size index, const IAnnotation* annotation) = 0;

    virtual mi::Sint32 add_annotation( const IAnnotation* annotation) = 0;

    virtual mi::Size get_memory_consumption() const = 0;
};

class IAnnotation_list : public
    mi::base::Interface_declare<0x87e92a87,0x1dc9,0x4c91,0x97,0x6b,0x89,0x8f,0xa3,0x69,0x5d,0xcb>
{
public:
    virtual mi::Size get_size() const = 0;

    virtual mi::Size get_index( const char* name) const = 0;

    virtual const char* get_name( mi::Size index) const = 0;

    virtual const IAnnotation_block* get_annotation_block( mi::Size index) const = 0;

    virtual const IAnnotation_block* get_annotation_block( const char* name) const = 0;

    virtual mi::Sint32 set_annotation_block(
        const char* name, const IAnnotation_block* block) = 0;

    virtual mi::Sint32 set_annotation_block(
        mi::Size index, const IAnnotation_block* block) = 0;

    virtual mi::Sint32 add_annotation_block(
        const char* name, const IAnnotation_block* block) = 0;

    virtual mi::Size get_memory_consumption() const = 0;
};

/// Internal helper class to store annotation definitions.
class IAnnotation_definition_list : public
    mi::base::Interface_declare<0xa43330cc,0xb1c4,0x41dd,0x9f,0x1b,0xed,0xcc,0x9,0x5b,0x51,0x1>
{
public:
    virtual mi::Sint32 add_definition(const IAnnotation_definition* anno_def) = 0;

    virtual const IAnnotation_definition* get_definition(mi::Size index) const = 0;

    virtual const IAnnotation_definition* get_definition(const char* name) const = 0;

    virtual mi::Size get_size() const = 0;

    virtual mi::Size get_memory_consumption() const = 0;
};


class IExpression_factory : public
    mi::base::Interface_declare<0xf2422380,0xae9c,0x44e0,0x8d,0xc7,0x15,0xf0,0x30,0xda,0x9e,0x1e>
{
public:
    virtual IValue_factory* get_value_factory() const = 0;

    virtual IExpression_constant* create_constant( IValue* value) const = 0;

    virtual const IExpression_constant* create_constant( const IValue* value) const = 0;

    virtual IExpression_call* create_call( const IType* type, DB::Tag tag) const = 0;

    virtual IExpression_parameter* create_parameter( const IType* type, mi::Size index) const = 0;

    virtual IExpression_direct_call* create_direct_call(
        const IType* type,
        DB::Tag module_tag,
        const Mdl_tag_ident& definition_ident,
        const std::string& definition_db_name,
        IExpression_list* arguments) const = 0;

    virtual IExpression_temporary* create_temporary( const IType* type, mi::Size index) const = 0;

    virtual IExpression_list* create_expression_list() const = 0;

    virtual IAnnotation* create_annotation(
        const char* name, const IExpression_list* arguments) const = 0;

    virtual IAnnotation_definition* create_annotation_definition(
        const char* name,
        const char* module_name,
        const char* simple_name,
        const std::vector<std::string>& parameter_type_names,
        mi::neuraylib::IAnnotation_definition::Semantics sema,
        bool is_exported,
        const IType_list* parameter_types,
        const IExpression_list* parameter_defaults,
        const IAnnotation_block* annotations) const = 0;

    virtual IAnnotation_block* create_annotation_block() const = 0;

    virtual IAnnotation_list* create_annotation_list() const = 0;

    virtual IAnnotation_definition_list* create_annotation_definition_list() const = 0;

    virtual IExpression* clone(
        const IExpression* expr,
        DB::Transaction* transaction,
        bool copy_immutable_calls) const = 0;

    template <class T>
    T* clone(
        const T* expr,
        DB::Transaction* transaction,
        bool copy_immutable_calls) const
    {
        mi::base::Handle<IExpression> ptr_expr(
            clone( static_cast<const IExpression*>( expr), transaction, copy_immutable_calls));
        if( !ptr_expr)
            return nullptr;
        return static_cast<T*>( ptr_expr->get_interface( typename T::IID()));
    }

    virtual IExpression_list* clone(
        const IExpression_list* list,
        DB::Transaction* transaction,
        bool copy_immutable_calls) const = 0;

    virtual mi::Sint32 compare( const IExpression* lhs, const IExpression* rhs) const = 0;

    virtual mi::Sint32 compare( const IExpression_list* lhs, const IExpression_list* rhs) const = 0;

    virtual const mi::IString* dump(
        DB::Transaction* transaction,
        const IExpression* expr,
        const char* name,
        mi::Size depth = 0) const = 0;

    virtual const mi::IString* dump(
        DB::Transaction* transaction,
        const IExpression_list* list,
        const char* name,
        mi::Size depth = 0) const = 0;

    virtual const mi::IString* dump(
        DB::Transaction* transaction,
        const IAnnotation* anno,
        const char* name,
        mi::Size depth = 0) const = 0;

    virtual const mi::IString* dump(
        DB::Transaction* transaction,
        const IAnnotation_block* block,
        const char* name,
        mi::Size depth = 0) const = 0;

    virtual const mi::IString* dump(
        DB::Transaction* transaction,
        const IAnnotation_list* list,
        const char* name,
        mi::Size depth = 0) const = 0;

    virtual IExpression* create_cast(
        DB::Transaction* transaction,
        IExpression* src_expr,
        const IType* target_type,
        const char* cast_db_name,
        bool force_cast,
        bool direct_call,
        mi::Sint32* errors) const = 0;

    /// Serializes an expression to \p serializer.
    virtual void serialize( SERIAL::Serializer* serializer, const IExpression* expr) const = 0;

    /// Deserializes an expression from \p deserializer.
    virtual IExpression* deserialize( SERIAL::Deserializer* deserializer) const = 0;

    /// Deserializes an expression from \p deserializer.
    template <class T>
    T* deserialize( SERIAL::Deserializer* deserializer) const
    {
        mi::base::Handle<IExpression> ptr_expr( deserialize( deserializer));
        if( !ptr_expr)
            return nullptr;
        return static_cast<T*>( ptr_expr->get_interface( typename T::IID()));
    }

    /// Serializes an expression list to \p serializer.
    virtual void serialize_list(
        SERIAL::Serializer* serializer, const IExpression_list* list) const = 0;

    /// Deserializes an expression list from \p deserializer.
    virtual IExpression_list* deserialize_list( SERIAL::Deserializer* deserializer) const = 0;

    /// Serializes an annotation to \p serializer.
    virtual void serialize_annotation(
        SERIAL::Serializer* serializer, const IAnnotation* annotation) const = 0;

    /// Deserializes an annotation from \p deserializer.
    virtual IAnnotation* deserialize_annotation(
        SERIAL::Deserializer* deserializer) const = 0;

    /// Serializes an annotation block to \p serializer.
    virtual void serialize_annotation_block(
        SERIAL::Serializer* serializer, const IAnnotation_block* block) const = 0;

    /// Deserializes an annotation block from \p deserializer.
    virtual IAnnotation_block* deserialize_annotation_block(
        SERIAL::Deserializer* deserializer) const = 0;

    /// Serializes an annotation list to \p serializer.
    virtual void serialize_annotation_list(
        SERIAL::Serializer* serializer, const IAnnotation_list* list) const = 0;

    /// Deserializes an annotation list from \p deserializer.
    virtual IAnnotation_list* deserialize_annotation_list(
        SERIAL::Deserializer* deserializer) const = 0;

    /// Serializes an annotation definition list to \p serializer.
    virtual void serialize_annotation_definition_list(
        SERIAL::Serializer* serializer, const IAnnotation_definition_list* anno_def_list) const = 0;

    /// Deserializes an annotation definition list from \p deserializer.
    virtual IAnnotation_definition_list* deserialize_annotation_definition_list(
        SERIAL::Deserializer* deserializer) const = 0;
};

/// Returns the global expression factory.
IExpression_factory* get_expression_factory();

// See base/lib/mem/i_mem_consumption.h
inline bool has_dynamic_memory_consumption( const IExpression*) { return true; }
inline size_t dynamic_memory_consumption( const IExpression* e)
{ return e->get_memory_consumption(); }

inline bool has_dynamic_memory_consumption( const IExpression_list*) { return true; }
inline size_t dynamic_memory_consumption( const IExpression_list* l)
{ return l->get_memory_consumption(); }

inline bool has_dynamic_memory_consumption( const IAnnotation*) { return true; }
inline size_t dynamic_memory_consumption( const IAnnotation* a)
{ return a->get_memory_consumption(); }

inline bool has_dynamic_memory_consumption(const IAnnotation_definition*) { return true; }
inline size_t dynamic_memory_consumption(const IAnnotation_definition* a)
{
    return a->get_memory_consumption();
}

inline bool has_dynamic_memory_consumption( const IAnnotation_block*) { return true; }
inline size_t dynamic_memory_consumption( const IAnnotation_block* b)
{ return b->get_memory_consumption(); }

inline bool has_dynamic_memory_consumption( const IAnnotation_list*) { return true; }
inline size_t dynamic_memory_consumption( const IAnnotation_list* l)
{ return l->get_memory_consumption(); }

inline bool has_dynamic_memory_consumption(const IAnnotation_definition_list*) { return true; }
inline size_t dynamic_memory_consumption(const IAnnotation_definition_list* l)
{
    return l->get_memory_consumption();
}

}  // namespace MDL

}  // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_EXPRESSION_H
