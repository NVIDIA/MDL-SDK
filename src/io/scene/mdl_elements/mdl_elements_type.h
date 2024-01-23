/***************************************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Header for the IType hierarchy and IType_factory implementation.

#ifndef IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_TYPE_H
#define IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_TYPE_H

#include "i_mdl_elements_type.h"

#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>

#include <shared_mutex>
#include <vector>

#include <base/lib/log/i_log_assert.h>
#include <base/lib/robin_hood/robin_hood.h>

// see documentation of mi::base::Interface_merger
#include <mi/base/config.h>
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4505 )
#endif

namespace MI {

namespace MDL {

class Type_list : public mi::base::Interface_implement<IType_list>
{
public:
    Type_list( mi::Size initial_capacity);

    // public API methods

    mi::Size get_size() const override;

    mi::Size get_index( const char* name) const override;

    const char* get_name( mi::Size index) const override;

    const IType* get_type( mi::Size index) const override;

    const IType* get_type( const char* name) const override;

    mi::Sint32 set_type( mi::Size index, const IType* type) override;

    mi::Sint32 set_type( const char* name, const IType* type) override;

    mi::Sint32 add_type( const char* name, const IType* type) override;

    // internal methods

    void add_type_unchecked( const char* name, const IType* type) override;

    mi::Size get_memory_consumption() const override;

    friend class Type_factory; // for serialization/deserialization

private:

    using Index_name_vector = std::vector<std::string>;
    Index_name_vector m_index_name;

    using Types_vector = std::vector<mi::base::Handle<const IType> >;
    Types_vector m_types;
};


class Type_factory : public mi::base::Interface_implement<IType_factory>
{
public:
    // public API methods

    const IType_alias* create_alias(
        const IType* type, mi::Uint32 modifiers, const char* symbol) const override;

    const IType_bool* create_bool() const override;

    const IType_int* create_int() const override;

    const IType_enum* create_enum( const char* symbol) const override;

    const IType_float* create_float() const override;

    const IType_double* create_double() const override;

    const IType_string* create_string() const override;

    const IType_vector* create_vector(
        const IType_atomic* element_type, mi::Size size) const override;

    const IType_matrix* create_matrix(
        const IType_vector* column_type, mi::Size columns) const override;

    const IType_color* create_color() const override;

    const IType_array* create_immediate_sized_array(
        const IType* element_type, mi::Size size) const override;

    const IType_array* create_deferred_sized_array(
        const IType* element_type, const char* size) const override;

    const IType_struct* create_struct( const char* symbol) const override;

    const IType_texture* create_texture( IType_texture::Shape shape) const override;

    const IType_light_profile* create_light_profile() const override;

    const IType_bsdf_measurement* create_bsdf_measurement() const override;

    const IType_bsdf* create_bsdf() const override;

    const IType_hair_bsdf* create_hair_bsdf() const override;

    const IType_edf* create_edf() const override;

    const IType_vdf* create_vdf() const override;

    IType_list* create_type_list( mi::Size initial_capacity) const override;

    const IType_enum* get_predefined_enum( IType_enum::Predefined_id id) const override;

    const IType_struct* get_predefined_struct(
        IType_struct::Predefined_id id) const override;

    IType_list* clone( const IType_list* list) const override;

    mi::Sint32 compare( const IType* lhs, const IType* rhs) const override;

    mi::Sint32 compare( const IType_list* lhs, const IType_list* rhs) const override;

    mi::Sint32 is_compatible(const IType* src, const IType* dst) const override;

    const mi::IString* dump( const IType* type, mi::Size depth = 0) const override;

    const mi::IString* dump( const IType_list* list, mi::Size depth = 0) const override;

    std::string get_mdl_type_name( const IType* type) const override;

    const IType* create_from_mdl_type_name( const char* name) const  override;

    const IType_enum* create_enum(
        const char* symbol,
        IType_enum::Predefined_id id,
        const IType_enum::Values& values,
        mi::base::Handle<const IAnnotation_block>& annotations,
        const IType_enum::Value_annotations& value_annotations,
        mi::Sint32* errors) override;

    const IType_struct* create_struct(
        const char* symbol,
        IType_struct::Predefined_id id,
        const IType_struct::Fields& fields,
        mi::base::Handle<const IAnnotation_block>& annotations,
        const IType_struct::Field_annotations& field_annotations,
        mi::Sint32* errors) override;

    void serialize( SERIAL::Serializer* serializer, const IType* type) const override;

    const IType* deserialize( SERIAL::Deserializer* deserializer) override;

    using IType_factory::deserialize;

    void serialize_list( SERIAL::Serializer* serializer, const IType_list* list) const override;

    IType_list* deserialize_list( SERIAL::Deserializer* deserializer) override;

    // internal methods

    static mi::Sint32 compare_static( const IType* lhs, const IType* rhs);

    static mi::Sint32 compare_static( const IType_list* lhs, const IType_list* rhs);

    /// Returns a type name suitable for the dump() methods.
    ///
    /// Similar to #get_mdl_type_name(), but more verbose for aliases, enums, and structs.
    ///
    /// This factory typically calls it with \p include_aliased_type set to \c true for full
    /// details, whereas the value factory typically calls it with \p include_aliased_type set to
    /// \c false for less details.
    static std::string get_dump_type_name( const IType* type, bool include_aliased_type);

    /// Acquires the mutex.
    void lock() const { m_mutex.lock(); }

    /// Releases the mutex.
    void unlock() const { m_mutex.unlock(); }

    /// Caller needs to hold the mutex.
    void unregister_enum_type( const IType_enum* type);

    /// Caller needs to hold the mutex.
    void unregister_struct_type( const IType_struct* type);

private:

    static std::string get_mdl_type_name_static( const IType* type);

    static mi::Sint32 compare_static( const IType_alias* lhs, const IType_alias* rhs);

    static mi::Sint32 compare_static( const IType_compound* lhs, const IType_compound* rhs);

    static mi::Sint32 compare_static( const IType_array* lhs, const IType_array* rhs);

    static mi::Sint32 compare_static( const IType_texture* lhs, const IType_texture* rhs);

    static void dump( const IType* type, mi::Size depth, std::ostringstream& s);

    static void dump( const IType_list* list, mi::Size depth, std::ostringstream& s);


    /// Performs the checks for create_enum() that need to happen under the lock.
    const IType_enum* lookup_enum(
        const char* symbol,
        IType_enum::Predefined_id id,
        const IType_enum::Values& values,
        mi::Sint32* errors);

    /// Performs the checks for create_struct() that need to happen under the lock.
    const IType_struct* lookup_struct(
        const char* symbol,
        IType_struct::Predefined_id id,
        const IType_struct::Fields& fields,
        mi::Sint32* errors);


    /// Checks whether \p type and (\p id, \p values) are equivalent types (ignoring annotations).
    static bool equivalent_enum_types(
        const IType_enum* type,
        IType_enum::Predefined_id id,
        const IType_enum::Values& values);

    /// Checks whether \p type and (\p id, \p fields) are equivalent types (ignoring annotations).
    static bool equivalent_struct_types(
        const IType_struct* type,
        IType_struct::Predefined_id id,
        const IType_struct::Fields& fields);


    using Weak_enum_symbol_map
        = robin_hood::unordered_map<std::string, const IType_enum*>;

    using Weak_enum_id_map
        = robin_hood::unordered_map<IType_enum::Predefined_id, const IType_enum*>;

    using Weak_struct_symbol_map
        = robin_hood::unordered_map<std::string, const IType_struct*>;

    using Weak_struct_id_map
        = robin_hood::unordered_map<IType_struct::Predefined_id, const IType_struct*>;


    /// Mutex for the four weak map members below.
    mutable std::shared_mutex m_mutex;

    /// All registered enum types by symbol. Needs #m_mutex.
    Weak_enum_symbol_map m_enum_symbols;

    /// All registered enum types by ID. Needs #m_mutex.
    Weak_enum_id_map m_enum_ids;

    /// All registered struct types by symbol. Needs #m_mutex.
    Weak_struct_symbol_map m_struct_symbols;

    /// All registered struct types by ID. Needs #m_mutex.
    Weak_struct_id_map m_struct_ids;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_TYPE_H
