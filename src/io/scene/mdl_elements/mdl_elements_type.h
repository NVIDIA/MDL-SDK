/***************************************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_TYPE_IMPL_H
#define IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_TYPE_IMPL_H

#include "i_mdl_elements_type.h"

#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>

#include <map>
#include <vector>

#include <base/lib/log/i_log_assert.h>
#include <base/system/main/neuray_cc_conf.h>

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
    // public API methods

    mi::Size get_size() const NEURAY_OVERRIDE;

    mi::Size get_index( const char* name) const NEURAY_OVERRIDE;

    const char* get_name( mi::Size index) const NEURAY_OVERRIDE;

    const IType* get_type( mi::Size index) const NEURAY_OVERRIDE;

    const IType* get_type( const char* name) const NEURAY_OVERRIDE;

    mi::Sint32 set_type( mi::Size index, const IType* type) NEURAY_OVERRIDE;

    mi::Sint32 set_type( const char* name, const IType* type) NEURAY_OVERRIDE;

    mi::Sint32 add_type( const char* name, const IType* type) NEURAY_OVERRIDE;

    mi::Size get_memory_consumption() const NEURAY_OVERRIDE;

    friend class Type_factory; // for serialization/deserialization

private:

    typedef std::map<std::string, mi::Size> Name_index_map;
    Name_index_map m_name_index;

    typedef std::vector<std::string> Index_name_vector;
    Index_name_vector m_index_name;

    typedef std::vector<mi::base::Handle<const IType> > Types_vector;
    Types_vector m_types;
};


class Type_factory : public mi::base::Interface_implement<IType_factory>
{
public:
    // public API methods

    const IType_alias* create_alias(
        const IType* type, mi::Uint32 modifiers, const char* symbol) const NEURAY_OVERRIDE;

    const IType_bool* create_bool() const NEURAY_OVERRIDE;

    const IType_int* create_int() const NEURAY_OVERRIDE;

    const IType_enum* create_enum( const char* symbol) const NEURAY_OVERRIDE;

    const IType_float* create_float() const NEURAY_OVERRIDE;

    const IType_double* create_double() const NEURAY_OVERRIDE;

    const IType_string* create_string() const NEURAY_OVERRIDE;

    const IType_vector* create_vector(
        const IType_atomic* element_type, mi::Size size) const NEURAY_OVERRIDE;

    const IType_matrix* create_matrix(
        const IType_vector* column_type, mi::Size columns) const NEURAY_OVERRIDE;

    const IType_color* create_color() const NEURAY_OVERRIDE;

    const IType_array* create_immediate_sized_array(
        const IType* element_type, mi::Size size) const NEURAY_OVERRIDE;

    const IType_array* create_deferred_sized_array(
        const IType* element_type, const char* size) const NEURAY_OVERRIDE;

    const IType_struct* create_struct( const char* symbol) const NEURAY_OVERRIDE;

    const IType_texture* create_texture( IType_texture::Shape shape) const NEURAY_OVERRIDE;

    const IType_light_profile* create_light_profile() const NEURAY_OVERRIDE;

    const IType_bsdf_measurement* create_bsdf_measurement() const NEURAY_OVERRIDE;

    const IType_bsdf* create_bsdf() const NEURAY_OVERRIDE;

    const IType_hair_bsdf* create_hair_bsdf() const NEURAY_OVERRIDE;

    const IType_edf* create_edf() const NEURAY_OVERRIDE;

    const IType_vdf* create_vdf() const NEURAY_OVERRIDE;

    IType_list* create_type_list() const NEURAY_OVERRIDE;

    const IType_enum* get_predefined_enum( IType_enum::Predefined_id id) const NEURAY_OVERRIDE;

    const IType_struct* get_predefined_struct(
        IType_struct::Predefined_id id) const NEURAY_OVERRIDE;

    mi::Sint32 compare( const IType* lhs, const IType* rhs) const NEURAY_OVERRIDE;

    mi::Sint32 compare( const IType_list* lhs, const IType_list* rhs) const NEURAY_OVERRIDE;

    mi::Sint32 is_compatible(const IType* src, const IType* dst) const NEURAY_OVERRIDE;

    const mi::IString* dump( const IType* type, mi::Size depth = 0) const NEURAY_OVERRIDE;

    const mi::IString* dump( const IType_list* list, mi::Size depth = 0) const NEURAY_OVERRIDE;

    const IType_enum* create_enum(
        const char* symbol,
        IType_enum::Predefined_id id,
        const IType_enum::Values& values,
        mi::base::Handle<const IAnnotation_block>& annotations,
        const IType_enum::Value_annotations& value_annotations,
        mi::Sint32* errors) NEURAY_OVERRIDE;

    const IType_struct* create_struct(
        const char* symbol,
        IType_struct::Predefined_id id,
        const IType_struct::Fields& fields,
        mi::base::Handle<const IAnnotation_block>& annotations,
        const IType_struct::Field_annotations& field_annotations,
        mi::Sint32* errors) NEURAY_OVERRIDE;

    void serialize( SERIAL::Serializer* serializer, const IType* type) const NEURAY_OVERRIDE;

    const IType* deserialize( SERIAL::Deserializer* deserializer) NEURAY_OVERRIDE;

    using IType_factory::deserialize;

    void serialize_list(
        SERIAL::Serializer* serializer,
        const IType_list* list) const NEURAY_OVERRIDE;

    IType_list* deserialize_list( SERIAL::Deserializer* deserializer) NEURAY_OVERRIDE;

    // internal methods

    static mi::Sint32 compare_static( const IType* lhs, const IType* rhs);

    static mi::Sint32 compare_static( const IType_list* lhs, const IType_list* rhs);

    static std::string get_type_name( const IType* type, bool include_aliased_type = true);

    mi::Uint32 destroy_enum_type( const IType_enum* e_type);

    mi::Uint32 destroy_struct_type( const IType_struct* s_type);

private:

    static mi::Sint32 compare_static( const IType_alias* lhs, const IType_alias* rhs);

    static mi::Sint32 compare_static( const IType_compound* lhs, const IType_compound* rhs);

    static mi::Sint32 compare_static( const IType_array* lhs, const IType_array* rhs);

    static mi::Sint32 compare_static( const IType_texture* lhs, const IType_texture* rhs);

    static void dump_static( const IType* type, mi::Size depth, std::ostringstream& s);

    static void dump_static( const IType_list* list, mi::Size depth, std::ostringstream& s);


    typedef std::map<std::string, const IType_enum *> Weak_enum_symbol_map;

    typedef std::map<IType_enum::Predefined_id, const IType_enum *> Weak_enum_id_map;

    typedef std::map<std::string, const IType_struct *> Weak_struct_symbol_map;

    typedef std::map<IType_struct::Predefined_id, const IType_struct *> Weak_struct_id_map;

    /// Lock for the four weak map members below.
    mutable mi::base::Lock m_weak_map_lock;

    /// All registered enum types by symbol. Needs #m_lock.
    Weak_enum_symbol_map m_enum_symbols;

    /// All registered enum types by ID. Needs #m_lock.
    Weak_enum_id_map m_enum_ids;

    /// All registered struct types by symbol. Needs #m_lock.
    Weak_struct_symbol_map m_struct_symbols;

    /// All registered struct types by ID. Needs #m_lock.
    Weak_struct_id_map m_struct_ids;
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_TYPE_IMPL_H
