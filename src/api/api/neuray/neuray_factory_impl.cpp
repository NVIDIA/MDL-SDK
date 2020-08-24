/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of IFactory
 **
 ** Implements the IFactory interface
 **/

#include "pch.h"

#include "neuray_array_impl.h"
#include "neuray_class_factory.h"
#include "neuray_factory_impl.h"
#include "neuray_map_impl.h"
#include "neuray_pointer_impl.h"
#include "neuray_ref_impl.h"
#include "neuray_structure_impl.h"
#include "neuray_type_utilities.h"


#include <mi/base/handle.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/idata.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/imatrix.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/iuuid.h>
#include <mi/neuraylib/ivector.h>

#include <mi/neuraylib/ibbox.h>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/ispectrum.h>
#include "neuray_expression_impl.h"
#include "neuray_type_impl.h"
#include "neuray_value_impl.h"

#include <sstream>
#include <base/system/stlext/i_stlext_likely.h>
#include <boost/core/ignore_unused.hpp>
#include <base/util/string_utils/i_string_lexicographic_cast.h>

namespace MI {

namespace NEURAY {

mi::neuraylib::IFactory* s_factory = nullptr;

Factory_impl::Factory_impl( Class_factory* class_factory)
  : m_class_factory( class_factory)
{
}

Factory_impl::~Factory_impl()
{
    m_class_factory = nullptr;
}

mi::base::IInterface* Factory_impl::create(
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    return m_class_factory->create_type_instance( nullptr, type_name, argc, argv);
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IData* source, mi::IData* target, mi::Uint32 options)
{
    if( !source || !target)
        return NULL_POINTER;

    bool adjust_target_keys = (options & 2 /* ADJUST_LENGTH_OF_DYNAMIC_ARRAYS */) != 0;
    bool fix_target_keys    = (options & FIX_SET_OF_TARGET_KEYS) != 0;
    if( adjust_target_keys && fix_target_keys)
        return INCOMPATIBLE_OPTIONS;

    // handle IData_simple
    mi::base::Handle<const mi::IData_simple> source_simple(
        source->get_interface<mi::IData_simple>());
    mi::base::Handle<mi::IData_simple> target_simple(
        target->get_interface<mi::IData_simple>());
    if( source_simple.is_valid_interface() && target_simple.is_valid_interface())
        return assign_from_to( source_simple.get(), target_simple.get(), options);

    // handle IData_collection
    mi::base::Handle<const mi::IData_collection> source_collection(
        source->get_interface<mi::IData_collection>());
    mi::base::Handle<mi::IData_collection> target_collection(
        target->get_interface<mi::IData_collection>());
    if( source_collection.is_valid_interface() && target_collection.is_valid_interface())
        return assign_from_to( source_collection.get(), target_collection.get(), options);

    return STRUCTURAL_MISMATCH;
}

mi::IData* Factory_impl::clone( const mi::IData* source, mi::Uint32 options)
{
    if( !source)
        return nullptr;

    // handle IData_simple
    mi::base::Handle<const mi::IData_simple> source_simple(
        source->get_interface<mi::IData_simple>());
    if( source_simple.is_valid_interface())
        return clone( source_simple.get(), options);

    // handle IData_collection
    mi::base::Handle<const mi::IData_collection> source_collection(
        source->get_interface<mi::IData_collection>());
    if( source_collection.is_valid_interface())
        return clone( source_collection.get(), options);

    ASSERT( M_NEURAY_API, false);
    return nullptr;
}

mi::Sint32 Factory_impl::compare( const mi::IData* lhs, const mi::IData* rhs)
{
    if( !lhs && !rhs) return 0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_NEURAY_API, lhs && rhs);

    const char* lhs_type = lhs->get_type_name(); //-V522 PVS
    const char* rhs_type = rhs->get_type_name(); //-V522 PVS
    int type_cmp = strcmp( lhs_type, rhs_type);
    if( type_cmp != 0)
    	return type_cmp;

    // handle IData_simple
    mi::base::Handle<const mi::IData_simple> lhs_simple(
        lhs->get_interface<mi::IData_simple>());
    mi::base::Handle<const mi::IData_simple> rhs_simple(
        rhs->get_interface<mi::IData_simple>());
    if( lhs_simple.is_valid_interface() && rhs_simple.is_valid_interface())
        return compare( lhs_simple.get(), rhs_simple.get());

    // handle IData_collection
    mi::base::Handle<const mi::IData_collection> lhs_collection(
        lhs->get_interface<mi::IData_collection>());
    mi::base::Handle<const mi::IData_collection> rhs_collection(
        rhs->get_interface<mi::IData_collection>());
    if( lhs_collection.is_valid_interface() && rhs_collection.is_valid_interface())
        return compare( lhs_collection.get(), rhs_collection.get());

    ASSERT( M_NEURAY_API, false);
    return 0;
}

const mi::IString* Factory_impl::dump( const mi::IData* data, const char* name, mi::Size depth)
{
    if( !data)
        return nullptr;

    std::ostringstream s;
    dump( /*transaction*/ nullptr, name, data, depth, s);
    mi::IString* result = create<mi::IString>( "String");
    result->set_c_str( s.str().c_str());
    return result;
}

const mi::IString* Factory_impl::dump(
    mi::neuraylib::ITransaction* transaction,
    const mi::IData* data,
    const char* name,
    mi::Size depth)
{
    if( !data)
        return nullptr;

    std::ostringstream s;
    dump( transaction, name, data, depth, s);
    mi::IString* result = create<mi::IString>( "String");
    result->set_c_str( s.str().c_str());
    return result;
}

const mi::IStructure_decl* Factory_impl::get_structure_decl( const char* structure_name) const
{
   if( !structure_name)
        return nullptr;

    return m_class_factory->get_structure_decl( structure_name);
}

const mi::IEnum_decl* Factory_impl::get_enum_decl( const char* enum_name) const
{
   if( !enum_name)
        return nullptr;

    return m_class_factory->get_enum_decl( enum_name);
}

mi::Sint32 Factory_impl::start()
{
    return 0;
}

mi::Sint32 Factory_impl::shutdown()
{
    return 0;
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IData_simple* source, mi::IData_simple* target, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source && target);

    // handle INumber
    mi::base::Handle<const mi::INumber> source_value(
        source->get_interface<mi::INumber>());
    mi::base::Handle<mi::INumber> target_value(
        target->get_interface<mi::INumber>());
    if( source_value.is_valid_interface() && target_value.is_valid_interface())
        return assign_from_to( source_value.get(), target_value.get());

    // handle IString
    mi::base::Handle<const mi::IString> source_string(
        source->get_interface<mi::IString>());
    mi::base::Handle<mi::IString> target_string(
        target->get_interface<mi::IString>());
    if( source_string.is_valid_interface() && target_string.is_valid_interface())
        return assign_from_to( source_string.get(), target_string.get());

    // handle IRef
    mi::base::Handle<const mi::IRef> source_ref(
        source->get_interface<mi::IRef>());
    mi::base::Handle<mi::IRef> target_ref(
        target->get_interface<mi::IRef>());
    if( source_ref.is_valid_interface() && target_ref.is_valid_interface())
        return assign_from_to( source_ref.get(), target_ref.get());

    // handle IEnum/IEnum
    mi::base::Handle<const mi::IEnum> source_enum(
        source->get_interface<mi::IEnum>());
    mi::base::Handle<mi::IEnum> target_enum(
        target->get_interface<mi::IEnum>());
    if( source_enum.is_valid_interface() && target_enum.is_valid_interface())
        return assign_from_to( source_enum.get(), target_enum.get(), options);

    // handle IEnum/ISint32
    mi::base::Handle<mi::ISint32> target_sint32( target->get_interface<mi::ISint32>());
    if( source_enum.is_valid_interface() && target_sint32.is_valid_interface())
        return assign_from_to( source_enum.get(), target_sint32.get(), options);

    // handle IUuid
    mi::base::Handle<const mi::IUuid> source_uuid(
        source->get_interface<mi::IUuid>());
    mi::base::Handle<mi::IUuid> target_uuid(
        target->get_interface<mi::IUuid>());
    if( source_uuid.is_valid_interface() && target_uuid.is_valid_interface())
        return assign_from_to( source_uuid.get(), target_uuid.get());

    // handle IVoid
    mi::base::Handle<const mi::IVoid> source_void(
        source->get_interface<mi::IVoid>());
    mi::base::Handle<mi::IVoid> target_void(
        target->get_interface<mi::IVoid>());
    if( source_void.is_valid_interface() && target_void.is_valid_interface())
        return 0; // nothing to do

    // handle IPointer/IPointer
    mi::base::Handle<const mi::IPointer> source_pointer(
        source->get_interface<mi::IPointer>());
    mi::base::Handle<mi::IPointer> target_pointer(
        target->get_interface<mi::IPointer>());
    if( source_pointer.is_valid_interface() && target_pointer.is_valid_interface())
        return assign_from_to( source_pointer.get(), target_pointer.get(), options);

    // handle IConst_pointer/IConst_pointer
    mi::base::Handle<const mi::IConst_pointer> source_const_pointer(
        source->get_interface<mi::IConst_pointer>());
    mi::base::Handle<mi::IConst_pointer> target_const_pointer(
        target->get_interface<mi::IConst_pointer>());
    if( source_const_pointer.is_valid_interface() && target_const_pointer.is_valid_interface())
        return assign_from_to( source_const_pointer.get(), target_const_pointer.get(), options);

    // handle IConst_pointer/IPointer
    if( source_const_pointer.is_valid_interface() && target_pointer.is_valid_interface())
        return assign_from_to( source_const_pointer.get(), target_pointer.get(), options);

    // handle IPointer/IConst_pointer
    if( source_pointer.is_valid_interface() && target_const_pointer.is_valid_interface())
        return assign_from_to( source_pointer.get(), target_const_pointer.get(), options);

    return NO_CONVERSION;
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IData_collection* source, mi::IData_collection* target, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source && target);

    mi::Uint32 result = 0;

    // check that both or none arguments are of type ICompound
    mi::base::Handle<const mi::ICompound> source_compound(
        source->get_interface<mi::ICompound>());
    mi::base::Handle<mi::ICompound> target_compound(
        target->get_interface<mi::ICompound>());
    if( source_compound.is_valid_interface() ^ target_compound.is_valid_interface())
        result |= DIFFERENT_COLLECTIONS;

    // check that both or none arguments are of type IDynamic_array
    mi::base::Handle<const mi::IDynamic_array> source_dynamic_array(
        source->get_interface<mi::IDynamic_array>());
    mi::base::Handle<mi::IDynamic_array> target_dynamic_array(
        target->get_interface<mi::IDynamic_array>());
    if( source_dynamic_array.is_valid_interface() ^ target_dynamic_array.is_valid_interface())
        result |= DIFFERENT_COLLECTIONS;

    // check that both or none arguments are of type IArray
    mi::base::Handle<const mi::IArray> source_array(
        source->get_interface<mi::IArray>());
    mi::base::Handle<mi::IArray> target_array(
        target->get_interface<mi::IArray>());
    if( source_array.is_valid_interface() ^ target_array.is_valid_interface())
        result |= DIFFERENT_COLLECTIONS;

    // check that both or none arguments are of type IStructure
    mi::base::Handle<const mi::IStructure> source_structure(
        source->get_interface<mi::IStructure>());
    mi::base::Handle<mi::IStructure> target_structure(
        target->get_interface<mi::IStructure>());
    if( source_structure.is_valid_interface() ^ target_structure.is_valid_interface())
        result |= DIFFERENT_COLLECTIONS;

    // check that both or none arguments are of type IMap
    mi::base::Handle<const mi::IMap> source_map(
        source->get_interface<mi::IMap>());
    mi::base::Handle<mi::IMap> target_map(
        target->get_interface<mi::IMap>());
    if( source_map.is_valid_interface() ^ target_map.is_valid_interface())
        result |= DIFFERENT_COLLECTIONS;

    // adjust length of target if it is a dynamic array or map (unless disabled by option)
    if( !(options & FIX_SET_OF_TARGET_KEYS)) {

        if( target_dynamic_array.is_valid_interface()) {

            if( source_array.is_valid_interface()) {
                mi::Size new_length = source_array->get_length();
                target_dynamic_array->set_length( new_length);
            } else {
                // find highest index-like key in source
                mi::Size max_index_plus_one = 0;
                mi::Size n = source->get_length();
                for( mi::Size i = 0; i < n; ++i) {
                    const char* key = source->get_key( i);
                    STLEXT::Likely<mi::Size> index_likely
                        = STRING::lexicographic_cast_s<mi::Size>( key);
                    if( !index_likely.get_status())
                        continue;
                    mi::Size index = *index_likely.get_ptr(); //-V522 PVS
                    if( index+1 > max_index_plus_one)
                        max_index_plus_one = index+1;
                }
                target_dynamic_array->set_length( max_index_plus_one);
            }
        }

        if( target_map.is_valid_interface()) {

            // remove target keys not in source
            mi::Size n = target_map->get_length();
            for( mi::Size i = 0; i < n; ) {
                const char* key = target_map->get_key( i);
                if( source->has_key( key))
                    ++i;
                else {
                    target_map->erase( key);
                    --n;
                }
            }

            std::string target_type_name = target->get_type_name();
            std::string target_element_name = Type_utilities::strip_map( target_type_name);
            mi::base::Handle<mi::neuraylib::ITransaction> transaction( get_transaction( target));

            // insert source keys not in target
            n = source->get_length();
            for( mi::Size i = 0; i < n; ++i) {

                const char* key = source->get_key( i);
                if( target_map->has_key( key))
                    continue;

                // for untyped maps use the type name of the source element
                std::string element_name = target_element_name;
                if( element_name == "Interface") {
                    mi::base::Handle<const mi::IData> source_value_data(
                        source->get_value<mi::IData>( key));
                    if( source_value_data.is_valid_interface())
                        element_name = source_value_data->get_type_name();
                }

                mi::base::Handle<mi::base::IInterface> target_value(
                    transaction.get() ? transaction->create( element_name.c_str())
                                      : create( element_name.c_str()));
                if( !target_value.is_valid_interface())
                    continue;

                mi::Sint32 result2 = target_map->insert( key, target_value.get());
                ASSERT( M_NEURAY_API, result2 == 0);
                boost::ignore_unused( result2);
            }
        }
    }

    // iterate over keys in source, and try to assign to the corresponding key in target
    mi::Size keys_found_in_target   = 0;
    mi::Size keys_missing_in_target = 0;

    mi::Size n = source->get_length();
    for( mi::Size i = 0; i < n; ++i) {

        // get i-th key from source and check whether target has that key
        const char* key = source->get_key( i);
        if( !target->has_key( key)) {
            ++keys_missing_in_target;
            continue;
        }
        ++keys_found_in_target;

        // get value from source
        mi::base::Handle<const mi::base::IInterface> source_value_interface(
            source->get_value( key));
        ASSERT(  M_NEURAY_API, source_value_interface.is_valid_interface());

        // check if source value is of type IData
        mi::base::Handle<const mi::IData> source_value_data(
            source_value_interface->get_interface<mi::IData>());
        if( !source_value_data.is_valid_interface()) {
            result |= NON_IDATA_VALUES;
            continue;
        }

        // get value from target
        mi::base::Handle<mi::base::IInterface> target_value_interface(
            target->get_value( key));
        ASSERT(  M_NEURAY_API, source_value_interface.is_valid_interface());

        // check if target value is of type IData
        mi::base::Handle<mi::IData> target_value_data(
            target_value_interface->get_interface<mi::IData>());
        if( !target_value_data.is_valid_interface()){
            result |= NON_IDATA_VALUES;
            continue;
        }

        source_value_interface = nullptr;
        target_value_interface = nullptr;

        // invoke assign_from_to() for this key, and set the target value for this key again
        result |= assign_from_to( source_value_data.get(), target_value_data.get(), options);
        mi::Uint32 result2 = target->set_value( key, target_value_data.get());
        ASSERT( M_NEURAY_API, result2 == 0);
        boost::ignore_unused( result2);
    }

    if( keys_missing_in_target > 0)
        result |= TARGET_KEY_MISSING;
    if( target->get_length() > keys_found_in_target)
        result |= SOURCE_KEY_MISSING;

    return result;
}

mi::Uint32 Factory_impl::assign_from_to( const mi::INumber* source, mi::INumber* target)
{
    ASSERT( M_NEURAY_API, source && target);

    std::string target_type_name = target->get_type_name();

    if( target_type_name == "Boolean")
        target->set_value( source->get_value<bool>());
    else if( target_type_name == "Sint8")
        target->set_value( source->get_value<mi::Sint8>());
    else if( target_type_name == "Sint16")
        target->set_value( source->get_value<mi::Sint16>());
    else if( target_type_name == "Sint32")
        target->set_value( source->get_value<mi::Sint32>());
    else if( target_type_name == "Sint64")
        target->set_value( source->get_value<mi::Sint64>());
    else if( target_type_name == "Uint8")
        target->set_value( source->get_value<mi::Uint8>());
    else if( target_type_name == "Uint16")
        target->set_value( source->get_value<mi::Uint16>());
    else if( target_type_name == "Uint32")
        target->set_value( source->get_value<mi::Uint32>());
    else if( target_type_name == "Uint64")
        target->set_value( source->get_value<mi::Uint64>());
    else if( target_type_name == "Float32")
        target->set_value( source->get_value<mi::Float32>());
    else if( target_type_name == "Float64")
        target->set_value( source->get_value<mi::Float64>());
    else if( target_type_name == "Size")
        target->set_value( source->get_value<mi::Size>());
    else if( target_type_name == "Difference")
        target->set_value( source->get_value<mi::Difference>());
    else {
        ASSERT( M_NEURAY_API, false);
    }

    return 0;
}

mi::Uint32 Factory_impl::assign_from_to( const mi::IString* source, mi::IString* target)
{
    ASSERT( M_NEURAY_API, source && target);

    const char* s = source->get_c_str();
    target->set_c_str( s);
    return 0;
}

mi::Uint32 Factory_impl::assign_from_to( const mi::IRef* source, mi::IRef* target)
{
    ASSERT( M_NEURAY_API, source && target);

    mi::base::Handle<const mi::base::IInterface> reference( source->get_reference());
    mi::Sint32 result = target->set_reference( reference.get());
    if( result == -4)
        return INCOMPATIBLE_PRIVACY_LEVELS;

    return 0;
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IEnum* source, mi::IEnum* target, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source && target);

    if( strcmp( source->get_type_name(), target->get_type_name()) != 0)
        return INCOMPATIBLE_ENUM_TYPES;

    const char* name = source->get_value_by_name();
    mi::Uint32 result = target->set_value_by_name( name);
    ASSERT( M_NEURAY_API, result == 0);
    boost::ignore_unused( result);
    return 0;
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IEnum* source, mi::ISint32* target, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source && target);

    mi::Sint32 value = source->get_value();
    target->set_value( value);
    return 0;
}

mi::Uint32 Factory_impl::assign_from_to( const mi::IUuid* source, mi::IUuid* target)
{
    ASSERT( M_NEURAY_API, source && target);

    mi::base::Uuid uuid = source->get_uuid();
    target->set_uuid( uuid);
    return 0;
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IPointer* source, mi::IPointer* target, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source && target);

    // shallow assignment
    if( (options & DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        mi::base::Handle<mi::base::IInterface> pointer( source->get_pointer());
        mi::Uint32 result = target->set_pointer( pointer.get());
        return (result == 0) ? 0 : static_cast<mi::Uint32>( INCOMPATIBLE_POINTER_TYPES);
    }

    // deep assignment, source has NULL pointer
    mi::base::Handle<mi::base::IInterface> source_pointer( source->get_pointer());
    if( !source_pointer.is_valid_interface()) {
        target->set_pointer( nullptr);
        return 0;
    }

    // deep assignment, source has non-IData pointer
    mi::base::Handle<mi::IData> source_pointer_data( source->get_pointer<mi::IData>());
    if( !source_pointer_data.is_valid_interface())
        return NON_IDATA_VALUES;

    // deep assignment, target has NULL pointer
    mi::base::Handle<mi::base::IInterface> target_pointer( target->get_pointer());
    if( !target_pointer.is_valid_interface())
        return NULL_POINTER;

    // deep assignment, source has non-IData pointer
    mi::base::Handle<mi::IData> target_pointer_data( target->get_pointer<mi::IData>());
    if( !target_pointer_data.is_valid_interface())
        return NON_IDATA_VALUES;

    // deep assignment
    return assign_from_to( source_pointer_data.get(), target_pointer_data.get(), options);
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IConst_pointer* source, mi::IPointer* target, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source && target);

    // shallow assignment
    if( (options & DEEP_ASSIGNMENT_OR_CLONE) == 0)
        return INCOMPATIBLE_POINTER_TYPES;

    // deep assignment, source has NULL pointer
    mi::base::Handle<const mi::base::IInterface> source_pointer( source->get_pointer());
    if( !source_pointer.is_valid_interface()) {
        target->set_pointer( nullptr);
        return 0;
    }

    // deep assignment, source has non-IData pointer
    mi::base::Handle<const mi::IData> source_pointer_data( source->get_pointer<mi::IData>());
    if( !source_pointer_data.is_valid_interface())
        return NON_IDATA_VALUES;

    // deep assignment, target has NULL pointer
    mi::base::Handle<mi::base::IInterface> target_pointer( target->get_pointer());
    if( !target_pointer.is_valid_interface())
        return NULL_POINTER;

    // deep assignment, source has non-IData pointer
    mi::base::Handle<mi::IData> target_pointer_data( target->get_pointer<mi::IData>());
    if( !target_pointer_data.is_valid_interface())
        return NON_IDATA_VALUES;

    // deep assignment
    return assign_from_to( source_pointer_data.get(), target_pointer_data.get(), options);
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IConst_pointer* source, mi::IConst_pointer* target, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source && target);

    // shallow assignment
    if( (options & DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        mi::base::Handle<const mi::base::IInterface> pointer( source->get_pointer());
        mi::Uint32 result = target->set_pointer( pointer.get());
        return (result == 0) ? 0 : static_cast<mi::Uint32>( INCOMPATIBLE_POINTER_TYPES);
    }

    // deep assignment
    return DEEP_ASSIGNMENT_TO_CONST_POINTER;
}

mi::Uint32 Factory_impl::assign_from_to(
    const mi::IPointer* source, mi::IConst_pointer* target, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source && target);

    // shallow assignment
    if( (options & DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        mi::base::Handle<mi::base::IInterface> pointer( source->get_pointer());
        mi::Uint32 result = target->set_pointer( pointer.get());
        return (result == 0) ? 0 : static_cast<mi::Uint32>( INCOMPATIBLE_POINTER_TYPES);
    }

    // deep assignment
    return DEEP_ASSIGNMENT_TO_CONST_POINTER;
}

mi::IData_simple* Factory_impl::clone( const mi::IData_simple* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    // handle IRef
    mi::base::Handle<const mi::IRef> source_ref(
        source->get_interface<mi::IRef>());
    if( source_ref.is_valid_interface())
        return clone( source_ref.get(), options);

    // handle IPointer
    mi::base::Handle<const mi::IPointer> source_pointer(
        source->get_interface<mi::IPointer>());
    if( source_pointer.is_valid_interface())
        return clone( source_pointer.get(), options);

    // handle IConst_pointer
    mi::base::Handle<const mi::IConst_pointer> source_const_pointer(
        source->get_interface<mi::IConst_pointer>());
    if( source_const_pointer.is_valid_interface())
        return clone( source_const_pointer.get(), options);

    // handle other subtypes of IData_simple
    mi::IData_simple* target = create<mi::IData_simple>( source->get_type_name());
    mi::Uint32 result = assign_from_to( source, target, options);
    ASSERT( M_NEURAY_API, result == 0);
    boost::ignore_unused( result);
    return target;
}

mi::IData_collection* Factory_impl::clone( const mi::IData_collection* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    // handle ICompound
    mi::base::Handle<const mi::ICompound> source_compound(
        source->get_interface<mi::ICompound>());
    if( source_compound.is_valid_interface())
        return clone( source_compound.get(), options);

    // handle IDynamic_array
    mi::base::Handle<const mi::IDynamic_array> source_dynamic_array(
        source->get_interface<mi::IDynamic_array>());
    if( source_dynamic_array.is_valid_interface())
        return clone( source_dynamic_array.get(), options);

    // handle IArray
    mi::base::Handle<const mi::IArray> source_array(
        source->get_interface<mi::IArray>());
    if( source_array.is_valid_interface())
        return clone( source_array.get(), options);

    // handle IStructure
    mi::base::Handle<const mi::IStructure> source_structure(
        source->get_interface<mi::IStructure>());
    if( source_structure.is_valid_interface())
        return clone( source_structure.get(), options);

    // handle IMap
    mi::base::Handle<const mi::IMap> source_map(
        source->get_interface<mi::IMap>());
    if( source_map.is_valid_interface())
        return clone( source_map.get(), options);

    ASSERT( M_NEURAY_API, false);
    return nullptr;
}

mi::IRef* Factory_impl::clone( const mi::IRef* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    mi::base::Handle<mi::neuraylib::ITransaction> transaction( get_transaction( source));
    mi::IRef* target = transaction->create<mi::IRef>( source->get_type_name());
    mi::Uint32 result = assign_from_to( source, target, options);
    ASSERT( M_NEURAY_API, result == 0);
    boost::ignore_unused( result);
    return target;
}

mi::IPointer* Factory_impl::clone( const mi::IPointer* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    const char* source_type_name = source->get_type_name();

    // shallow clone
    if( (options & DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        mi::IPointer* target = create_with_transaction<mi::IPointer>( source_type_name, source);
        if( !target)
            return nullptr;
        mi::Uint32 result = target->set_pointer( source->get_pointer());
        ASSERT( M_NEURAY_API, result == 0);
        boost::ignore_unused( result);
        return target;
    }

    // deep clone, NULL pointer
    mi::base::Handle<mi::base::IInterface> source_pointer( source->get_pointer());
    if( !source_pointer.is_valid_interface())
        return create<mi::IPointer>( source_type_name);

    // deep clone, non-NULL pointer
    mi::base::Handle<mi::IData> source_pointer_data( source->get_pointer<mi::IData>());
    if( !source_pointer_data.is_valid_interface())
        return nullptr;
    mi::base::Handle<mi::IData> target_pointer_data(
        clone( source_pointer_data.get(), options));
    if( !target_pointer_data.is_valid_interface())
        return nullptr;

    mi::IPointer* target = create_with_transaction<mi::IPointer>( source_type_name, source);
    if( !target)
        return nullptr;
    mi::Uint32 result = target->set_pointer( target_pointer_data.get());
    ASSERT( M_NEURAY_API, result == 0);
    boost::ignore_unused( result);
    return target;
}

mi::IConst_pointer* Factory_impl::clone( const mi::IConst_pointer* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    const char* source_type_name = source->get_type_name();

    // shallow clone
    if( (options & DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        mi::IConst_pointer* target
            = create_with_transaction<mi::IConst_pointer>( source_type_name, source);
        if( !target)
            return nullptr;
        mi::Uint32 result = target->set_pointer( source->get_pointer());
        ASSERT( M_NEURAY_API, result == 0);
        boost::ignore_unused( result);
        return target;
    }

    // deep clone, NULL pointer
    mi::base::Handle<const mi::base::IInterface> source_pointer( source->get_pointer());
    if( !source_pointer.is_valid_interface())
        return create<mi::IConst_pointer>( source_type_name);

    // deep clone, non-NULL pointer
    mi::base::Handle<const mi::IData> source_pointer_data( source->get_pointer<mi::IData>());
    if( !source_pointer_data.is_valid_interface())
        return nullptr;
    mi::base::Handle<const mi::IData> target_pointer_data(
        clone( source_pointer_data.get(), options));
    if( !target_pointer_data.is_valid_interface())
        return nullptr;

    mi::IConst_pointer* target
        = create_with_transaction<mi::IConst_pointer>( source_type_name, source);
    if( !target)
        return nullptr;
    mi::Uint32 result = target->set_pointer( target_pointer_data.get());
    ASSERT( M_NEURAY_API, result == 0);
    boost::ignore_unused( result);
    return target;
}

mi::ICompound* Factory_impl::clone( const mi::ICompound* source, mi::Uint32 options)
{
     ASSERT( M_NEURAY_API, source);

     mi::ICompound* target = create<mi::ICompound>( source->get_type_name());
     mi::Uint32 result = assign_from_to( source, target, options);
     ASSERT( M_NEURAY_API, result == 0);
     boost::ignore_unused( result);
     return target;
}

mi::IDynamic_array* Factory_impl::clone( const mi::IDynamic_array* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    mi::IDynamic_array* target
        = create_with_transaction<mi::IDynamic_array>( source->get_type_name(), source);
    if( !target)
        return nullptr;

    mi::Uint32 result = assign_from_to( source, target, options);
    if( result != 0) {
        // might happen for non-IData's
        target->release();
        return nullptr;
    }

    return target;
}

mi::IArray* Factory_impl::clone( const mi::IArray* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    mi::IArray* target = create_with_transaction<mi::IArray>( source->get_type_name(), source);
    if( !target)
        return nullptr;

    mi::Uint32 result = assign_from_to( source, target, options);
    if( result != 0) {
        // might happen for non-IData's
        target->release();
        return nullptr;
    }

    return target;
}

mi::IStructure* Factory_impl::clone( const mi::IStructure* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    mi::IStructure* target
        = create_with_transaction<mi::IStructure>( source->get_type_name(), source);
    if( !target)
        return nullptr;

    mi::Size n = source->get_length();
    for( mi::Size i = 0; i < n; ++i) {

        // get i-th key from source and its value
        const char* key = source->get_key( i);
        mi::base::Handle<const mi::base::IInterface> source_value_interface(
            source->get_value( key));

        // check if source value is of type IData
        mi::base::Handle<const mi::IData> source_value_data(
            source_value_interface->get_interface<mi::IData>());
        if( !source_value_data.is_valid_interface()) {
            target->release();
            return nullptr;
        }

        // clone source value
        mi::base::Handle<mi::IData> target_value_data( clone( source_value_data.get(), options));
        if( !target_value_data.is_valid_interface()) {
            // might happen for non-IData's or no longer registered type names
            target->release();
            return nullptr;
        }

        // and set clone in target
        target->set_value( key, target_value_data.get());
    }

    return target;
}

mi::IMap* Factory_impl::clone( const mi::IMap* source, mi::Uint32 options)
{
    ASSERT( M_NEURAY_API, source);

    mi::IMap* target = create_with_transaction<mi::IMap>( source->get_type_name(), source);
    if( !target)
        return nullptr;

    mi::Size n = source->get_length();
    for( mi::Size i = 0; i < n; ++i) {

        // get i-th key from source and its value
        const char* key = source->get_key( i);
        mi::base::Handle<const mi::base::IInterface> source_value_interface(
            source->get_value( key));

        // check if source value is of type IData
        mi::base::Handle<const mi::IData> source_value_data(
            source_value_interface->get_interface<mi::IData>());
        if( !source_value_data.is_valid_interface()) {
            target->release();
            return nullptr;
        }

        // clone source value
        mi::base::Handle<mi::IData> target_value_data( clone( source_value_data.get(), options));
        if( !target_value_data.is_valid_interface()) {
            // might happen for non-IData's or no longer registered type names
            target->release();
            return nullptr;
        }

        // and insert clone into map
        target->insert( key, target_value_data.get());
    }

    return target;
}

mi::Sint32 Factory_impl::compare( const mi::IData_simple* lhs, const mi::IData_simple* rhs)
{
    ASSERT( M_NEURAY_API, lhs && rhs);

    // handle INumber
    mi::base::Handle<const mi::INumber> lhs_value(
        lhs->get_interface<mi::INumber>());
    mi::base::Handle<const mi::INumber> rhs_value(
        rhs->get_interface<mi::INumber>());
    if( lhs_value.is_valid_interface() && rhs_value.is_valid_interface())
        return compare( lhs_value.get(), rhs_value.get());

    // handle IString
    mi::base::Handle<const mi::IString> lhs_string(
        lhs->get_interface<mi::IString>());
    mi::base::Handle<const mi::IString> rhs_string(
        rhs->get_interface<mi::IString>());
    if( lhs_string.is_valid_interface() && rhs_string.is_valid_interface())
        return compare( lhs_string.get(), rhs_string.get());

    // handle IRef
    mi::base::Handle<const mi::IRef> lhs_ref(
        lhs->get_interface<mi::IRef>());
    mi::base::Handle<const mi::IRef> rhs_ref(
        rhs->get_interface<mi::IRef>());
    if( lhs_ref.is_valid_interface() && rhs_ref.is_valid_interface())
        return compare( lhs_ref.get(), rhs_ref.get());

    // handle IEnum
    mi::base::Handle<const mi::IEnum> lhs_enum(
        lhs->get_interface<mi::IEnum>());
    mi::base::Handle<const mi::IEnum> rhs_enum(
        rhs->get_interface<mi::IEnum>());
    if( lhs_enum.is_valid_interface() && rhs_enum.is_valid_interface())
        return compare( lhs_enum.get(), rhs_enum.get());

    // handle IUuid
    mi::base::Handle<const mi::IUuid> lhs_uuid(
        lhs->get_interface<mi::IUuid>());
    mi::base::Handle<const mi::IUuid> rhs_uuid(
        rhs->get_interface<mi::IUuid>());
    if( lhs_uuid.is_valid_interface() && rhs_uuid.is_valid_interface())
        return compare( lhs_uuid.get(), rhs_uuid.get());

    // handle IVoid
    mi::base::Handle<const mi::IVoid> lhs_void(
        lhs->get_interface<mi::IVoid>());
    mi::base::Handle<const mi::IVoid> rhs_void(
        rhs->get_interface<mi::IVoid>());
    if( lhs_void.is_valid_interface() && rhs_void.is_valid_interface())
        return 0; // nothing to do

    // handle IPointer/IPointer
    mi::base::Handle<const mi::IPointer> lhs_pointer(
        lhs->get_interface<mi::IPointer>());
    mi::base::Handle<const mi::IPointer> rhs_pointer(
        rhs->get_interface<mi::IPointer>());
    if( lhs_pointer.is_valid_interface() && rhs_pointer.is_valid_interface())
        return compare( lhs_pointer.get(), rhs_pointer.get());

    // handle IConst_pointer/IConst_pointer
    mi::base::Handle<const mi::IConst_pointer> lhs_const_pointer(
        lhs->get_interface<mi::IConst_pointer>());
    mi::base::Handle<const mi::IConst_pointer> rhs_const_pointer(
        rhs->get_interface<mi::IConst_pointer>());
    if( lhs_const_pointer.is_valid_interface() && rhs_const_pointer.is_valid_interface())
        return compare( lhs_const_pointer.get(), rhs_const_pointer.get());

    ASSERT( M_NEURAY_API, false);
    return 0;
}

mi::Sint32 Factory_impl::compare( const mi::IData_collection* lhs, const mi::IData_collection* rhs)
{
    // compare length
    mi::Size lhs_n = lhs->get_length();
    mi::Size rhs_n = rhs->get_length();
    if( lhs_n < rhs_n) return -1;
    if( lhs_n > rhs_n) return +1;

    ASSERT( M_NEURAY_API, lhs_n == rhs_n);

    for( mi::Size i = 0; i < lhs_n; ++i) {

        // compare keys for index i
        const char* lhs_key = lhs->get_key( i);
        const char* rhs_key = rhs->get_key( i);
        int key_cmp = strcmp( lhs_key, rhs_key);
        if( key_cmp != 0)
            return key_cmp;

        // get value for index i from lhs and rhs
        mi::base::Handle<const mi::base::IInterface> lhs_value_interface(
            lhs->get_value( i));
        ASSERT(  M_NEURAY_API, lhs_value_interface.is_valid_interface());
        mi::base::Handle<const mi::base::IInterface> rhs_value_interface(
            rhs->get_value( i));
        ASSERT(  M_NEURAY_API, rhs_value_interface.is_valid_interface());

        // check if lhs and rhs value is of type IData
        mi::base::Handle<const mi::IData> lhs_value_data(
            lhs_value_interface->get_interface<mi::IData>());
        mi::base::Handle<const mi::IData> rhs_value_data(
            rhs_value_interface->get_interface<mi::IData>());

        // if one of the values is not of type IData compare the interface pointers
        if( !lhs_value_data.is_valid_interface() || !rhs_value_data.is_valid_interface()) {
            if( lhs_value_interface.get() < rhs_value_interface.get()) return -1;
            if( lhs_value_interface.get() > rhs_value_interface.get()) return +1;
            continue;
        }

        // of both values are of type IData invoke compare() on them
        mi::Sint32 value_cmp = compare( lhs_value_data.get(), rhs_value_data.get());
        if( value_cmp != 0)
            return value_cmp;
    }

    return 0;
}

mi::Sint32 Factory_impl::compare( const mi::INumber* lhs, const mi::INumber* rhs)
{
    const char* lhs_type_name = lhs->get_type_name();

    // bool
    if( strcmp( lhs_type_name, "Boolean") == 0) {
        bool lhs_value = lhs->get_value<bool>();
        bool rhs_value = rhs->get_value<bool>();
        if( lhs_value < rhs_value) return -1;
        if( lhs_value > rhs_value) return +1;
        return 0;
    }

    // signed integral types
    if( strncmp( lhs_type_name, "Sint", 4) == 0 || strcmp( lhs_type_name, "Difference") == 0) {
        mi::Sint64 lhs_value = lhs->get_value<mi::Sint64>();
        mi::Sint64 rhs_value = rhs->get_value<mi::Sint64>();
        if( lhs_value < rhs_value) return -1;
        if( lhs_value > rhs_value) return +1;
        return 0;
    }

    // unsigned integral types
    if( strncmp( lhs_type_name, "Uint", 4) == 0 || strcmp( lhs_type_name, "Size") == 0) {
        mi::Uint64 lhs_value = lhs->get_value<mi::Uint64>();
        mi::Uint64 rhs_value = rhs->get_value<mi::Uint64>();
        if( lhs_value < rhs_value) return -1;
        if( lhs_value > rhs_value) return +1;
        return 0;
    }

    // floating-point types
    if( strncmp( lhs_type_name, "Float", 5) == 0) {
        mi::Float64 lhs_value = lhs->get_value<mi::Float64>();
        mi::Float64 rhs_value = rhs->get_value<mi::Float64>();
        if( lhs_value < rhs_value) return -1;
        if( lhs_value > rhs_value) return +1;
        return 0;
    }

    ASSERT( M_NEURAY_API, false);
    return 0;
}

mi::Sint32 Factory_impl::compare( const mi::IString* lhs, const mi::IString* rhs)
{
    return strcmp( lhs->get_c_str(), rhs->get_c_str());
}

mi::Sint32 Factory_impl::compare( const mi::IRef* lhs, const mi::IRef* rhs)
{
    const char* lhs_name = lhs->get_reference_name();
    const char* rhs_name = rhs->get_reference_name();

    if( !lhs_name &&  rhs_name) return -1;
    if(  lhs_name && !rhs_name) return +1;
    if( !lhs_name && !rhs_name) return 0;

    ASSERT( M_NEURAY_API, lhs_name && rhs_name);
    return strcmp( lhs_name, rhs_name); //-V575
}

mi::Sint32 Factory_impl::compare( const mi::IEnum* lhs, const mi::IEnum* rhs)
{
    mi::Sint32 lhs_value = lhs->get_value();
    mi::Sint32 rhs_value = rhs->get_value();
    if( lhs_value < rhs_value) return -1;
    if( lhs_value > rhs_value) return +1;
    return 0;
}

mi::Sint32 Factory_impl::compare( const mi::IUuid* lhs, const mi::IUuid* rhs)
{
    mi::base::Uuid lhs_uuid = lhs->get_uuid();
    mi::base::Uuid rhs_uuid = rhs->get_uuid();
    if( lhs_uuid < rhs_uuid) return -1;
    if( lhs_uuid > rhs_uuid) return +1;
    return 0;
}

mi::Sint32 Factory_impl::compare( const mi::IPointer* lhs, const mi::IPointer* rhs)
{
    // get pointer for lhs and rhs as IInterface
    mi::base::Handle<const mi::base::IInterface> lhs_interface( lhs->get_pointer());
    mi::base::Handle<const mi::base::IInterface> rhs_interface( rhs->get_pointer());

    // if at least one of the pointers is \c NULL compare the interface pointers
    if( !lhs_interface.is_valid_interface() || !rhs_interface.is_valid_interface()) {
        if( !lhs_interface.is_valid_interface() ||  rhs_interface.is_valid_interface()) return -1;
        if(  lhs_interface.is_valid_interface() || !rhs_interface.is_valid_interface()) return +1;
        return 0;
    }

    // get pointer for lhs and rhs as IData (if possible)
    mi::base::Handle<const mi::IData> lhs_data( lhs_interface->get_interface<mi::IData>());
    mi::base::Handle<const mi::IData> rhs_data( rhs_interface->get_interface<mi::IData>());

    // if at least one of the values is not of type IData compare the interface pointers
    if( !lhs_data.is_valid_interface() || !rhs_data.is_valid_interface()) {
        if( lhs_interface.get() < rhs_interface.get()) return -1;
        if( lhs_interface.get() > rhs_interface.get()) return +1;
        return 0;
    }

    // if both values are of type IData invoke compare() on them
    return compare( lhs_data.get(), rhs_data.get());
}

mi::Sint32 Factory_impl::compare( const mi::IConst_pointer* lhs, const mi::IConst_pointer* rhs)
{
    // get pointer for lhs and rhs as IInterface
    mi::base::Handle<const mi::base::IInterface> lhs_interface( lhs->get_pointer());
    mi::base::Handle<const mi::base::IInterface> rhs_interface( rhs->get_pointer());

    // if at least one of the pointers is \c NULL compare the interface pointers
    if( !lhs_interface.is_valid_interface() || !rhs_interface.is_valid_interface()) {
        if( !lhs_interface.is_valid_interface() ||  rhs_interface.is_valid_interface()) return -1;
        if(  lhs_interface.is_valid_interface() || !rhs_interface.is_valid_interface()) return +1;
        return 0;
    }

    // get pointer for lhs and rhs as IData (if possible)
    mi::base::Handle<const mi::IData> lhs_data( lhs_interface->get_interface<mi::IData>());
    mi::base::Handle<const mi::IData> rhs_data( rhs_interface->get_interface<mi::IData>());

    // if at least one of the values is not of type IData compare the interface pointers
    if( !lhs_data.is_valid_interface() || !rhs_data.is_valid_interface()) {
        if( lhs_interface.get() < rhs_interface.get()) return -1;
        if( lhs_interface.get() > rhs_interface.get()) return +1;
        return 0;
    }

    // if both values are of type IData invoke compare() on them
    return compare( lhs_data.get(), rhs_data.get());
}

mi::neuraylib::ITransaction* Factory_impl::get_transaction( const mi::IData* data)
{
    ASSERT( M_NEURAY_API, data);

    // extract transaction from IRef
    mi::base::Handle<const mi::IRef> ref( data->get_interface<mi::IRef>());
    if( ref.is_valid_interface()) {
            const Ref_impl* impl = static_cast<const Ref_impl*>( data);
           return impl->get_transaction();
    }

    // extract transaction from IStructure
    mi::base::Handle<const mi::IStructure> structure( data->get_interface<mi::IStructure>());
    if( structure.is_valid_interface()) {
            const Structure_impl* impl = static_cast<const Structure_impl*>( data);
            return impl->get_transaction();
    }

    // extract transaction from IDynamic_array
    mi::base::Handle<const mi::IDynamic_array> dynamic_array(
        data->get_interface<mi::IDynamic_array>());
    if( dynamic_array.is_valid_interface()) {
            const Dynamic_array_impl* impl = static_cast<const Dynamic_array_impl*>( data);
            return impl->get_transaction();
    }

    // extract transaction from IArray
    mi::base::Handle<const mi::IArray> array( data->get_interface<mi::IArray>());
    if( array.is_valid_interface()) {
            const Array_impl* impl = static_cast<const Array_impl*>( data);
            return impl->get_transaction();
    }

    // extract transaction from IMap
    mi::base::Handle<const mi::IMap> map( data->get_interface<mi::IMap>());
    if( map.is_valid_interface()) {
        const Map_impl* impl = static_cast<const Map_impl*>( data);
        return impl->get_transaction();
    }

    // extract transaction from ICompound
    if( data->compare_iid( mi::ICompound::IID()))
        return nullptr;

    // all interfaces derived from IData_collection should be handled now
    ASSERT( M_NEURAY_API, !data->compare_iid( mi::IData_collection::IID()));

    // extract transaction from IPointer
    mi::base::Handle<const mi::IPointer> pointer( data->get_interface<mi::IPointer>());
    if( pointer.is_valid_interface()) {
        const Pointer_impl* impl = static_cast<const Pointer_impl*>( data);
        return impl->get_transaction();
    }

    // extract transaction from IConst_pointer
    mi::base::Handle<const mi::IConst_pointer> const_pointer(
        data->get_interface<mi::IConst_pointer>());
    if( const_pointer.is_valid_interface()) {
        const Const_pointer_impl* impl = static_cast<const Const_pointer_impl*>( data);
        return impl->get_transaction();
    }

    return nullptr;
}

mi::base::IInterface* Factory_impl::create_with_transaction(
    const char* type_name,
    const mi::IData* prototype,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    mi::base::IInterface* result = create( type_name, argc, argv);
    if( result)
        return result;

    // The first create() call might have failed for IRef's, non-IData's or no longer registered
    // type names. Extract transaction from prototype for fallback.
    mi::base::Handle<mi::neuraylib::ITransaction> transaction( get_transaction( prototype));
    if( !transaction.is_valid_interface())
        return nullptr;

    // This create() call might fail for non-IData's (not supported via the general factory) or no
    // longer registered type names.
    return transaction->create( type_name, argc, argv);
}

std::string get_prefix( mi::Size depth)
{
    std::string prefix;
    for( mi::Size i = 0; i < depth; i++)
        prefix += "    ";
    return prefix;
}

void Factory_impl::dump(
    mi::neuraylib::ITransaction* transaction,
    const char* name,
    const mi::IData* data,
    mi::Size depth,
    std::ostringstream& s)
{
    mi::base::Handle<const mi::IData_collection> collection(
        data->get_interface<mi::IData_collection>());

    if( name) {
        const char* type_name = data->get_type_name();
        s << type_name << " " << name << " = ";
    }

    switch( uuid_hash32( data->get_iid())) {

        case mi::IBoolean::IID::hash32: {
            mi::base::Handle<const mi::IBoolean> boolean( data->get_interface<mi::IBoolean>());
             s << (boolean->get_value<bool>() ? "true" : "false");
            return;
        }

        case mi::IUint8::IID::hash32: {
            mi::base::Handle<const mi::IUint8> uint8( data->get_interface<mi::IUint8>());
            s << uint8->get_value<mi::Uint8>();
            return;
        }

        case mi::IUint16::IID::hash32: {
            mi::base::Handle<const mi::IUint16> uint16( data->get_interface<mi::IUint16>());
            s << uint16->get_value<mi::Uint16>();
            return;
        }

        case mi::IUint32::IID::hash32: {
            mi::base::Handle<const mi::IUint32> uint32( data->get_interface<mi::IUint32>());
            s << uint32->get_value<mi::Uint32>();
            return;
        }

        case mi::IUint64::IID::hash32: {
            mi::base::Handle<const mi::IUint64> uint64( data->get_interface<mi::IUint64>());
            s << uint64->get_value<mi::Uint64>();
            return;
        }

        case mi::ISint8::IID::hash32: {
            mi::base::Handle<const mi::ISint8> sint8( data->get_interface<mi::ISint8>());
            s << sint8->get_value<mi::Sint8>();
            return;
        }

        case mi::ISint16::IID::hash32: {
            mi::base::Handle<const mi::ISint16> sint16( data->get_interface<mi::ISint16>());
            s << sint16->get_value<mi::Sint16>();
            return;
        }

        case mi::ISint32::IID::hash32: {
            mi::base::Handle<const mi::ISint32> sint32( data->get_interface<mi::ISint32>());
            s << sint32->get_value<mi::Sint32>();
            return;
        }

        case mi::ISint64::IID::hash32: {
            mi::base::Handle<const mi::ISint64> sint64( data->get_interface<mi::ISint64>());
            s << sint64->get_value<mi::Sint64>();
            return;
        }

        case mi::IFloat32::IID::hash32: {
            mi::base::Handle<const mi::IFloat32> float32( data->get_interface<mi::IFloat32>());
            s << float32->get_value<mi::Float32>();
            return;
        }

        case mi::IFloat64::IID::hash32: {
            mi::base::Handle<const mi::IFloat64> float64( data->get_interface<mi::IFloat64>());
            s << float64->get_value<mi::Float64>();
            return;
        }

        case mi::ISize::IID::hash32: {
            mi::base::Handle<const mi::ISize> size( data->get_interface<mi::ISize>());
            s << size->get_value<mi::Size>();
            return;
        }

        case mi::IDifference::IID::hash32: {
            mi::base::Handle<const mi::IDifference> diff( data->get_interface<mi::IDifference>());
            s << diff->get_value<mi::Difference>();
            return;
        }

        case mi::IUuid::IID::hash32: {
            mi::base::Handle<const mi::IUuid> uuid( data->get_interface<mi::IUuid>());
            mi::base::Uuid u = uuid->get_uuid();
            s << m_class_factory->uuid_to_string( u);
            return;
        }

        case mi::IPointer::IID::hash32: {
            mi::base::Handle<const mi::IPointer> pointer(
                data->get_interface<mi::IPointer>());
            mi::base::Handle<const mi::base::IInterface> p( pointer->get_pointer());
            s << p.get();
            return;
        }

        case mi::IConst_pointer::IID::hash32: {
            mi::base::Handle<const mi::IConst_pointer> pointer(
                data->get_interface<mi::IConst_pointer>());
            mi::base::Handle<const mi::base::IInterface> p( pointer->get_pointer());
            s << p.get();
            return;
        }

        case mi::IString::IID::hash32: {
            mi::base::Handle<const mi::IString> string( data->get_interface<mi::IString>());
            s << "\"" << string->get_c_str() << "\"";
            return;
        }

        case mi::IRef::IID::hash32: {
            mi::base::Handle<const mi::IRef> ref( data->get_interface<mi::IRef>());
            const char* reference_name = ref->get_reference_name();
            if( reference_name)
                s << "points to \"" << reference_name << "\"";
            else
                s << "(unset)";
            return;
        }

        case mi::IEnum::IID::hash32: {
            mi::base::Handle<const mi::IEnum> e( data->get_interface<mi::IEnum>());
            s << e->get_value_by_name() << "(" << e->get_value() << ")";
            return;
        }

        case mi::IVoid::IID::hash32: {
            s << "(void)";
            return;
        }

        case mi::IColor::IID::hash32: {
            mi::base::Handle<const mi::IColor> color( data->get_interface<mi::IColor>());
            mi::math::Color c = color->get_value();
            s << "(" << c.r << ", " << c.g << ", " << c.b << ")";
            return;
        }

        case mi::IColor3::IID::hash32: {
            mi::base::Handle<const mi::IColor3> color( data->get_interface<mi::IColor3>());
            mi::math::Color c = color->get_value();
            s << "(" << c.r << ", " << c.g << ", " << c.b << ")";
            return;
        }

        case mi::ISpectrum::IID::hash32: {
            mi::base::Handle<const mi::ISpectrum> spectrum( data->get_interface<mi::ISpectrum>());
            mi::math::Spectrum sp = spectrum->get_value();
            s << "(" << sp.get( 0) << ", " << sp.get( 1) << ", " << sp.get( 2) << ")";
            return;
        }

        case mi::IBbox3::IID::hash32: {
            mi::base::Handle<const mi::IBbox3> bbox( data->get_interface<mi::IBbox3>());
            mi::Bbox3 b = bbox->get_value();
            s << "(" << b.min[0] << ", " << b.min[1] << ", " << b.min[2] << ") - "
              << "(" << b.max[0] << ", " << b.max[1] << ", " << b.max[2] << ")";
            return;
        }

        case mi::IBoolean_2::IID::hash32:
        case mi::IBoolean_3::IID::hash32:
        case mi::IBoolean_4::IID::hash32:
        case mi::IBoolean_2_2::IID::hash32:
        case mi::IBoolean_2_3::IID::hash32:
        case mi::IBoolean_2_4::IID::hash32:
        case mi::IBoolean_3_2::IID::hash32:
        case mi::IBoolean_3_3::IID::hash32:
        case mi::IBoolean_3_4::IID::hash32:
        case mi::IBoolean_4_2::IID::hash32:
        case mi::IBoolean_4_3::IID::hash32:
        case mi::IBoolean_4_4::IID::hash32: {
            s << "(";
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle<const mi::IBoolean> field(
                    collection->get_value<mi::IBoolean>( i));
                s << (i > 0 ? ", " : "") << field->get_value<bool>();
            }
            s << ")";
            return;
        }

        case mi::ISint32_2::IID::hash32:
        case mi::ISint32_3::IID::hash32:
        case mi::ISint32_4::IID::hash32:
        case mi::ISint32_2_2::IID::hash32:
        case mi::ISint32_2_3::IID::hash32:
        case mi::ISint32_2_4::IID::hash32:
        case mi::ISint32_3_2::IID::hash32:
        case mi::ISint32_3_3::IID::hash32:
        case mi::ISint32_3_4::IID::hash32:
        case mi::ISint32_4_2::IID::hash32:
        case mi::ISint32_4_3::IID::hash32:
        case mi::ISint32_4_4::IID::hash32: {
            s << "(";
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle<const mi::ISint32> field(
                    collection->get_value<mi::ISint32>( i));
                s << (i > 0 ? ", " : "") << field->get_value<mi::Sint32>();
            }
            s << ")";
            return;
        }

        case mi::IUint32_2::IID::hash32:
        case mi::IUint32_3::IID::hash32:
        case mi::IUint32_4::IID::hash32:
        case mi::IUint32_2_2::IID::hash32:
        case mi::IUint32_2_3::IID::hash32:
        case mi::IUint32_2_4::IID::hash32:
        case mi::IUint32_3_2::IID::hash32:
        case mi::IUint32_3_3::IID::hash32:
        case mi::IUint32_3_4::IID::hash32:
        case mi::IUint32_4_2::IID::hash32:
        case mi::IUint32_4_3::IID::hash32:
        case mi::IUint32_4_4::IID::hash32: {
            s << "(";
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle<const mi::IUint32> field(
                    collection->get_value<mi::IUint32>( i));
                s << (i > 0 ? ", " : "") << field->get_value<mi::Uint32>();
            }
            s << ")";
            return;
        }

        case mi::IFloat32_2::IID::hash32:
        case mi::IFloat32_3::IID::hash32:
        case mi::IFloat32_4::IID::hash32:
        case mi::IFloat32_2_2::IID::hash32:
        case mi::IFloat32_2_3::IID::hash32:
        case mi::IFloat32_2_4::IID::hash32:
        case mi::IFloat32_3_2::IID::hash32:
        case mi::IFloat32_3_3::IID::hash32:
        case mi::IFloat32_3_4::IID::hash32:
        case mi::IFloat32_4_2::IID::hash32:
        case mi::IFloat32_4_3::IID::hash32:
        case mi::IFloat32_4_4::IID::hash32: {
            s << "(";
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle<const mi::IFloat32> field(
                    collection->get_value<mi::IFloat32>( i));
                s << (i > 0 ? ", " : "") << field->get_value<mi::Float32>();
            }
            s << ")";
            return;
        }

        case mi::IFloat64_2::IID::hash32:
        case mi::IFloat64_3::IID::hash32:
        case mi::IFloat64_4::IID::hash32:
        case mi::IFloat64_2_2::IID::hash32:
        case mi::IFloat64_2_3::IID::hash32:
        case mi::IFloat64_2_4::IID::hash32:
        case mi::IFloat64_3_2::IID::hash32:
        case mi::IFloat64_3_3::IID::hash32:
        case mi::IFloat64_3_4::IID::hash32:
        case mi::IFloat64_4_2::IID::hash32:
        case mi::IFloat64_4_3::IID::hash32:
        case mi::IFloat64_4_4::IID::hash32: {
            s << "(";
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle<const mi::IFloat64> field(
                    collection->get_value<mi::IFloat64>( i));
                s << (i > 0 ? ", " : "") << field->get_value<mi::Float64>();
            }
            s << ")";
            return;
        }

        case mi::IMap::IID::hash32:
        case mi::IStructure::IID::hash32: {
            s << "{";
            mi::Size length = collection->get_length();
            if( length > 0)
                s << std::endl;
            else
                s << " ";
            for( mi::Size i = 0; i < length; i++) {
                s << get_prefix( depth+1);
                const char* key = collection->get_key( i);
                mi::base::Handle<const mi::base::IInterface> field( collection->get_value( i));
                mi::base::Handle<const mi::IData> field_data( field->get_interface<mi::IData>());
                if( field_data)
                    dump( transaction, key, field_data.get(), depth+1, s);
                else
                    dump_non_idata( transaction, key, field.get(), depth+1, s);
                s << ";" << std::endl;
            }
            if( length > 0)
                s << get_prefix( depth);
            s << "}";
            return;
        }

        case mi::IDynamic_array::IID::hash32:
        case mi::IArray::IID::hash32: {
            s << "{";
            mi::Size length =  collection->get_length();
            if( length > 0)
                s << std::endl;
            else
                s << " ";
            for( mi::Size i = 0; i < length; i++) {
                s << get_prefix( depth+1) << "[" << i << "] = ";
                const char* key = collection->get_key( i);
                mi::base::Handle<const mi::base::IInterface> field( collection->get_value( i));
                mi::base::Handle<const mi::IData> field_data( field->get_interface<mi::IData>());
                if( field_data)
                    dump( transaction, key, field_data.get(), depth+1, s);
                else
                    dump_non_idata( transaction, key, field.get(), depth+1, s);
                s << ";" << std::endl;
            }
            if( length > 0)
                s << get_prefix( depth);
            s << "}";
            return;
        }

        default:
            s << "(dumper for this type missing)";
            return;
    }
}

void Factory_impl::dump_non_idata(
    mi::neuraylib::ITransaction* transaction,
    const char* name,
    const mi::base::IInterface* data,
    mi::Size depth,
    std::ostringstream& s)
{
    ASSERT( M_NEURAY_API, data);

    mi::base::Handle<const mi::neuraylib::IType> type(
        data->get_interface<mi::neuraylib::IType>());
    if( type) {
        mi::base::Handle<const mi::neuraylib::IType_factory> tf(
            m_class_factory->create_type_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            tf->dump( type.get(), depth));
        s << result->get_c_str();
        return;
    }

    mi::base::Handle<const mi::neuraylib::IType_list> type_list(
        data->get_interface<mi::neuraylib::IType_list>());
    if( type_list) {
        mi::base::Handle<const mi::neuraylib::IType_factory> tf(
            m_class_factory->create_type_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            tf->dump( type_list.get(), depth));
        s << result->get_c_str();
        return;
    }

    mi::base::Handle<const mi::neuraylib::IValue> value(
        data->get_interface<mi::neuraylib::IValue>());
    if( value) {
        mi::base::Handle<const mi::neuraylib::IValue_factory> vf(
            m_class_factory->create_value_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            vf->dump( value.get(), name, depth));
        s << result->get_c_str();
        return;
    }

    mi::base::Handle<const mi::neuraylib::IValue_list> value_list(
        data->get_interface<mi::neuraylib::IValue_list>());
    if( value_list) {
        mi::base::Handle<const mi::neuraylib::IValue_factory> vf(
            m_class_factory->create_value_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            vf->dump( value_list.get(), name, depth));
        s << result->get_c_str();
        return;
    }

    mi::base::Handle<const mi::neuraylib::IExpression> expr(
        data->get_interface<mi::neuraylib::IExpression>());
    if( expr) {
        mi::base::Handle<const mi::neuraylib::IExpression_factory> ef(
            m_class_factory->create_expression_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            ef->dump( expr.get(), name, depth));
        s << result->get_c_str();
        return;
    }

    mi::base::Handle<const mi::neuraylib::IExpression_list> expr_list(
        data->get_interface<mi::neuraylib::IExpression_list>());
    if( expr_list) {
        mi::base::Handle<const mi::neuraylib::IExpression_factory> ef(
            m_class_factory->create_expression_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            ef->dump( expr_list.get(), name, depth));
        s << result->get_c_str();
        return;
    }

    mi::base::Handle<const mi::neuraylib::IAnnotation> anno(
        data->get_interface<mi::neuraylib::IAnnotation>());
    if( anno) {
        mi::base::Handle<const mi::neuraylib::IExpression_factory> ef(
            m_class_factory->create_expression_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            ef->dump( anno.get(), name, depth));
        s << result->get_c_str();
        return;
    }


    mi::base::Handle<const mi::neuraylib::IAnnotation_block> block(
        data->get_interface<mi::neuraylib::IAnnotation_block>());
    if( block) {
        mi::base::Handle<const mi::neuraylib::IExpression_factory> ef(
            m_class_factory->create_expression_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            ef->dump( block.get(), name, depth));
        s << result->get_c_str();
        return;
    }

    mi::base::Handle<const mi::neuraylib::IAnnotation_list> anno_list(
        data->get_interface<mi::neuraylib::IAnnotation_list>());
    if( anno_list) {
        mi::base::Handle<const mi::neuraylib::IExpression_factory> ef(
            m_class_factory->create_expression_factory( transaction));
        mi::base::Handle<const mi::IString> result(
            ef->dump( anno_list.get(), name, depth));
        s << result->get_c_str();
        return;
    }

    s << "(dumping of this type is not supported)";
}

} // namespace NEURAY

} // namespace MI

