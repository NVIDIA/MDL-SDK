/******************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief
 **/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for prod/lib/neuray"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <type_traits>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>

#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/ibbox.h>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/idata.h>
#include <mi/neuraylib/idatabase.h>
#include <mi/neuraylib/idebug_configuration.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/iextension_api.h>
#include <mi/neuraylib/ifactory.h>
#include <mi/neuraylib/imap.h>
#include <mi/neuraylib/imatrix.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/ipointer.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/iscope.h>
#include <mi/neuraylib/ispectrum.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure_decl.h>
#include <mi/neuraylib/istructure.h>
#include <mi/neuraylib/itransaction.h>
#include <mi/neuraylib/ivector.h>
#include <mi/neuraylib/iuuid.h>


#include "test_shared.h"

#define CHECK_ASSIGN( result, source, target) \
    MI_CHECK_EQUAL( result, factory->assign_from_to( source, target))
#define CHECK_ASSIGN_DEEP( result, source, target) \
    MI_CHECK_EQUAL( result, \
        factory->assign_from_to( source, target, IFactory::DEEP_ASSIGNMENT_OR_CLONE))
#define CHECK_ASSIGN_FIXED( result, source, target) \
    MI_CHECK_EQUAL( result, \
        factory->assign_from_to( source, target, IFactory::FIX_SET_OF_TARGET_KEYS))
#define CHECK_COMPARE( result, lhs, rhs) \
    MI_CHECK_EQUAL( result, factory->compare( lhs, rhs)); \
    MI_CHECK_EQUAL( -(result), factory->compare( rhs, lhs));

using mi::neuraylib::IFactory;

// for test_structural_mismatch()
class Data_simple : public mi::base::Interface_implement<mi::IData_simple>
{
public:
    const char* get_type_name() const final { return "Data_simple"; }
};

// for test_structural_mismatch()
class Data_collection : public mi::base::Interface_implement<mi::IData_collection>
{
public:
    const char* get_type_name() const final { return "Data_collection"; }
    mi::Size get_length() const final { return 0; }
    const char* get_key( mi::Size index) const final { return nullptr; }
    bool has_key( const char* key) const final { return false; }
    const mi::base::IInterface* get_value( const char* key) const final { return nullptr; }
    mi::base::IInterface* get_value( const char* key) final { return nullptr; }
    const mi::base::IInterface* get_value( mi::Size index) const final { return nullptr; }
    mi::base::IInterface* get_value( mi::Size index) final { return nullptr; }
    mi::Sint32 set_value( const char* key, mi::base::IInterface* value) final { return 0; }
    mi::Sint32 set_value( mi::Size index, mi::base::IInterface* value) final { return 0; }
};

void test_null_pointer( mi::neuraylib::IFactory* factory)
{
    Data_simple simple;
    CHECK_ASSIGN( IFactory::NULL_POINTER, nullptr, &simple);
    CHECK_ASSIGN( IFactory::NULL_POINTER, &simple, nullptr);
    CHECK_ASSIGN( IFactory::NULL_POINTER, nullptr, nullptr);

    mi::base::Handle<mi::ISint32> sint32       ( factory->create<mi::ISint32>(  "Sint32"));
    mi::base::Handle<const mi::ISint32> csint32( factory->create<mi::ISint32>(  "Sint32"));
    mi::base::Handle<mi::IPointer> p_1(          factory->create<mi::IPointer>( "Pointer<Sint32>"));
    mi::base::Handle<mi::IPointer> p_2(          factory->create<mi::IPointer>( "Pointer<Sint32>"));
    mi::base::Handle<mi::IConst_pointer> cp_2(
        factory->create<mi::IConst_pointer>( "Const_pointer<Sint32>"));
    p_2->set_pointer( sint32.get());
    cp_2->set_pointer( csint32.get());

    CHECK_ASSIGN_DEEP( IFactory::NULL_POINTER, p_2.get() , p_1.get());
    CHECK_ASSIGN_DEEP( IFactory::NULL_POINTER, cp_2.get(), p_1.get());
}

void test_structural_mismatch( mi::neuraylib::IFactory* factory)
{
    Data_simple simple;
    Data_collection collection;
    CHECK_ASSIGN( IFactory::STRUCTURAL_MISMATCH, &collection, &simple);
    CHECK_ASSIGN( IFactory::STRUCTURAL_MISMATCH, &simple, &collection);
}

void test_no_conversion( mi::neuraylib::IFactory* factory, mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::ISint32> sint32( factory->create<mi::ISint32>( "Sint32"));
    mi::base::Handle<mi::IString> string( factory->create<mi::IString>( "String"));
    mi::base::Handle<mi::IRef> ref( transaction->create<mi::IRef>( "Ref"));
    mi::base::Handle<mi::IUuid> uuid( factory->create<mi::IUuid>( "Uuid"));
    mi::base::Handle<mi::IVoid> void_( factory->create<mi::IVoid>( "Void"));
    mi::base::Handle<mi::IPointer> pointer( factory->create<mi::IPointer>( "Pointer<Interface>"));
    mi::base::Handle<mi::IConst_pointer> cpointer(
       factory->create<mi::IConst_pointer>( "Const_pointer<Interface>"));
    mi::base::Handle<mi::IEnum> enum_( factory->create<mi::IEnum>( "Test_enum"));

    CHECK_ASSIGN( IFactory::NO_CONVERSION, sint32.get(),   string.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, sint32.get(),   ref.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, sint32.get(),   uuid.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, sint32.get(),   void_.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, sint32.get(),   pointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, sint32.get(),   cpointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, string.get(),   sint32.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, string.get(),   ref.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, string.get(),   uuid.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, string.get(),   void_.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, string.get(),   pointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, string.get(),   cpointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, ref.get(),      sint32.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, ref.get(),      string.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, ref.get(),      uuid.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, ref.get(),      void_.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, ref.get(),      pointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, ref.get(),      cpointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, uuid.get(),     sint32.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, uuid.get(),     string.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, uuid.get(),     ref.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, uuid.get(),     void_.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, uuid.get(),     pointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, uuid.get(),     cpointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, void_.get(),    sint32.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, void_.get(),    string.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, void_.get(),    ref.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, void_.get(),    uuid.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, void_.get(),    pointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, void_.get(),    cpointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, pointer.get(),  sint32.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, pointer.get(),  string.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, pointer.get(),  ref.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, pointer.get(),  uuid.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, pointer.get(),  void_.get());
    CHECK_ASSIGN( 0                      , pointer.get(),  cpointer.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, cpointer.get(), sint32.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, cpointer.get(), string.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, cpointer.get(), ref.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, cpointer.get(), uuid.get());
    CHECK_ASSIGN( IFactory::NO_CONVERSION, cpointer.get(), void_.get());
    CHECK_ASSIGN( IFactory::INCOMPATIBLE_POINTER_TYPES, cpointer.get(), pointer.get());


}

void test_key_missing( mi::neuraylib::IFactory* factory)
{
    mi::base::Handle<mi::ISint32> dummy( factory->create<mi::ISint32>( "Sint32"));

    mi::base::Handle<mi::IMap> map( factory->create<mi::IMap>( "Map<Sint32>"));

    mi::base::Handle<mi::IMap> map_a( factory->create<mi::IMap>( "Map<Sint32>"));
    MI_CHECK_EQUAL( 0, map_a->insert( "a", dummy.get()));

    mi::base::Handle<mi::IMap> map_a_b( factory->create<mi::IMap>( "Map<Sint32>"));
    MI_CHECK_EQUAL( 0, map_a_b->insert( "a", dummy.get()));
    MI_CHECK_EQUAL( 0, map_a_b->insert( "a_b", dummy.get()));

    mi::base::Handle<mi::IMap> map_a_c( factory->create<mi::IMap>( "Map<Sint32>"));
    MI_CHECK_EQUAL( 0, map_a_c->insert( "a", dummy.get()));
    MI_CHECK_EQUAL( 0, map_a_c->insert( "a_c", dummy.get()));

    CHECK_ASSIGN_FIXED( IFactory::SOURCE_KEY_MISSING, map.get(), map_a.get());
    CHECK_ASSIGN_FIXED( IFactory::SOURCE_KEY_MISSING, map_a.get(), map_a_b.get());
    CHECK_ASSIGN_FIXED( IFactory::SOURCE_KEY_MISSING, map_a.get(), map_a_c.get());
    CHECK_ASSIGN_FIXED(
        IFactory::SOURCE_KEY_MISSING|IFactory::TARGET_KEY_MISSING, map_a_b.get(), map_a_c.get());

    CHECK_ASSIGN_FIXED( IFactory::TARGET_KEY_MISSING, map_a.get(), map.get());
    CHECK_ASSIGN_FIXED( IFactory::TARGET_KEY_MISSING, map_a_b.get(), map_a.get());
    CHECK_ASSIGN_FIXED( IFactory::TARGET_KEY_MISSING, map_a_c.get(), map_a.get());
    CHECK_ASSIGN_FIXED(
        IFactory::TARGET_KEY_MISSING|IFactory::SOURCE_KEY_MISSING, map_a_c.get(), map_a_b.get());

    CHECK_ASSIGN( 0, map_a.get(), map.get());
    MI_CHECK_EQUAL( 1, map->get_length());
    CHECK_ASSIGN( 0, map_a_b.get(), map_a.get());
    MI_CHECK_EQUAL( 2, map_a->get_length());
    CHECK_ASSIGN( 0, map_a_c.get(), map_a.get());
    MI_CHECK_EQUAL( 2, map_a->get_length());
    CHECK_ASSIGN( 0, map_a_c.get(), map_a_b.get());
    MI_CHECK_EQUAL( 2, map_a_b->get_length());

    mi::base::Handle<mi::IDynamic_array> dynamic_array1(
        factory->create<mi::IDynamic_array>( "Sint32[]"));
    mi::base::Handle<mi::IDynamic_array> dynamic_array2(
        factory->create<mi::IDynamic_array>( "Sint32[]"));
    dynamic_array1->set_length( 1);
    dynamic_array2->set_length( 2);

    CHECK_ASSIGN_FIXED( IFactory::SOURCE_KEY_MISSING, dynamic_array1.get(), dynamic_array2.get());

    CHECK_ASSIGN_FIXED( IFactory::TARGET_KEY_MISSING, dynamic_array2.get(), dynamic_array1.get());

    CHECK_ASSIGN( 0, dynamic_array2.get(), dynamic_array1.get());
    MI_CHECK_EQUAL( 2, dynamic_array1->get_length());
}

void test_different_collections( mi::neuraylib::IFactory* factory)
{
    mi::base::Handle<mi::ISint32> dummy( factory->create<mi::ISint32>( "Sint32"));

    mi::base::Handle<mi::IMap> map( factory->create<mi::IMap>( "Map<Sint32>"));
    mi::base::Handle<mi::IArray> array( factory->create<mi::IArray>( "Sint32[1]"));
    mi::base::Handle<mi::IDynamic_array> dynamic_array(
        factory->create<mi::IDynamic_array>( "Sint32[]"));
    mi::base::Handle<mi::ISint32_2> sint32_2( factory->create<mi::ISint32_2>( "Sint32<2>"));
    mi::base::Handle<mi::IStructure> structure( factory->create<mi::IStructure>( "s_empty"));

    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, map.get(), dynamic_array.get());
    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, dynamic_array.get(), map.get());

    MI_CHECK_EQUAL( 0, dynamic_array->push_back( dummy.get()));
    MI_CHECK_EQUAL( 0, map->insert( "0", dummy.get()));

    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, map.get(), array.get());
    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, map.get(), dynamic_array.get());
    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, dynamic_array.get(), map.get());
    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, dynamic_array.get(), array.get());
    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, array.get(), map.get());
    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, array.get(), dynamic_array.get());

    MI_CHECK_EQUAL( 0, map->erase( "0"));
    MI_CHECK_EQUAL( 0, map->insert( "x", dummy.get()));
    MI_CHECK_EQUAL( 0, map->insert( "y", dummy.get()));

    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, map.get(), sint32_2.get());
    CHECK_ASSIGN( IFactory::DIFFERENT_COLLECTIONS, sint32_2.get(), map.get());
}

void test_non_idata_values(
    mi::neuraylib::IFactory* factory, mi::neuraylib::ITransaction* transaction)
{
}

void test_incompatible_pointer_types(
    mi::neuraylib::IFactory* factory, mi::neuraylib::ITransaction* transaction)
{
}

void test_deep_assignment_to_const_pointer(
    mi::neuraylib::IFactory* factory, mi::neuraylib::ITransaction* transaction)
{
    mi::base::Handle<mi::IConst_pointer> cp_1(
        factory->create<mi::IConst_pointer>( "Const_pointer<Sint32>"));
    mi::base::Handle<mi::IConst_pointer> cp_2(
        factory->create<mi::IConst_pointer>( "Const_pointer<Sint32>"));
    CHECK_ASSIGN_DEEP( IFactory::DEEP_ASSIGNMENT_TO_CONST_POINTER, cp_1.get(), cp_2.get());
    CHECK_ASSIGN     ( 0,                                          cp_1.get(), cp_2.get());
}

void test_idata_simple( mi::neuraylib::IFactory* factory, mi::neuraylib::ITransaction* transaction)
{
    // ISint32
    mi::base::Handle<mi::ISint32> sint32_1( factory->create<mi::ISint32>( "Sint32"));
    mi::base::Handle<mi::ISint32> sint32_2( factory->create<mi::ISint32>( "Sint32"));
    sint32_1->set_value( 42);
    sint32_2->set_value( 43);
    MI_CHECK_NOT_EQUAL( sint32_1->get_value<mi::Sint32>(), sint32_2->get_value<mi::Sint32>());
    CHECK_COMPARE( -1, sint32_1.get(), sint32_2.get());
    CHECK_ASSIGN( 0, sint32_1.get(), sint32_2.get());
    MI_CHECK_EQUAL( sint32_1->get_value<mi::Sint32>(), sint32_2->get_value<mi::Sint32>());
    CHECK_COMPARE( 0, sint32_1.get(), sint32_2.get());

    // IString
    mi::base::Handle<mi::IString> string_1( factory->create<mi::IString>( "String"));
    mi::base::Handle<mi::IString> string_2( factory->create<mi::IString>( "String"));
    string_1->set_c_str( "42");
    string_2->set_c_str( "43");
    MI_CHECK_NOT_EQUAL_CSTR( string_1->get_c_str(), string_2->get_c_str());
    CHECK_COMPARE( -1, string_1.get(), string_2.get());
    CHECK_ASSIGN( 0, string_1.get(), string_2.get());
    MI_CHECK_EQUAL_CSTR( string_1->get_c_str(), string_2->get_c_str());
    CHECK_COMPARE( 0, string_1.get(), string_2.get());


    // IUuid
    mi::base::Uuid uuid_small;
    uuid_small.m_id1 = 0x12345678;
    uuid_small.m_id2 = 0x87654321;
    uuid_small.m_id3 = 0x00000001;
    uuid_small.m_id4 = 0x10000000;
    mi::base::Uuid uuid_large = uuid_small;
    uuid_large.m_id1 = 0xffffffff;
    mi::base::Handle<mi::IUuid> uuid_1( factory->create<mi::IUuid>( "Uuid"));
    mi::base::Handle<mi::IUuid> uuid_2( factory->create<mi::IUuid>( "Uuid"));
    uuid_1->set_uuid( uuid_small);
    uuid_2->set_uuid( uuid_large);
    MI_CHECK( uuid_1->get_uuid() < uuid_2->get_uuid());
    CHECK_COMPARE( -1, uuid_1.get(), uuid_2.get());
    CHECK_ASSIGN( 0, uuid_1.get(), uuid_2.get());
    MI_CHECK( uuid_1->get_uuid() == uuid_2->get_uuid());
    CHECK_COMPARE( 0, uuid_1.get(), uuid_2.get());

    // IVoid
    mi::base::Handle<mi::IVoid> void_1( factory->create<mi::IVoid>( "Void"));
    mi::base::Handle<mi::IVoid> void_2( factory->create<mi::IVoid>( "Void"));
    CHECK_ASSIGN( 0, void_1.get(), void_2.get());
    CHECK_COMPARE( 0, void_1.get(), void_2.get());

    // IPointer
    sint32_1->set_value( 42);
    sint32_2->set_value( 43);
    mi::base::Handle<mi::IPointer> pointer_1( factory->create<mi::IPointer>( "Pointer<Sint32>"));
    mi::base::Handle<mi::IPointer> pointer_2( factory->create<mi::IPointer>( "Pointer<Sint32>"));
    mi::base::Handle<mi::ISint32> m_sint32( pointer_2->get_pointer<mi::ISint32>());
    MI_CHECK( !m_sint32.is_valid_interface());
    MI_CHECK_EQUAL( 0, pointer_1->set_pointer( sint32_1.get()));
    CHECK_COMPARE( 1, pointer_1.get(), pointer_2.get());
    CHECK_ASSIGN( 0, pointer_1.get(), pointer_2.get());
    m_sint32 = pointer_2->get_pointer<mi::ISint32>();
    MI_CHECK( m_sint32.is_valid_interface());
    CHECK_COMPARE( 0, pointer_1.get(), pointer_2.get());
    pointer_1->set_pointer( nullptr);
    CHECK_COMPARE( -1, pointer_1.get(), pointer_2.get());

    // IConst_pointer
    mi::base::Handle<mi::IConst_pointer> cpointer_1(
        factory->create<mi::IConst_pointer>( "Const_pointer<Sint32>"));
    mi::base::Handle<mi::IConst_pointer> cpointer_2(
        factory->create<mi::IConst_pointer>( "Const_pointer<Sint32>"));
    mi::base::Handle<const mi::ISint32> c_sint32( cpointer_2->get_pointer<mi::ISint32>());
    MI_CHECK( !c_sint32.is_valid_interface());
    MI_CHECK_EQUAL( 0, cpointer_1->set_pointer( sint32_1.get()));
    CHECK_COMPARE( 1, cpointer_1.get(), cpointer_2.get());
    CHECK_ASSIGN( 0, cpointer_1.get(), cpointer_2.get());
    c_sint32 = cpointer_2->get_pointer<mi::ISint32>();
    MI_CHECK( c_sint32.is_valid_interface());
    CHECK_COMPARE( 0, cpointer_1.get(), cpointer_2.get());
    cpointer_1->set_pointer( nullptr);
    CHECK_COMPARE( -1, cpointer_1.get(), cpointer_2.get());


    // IFloat32 => ISint32 conversion
    mi::base::Handle<mi::IFloat32> float32( factory->create<mi::IFloat32>( "Float32"));
    float32->set_value( -42.42);
    MI_CHECK_NOT_EQUAL( sint32_1->get_value<mi::Sint32>(), float32->get_value<mi::Float32>());
    CHECK_ASSIGN( 0, sint32_1.get(), float32.get());
    MI_CHECK_EQUAL( sint32_1->get_value<mi::Sint32>(), float32->get_value<mi::Float32>());

    // clone()
    mi::base::Handle<mi::ISint32> sint32_1_clone( factory->clone<mi::ISint32>( sint32_1.get()));
    MI_CHECK( sint32_1_clone.is_valid_interface());
    mi::base::Handle<mi::IString> string_1_clone( factory->clone<mi::IString>( string_1.get()));
    MI_CHECK( string_1_clone.is_valid_interface());
    mi::base::Handle<mi::IUuid> uuid_1_clone( factory->clone<mi::IUuid>( uuid_1.get()));
    MI_CHECK( uuid_1_clone.is_valid_interface());
    mi::base::Handle<mi::IVoid> void_1_clone( factory->clone<mi::IVoid>( void_1.get()));
    MI_CHECK( void_1_clone.is_valid_interface());
    mi::base::Handle<mi::IPointer> pointer_1_clone( factory->clone<mi::IPointer>( pointer_1.get()));
    MI_CHECK( pointer_1_clone.is_valid_interface());
    mi::base::Handle<mi::IConst_pointer> cpointer_1_clone(
        factory->clone<mi::IConst_pointer>( cpointer_1.get()));
    MI_CHECK( cpointer_1_clone.is_valid_interface());
}

void test_idata_collection( mi::neuraylib::IFactory* factory)
{
    mi::base::Handle<mi::ISint32> sint32_1( factory->create<mi::ISint32>( "Sint32"));
    mi::base::Handle<mi::ISint32> sint32_2( factory->create<mi::ISint32>( "Sint32"));
    mi::base::Handle<mi::ISint32> tmp;

    // IArray
    sint32_1->set_value( 1);
    sint32_2->set_value( 2);
    mi::base::Handle<mi::IArray> array_1( factory->create<mi::IArray>( "Sint32[1]"));
    mi::base::Handle<mi::IArray> array_2( factory->create<mi::IArray>( "Sint32[1]"));
    array_1->set_element( 0, sint32_1.get());
    array_2->set_element( 0, sint32_2.get());
    tmp = array_2->get_element<mi::ISint32>( 0);
    MI_CHECK_NOT_EQUAL( tmp->get_value<mi::Sint32>(), sint32_1->get_value<mi::Sint32>());
    CHECK_COMPARE( -1, array_1.get(), array_2.get());
    CHECK_ASSIGN( 0, array_1.get(), array_2.get());
    tmp = array_2->get_element<mi::ISint32>( 0);
    MI_CHECK_EQUAL( tmp->get_value<mi::Sint32>(), sint32_1->get_value<mi::Sint32>());
    CHECK_COMPARE( 0, array_1.get(), array_2.get());

    // IDynamic_array
    sint32_1->set_value( 1);
    sint32_2->set_value( 2);
    mi::base::Handle<mi::IDynamic_array> dynamic_array_1(
        factory->create<mi::IDynamic_array>( "Sint32[]"));
    mi::base::Handle<mi::IDynamic_array> dynamic_array_2(
        factory->create<mi::IDynamic_array>( "Sint32[]"));
    dynamic_array_1->set_length( 1);
    dynamic_array_2->set_length( 1);
    dynamic_array_1->set_element( 0, sint32_1.get());
    dynamic_array_2->set_element( 0, sint32_2.get());
    tmp = dynamic_array_2->get_element<mi::ISint32>( 0);
    MI_CHECK_NOT_EQUAL( tmp->get_value<mi::Sint32>(), sint32_1->get_value<mi::Sint32>());
    CHECK_COMPARE( -1, dynamic_array_1.get(), dynamic_array_2.get());
    CHECK_ASSIGN( 0, dynamic_array_1.get(), dynamic_array_2.get());
    tmp = dynamic_array_2->get_element<mi::ISint32>( 0);
    MI_CHECK_EQUAL( tmp->get_value<mi::Sint32>(), sint32_1->get_value<mi::Sint32>());
    CHECK_COMPARE( 0, dynamic_array_1.get(), dynamic_array_2.get());

    // IMap
    sint32_1->set_value( 1);
    sint32_2->set_value( 2);
    mi::base::Handle<mi::IMap> map_1( factory->create<mi::IMap>( "Map<Sint32>"));
    mi::base::Handle<mi::IMap> map_2( factory->create<mi::IMap>( "Map<Sint32>"));
    map_1->insert( "key", sint32_1.get());
    map_2->insert( "key", sint32_2.get());
    tmp = map_2->get_value<mi::ISint32>( "key");
    MI_CHECK_NOT_EQUAL( tmp->get_value<mi::Sint32>(), sint32_1->get_value<mi::Sint32>());
    CHECK_COMPARE( -1, map_1.get(), map_2.get());
    CHECK_ASSIGN( 0, map_1.get(), map_2.get());
    tmp = map_2->get_value<mi::ISint32>( "key");
    MI_CHECK_EQUAL( tmp->get_value<mi::Sint32>(), sint32_1->get_value<mi::Sint32>());
    CHECK_COMPARE( 0, map_1.get(), map_2.get());

    // ISint32_2
    mi::base::Handle<mi::ISint32_2> vector_1( factory->create<mi::ISint32_2>( "Sint32<2>"));
    mi::base::Handle<mi::ISint32_2> vector_2( factory->create<mi::ISint32_2>( "Sint32<2>"));
    MI_CHECK( vector_1.is_valid_interface());
    MI_CHECK( vector_2.is_valid_interface());
    vector_1->set_value( 0, 0, 1);
    vector_2->set_value( 0, 0, 2);
    MI_CHECK_NOT_EQUAL(
        vector_2->get_value<mi::Sint32>( 0, 0), vector_1->get_value<mi::Sint32>( 0, 0));
    CHECK_COMPARE( -1, vector_1.get(), vector_2.get());
    CHECK_ASSIGN( 0, vector_1.get(), vector_2.get());
    MI_CHECK_EQUAL( vector_2->get_value<mi::Sint32>( 0, 0), vector_1->get_value<mi::Sint32>( 0, 0));
    CHECK_COMPARE( 0, vector_1.get(), vector_2.get());

    // IBbox3
    mi::base::Handle<mi::IBbox3> bbox_1( factory->create<mi::IBbox3>( "Bbox3"));
    mi::base::Handle<mi::IBbox3> bbox_2( factory->create<mi::IBbox3>( "Bbox3"));
    MI_CHECK( bbox_1.is_valid_interface());
    MI_CHECK( bbox_2.is_valid_interface());
    bbox_1->set_value( 1, 2, 1);
    bbox_2->set_value( 1, 2, 2);
    MI_CHECK_NOT_EQUAL( bbox_2->get_value<mi::Sint32>( 1, 2), bbox_1->get_value<mi::Sint32>( 1, 2));
    CHECK_COMPARE( -1, bbox_1.get(), bbox_2.get());
    CHECK_ASSIGN( 0, bbox_1.get(), bbox_2.get());
    MI_CHECK_EQUAL( bbox_2->get_value<mi::Sint32>( 1, 2), bbox_1->get_value<mi::Sint32>( 1, 2));
    CHECK_COMPARE( 0, bbox_1.get(), bbox_2.get());


    // clone()
    mi::base::Handle<mi::IArray> array_1_clone( factory->clone<mi::IArray>( array_1.get()));
    MI_CHECK( array_1_clone.is_valid_interface());
    mi::base::Handle<mi::IDynamic_array> dynamic_array_1_clone(
        factory->clone<mi::IDynamic_array>( dynamic_array_1.get()));
    MI_CHECK( dynamic_array_1_clone.is_valid_interface());
    mi::base::Handle<mi::IMap> map_1_clone( factory->clone<mi::IMap>( map_1.get()));
    MI_CHECK( map_1_clone.is_valid_interface());
    mi::base::Handle<mi::ISint32_2> vector_1_clone( factory->clone<mi::ISint32_2>( vector_1.get()));
    MI_CHECK( vector_1_clone.is_valid_interface());
}

template<class I>
void test_type_traits_helper(
    mi::neuraylib::IFactory* factory,
    mi::neuraylib::ITransaction* transaction,
    const char* type_name)
{
    mi::base::Handle<mi::IData> data;
    std::string type_name_str( type_name);

    // check Type_traits
    std::string type_name_I = mi::Type_traits<I>::get_type_name();
    using P = typename mi::Type_traits<I>::Primitive_type;
    std::string type_name_P = mi::Type_traits<P>::get_type_name();
    using II = typename mi::Type_traits<P>::Interface_type;

    if( type_name_str.substr( 0, 3) == "Ref") {
        // the type traits map IRef to "Ref", and IRef/IString ambiguity for const char*
        MI_CHECK_EQUAL( type_name_I, "Ref");
        MI_CHECK_EQUAL( type_name_P, "String");
    } else if( type_name_str == "Size") {
        // Size/Uint64 ambiguity for mi::Uint64
        MI_CHECK_EQUAL( type_name_I, type_name_str);
        MI_CHECK_EQUAL( type_name_P, "Uint64");
    } else if( type_name_str == "Difference") {
        // Difference/Sint64 ambiguity for mi::Sint64
        MI_CHECK_EQUAL( type_name_I, type_name_str);
        MI_CHECK_EQUAL( type_name_P, "Sint64");
    } else if( type_name_str == "Color3") {
        // Color3/Color ambiguity for mi::math::Color
        MI_CHECK_EQUAL( type_name_I, type_name_str);
        MI_CHECK_EQUAL( type_name_P, "Color");
    } else {
        MI_CHECK_EQUAL( type_name_I, type_name_str);
        MI_CHECK_EQUAL( type_name_P, type_name_str);
        MI_CHECK( (std::is_same<II, I>::value));
    }

    // check IFactory::create<T>( const char*)
    data = factory->create<I>( type_name);
    if( type_name_str.substr( 0, 3) == "Ref") {
        // IFactory cannot handle IRef
        MI_CHECK( !data.is_valid_interface());
    } else {
        MI_CHECK( data.is_valid_interface());
        MI_CHECK( type_name_str == data->get_type_name());
    }

    // check IFactory::create<T>()
    data = factory->create<I>();
    if( type_name_str.substr( 0, 3) == "Ref") {
        // IFactory cannot handle IRef
        MI_CHECK( !data.is_valid_interface());
    } else {
        MI_CHECK( data.is_valid_interface());
        MI_CHECK( type_name_str == data->get_type_name());
    }

    // check ITransaction::create<T>( const char*)
    data = transaction->create<I>( type_name);
    MI_CHECK( data.is_valid_interface());
    MI_CHECK( type_name_str == data->get_type_name());

    // check ITransaction::create<T>()
    data = transaction->create<I>();
    MI_CHECK( data.is_valid_interface());
    MI_CHECK( type_name_str == data->get_type_name());
}

void test_type_traits( mi::neuraylib::IFactory* factory, mi::neuraylib::ITransaction* transaction)
{
    // IData_simple
    test_type_traits_helper<mi::IBoolean    >( factory, transaction, "Boolean");
    test_type_traits_helper<mi::ISint8      >( factory, transaction, "Sint8");
    test_type_traits_helper<mi::ISint16     >( factory, transaction, "Sint16");
    test_type_traits_helper<mi::ISint32     >( factory, transaction, "Sint32");
    test_type_traits_helper<mi::ISint64     >( factory, transaction, "Sint64");
    test_type_traits_helper<mi::IUint8      >( factory, transaction, "Uint8");
    test_type_traits_helper<mi::IUint16     >( factory, transaction, "Uint16");
    test_type_traits_helper<mi::IUint32     >( factory, transaction, "Uint32");
    test_type_traits_helper<mi::IUint64     >( factory, transaction, "Uint64");
    test_type_traits_helper<mi::IFloat32    >( factory, transaction, "Float32");
    test_type_traits_helper<mi::IFloat64    >( factory, transaction, "Float64");
    test_type_traits_helper<mi::ISize       >( factory, transaction, "Size");
    test_type_traits_helper<mi::IDifference >( factory, transaction, "Difference");
    test_type_traits_helper<mi::IString     >( factory, transaction, "String");
    test_type_traits_helper<mi::IUuid       >( factory, transaction, "Uuid");
    test_type_traits_helper<mi::IVoid       >( factory, transaction, "Void");
    test_type_traits_helper<mi::IRef        >( factory, transaction, "Ref");

    // IVector
    test_type_traits_helper<mi::IBoolean_2  >( factory, transaction, "Boolean<2>");
    test_type_traits_helper<mi::IBoolean_3  >( factory, transaction, "Boolean<3>");
    test_type_traits_helper<mi::IBoolean_4  >( factory, transaction, "Boolean<4>");
    test_type_traits_helper<mi::ISint32_2   >( factory, transaction, "Sint32<2>");
    test_type_traits_helper<mi::ISint32_3   >( factory, transaction, "Sint32<3>");
    test_type_traits_helper<mi::ISint32_4   >( factory, transaction, "Sint32<4>");
    test_type_traits_helper<mi::IUint32_2   >( factory, transaction, "Uint32<2>");
    test_type_traits_helper<mi::IUint32_3   >( factory, transaction, "Uint32<3>");
    test_type_traits_helper<mi::IUint32_4   >( factory, transaction, "Uint32<4>");
    test_type_traits_helper<mi::IFloat32_2  >( factory, transaction, "Float32<2>");
    test_type_traits_helper<mi::IFloat32_3  >( factory, transaction, "Float32<3>");
    test_type_traits_helper<mi::IFloat32_4  >( factory, transaction, "Float32<4>");
    test_type_traits_helper<mi::IFloat64_2  >( factory, transaction, "Float64<2>");
    test_type_traits_helper<mi::IFloat64_3  >( factory, transaction, "Float64<3>");
    test_type_traits_helper<mi::IFloat64_4  >( factory, transaction, "Float64<4>");

    // IMatrix
    test_type_traits_helper<mi::IBoolean_2_2>( factory, transaction, "Boolean<2,2>");
    test_type_traits_helper<mi::IBoolean_2_3>( factory, transaction, "Boolean<2,3>");
    test_type_traits_helper<mi::IBoolean_2_4>( factory, transaction, "Boolean<2,4>");
    test_type_traits_helper<mi::IBoolean_3_2>( factory, transaction, "Boolean<3,2>");
    test_type_traits_helper<mi::IBoolean_3_3>( factory, transaction, "Boolean<3,3>");
    test_type_traits_helper<mi::IBoolean_3_4>( factory, transaction, "Boolean<3,4>");
    test_type_traits_helper<mi::IBoolean_4_2>( factory, transaction, "Boolean<4,2>");
    test_type_traits_helper<mi::IBoolean_4_3>( factory, transaction, "Boolean<4,3>");
    test_type_traits_helper<mi::IBoolean_4_4>( factory, transaction, "Boolean<4,4>");
    test_type_traits_helper<mi::ISint32_2_2 >( factory, transaction, "Sint32<2,2>");
    test_type_traits_helper<mi::ISint32_2_3 >( factory, transaction, "Sint32<2,3>");
    test_type_traits_helper<mi::ISint32_2_4 >( factory, transaction, "Sint32<2,4>");
    test_type_traits_helper<mi::ISint32_3_2 >( factory, transaction, "Sint32<3,2>");
    test_type_traits_helper<mi::ISint32_3_3 >( factory, transaction, "Sint32<3,3>");
    test_type_traits_helper<mi::ISint32_3_4 >( factory, transaction, "Sint32<3,4>");
    test_type_traits_helper<mi::ISint32_4_2 >( factory, transaction, "Sint32<4,2>");
    test_type_traits_helper<mi::ISint32_4_3 >( factory, transaction, "Sint32<4,3>");
    test_type_traits_helper<mi::ISint32_4_4 >( factory, transaction, "Sint32<4,4>");
    test_type_traits_helper<mi::IUint32_2_2 >( factory, transaction, "Uint32<2,2>");
    test_type_traits_helper<mi::IUint32_2_3 >( factory, transaction, "Uint32<2,3>");
    test_type_traits_helper<mi::IUint32_2_4 >( factory, transaction, "Uint32<2,4>");
    test_type_traits_helper<mi::IUint32_3_2 >( factory, transaction, "Uint32<3,2>");
    test_type_traits_helper<mi::IUint32_3_3 >( factory, transaction, "Uint32<3,3>");
    test_type_traits_helper<mi::IUint32_3_4 >( factory, transaction, "Uint32<3,4>");
    test_type_traits_helper<mi::IUint32_4_2 >( factory, transaction, "Uint32<4,2>");
    test_type_traits_helper<mi::IUint32_4_3 >( factory, transaction, "Uint32<4,3>");
    test_type_traits_helper<mi::IUint32_4_4 >( factory, transaction, "Uint32<4,4>");
    test_type_traits_helper<mi::IFloat32_2_2>( factory, transaction, "Float32<2,2>");
    test_type_traits_helper<mi::IFloat32_2_3>( factory, transaction, "Float32<2,3>");
    test_type_traits_helper<mi::IFloat32_2_4>( factory, transaction, "Float32<2,4>");
    test_type_traits_helper<mi::IFloat32_3_2>( factory, transaction, "Float32<3,2>");
    test_type_traits_helper<mi::IFloat32_3_3>( factory, transaction, "Float32<3,3>");
    test_type_traits_helper<mi::IFloat32_3_4>( factory, transaction, "Float32<3,4>");
    test_type_traits_helper<mi::IFloat32_4_2>( factory, transaction, "Float32<4,2>");
    test_type_traits_helper<mi::IFloat32_4_3>( factory, transaction, "Float32<4,3>");
    test_type_traits_helper<mi::IFloat32_4_4>( factory, transaction, "Float32<4,4>");
    test_type_traits_helper<mi::IFloat64_2_2>( factory, transaction, "Float64<2,2>");
    test_type_traits_helper<mi::IFloat64_2_3>( factory, transaction, "Float64<2,3>");
    test_type_traits_helper<mi::IFloat64_2_4>( factory, transaction, "Float64<2,4>");
    test_type_traits_helper<mi::IFloat64_3_2>( factory, transaction, "Float64<3,2>");
    test_type_traits_helper<mi::IFloat64_3_3>( factory, transaction, "Float64<3,3>");
    test_type_traits_helper<mi::IFloat64_3_4>( factory, transaction, "Float64<3,4>");
    test_type_traits_helper<mi::IFloat64_4_2>( factory, transaction, "Float64<4,2>");
    test_type_traits_helper<mi::IFloat64_4_3>( factory, transaction, "Float64<4,3>");
    test_type_traits_helper<mi::IFloat64_4_4>( factory, transaction, "Float64<4,4>");

    // remaining ICompound interfaces
    test_type_traits_helper<mi::IColor    >( factory, transaction, "Color");
    test_type_traits_helper<mi::IColor3   >( factory, transaction, "Color3");
    test_type_traits_helper<mi::ISpectrum >( factory, transaction, "Spectrum");
    test_type_traits_helper<mi::IBbox3    >( factory, transaction, "Bbox3");
}

void run_tests( mi::neuraylib::INeuray* neuray)
{
    MI_CHECK_EQUAL( 0, neuray->start());

    {
        mi::base::Handle<mi::neuraylib::IDatabase> database(
            neuray->get_api_component<mi::neuraylib::IDatabase>());
        MI_CHECK( database.is_valid_interface());
        mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());
        MI_CHECK( scope.is_valid_interface());
        mi::base::Handle<mi::neuraylib::ITransaction> transaction( scope->create_transaction());
        MI_CHECK( transaction.is_valid_interface());


        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        MI_CHECK( factory.is_valid_interface());

        // test assign() with non-zero result code, clone()
        test_null_pointer( factory.get());
        test_structural_mismatch( factory.get());
        test_no_conversion( factory.get(), transaction.get());
        test_key_missing( factory.get());
        test_different_collections( factory.get());
        test_non_idata_values( factory.get(), transaction.get());
        test_incompatible_pointer_types( factory.get(), transaction.get());
        test_deep_assignment_to_const_pointer( factory.get(), transaction.get());

        // test assign() with zero result code, compare(), clone()
        test_idata_simple( factory.get(), transaction.get());
        test_idata_collection( factory.get());

        // test create() with type traits
        test_type_traits( factory.get(), transaction.get());

        MI_CHECK_EQUAL( 0, transaction->commit());
    }

    MI_CHECK_EQUAL( 0, neuray->shutdown());
}

MI_TEST_AUTO_FUNCTION( test_ifactory )
{
    mi::base::Handle<mi::neuraylib::INeuray> neuray( load_and_get_ineuray());
    MI_CHECK( neuray.is_valid_interface());

    {
        mi::base::Handle<mi::neuraylib::IDebug_configuration> debug_configuration(
            neuray->get_api_component<mi::neuraylib::IDebug_configuration>());
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_store=1"));
        MI_CHECK_EQUAL( 0, debug_configuration->set_option( "check_serializer_edit=1"));


        mi::base::Handle<mi::neuraylib::IFactory> factory(
            neuray->get_api_component<mi::neuraylib::IFactory>());
        MI_CHECK( factory.is_valid_interface());


        run_tests( neuray.get());
        run_tests( neuray.get());
    }

    neuray = nullptr;
    MI_CHECK( unload());
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

