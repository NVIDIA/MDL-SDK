/******************************************************************************
 * Copyright (c) 2008-2024, NVIDIA CORPORATION. All rights reserved.
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

/// \file test_type.cpp
/// \brief The tests for the class ATTR::Type

#include "pch.h"

#include <base/data/attr/attr.h>
#include <base/system/test/i_test_auto_case.h>
#include <base/lib/mem/mem.h>
#include <vector>
#include <base/lib/log/i_log_module.h>
#include <base/system/main/access_module.h>

using namespace MI;
using namespace MI::ATTR;
using namespace MI::MEM;

#undef EXPERIMENTAL_ARRAYS_OF_ARRAYS_MODE

class Type_test_suite : public TEST::Test_suite
{
private:
    SYSTEM::Access_module<LOG::Log_module> m_log_module;

public:
    Type_test_suite() : TEST::Test_suite("ATTR::Type Test Suite")
    {
        m_log_module.set();
        add( MI_TEST_METHOD(Type_test_suite, test_constructor) );
        add( MI_TEST_METHOD(Type_test_suite, test_copy_constructor) );
        add( MI_TEST_METHOD(Type_test_suite, test_set_name) );
        add( MI_TEST_METHOD(Type_test_suite, test_equality) );
        add( MI_TEST_METHOD(Type_test_suite, test_size_of_elem) );
        add( MI_TEST_METHOD(Type_test_suite, test_array_type) );
        add( MI_TEST_METHOD(Type_test_suite, test_lookup) );
        add( MI_TEST_METHOD(Type_test_suite, test_sizeof_all) );
    }

    ~Type_test_suite()
    {
        m_log_module.reset();
    }

    void test_constructor()
    {
        // default constructor
        {
        const Type type;
        MI_CHECK_EQUAL(type.get_name(), 0);
        MI_CHECK_EQUAL(type.get_typecode(), ATTR::TYPE_UNDEF);
        MI_CHECK_EQUAL(type.get_arraysize(), 1);
        MI_CHECK_EQUAL(type.get_next(), 0);
        MI_CHECK_EQUAL(type.get_child(), 0);
        MI_CHECK_EQUAL(type.get_const(), false);
        }
    }

    void test_copy_constructor()
    {
        // default copy constructor
        {
        const Type orig;
        const Type type(orig);
        MI_CHECK_EQUAL(type.get_name(), 0);
        MI_CHECK_EQUAL(type.get_typecode(), ATTR::TYPE_UNDEF);
        MI_CHECK_EQUAL(type.get_arraysize(), 1);
        MI_CHECK_EQUAL(type.get_next(), 0);
        MI_CHECK_EQUAL(type.get_child(), 0);
        MI_CHECK_EQUAL(type.get_const(), false);
        }
        // copy constructor of initialized object
        {
        const Type orig(ATTR::TYPE_INT32, "ali", 0);
        const Type type(orig);
        MI_CHECK(strcmp(type.get_name(), "ali") == 0);
        MI_CHECK_EQUAL(type.get_typecode(), ATTR::TYPE_INT32);
        MI_CHECK_EQUAL(type.get_arraysize(), 0);
        MI_CHECK_EQUAL(type.get_next(), 0);
        MI_CHECK_EQUAL(type.get_child(), 0);
        MI_CHECK_EQUAL(type.get_const(), false);
        }
    }

    void test_set_name()
    {
        {
        Type type;
        type.set_name("ali");
        MI_CHECK(strcmp(type.get_name(), "ali") == 0);
        }
    }

    void test_equality()
    {
        {
        const Type one(TYPE_BOOLEAN, "single type");
        const Type other(TYPE_BOOLEAN, "array type");
        MI_CHECK(one == other);
        }
    }

    void test_size_of_elem()
    {
        // skip testing non-struct types since those fallback onto sizeof_one()
        {}
        // since structs have size == 0 even nested structs should have no additional size
        {
        Type root(TYPE_STRUCT, "root", 1);
        Type child_root(TYPE_STRUCT, "child_root", 1);
        Type leaf(TYPE_INT32, "leaf", 3);
        Type empty_leaf(TYPE_STRUCT, "", 1);
        leaf.set_next(empty_leaf);
        child_root.set_child(leaf);
        root.set_child(child_root);
        MI_CHECK_EQUAL(root.sizeof_elem(), 3*sizeof(Sint32));
        }
    }

    void test_sizeof_all()
    {
        {
        Type root(TYPE_INT32, "root", 1);
        MI_CHECK_EQUAL(root.sizeof_all(), root.get_arraysize()*sizeof(Sint32));
        }
        {
        Type root(TYPE_INT32, "root", 5);
        MI_CHECK_EQUAL(root.sizeof_all(), root.get_arraysize()*sizeof(Sint32));
        }
        {
        Type root(TYPE_STRUCT, "root", 5);
        Type child_01(TYPE_INT32, "child1", 2);
        Type child_02(TYPE_BOOLEAN, "child2", 5);
        child_01.set_next(child_02);
        root.set_child(child_01);

        // 2 x sizeof(int)  --> 8
        // 5 x sizeof(bool) --> 5
        //                    ---
        //                     13
        //                    ---
        // align to int==4 --> 16 * arraysize = 80
        //                                      ==
        MI_CHECK_EQUAL(root.sizeof_all(), root.get_arraysize() * 16);
        }
    }

    void test_array_type()
    {
        // empty array
        {
        const Type array_type(TYPE_ARRAY, "array_type", 0);
        MI_CHECK_EQUAL(array_type.get_arraysize(), 0);
        MI_CHECK_EQUAL(array_type.component_count(), 0);
        MI_CHECK_EQUAL(array_type.component_type(), TYPE_UNDEF);
        MI_CHECK(strcmp(array_type.type_name(), "array") == 0);
        }
        // simple arrray
        {
        Type array_type(TYPE_ARRAY, "array_type", 0);
        const Type elem_type(TYPE_INT32, "element", 12);
        array_type.set_child(elem_type);
        MI_CHECK_EQUAL(array_type.get_arraysize(), 12);
        array_type.set_arraysize(12);
        MI_CHECK_EQUAL(array_type.get_arraysize(), 12);
        MI_CHECK_EQUAL(array_type.component_count(), 1);
        MI_CHECK_EQUAL(array_type.component_type(), TYPE_INT32);
        }
        // nested array
        {
        Type array_type(TYPE_ARRAY, "array_type", 0);
        Type inner_array(TYPE_ARRAY, "inner_array", 0);
        Type elem_type(TYPE_INT32, "element", 12);
        inner_array.set_child(elem_type);
        inner_array.set_arraysize(12);
        array_type.set_child(inner_array);
        // the "root" array has array size of 0 (since it must be set via set_arraysize()), with
        // unknown component count but of array type_code
        MI_CHECK_EQUAL(array_type.get_arraysize(), 0);
        MI_CHECK_EQUAL(array_type.component_count(), 0);
#ifdef EXPERIMENTAL_ARRAYS_OF_ARRAYS_MODE
        MI_CHECK_EQUAL(array_type.component_type(), TYPE_ARRAY);
#else
        MI_CHECK_EQUAL(array_type.component_type(), TYPE_UNDEF);
#endif
        }
        // dynamic array
        {
        Type elem_type(TYPE_INT32, "ints");
        Type dyn_array_type(elem_type, 0);
        MI_CHECK_EQUAL(dyn_array_type.get_arraysize(), 0);
        MI_CHECK_EQUAL(dyn_array_type.get_typecode(), TYPE_INT32);
        MI_CHECK_EQUAL(dyn_array_type.get_typecode(), elem_type.get_typecode());
        MI_CHECK_EQUAL(strcmp(dyn_array_type.get_name(), elem_type.get_name()), 0);
        }
    }

    void test_lookup()
    {
        // return 0 if not found
        {
        Type type;
        Uint offs;
        MI_CHECK_EQUAL(type.lookup("dummy", &offs), 0);
        // same for array
        Type array(TYPE_ARRAY, "array", 0);
        array.set_child(type);
        array.set_arraysize(4);
        MI_CHECK_EQUAL(array.lookup("dummy", &offs), 0);
        }
        // standard behaviour
        {
        Type type(TYPE_INT32, "dummy", 4);
        Uint offs;
        const Type* res = type.lookup("dummy", &offs);
        MI_CHECK_EQUAL(res, &type);
        MI_CHECK_EQUAL(offs, 0);
        // same for array
        Type array(TYPE_ARRAY, "array", 0);
        array.set_child(type);
        array.set_arraysize(4);
        // looking up w/o an index returns the array itself
        res = array.lookup("dummy", &offs);
        MI_REQUIRE(res);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_ARRAY);
        MI_CHECK_EQUAL(offs, 0);
        MI_CHECK_EQUAL(*res, array);
        // looking up with an index returns the array element type
        res = array.lookup("dummy[0]", &offs);
        MI_REQUIRE(res);
        MI_CHECK(res->get_typecode() == TYPE_INT32);
        MI_CHECK_EQUAL(offs, 0);
        MI_CHECK_EQUAL(*res, type);
        }
        {
        // original structure
        // build an attribute type tree: struct [100] {}.
        // typedef struct { char a[5]; } Rgbea;
        // struct Atype {struct {Rgbea rgbea[2]; Color color; char byte;} s[100]; };
#define ATTRNAME "attr"
        Type type0(TYPE_STRUCT, ATTRNAME, 100);    // struct {
        Type type1(TYPE_RGBEA,  "rgbea", 2);		//   Rgbea[2];
        Type type2(TYPE_COLOR,  "color");		//   Scalar[4];
        Type type3(TYPE_INT8,   "byte");		//   char;
        type1.set_const();
        type2.set_next(type3);
        type1.set_next(type2);
        type0.set_child(type1);			    // } [100]
        Uint offs;
        MI_REQUIRE_EQUAL(type0, *type0.lookup(ATTRNAME,			&offs));
        MI_REQUIRE_EQUAL(offs, 0);
        MI_REQUIRE_EQUAL(type1, *type0.lookup(ATTRNAME ".rgbea",	&offs));
        MI_REQUIRE_EQUAL(offs, 0);
        MI_REQUIRE_EQUAL(type1, *type0.lookup(ATTRNAME ".rgbea[0]",	&offs));
        MI_REQUIRE_EQUAL(offs, 0);
        MI_REQUIRE_EQUAL(type1, *type0.lookup(ATTRNAME ".rgbea[1]",	&offs));
        MI_REQUIRE_EQUAL(offs, 5);
        MI_REQUIRE_EQUAL(type2, *type0.lookup(ATTRNAME ".color",	&offs));
        MI_REQUIRE_EQUAL(offs, 12);
        MI_REQUIRE_EQUAL(type3, *type0.lookup(ATTRNAME ".byte",	&offs));
        MI_REQUIRE_EQUAL(offs, 28);
        MI_REQUIRE(!type0.lookup(ATTRNAME ".rgbea[2]", &offs));
        MI_REQUIRE(!type0.lookup(ATTRNAME ".foo",      &offs));

        // check correct offset computation
        Uint offset_01 = 0;
        type0.lookup(ATTRNAME ".rgbea[0]", &offset_01);
        Uint offset_02 = 0;
        type0.lookup(ATTRNAME ".rgbea[1]", &offset_02);
        MI_REQUIRE_LESS(offset_01, offset_02);

        const Type* res = type0.lookup("attr[1].byte", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_INT8);
        MI_CHECK_EQUAL(res->get_arraysize(), 1);
        MI_REQUIRE_EQUAL(offs, 60);

#undef ATTRNAME
        }
        {
        // use exactly the same layout as above but inbetween using array types now -
        // this should NOT change the lookup() at all
        // typedef struct { char a[5]; } Rgbe;
        // struct Atype {struct {Rgbe rgbe[2]; Color color; char byte;} s[100]; };
#define ATTRNAME "attr"
        Type array0(TYPE_ARRAY, ATTRNAME, 0); // here the name does not matter
        Type type0(TYPE_STRUCT, ATTRNAME, 100);
        Type array1(TYPE_ARRAY, 0, 0);
        Type type1(TYPE_RGBEA, "rgbea", 2);
        Type type2(TYPE_COLOR, "color");
        Type type3(TYPE_INT8, "byte");
        type1.set_const();
        type2.set_next(type3);
        type1.set_next(type2);
        array1.set_child(type1);
        array1.set_arraysize(type1.get_arraysize());
        type0.set_child(array1);
        array0.set_child(type0); // this will overwrite array0's name with type0's name
        Uint offs;
        // array0
        const Type* res = array0.lookup("attr", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_ARRAY);
        MI_CHECK_EQUAL(res->get_arraysize(), 100);
        MI_REQUIRE_EQUAL(offs, 0);
        // type0
        res = array0.lookup("attr[0]", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_STRUCT);
        MI_CHECK_EQUAL(res->get_arraysize(), 100);
        MI_REQUIRE_EQUAL(offs, 0);
        // array1
        res = array0.lookup("attr[0].rgbea", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_ARRAY);
        MI_CHECK_EQUAL(res->get_arraysize(), 2);
        MI_REQUIRE_EQUAL(offs, 0);
        // type1[0]
        res = array0.lookup("attr[0].rgbea[0]", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_RGBEA);
        MI_CHECK_EQUAL(res->get_arraysize(), 2);
        MI_REQUIRE_EQUAL(offs, 0);
        // type1[1]
        res = array0.lookup("attr[0].rgbea[1]", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_RGBEA);
        MI_CHECK_EQUAL(res->get_arraysize(), 2);
        MI_REQUIRE_EQUAL(offs, 5);
        // type2
        res = array0.lookup("attr[0].color", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_COLOR);
        MI_CHECK_EQUAL(res->get_arraysize(), 1);
        MI_REQUIRE_EQUAL(offs, 12);
        // type3
        res = array0.lookup("attr[0].byte", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_INT8);
        MI_CHECK_EQUAL(res->get_arraysize(), 1);
        MI_REQUIRE_EQUAL(offs, 28);

        res = array0.lookup("attr[1].byte", &offs);
        MI_CHECK_EQUAL(res->get_typecode(), TYPE_INT8);
        MI_CHECK_EQUAL(res->get_arraysize(), 1);
        MI_REQUIRE_EQUAL(offs, 60);
#undef ATTRNAME
        }
    }
};

MI_TEST_AUTO_CASE( new Type_test_suite );
