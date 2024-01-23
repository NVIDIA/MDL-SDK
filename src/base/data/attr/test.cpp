/******************************************************************************
 * Copyright (c) 2004-2024, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief test attributes
///
/// create a tag, attach attributes to it, and check that they are retrievable.

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "ATTR Test Suite"

#include <base/lib/log/i_log_logger.h>
#include <base/lib/log/i_log_module.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_database.h>
#include <base/data/attr/attr.h>
#include <base/data/serial/i_serializer.h>
#include <base/data/serial/i_serial_buffer_serializer.h>
#include <base/system/main/access_module.h>

#include <base/system/stlext/i_stlext_no_unused_variable_warning.h>
#include <base/system/test/i_test_auto_driver.h>

#include <cstdio>

using namespace MI;
using namespace MI::LOG;
using namespace MI::DB;
using namespace MI::ATTR;
using namespace MI::SYSTEM;
using namespace MI::SERIAL;
using namespace MI::STLEXT;


// Define pretty-printers for error-reporting.

namespace std {

inline std::ostream & operator<<(
    std::ostream & os,
    MI::ATTR::Type const & t)
{
    os << t.print();
    return os;
}

}

#include "test_type.h"
#include "test_attr_set.h"
#include "test_attribute.h"
#include "test_attribute_type.h"

MI_TEST_AUTO_FUNCTION( test_pointer_attribute )
{
    Access_module<Attr_module> attr_module(false);
    Access_module<Log_module> log_module(false);

    //
    // check attribute of type TYPE_STRING
    //
    Attribute string_attr(Attribute::id_create("test_string"), TYPE_STRING, 1);
    Attribute::set_string(*(char **)string_attr.set_values(), "test_string");
    Attribute string_attr_copy(string_attr);

    const char** p1 = (const char **)string_attr.get_values();
    const char** p2 = (const char **)string_attr_copy.get_values();
    MI_REQUIRE(strcmp(*p1, *p2) == 0);

    attr_module.reset();
}

//
// Create a database element, attach an attribute to it, and make sure that it
// is possible to retrieve the attribute from the element. Warning - this code
// puts the element, the attribute, and the types all on the stack; a real
// program must allocate them on the heap because the ATTR module does not make
// copies!
//

MI_TEST_AUTO_FUNCTION( verify_attribute_system )
{
    SYSTEM::Access_module<Attr_module> attr_module(false);
    SYSTEM::Access_module<Log_module> log_module(false);
    using std::swap;

    //
    // build an attribute type tree: struct [100] {}.
    // the toplevel type (type0) must have the attribute's name.
    //
    struct Atype {struct {char rgbea[5][2]; mi::math::Color color; char byte;} s[100]; };
#   define ATTRNAME "attr"
    Type *type0 = new Type(TYPE_STRUCT, ATTRNAME, 100);	// struct {
    Type *type1 = new Type(TYPE_RGBEA,  "rgbea", 2);	//   char[5][2];
    Type *type2 = new Type(TYPE_COLOR,  "color");	//   Scalar[4];
    Type *type3 = new Type(TYPE_INT8,   "byte");	//   char;
    type1->set_const();
    type2->set_next(*type3);
    type1->set_next(*type2);
    type0->set_child(*type1);				// } [100]

    MI_REQUIRE_EQUAL(type0->align_one(), 1);
    MI_REQUIRE_EQUAL(type1->align_one(), 1);
    MI_REQUIRE_EQUAL(type2->align_one(), sizeof(Scalar));
    MI_REQUIRE_EQUAL(type3->align_one(), 1);
    MI_REQUIRE_EQUAL(type0->align_all(), sizeof(Scalar));

    MI_REQUIRE_EQUAL(type0->sizeof_one(), 0);
    MI_REQUIRE_EQUAL(type1->sizeof_one(), 2 * 5);
    MI_REQUIRE_EQUAL(type2->sizeof_one(), 4 * sizeof(Scalar));
    MI_REQUIRE_EQUAL(type3->sizeof_one(), 1);
    MI_REQUIRE_EQUAL(type0->sizeof_all(), 100 * (2 * 5 +	// rgbea[2]
                                                 2 +		// padding
                                                 4 * sizeof(Scalar)+ //float[4]
                                                 1 +		// char
                                                 3));		// padding
    //
    // test lookup by name
    //
    Uint offs;
    no_unused_variable_warning_please(offs);
    MI_REQUIRE_EQUAL(*type0, *type0->lookup(ATTRNAME,		  &offs));
    MI_REQUIRE_EQUAL(offs, 0);
    MI_REQUIRE_EQUAL(*type1, *type0->lookup(ATTRNAME ".rgbea",    &offs));
    MI_REQUIRE_EQUAL(offs, 0);
    MI_REQUIRE_EQUAL(*type1, *type0->lookup(ATTRNAME ".rgbea[0]", &offs));
    MI_REQUIRE_EQUAL(offs, 0);
    MI_REQUIRE_EQUAL(*type1, *type0->lookup(ATTRNAME ".rgbea[1]", &offs));
    MI_REQUIRE_EQUAL(offs, 5);
    MI_REQUIRE_EQUAL(*type2, *type0->lookup(ATTRNAME ".color",    &offs));
    MI_REQUIRE_EQUAL(offs, 12);
    MI_REQUIRE_EQUAL(*type3, *type0->lookup(ATTRNAME ".byte",     &offs));
    MI_REQUIRE_EQUAL(offs, 28);
    MI_REQUIRE(!type0->lookup(ATTRNAME ".rgbea[2]", &offs));
    MI_REQUIRE(!type0->lookup(ATTRNAME ".foo",      &offs));

    // check correct offset computation
    Uint offset_01 = 0;
    type0->lookup(ATTRNAME ".rgbea[0]", &offset_01);
    Uint offset_02 = 0;
    type0->lookup(ATTRNAME ".rgbea[1]", &offset_02);
    MI_REQUIRE_LESS(offset_01, offset_02);

    {
#   define ATTR_NAME "attr_new"
    Type type0(TYPE_STRUCT, ATTR_NAME, 100);// struct {
    Type type1(TYPE_RGBEA,  "rgbea", 2);	//   char[5][2];
    Type type2(TYPE_COLOR,  "color");	//   Scalar[4];
    Type type3(TYPE_STRUCT, "colors", 2); //   struct {
    Type type3_1(TYPE_COLOR, "ambient");    //     Scalar[4];} [2];
    Type type4(TYPE_INT8,   "byte");	//   char;
    // due to new deep copy behaviour Types are *not* shared anymore!
    // This means that set_XXX() functions do not work as expected anymore.
    /* old order
    type0->set_child(*type1);				// } [100]
    type1->set_next(*type2);
    type2->set_next(*type3);
    type3->set_child(*type3_1);
    type3->set_next(*type4);
    type1->set_const();
    type2->set_global();
    */
    type1.set_const();
    type3.set_next(type4);
    type3.set_child(type3_1);
    type2.set_next(type3);
    type1.set_next(type2);
    type0.set_child(type1);				// } [100]

    //
    // test lookup by name
    //
    Uint offs;
    no_unused_variable_warning_please(offs);
    MI_REQUIRE_EQUAL(type0, *type0.lookup(ATTR_NAME,		  &offs));
    MI_REQUIRE_EQUAL(offs, 0);
    MI_REQUIRE_EQUAL(type1, *type0.lookup(ATTR_NAME ".rgbea",    &offs));
    MI_REQUIRE_EQUAL(offs, 0);
    MI_REQUIRE_EQUAL(type1, *type0.lookup(ATTR_NAME ".rgbea[0]", &offs));
    MI_REQUIRE_EQUAL(offs, 0);
    MI_REQUIRE_EQUAL(type1, *type0.lookup(ATTR_NAME ".rgbea[1]", &offs));
    MI_REQUIRE_EQUAL(offs, 5);
    MI_REQUIRE_EQUAL(type2, *type0.lookup(ATTR_NAME ".color",    &offs));
    MI_REQUIRE_EQUAL(offs, 12);
    MI_REQUIRE_EQUAL(type3, *type0.lookup(ATTR_NAME ".colors", &offs));
    MI_REQUIRE_EQUAL(offs, 28);
    Uint offs_01=0;
    Uint offs_02=0;
    MI_REQUIRE_EQUAL(
        type3_1, *type0.lookup(ATTR_NAME ".colors[0].ambient", &offs_01));
    MI_REQUIRE_EQUAL(
        type3_1, *type0.lookup(ATTR_NAME ".colors[1].ambient", &offs_02));
    MI_REQUIRE_LESS(offs_01, offs_02);
    MI_REQUIRE_EQUAL(type4, *type0.lookup(ATTR_NAME ".byte",     &offs));
    }

    //
    // create an attribute ID from the attribute name. Attribute IDs are
    // used internally to avoid frequent hashes and string comparisons.
    //
    Attribute_id id = Attribute::id_create(ATTRNAME);
    MI_REQUIRE_EQUAL(id, Attribute::id_lookup(ATTRNAME));

    //
    // human-readable (sort of) dump of the type tree
    //
    std::string descr = type0->print();
    MI_REQUIRE(!strcmp(descr.c_str(),
                "struct[100] { const rgbea[2] rgbea; color color; int8 byte; }"));

    //
    // Attribute_set flags
    //
    Attribute_set flag_test;
    set_bool_attrib(flag_test, 0, true);
//    flag_test.set_flag_present(0, true);
    set_bool_attrib(flag_test, 10, true);
//    flag_test.set_flag_present(10, true);

    MI_REQUIRE(get_bool_attrib(flag_test, 0));
    MI_REQUIRE(get_bool_attrib(flag_test, 10));

    // reset all present/value flags
    for (Attribute_id f=0; f < reserved_flag_ids; ++f)
        set_bool_attrib(flag_test, f, false);
    for (Attribute_id f=0; f < reserved_flag_ids; ++f)
        flag_test.detach(f);

    MI_REQUIRE(!get_bool_attrib(flag_test, 0));
    MI_REQUIRE(!get_bool_attrib(flag_test, 10));
    {

    // create a type hierarchy first - note that each parameter will end up in its own attribute!
    Type root(TYPE_STRUCT,      "root");
    Type null(TYPE_BOOLEAN,       "boolean 00");
    Type first(TYPE_BOOLEAN,      "boolean 01");
    Type second(TYPE_INT32,       "int32 01");
    Type third(TYPE_STRUCT,       "struct 01");
    Type next_first(TYPE_BOOLEAN,   "boolean child 01");
    Type next_second(TYPE_INT32,    "int32 child 01");
    Type fourth(TYPE_INT32,       "int32 02");

    third.set_next(fourth);
    next_first.set_next(next_second);
    third.set_child(next_first);
    second.set_next(third);
    first.set_next(second);
    null.set_next(first);
    root.set_child(null);

    MI_REQUIRE_EQUAL(root.sizeof_one(), 0);
    MI_REQUIRE_EQUAL(null.sizeof_one(), 1);
    MI_REQUIRE_EQUAL(first.sizeof_one(), 1);
    MI_REQUIRE_EQUAL(second.sizeof_one(), 4);
    MI_REQUIRE_EQUAL(third.sizeof_one(), 0);
    MI_REQUIRE_EQUAL(fourth.sizeof_one(), 4);
    MI_REQUIRE_EQUAL(next_first.sizeof_one(), 1);
    MI_REQUIRE_EQUAL(next_second.sizeof_one(), 4);

    MI_REQUIRE_EQUAL(root.align_one(), 1);
    MI_REQUIRE_EQUAL(null.align_one(), 1);
    MI_REQUIRE_EQUAL(first.align_one(), 1);
    MI_REQUIRE_EQUAL(second.align_one(), 4);
    MI_REQUIRE_EQUAL(third.align_one(), 1);
    MI_REQUIRE_EQUAL(fourth.align_one(), 4);
    MI_REQUIRE_EQUAL(fourth.align_all(), 4);

    Uint offs;
    const Type* t = null.lookup("boolean 00", &offs);
    MI_REQUIRE_EQUAL(*t, null);
    MI_REQUIRE_EQUAL(offs, 0);
    t = root.lookup("root.boolean 01", &offs);
    MI_REQUIRE_EQUAL(*t, first);
    MI_REQUIRE_EQUAL(offs, 1);

    t = root.lookup("root.int32 01", &offs);
    MI_REQUIRE(t);
    MI_REQUIRE_EQUAL(*t, second);
    MI_REQUIRE_EQUAL(offs, 4);

    t = root.lookup("root.struct 01", &offs);
    MI_REQUIRE_EQUAL(*t, third);
    MI_REQUIRE_EQUAL(offs, 8);
    t = root.lookup("root.struct 01.boolean child 01", &offs);
    MI_REQUIRE_EQUAL(*t, next_first);
    MI_REQUIRE_EQUAL(offs, 8);
    t = root.lookup("root.struct 01.int32 child 01", &offs);
    MI_REQUIRE_EQUAL(*t, next_second);
    MI_REQUIRE_EQUAL(offs, 12);
    t = root.lookup("root.int32 02", &offs);
    MI_REQUIRE_EQUAL(*t, fourth);
    MI_REQUIRE_EQUAL(offs, 16);
    }

    // create the attribute first
    //   scalar "Length",
    //   scalar "Width",
    //   scalar "Height",
    //   transform "Matrix",
    //   struct "NorthTexture" {
    //       color texture "Texture",
    //       boolean "RepeatU"
    //   },
    //   color "BgColor",

    {
    Attribute_set check_set_01;

    // add some attributes
    auto attr_01 = std::make_shared<Attribute>(TYPE_BOOLEAN, "boolean 01");
    bool result = check_set_01.attach(attr_01);
    MI_REQUIRE(result);
    auto attr_02 = std::make_shared<Attribute>(TYPE_BOOLEAN, "boolean 02");
    result = check_set_01.attach(attr_02);
    MI_REQUIRE(result);
    auto attr_03 = std::make_shared<Attribute>(TYPE_INT32, "int32 01");
    result = check_set_01.attach(attr_03);
    MI_REQUIRE(result);
    auto attr_04 = std::make_shared<Attribute>(TYPE_BOOLEAN, "boolean 03");
    result = check_set_01.attach(attr_04);
    MI_REQUIRE(result);
    auto attr_05 = std::make_shared<Attribute>(TYPE_COLOR, "color 01");
    result = check_set_01.attach(attr_05);
    MI_REQUIRE(result);

    // create a type hierarchy first
    Type root(TYPE_BOOLEAN, "boolean root");
    Type first(TYPE_BOOLEAN, "boolean 01");
    Type second(TYPE_INT32, "int32 01");
    Type third(TYPE_STRUCT, "struct 01");
    Type next_first(TYPE_BOOLEAN, "boolean child 01");
    Type next_second(TYPE_INT32, "int32 child 01");
    Type fourth(TYPE_INT32, "int32 02");

    next_first.set_next(next_second);
    third.set_child(next_first);
    third.set_next(fourth);
    second.set_next(third);
    first.set_next(second);
    root.set_next(first);

    const char* name = "attr_06";
    Attribute_id id = Attribute::id_create(name);

    auto attr_06 = std::make_shared<Attribute>(id, root);
    result = check_set_01.attach(attr_06);
    MI_REQUIRE(result);

    }

    //
    // test normal serialization
    //
    Buffer_serializer ser1;
    Buffer_deserializer deser1;
    type0->serialize(&ser1);
    Type full;
    deser1.deserialize(&full, ser1.get_buffer(), ser1.get_buffer_size());
    MI_REQUIRE_EQUAL(full, *type0);
    MI_REQUIRE(
           !strcmp(full.get_name(),
                 type0->get_name()));
    // need to introduce this const ref to get access to the const version
    const Type& full_ref(full);
    const Type* type0_child = const_cast<const Type*>(type0)->get_child();
    MI_REQUIRE(
           !strcmp(full_ref.get_child()->get_name(), type0_child->get_name()));
    MI_REQUIRE(
           !strcmp(full_ref.get_child()->get_next()->get_name(),
                 type0_child->get_next()->get_name()));
    MI_REQUIRE(
           !strcmp(full_ref.get_child()->get_next()->get_next()->get_name(),
                 type0_child->get_next()->get_next()->get_name()));

    //
    // misc
    //
    delete type0;
    delete type3;
    delete type1;
    delete type2;
    MI_REQUIRE_EQUAL(Type::component_count(TYPE_COLOR), 4);
    MI_REQUIRE_EQUAL(Type::component_type (TYPE_COLOR), TYPE_SCALAR);

    attr_module.reset();
}
