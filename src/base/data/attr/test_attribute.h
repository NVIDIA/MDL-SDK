/***************************************************************************************************
 * Copyright (c) 2008-2026, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The tests for the class \c ATTR::Attribute

#include "pch.h"

#include "attr.h"
#include "i_attr_utilities.h"
#include "test_utilities.h"

#include <base/data/serial/i_serializer.h>
#include <base/system/main/access_module.h>
#include <base/system/test/i_test_auto_driver.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_logger.h>

using namespace MI;
using namespace ATTR;
using namespace MEM;
using namespace std;

bool compare_iterators(
    Type_iterator& it_one,
    Type_iterator& it_other)
{
    // iterate over all elements of the type
    for (; !it_one.at_end(); it_one.to_next(), it_other.to_next()) {
        char* one_value = it_one.get_value();
        char* other_value = it_other.get_value();
        int arraysize = it_one->get_arraysize();
        if (arraysize != it_other->get_arraysize())
            return false;
        if (it_one->get_typecode() != it_other->get_typecode())
            return false;

        if (!arraysize && it_one->get_typecode() != TYPE_ARRAY) {
            // this is a dynamic array
            Dynamic_array* one_array = (Dynamic_array*)one_value;
            Dynamic_array* other_array = (Dynamic_array*)other_value;
            if (one_array->m_count != other_array->m_count)
                return false;
            arraysize = one_array->m_count;
            one_value = one_array->m_value;
            other_value = other_array->m_value;
        }

        if (it_one->get_typecode() == TYPE_STRUCT) {
            if (it_one->sizeof_elem() != it_other->sizeof_elem())
                return false;
            size_t size = it_one->sizeof_elem();
            for (int a=0; a < arraysize; ++a) {
                Type_iterator one_sub(&it_one, one_value);
                Type_iterator other_sub(&it_other, other_value);
                if (!compare_iterators(one_sub, other_sub))
                    return false;
                one_value += size;
                other_value += size;
            }
        }
        else {
            int type, count, size;
            Type_code type_code = it_one->get_typecode();
            if (type_code == TYPE_ARRAY) {
                type_code = it_one->get_child()->get_typecode();
                if (it_other->get_child()->get_typecode() != type_code)
                    return false;
            }
            eval_typecode(type_code, &type, &count, &size);

            for (int a=0; a < arraysize; ++a) {
                for (int i=0; i < count; ++i) {
                    switch(type) {
                          case '*': {
                              if (*(char **)one_value) {
                                  if (!*(char **)other_value)
                                      return false;
                                  if (strcmp(*(char **)one_value, *(char **)other_value))
                                      return false;
                              }
                              break;
                                    }
                          case 'c':
                          case 's':
                          case 'i':
                          case 'q':
                          case 'f':
                          case 'd':
                          case 't':
                              for (int s=0; s<size; ++s)
                                  if (one_value[s] != other_value[s])
                                      return false;
                              break;
                          default:  ASSERT(M_ATTR, 0);
                    }
                    one_value += size;
                    other_value += size;
                }
            }
        }
    }
    return true;
}

/// Return whether the two given \c Attributes are equal or not.
bool are_equal(
    const Attribute& one,
    const Attribute& other)
{
    Type_iterator it_one(&one.get_type(), const_cast<char*>(one.get_values()));
    Type_iterator it_other(&other.get_type(), const_cast<char*>(other.get_values()));

    if (!compare_iterators(it_one, it_other))
        return false;

    return true;
}

string write_struct(
    Type_iterator& it)
{
    string res;
    return res;
}

class Attribute_test_suite : public TEST::Test_suite
{
public:
    Attribute_test_suite() : TEST::Test_suite("ATTR::Attribute Test Suite")
    {
        m_attr_module.set();
        m_log_module.set();
        m_log_module->set_severity_limit(LOG::ILogger::S_ALL);
        // configure mod_log
        m_log_module->set_severity_by_category(LOG::ILogger::C_DATABASE, LOG::ILogger::S_ALL);

        add(MI_TEST_METHOD(Attribute_test_suite, test_constructors));
        add(MI_TEST_METHOD(Attribute_test_suite, test_copy_constructor));
        add(MI_TEST_METHOD(Attribute_test_suite, test_list_shrink));
        add(MI_TEST_METHOD(Attribute_test_suite, test_get_size));
        add(MI_TEST_METHOD(Attribute_test_suite, test_print));
        add(MI_TEST_METHOD(Attribute_test_suite, test_get_vector));
        add(MI_TEST_METHOD(Attribute_test_suite, test_set_tags));
        add(MI_TEST_METHOD(Attribute_test_suite, test_value_set_get));
        add(MI_TEST_METHOD(Attribute_test_suite, test_hard_value_labor));
    }

    ~Attribute_test_suite()
    {
        m_log_module.reset();
        m_attr_module.reset();
    }

    void test_constructors()
    {
        // constructors
        {
        const char* name = "Name";
        Attribute attr(TYPE_INT32, name);
        MI_CHECK_EQUAL(attr.get_id(), Attribute::id_create(name));
        MI_CHECK(attr.get_override() != PROPAGATION_OVERRIDE);
        }
    }

    void test_copy_constructor()
    {
        // single simple type
        {
        Attribute attr(TYPE_INT32, "simple_int");
        attr.set_value_int(12);
        Attribute other = attr;

        MI_CHECK(are_equal(attr, other));
        }
        // array of simple type
        {
        Attribute attr(TYPE_INT32, "simple_ints", 5);
        attr.set_value_int(1, 0);
        attr.set_value_int(2, 1);
        attr.set_value_int(3, 2);
        attr.set_value_int(4, 3);
        attr.set_value_int(5, 4);
        Attribute other = attr;

        MI_CHECK(are_equal(attr, other));
        }
        // array of simple type - failing test
        {
        Attribute attr(TYPE_INT32, "simple_ints", 5);
        attr.set_value_int(1, 0);
        attr.set_value_int(2, 1);
        attr.set_value_int(3, 2);
        attr.set_value_int(4, 3);
        attr.set_value_int(5, 4);
        Attribute other(TYPE_INT32, "simple_ints", 5);
        other.set_value_int(1, 0);
        other.set_value_int(2, 1);
        other.set_value_int(3, 2);
        other.set_value_int(5, 3);  // !!  Different from attr
        other.set_value_int(4, 4);  // !!  Different from attr

        MI_CHECK(!are_equal(attr, other));
        }
        // simple struct
        {
        Type root(TYPE_STRUCT, "struct");
        Type truth(TYPE_BOOLEAN, "01_bool");
        Type integ(TYPE_INT32, "02_int32");
        truth.set_next(integ);
        root.set_child(truth);

        Attribute_id id = Attribute::id_create(root.get_name());
        Attribute attr(id, root);
        // setting values
        set_value(root, attr.set_values(), "struct.01_bool", true);
        set_value(root, attr.set_values(), "struct.02_int32", 12);

        Attribute other = attr;

        MI_CHECK(are_equal(attr, other));
        }
        {
        Type root(TYPE_STRUCT, "struct");
        Type truth(TYPE_BOOLEAN, "01_bool");
        Type integ(TYPE_INT32, "02_int32", 5);
        Type truth_02(TYPE_BOOLEAN, "03_bool");
        integ.set_next(truth_02);
        truth.set_next(integ);
        root.set_child(truth);

        Attribute_id id = Attribute::id_create(root.get_name());
        Attribute attr(id, root);
        // setting values
        set_value(root, attr.set_values(), "struct.01_bool", true);
        set_value(root, attr.set_values(), "struct.02_int32[0]", 0);
        set_value(root, attr.set_values(), "struct.02_int32[1]", 1);
        set_value(root, attr.set_values(), "struct.02_int32[2]", 2);
        set_value(root, attr.set_values(), "struct.02_int32[3]", 3);
        set_value(root, attr.set_values(), "struct.02_int32[4]", 4);
        set_value(root, attr.set_values(), "struct.03_bool", false);

        Attribute other = attr;

        MI_CHECK(are_equal(attr, other));
        }
        // dynamic array
        {
        Type root(TYPE_INT32, "simple_ints", 0);
        Dynamic_array values;
        int* ints = new int[5];
        ints[0] = 1; ints[1] = 2; ints[2] = 3; ints[3] = 4; ints[4] = 5;
        values.m_count = 5;
        values.m_value = (char*)ints;

        Attribute_id id = Attribute::id_create(root.get_name());
        Attribute attr(id, root);
        set_value(root,attr.set_values(),"simple_ints",values);

        Attribute other = attr;

        MI_CHECK(are_equal(attr, other));
        }
        {
        Type root(TYPE_STRUCT, "framebuffer");
        Type first(TYPE_STRING, "fb_name");
        Type second(TYPE_STRING, "fb_img_type");
        Type third(TYPE_STRING, "fb_file_format");
        Type fourth(TYPE_STRING, "fb_filename");

        third.set_next(fourth);
        second.set_next(third);
        first.set_next(second);
        root.set_child(first);

        Attribute_id id = Attribute::id_create(root.get_name());
        Attribute attr(id, root);

        // setting values
        set_value(root, attr.set_values(), "framebuffer.fb_name", string("fb_name"));
        set_value(root, attr.set_values(), "framebuffer.fb_img_type", string("fb_img_type"));
        set_value(root, attr.set_values(), "framebuffer.fb_file_format", string("fb_file_format"));
        set_value(root, attr.set_values(), "framebuffer.fb_filename", string("fb_filename"));


        SERIAL::Deserialization_manager* dm = SERIAL::Deserialization_manager::create();
        dm->register_class(Attribute::id, Attribute::factory);

        SERIAL::Buffer_serializer buf_serializer;
        buf_serializer.serialize(&attr);
        SERIAL::Buffer_deserializer buf_deserializer(dm);
        Attribute* other = (Attribute*)buf_deserializer.deserialize(
            buf_serializer.get_buffer(),
            buf_serializer.get_buffer_size());

        MI_CHECK(are_equal(attr, *other));
        delete other;
        SERIAL::Deserialization_manager::release(dm);
        }
    }

    void test_list_shrink()
    {
        // create an attribute with listsize > 1
        // create an attribute with listsize == 1 with identical values for that
        // shrink first one and compare it with second
    }

    void test_get_size()
    {
        // dynamic array
        {
        Type root(TYPE_INT32, "simple_ints", 0);
        Dynamic_array values;
        int* ints = new int[5];
        ints[0] = 1; ints[1] = 2; ints[2] = 3; ints[3] = 4; ints[4] = 5;
        values.m_count = 5;
        values.m_value = (char*)ints;

        Attribute_id id = Attribute::id_create(root.get_name());
        Attribute attr(id, root);
        set_value(root,attr.set_values(),"simple_ints",values);
        MI_CHECK_EQUAL(attr.get_size(),
            sizeof(Attribute)
            + sizeof(Dynamic_array)
            + values.m_count*sizeof(int));
        }

    }

    void test_print()
    {
        // dynamic array
        {
        Type root(TYPE_INT32, "simple_ints", 0);
        Dynamic_array values;
        int* ints = new int[5];
        ints[0] = 1; ints[1] = 2; ints[2] = 3; ints[3] = 4; ints[4] = 5;
        values.m_count = 5;
        values.m_value = (char*)ints;

        Attribute_id id = Attribute::id_create(root.get_name());
        Attribute attr(id, root);
        set_value(root,attr.set_values(),"simple_ints",values);
/*
        Type_iterator it(&attr.get_type(), attr.set_values());
        string res = write_non_struct(it);
        MI_CHECK_EQUAL(res, "");
 */
        }
    }

    void test_get_vector()
    {
        typedef Attribute::Vector3 Vector3;
        {
        Type t(TYPE_VECTOR3, "vec");
        Vector3 v; v.x = 0.f; v.y = 1.f; v.z = 2.f;
        Attribute_id id = Attribute::id_create(t.get_name());
        Attribute attr(id, t);
        attr.set_value_vector3(v);
        Vector3 v_copy = attr.get_value<Vector3>();
        MI_CHECK(v == v_copy);
        }
        {
        Type t(TYPE_VECTOR2, "vec");
        Vector2 v; v.x = 0.f; v.y = 1.f;
        Attribute_id id = Attribute::id_create(t.get_name());
        Attribute attr(id, t);
        attr.set_value(v);
        Vector2 v_copy = attr.get_value<Vector2>();
        MI_CHECK(v == v_copy);
        }
        {
        Type t(TYPE_VECTOR4, "vec");
        Vector4 v; v.x = 0.f; v.y = 1.f; v.z = 2.f; v[3] = 3.f;
        Attribute_id id = Attribute::id_create(t.get_name());
        Attribute attr(id, t);
        attr.set_value(v);
        Vector4 v_copy = attr.get_value<Vector4>();
        MI_CHECK(v == v_copy);
        }

    }

    void test_set_tags()
    {
        Type t(TYPE_TAG, "tag");
        DB::Tag tag(12);
        Attribute_id id = Attribute::id_create(t.get_name());
        Attribute attr(id, t);
        attr.set_value(tag);
        Attribute attr2(attr);

        DB::Tag res = attr2.get_value<DB::Tag>();
        MI_CHECK(tag == res);
    }

    void test_value_set_get()
    {
        // single simple type
        {
        int value = 12;
        Attribute attr(TYPE_INT32, "simple_int");
        attr.set_value_int(value);
        MI_CHECK_EQUAL(value, attr.get_value_int());
        }
        // single simple type
        {
        bool value = false;
        Attribute attr(TYPE_BOOLEAN, "simple_bool");
        attr.set_value_bool(value);
        MI_CHECK_EQUAL(value, attr.get_value_bool());
        }
        // single simple type
        {
        const char* value = "me";
        Attribute attr(TYPE_STRING, "simple_str");
        attr.set_value_string(value);
        MI_CHECK_EQUAL(strcmp(value, attr.get_value_string()), 0);
        }
        // single simple type
        {
        Vector3 value; value.x = 1.f; value.y = 2.f; value.z = 3.f;
        Attribute attr(TYPE_VECTOR3, "simple_vec3");
        attr.set_value_vector3(value);
        MI_CHECK(value == attr.get_value_vector3());
        }
    }

    void test_hard_value_labor()
    {
        // single simple type
        {
        int value = 12;
        Attribute attr(TYPE_INT32, "foo");
        attr.set_value_int(value);
        MI_CHECK_EQUAL(value, *(int*)attr.get_values());
        }
        // single simple type - raw access
        {
        int value = 12;
        Attribute attr(TYPE_INT32, "foo");
        *reinterpret_cast<int*>(attr.set_values()) = value;
        MI_CHECK_EQUAL(value, *(int*)attr.get_values());
        }
        // single simple type
        {
        const char* value = "value";
        Attribute attr(TYPE_STRING, "foo");
        attr.set_value_string(value);
        MI_CHECK_EQUAL(strcmp(value, *(char**)attr.get_values()), 0);
        }
        // single simple type - raw access: Note that the attribute owns the char pointer!
        {
        const char* value = "value";
        Attribute attr(TYPE_STRING, "foo");
        int len = strlen(value)+1;
        char* v = new char[len];
        memcpy(v, value, len);
        *reinterpret_cast<const char**>(attr.set_values()) = v;
        MI_CHECK_EQUAL(strcmp(value, *(const char* const*)attr.get_values()), 0);
        }
        // simple dynamic array
        {
        Type root(TYPE_INT32, "foo", 0);
        Dynamic_array values;
        int* ints = new int[5];
        ints[0] = 1; ints[1] = 2; ints[2] = 3; ints[3] = 4; ints[4] = 5;
        values.m_count = 5;
        values.m_value = (char*)ints;

        Attribute_id id = Attribute::id_create(root.get_name());
        Attribute attr(id, root);
        set_value(root, attr.set_values(), "foo", values);

        const Dynamic_array* result = (const Dynamic_array*)attr.get_values();
        MI_CHECK_EQUAL(values.m_count, result->m_count);
        for (size_t i=0; i<values.m_count; ++i)
            MI_CHECK_EQUAL(values.m_value[i], result->m_value[i]);
        }
        // dynamic array inside a struct { BOOLEAN, INT32[], BOOLEAN } - raw access
        {
        Type root(TYPE_STRUCT, "root");
        Type first_child(TYPE_BOOLEAN, "first");
        Type second_child(TYPE_INT32, "second", 0);
        Type third_child(TYPE_BOOLEAN, "third");

        second_child.set_next(third_child);
        first_child.set_next(second_child);
        root.set_child(first_child);

        Dynamic_array values;
        int* ints = new int[5];
        ints[0] = 1; ints[1] = 2; ints[2] = 3; ints[3] = 4; ints[4] = 5;
        values.m_count = 5;
        values.m_value = (char*)ints;

        Attribute_id id = Attribute::id_create(root.get_name());
        Attribute attr(id, root);

        *reinterpret_cast<bool*>(attr.set_values()) = true;
        char* ret_address=0;
        root.lookup("root.second", attr.set_values(), &ret_address);
        reinterpret_cast<Dynamic_array*>(ret_address)->m_count = values.m_count;
        reinterpret_cast<Dynamic_array*>(ret_address)->m_value = values.m_value;
        root.lookup("root.third", attr.set_values(), &ret_address);
        *reinterpret_cast<bool*>(ret_address) = false;

        MI_CHECK_EQUAL(*(const bool*)(attr.get_values()), true);
        root.lookup("root.second", attr.set_values(), &ret_address);
        const Dynamic_array* result = (const Dynamic_array*)ret_address;
        MI_CHECK_EQUAL(values.m_count, result->m_count);
        root.lookup("root.second[0]", attr.set_values(), &ret_address);
        MI_CHECK_EQUAL(*(const int*)(ret_address), 1);
        root.lookup("root.second[1]", attr.set_values(), &ret_address);
        MI_CHECK_EQUAL(*(const int*)(ret_address), 2);
        root.lookup("root.second[2]", attr.set_values(), &ret_address);
        MI_CHECK_EQUAL(*(const int*)(ret_address), 3);
        root.lookup("root.second[3]", attr.set_values(), &ret_address);
        MI_CHECK_EQUAL(*(const int*)(ret_address), 4);
        root.lookup("root.second[4]", attr.set_values(), &ret_address);
        MI_CHECK_EQUAL(*(const int*)(ret_address), 5);
        }
    }

    SYSTEM::Access_module<Attr_module> m_attr_module;
    SYSTEM::Access_module<Log_module> m_log_module;
};

// don't remove this comment or it will break compilation
MI_TEST_AUTO_CASE( new Attribute_test_suite );

