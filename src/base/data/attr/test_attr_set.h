/******************************************************************************
 * Copyright (c) 2008-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The tests for the class ATTR::Attribute_set

#include "pch.h"

#include "attr.h"
#include "test_utilities.h"

#include <base/system/main/access_module.h>
#include <base/system/test/i_test_auto_driver.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_logger.h>

#include <memory>

using namespace MI;
using namespace MI::ATTR;
using namespace MI::MEM;


class Attr_set_test_suite : public TEST::Test_suite
{
public:
    Attr_set_test_suite() : TEST::Test_suite("ATTR::Attribute_set Test Suite")
    {
        m_attr_module.set();
        add( MI_TEST_METHOD(Attr_set_test_suite, test_constructor) );
        add( MI_TEST_METHOD(Attr_set_test_suite, test_copy_constructor) );
        add( MI_TEST_METHOD(Attr_set_test_suite, test_detaching) );
        add( MI_TEST_METHOD(Attr_set_test_suite, test_get_references) );
        add( MI_TEST_METHOD(Attr_set_test_suite, test_strange_value_merging) );
    }
    ~Attr_set_test_suite()
    {
        m_attr_module.reset();
    }


    void test_constructor()
    {
        // default constructor
        {
        Attribute_set attr_set;
        MI_CHECK(attr_set.get_attributes().empty());
        }
    }

    void test_copy_constructor()
    {
        // default copy constructor
    }


    void test_detaching()
    {
        // test that a detached boost::shared_ptr<Attribute> has always a ref_count >= 1
        {
        Attribute_set attr_set;
        Attribute_id id;
        // this artificial scope avoids that ref_count will be two due to the attr
        {
        const char* name = "dummy";
        auto attr = std::make_shared<Attribute>(TYPE_INT32, name);
        attr_set.attach(attr);
        MI_CHECK(attr_set.lookup(name));
        id = attr->get_id();
        }
        std::shared_ptr<Attribute> retrieved = attr_set.detach(id);
        MI_CHECK(retrieved && retrieved.use_count() == 1);
        }
    }


    void test_get_references()
    {
        {
        Attribute_set attr_set;
        { // setting up the attr_set
        const char* name = "attr_01";
        auto attr_01 = std::make_shared<Attribute>(TYPE_TAG, name);
        DB::Tag tag(12);
        set_value(attr_01->get_type(), attr_01->set_values(), name, tag);
        attr_set.attach(attr_01);
        const char* name_02 = "attr_02";
        auto attr_02 = std::make_shared<Attribute>(TYPE_INT32, name_02);
        int int_val(7);
        set_value(attr_02->get_type(), attr_02->set_values(), name_02, int_val);
        attr_set.attach(attr_02);
        const char* name_03 = "attr_03";
        auto attr_03 = std::make_shared<Attribute>(TYPE_TAG, name_03);
        DB::Tag tag_03(27);
        set_value(attr_03->get_type(), attr_03->set_values(), name_03, tag_03);
        attr_set.attach(attr_03);
        }
        DB::Tag_set result;
        attr_set.get_references(&result);
        MI_CHECK(result.size() == 2);
        std::set<DB::Tag>::const_iterator it=result.begin();
        MI_CHECK(it->get_uint() == 12);
        ++it;
        MI_CHECK(it->get_uint() == 27);
        }
    }
    void test_strange_value_merging()
    {
        {
        Attribute_set attr_set;
        size_t check_size = 0;
        // setting up the set
        {
            // dynamic array inside a struct { BOOLEAN, INT32[], BOOLEAN } - raw access
            {
            Type root(TYPE_STRUCT, "one");
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
            auto attr = std::make_shared<Attribute>(id, root);
            check_size += attr->get_type().sizeof_all();

            *reinterpret_cast<bool*>(attr->set_values()) = true;
            char* ret_address=0;
            root.lookup("one.second", attr->set_values(), &ret_address);
            reinterpret_cast<Dynamic_array*>(ret_address)->m_count = values.m_count;
            reinterpret_cast<Dynamic_array*>(ret_address)->m_value = values.m_value;
            Uint offset=0;
            root.lookup("one.third", &offset);
            root.lookup("one.third", attr->set_values(), &ret_address);
            MI_CHECK_EQUAL(ret_address, attr->set_values()+offset);
            *reinterpret_cast<bool*>(attr->set_values()+offset) = false;

            attr_set.attach(attr);
            }
            // single simple type
            {
            int value = 12;
            auto attr = std::make_shared<Attribute>(TYPE_INT32, "two");
            check_size += attr->get_type().sizeof_all();
            attr->set_value_int(value);
            MI_CHECK_EQUAL(value, *(int*)attr->get_values());

            attr_set.attach(attr);
            }
            // single simple type - raw access
            {
            int value = 123;
            auto attr = std::make_shared<Attribute>(TYPE_INT32, "three");
            check_size += attr->get_type().sizeof_all();
            *reinterpret_cast<int*>(attr->set_values()) = value;
            MI_CHECK_EQUAL(value, *(int*)attr->get_values());

            attr_set.attach(attr);
            }
            // single simple type
            {
            const char* value = "value";
            auto attr = std::make_shared<Attribute>(TYPE_STRING, "four");
            check_size += attr->get_type().sizeof_all();
            int len = strlen(value)+1;
            char* v = new char[len];
            memcpy(v, value, len);
            *reinterpret_cast<const char**>(attr->set_values()) = v;
            MI_CHECK_EQUAL(strcmp(value, *(const char* const*)attr->get_values()), 0);

            attr_set.attach(attr);
            }
        }
        }

        {
        Attribute_set attr_set;
        size_t check_size = 0;
        // setting up the set
        {
            // "on" (boolean) true
            {
            bool value = true;
            auto attr = std::make_shared<Attribute>(TYPE_BOOLEAN, "on");
            check_size += attr->get_type().sizeof_all();
            *reinterpret_cast<bool*>(attr->set_values()) = value;
            MI_CHECK_EQUAL(value, *(bool*)attr->get_values());
            attr_set.attach(attr);
            }
            // "flags" (int32) 0
            {
            int value = 0;
            auto attr = std::make_shared<Attribute>(TYPE_INT32, "flags");
            check_size += attr->get_type().sizeof_all();
            *reinterpret_cast<int*>(attr->set_values()) = value;
            MI_CHECK_EQUAL(value, *(int*)attr->get_values());
            attr_set.attach(attr);
            }
            // "sun_direction" (vector3) 0.5 0.5 0.
            {
            //float value[3] = {0.5f, 0.5f, 0.f};
            Vector3 value; value.x = 0.5f; value.y = 0.5f; value.z = 0.f;
            auto attr = std::make_shared<Attribute>(TYPE_VECTOR3, "sun_direction");
            check_size += attr->get_type().sizeof_all();
            *reinterpret_cast<Vector3*>(attr->set_values()) = value;
            attr_set.attach(attr);
            }
            // "saturation" (scalar) 1.2
            {
            float value = 1.2f;
            auto attr = std::make_shared<Attribute>(TYPE_SCALAR, "saturation");
            check_size += attr->get_type().sizeof_all();
            *reinterpret_cast<float*>(attr->set_values()) = value;
            attr_set.attach(attr);
            }
        }
        }

    }
    SYSTEM::Access_module<Attr_module> m_attr_module;
};

MI_TEST_AUTO_CASE( new Attr_set_test_suite );
