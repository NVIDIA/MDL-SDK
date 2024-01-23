/******************************************************************************
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for base/data/dblight"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>

#include "i_dblight.h"
#include "dblight_database.h"
#include "dblight_info.h"

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_element.h>
#include <base/data/db/i_db_scope.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_transaction_ptr.h>

#include <base/data/serial/i_serializer.h>
#include <base/data/serial/serial.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_module.h>
#include <base/system/main/access_module.h>
#include <base/system/main/i_assert.h>
#include <base/util/registry/i_config_registry.h>

using namespace MI;
namespace fs = boost::filesystem;

class My_element : public DB::Element<My_element, 0x12345678>
{
public:
    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const
    { serializer->write( m_value); serializer->write( m_tag_set); return this + 1; }
    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer)
    { deserializer->read( &m_value);  deserializer->read( &m_tag_set); return this + 1; }
    DB::Element_base* copy() const { return new My_element( *this); }
    std::string get_class_name() const { return "My_element"; }
    void get_references( DB::Tag_set* result) const
    { result->insert( m_tag_set.begin(), m_tag_set.end()); }

    My_element() : m_value( 0) { }
    explicit My_element( int value, const DB::Tag_set& tag_set = {})
      : m_value( value), m_tag_set( tag_set) { }
    void set_value( int value) { m_value = value; }
    int get_value() const { return m_value; }
    void set_tag_set( const DB::Tag_set& tag_set) { m_tag_set = tag_set; }
    const DB::Tag_set& get_tag_set() const { return m_tag_set; }

private:
    int m_value;
    DB::Tag_set m_tag_set;
};

/// Compares two streams in a very simple way.
bool compare_files( std::ifstream& s1, std::stringstream& s2)
{
    auto is_whitespace = boost::algorithm::is_any_of( " \t\r");

    size_t line_number = 1;
    std::string line1;
    std::string line2;

    std::getline( s1, line1);
    std::getline( s2, line2);
    boost::algorithm::trim_if( line1, is_whitespace);
    boost::algorithm::trim_if( line2, is_whitespace);

    while( !s1.eof() && !s2.eof()) {

        if( line1 != line2) {
            LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
                "Difference in line %zu: \"%s\" vs \"%s\".",
                line_number, line1.c_str(), line2.c_str());
            return false;
        }

        ++line_number;
        std::getline( s1, line1);
        std::getline( s2, line2);
        boost::algorithm::trim_if( line1, is_whitespace);
        boost::algorithm::trim_if( line2, is_whitespace);
    }

    if( s1.eof() != s2.eof()) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
            "Difference in line %zu: \"%s\" vs \"%s\".",
            line_number, line1.c_str(), line2.c_str());
        return false;
    }

    return true;
}

struct Test_db
{
public:
    Test_db( const char* test_name, bool compare = true)
      : m_test_name( test_name), m_compare( compare)
    {
        std::cerr << "Begin test: " << m_test_name << std::endl;
        std::cerr << std::endl;

        m_db = DBLIGHT::factory();
        m_db_impl = static_cast<DBLIGHT::Database_impl*>( m_db);
        m_scope = m_db->get_global_scope();

        SERIAL::Deserialization_manager* manager = m_db_impl->get_deserialization_manager();
        manager->register_class<My_element>();
    }

    ~Test_db()
    {
        m_db_impl->close();

        std::cerr << "End test: " << m_test_name << std::endl;
        std::cerr << std::endl;

        export_dumps_and_compare();
    }

    /// Calls Database_impl::dump(), stores the result for later export, and prints it to stderr.
    void dump( bool mask_pointer_values = true)
    {
        size_t pos = m_dump.str().size();
        m_db_impl->dump( m_dump, mask_pointer_values);
        std::cerr << m_dump.str().substr( pos);
    }

private:
    void export_dumps_and_compare();

    const char* const m_test_name;
    const bool m_compare;

public:
    DB::Database* m_db;
    DBLIGHT::Database_impl* m_db_impl;
    DB::Scope* m_scope;
    std::stringstream m_dump;
};

void Test_db::export_dumps_and_compare()
{
    fs::path fs_output_dir( "data");
    fs::path fs_output_file = fs_output_dir / m_test_name;
    fs_output_file += ".dump.txt";
    fs::create_directory( fs_output_dir);
    std::ofstream file_output( fs_output_file.string().c_str());
    file_output << m_dump.str();
    file_output.close();

    if( !m_compare)
        return;

    fs::path fs_input_dir( MI::TEST::mi_src_path( "base/data/dblight/data"));
    fs::path fs_input_file = fs_input_dir / m_test_name;
    fs_input_file += ".dump.txt";
    std::ifstream file_input( fs_input_file.string().c_str());
    if( !file_input.good()) {
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
            "Reference data \"%s\" for generated data \"%s\" not found.",
            fs_input_file.string().c_str(), fs_output_file.string().c_str());
        abort();
    }

    m_dump.seekg( 0);
    bool success = compare_files( file_input, m_dump);
    if( !success) {
#ifndef MI_PLATFORM_WINDOWS
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE, "Full diff below:");
        std::string command = "diff -u " + fs_input_file.string() + " " + fs_output_file.string();
        int result = system( command.c_str());
        (void) result;
#endif // MI_PLATFORM_WINDOWS
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
            "Diff command: diff %s %s",
            fs_input_file.string().c_str(), fs_output_file.string().c_str());
        LOG::mod_log->error( M_DB, LOG::Mod_log::C_DATABASE,
            "Approval command: cp %s %s",
            fs_output_file.string().c_str(), fs_input_file.string().c_str());
        abort();
    }
}

void test_info_impl()
{
    DBLIGHT::Info_impl a1( DB::Scope_id( 20), DB::Transaction_id( 30), 40);
    DBLIGHT::Info_impl a2( DB::Scope_id( 21), DB::Transaction_id( 30), 40);

    DBLIGHT::Info_impl b1( DB::Scope_id( 20), DB::Transaction_id( 30), 40);
    DBLIGHT::Info_impl b2( DB::Scope_id( 20), DB::Transaction_id( 31), 40);

    DBLIGHT::Info_impl c1( DB::Scope_id( 20), DB::Transaction_id( 30), 40);
    DBLIGHT::Info_impl c2( DB::Scope_id( 20), DB::Transaction_id( 30), 41);

    MI_CHECK(   a1 == a1);
    MI_CHECK( !(a1 == a2));
    MI_CHECK(   a1 != a2);
    MI_CHECK( !(a1 != a1));
    MI_CHECK(   a1 <  a2);
    MI_CHECK( !(a2 <  a1));
    MI_CHECK(   a1 <= a2);
    MI_CHECK( !(a2 <= a1));
    MI_CHECK(   a2 >  a1);
    MI_CHECK( !(a1 >  a2));
    MI_CHECK(   a2 >= a1);
    MI_CHECK( !(a1 >= a2));

    MI_CHECK(   b1 == b1);
    MI_CHECK( !(b1 == b2));
    MI_CHECK(   b1 != b2);
    MI_CHECK( !(b1 != b1));
    MI_CHECK(   b1 <  b2);
    MI_CHECK( !(b2 <  b1));
    MI_CHECK(   b1 <= b2);
    MI_CHECK( !(b2 <= b1));
    MI_CHECK(   b2 >  b1);
    MI_CHECK( !(b1 >  b2));
    MI_CHECK(   b2 >= b1);
    MI_CHECK( !(b1 >= b2));

    MI_CHECK(   c1 == c1);
    MI_CHECK( !(c1 == c2));
    MI_CHECK(   c1 != c2);
    MI_CHECK( !(c1 != c1));
    MI_CHECK(   c1 <  c2);
    MI_CHECK( !(c2 <  c1));
    MI_CHECK(   c1 <= c2);
    MI_CHECK( !(c2 <= c1));
    MI_CHECK(   c2 >  c1);
    MI_CHECK( !(c1 >  c2));
    MI_CHECK(   c2 >= c1);
    MI_CHECK( !(c1 >= c2));
}

void test_empty_db()
{
    Test_db db( __func__);
    db.dump();
}

void test_transaction_create_commit()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction = db.m_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction);
    MI_CHECK( transaction->is_open());

    bool success = transaction->commit();
    db.dump();
    MI_CHECK( success);
    MI_CHECK( !transaction->is_open());

    success = transaction->commit();
    MI_CHECK( !success);

    MI_CHECK_EQUAL( transaction->get_scope(), db.m_scope);

    transaction.reset();
    db.dump();
}

void test_transaction_create_abort()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction = db.m_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction);
    MI_CHECK( transaction->is_open());

    transaction->abort();
    db.dump();
    MI_CHECK( !transaction->is_open());

    transaction->abort();

    MI_CHECK_EQUAL( transaction->get_scope(), db.m_scope);

    transaction.reset();
    db.dump();
}

void test_two_transactions_create_commit()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction0);

    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction1);

    transaction0->commit();
    db.dump();

    transaction0.reset();
    db.dump();

    transaction1->commit();
    db.dump();

    transaction1.reset();
    db.dump();
}

void test_two_transactions_create_reverse_commit()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction0);

    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction1);

    transaction1->commit();
    db.dump();

    transaction1.reset();
    db.dump();

    transaction0->commit();
    db.dump();

    transaction0.reset();
    db.dump();
}

void test_transaction_store_with_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    transaction->store( element1, "foo");
    db.dump();

    // Triggers an error message.
    My_element* element2 = new My_element( 43);
    transaction->store( element2, "");
    db.dump();

    // Triggers an error message.
    transaction->store( static_cast<DB::Element_base*>( nullptr));
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_without_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    My_element* element = new My_element( 42);
    transaction->store( tag, element);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_multiple_versions_with_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    My_element* element1 = new My_element( 42);
    transaction->store( tag, element1, "foo");
    db.dump();

    My_element* element2 = new My_element( 43);
    transaction->store( tag, element2, "foo");
    db.dump();

    // Triggers an error message.
    My_element* element3 = new My_element( 44);
    transaction->store( DB::Tag(), element3, "foo");
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_multiple_versions_without_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    My_element* element1 = new My_element( 42);
    transaction->store( tag, element1);
    db.dump();

    My_element* element2 = new My_element( 43);
    transaction->store( tag, element2);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_multiple_names_per_tag()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    My_element* element1 = new My_element( 42);
    transaction->store( tag, element1, "foo");
    db.dump();

    My_element* element2 = new My_element( 43);
    transaction->store( tag, element2, "bar");
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_multiple_tags_per_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->reserve_tag();
    transaction->store( tag1, element1, "foo");
    db.dump();

    My_element* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->reserve_tag();
    transaction->store( tag2, element2, "foo");
    db.dump();

    transaction->commit();
    db.dump();
}

void test_two_transactions_store_parallel()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    DB::Tag tag = transaction0->reserve_tag();

    My_element* element1 = new My_element( 42);
    transaction0->store( tag, element1, "foo");

    My_element* element2 = new My_element( 43);
    transaction1->store( tag, element2, "foo");
    transaction0->commit();
    transaction1->commit();
    db.dump();

    DB::Transaction_ptr transaction2 = db.m_scope->start_transaction();

    {
        DB::Access<My_element> access( tag, transaction2.get());
        MI_CHECK_EQUAL( access->get_value(), 43);
    }

    transaction2->commit();
}

void test_two_transactions_store_parallel_reverse_commit()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    DB::Tag tag = transaction0->reserve_tag();

    My_element* element1 = new My_element( 42);
    transaction0->store( tag, element1, "foo");

    My_element* element2 = new My_element( 43);
    transaction1->store( tag, element2, "foo");

    transaction1->commit();
    transaction0->commit();
    db.dump();

    DB::Transaction_ptr transaction2 = db.m_scope->start_transaction();

    {
        DB::Access<My_element> access( tag, transaction2.get());
        MI_CHECK_EQUAL( access->get_value(), 43);
    }

    transaction2->commit();
}

void test_two_transactions_store_parallel_not_globally_visible()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    DB::Transaction_ptr transaction2 = db.m_scope->start_transaction();
    DB::Tag tag = transaction0->reserve_tag();

    My_element* element1 = new My_element( 42);
    transaction1->store( tag, element1, "foo");

    My_element* element2 = new My_element( 43);
    transaction2->store( tag, element2, "foo");
    transaction1->commit();
    transaction2->commit();
    // Transaction 0 prevents that changes from transaction 1 and transaction 2 become globally
    // visible. Still the version with value 42 should be collected by the GC as in
    // test_two_transactions_store_parallel() without transaction 0.
    db.dump();

    transaction0->commit();
    db.dump();

    DB::Transaction_ptr transaction3 = db.m_scope->start_transaction();

    {
        DB::Access<My_element> access( tag, transaction3.get());
        MI_CHECK_EQUAL( access->get_value(), 43);
    }

    transaction3->commit();
}

void test_transaction_create_store_abort()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction = db.m_scope->start_transaction();
    db.dump();

    My_element* element = new My_element( 42);
    transaction->store( element, "foo");
    db.dump();

    transaction->abort();
    db.dump();
}

void test_transaction_access()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    {
        DB::Access<My_element> access( tag, transaction.get());
        db.dump();
        MI_CHECK( access);
        MI_CHECK_EQUAL( access->get_value(), 42);

        access.reset();
        db.dump();
    }

    transaction->commit();
    db.dump();
}

void test_transaction_edit()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    {
        DB::Edit<My_element> edit( tag, transaction.get());
        db.dump();
        MI_CHECK( edit);
        MI_CHECK_EQUAL( edit->get_value(), 42);

        edit->set_value( 43);
        MI_CHECK_EQUAL( edit->get_value(), 43);

        edit.reset();
        db.dump();

        DB::Access<My_element> access( tag, transaction.get());
        db.dump();
        MI_CHECK( access);
        MI_CHECK_EQUAL( access->get_value(), 43);

        access.reset();
        db.dump();
    }

    transaction->commit();
    db.dump();
}

void test_transaction_two_edits_sequential()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    {
        // sequential edits
        DB::Edit<My_element> edit1( tag, transaction.get());
        db.dump();
        MI_CHECK( edit1);
        MI_CHECK_EQUAL( edit1->get_value(), 42);

        edit1->set_value( 43);
        MI_CHECK_EQUAL( edit1->get_value(), 43);

        edit1.reset();
        db.dump();

        DB::Edit<My_element> edit2( tag, transaction.get());
        db.dump();
        MI_CHECK( edit2);
        MI_CHECK_EQUAL( edit2->get_value(), 43);

        edit2->set_value( 44);
        MI_CHECK_EQUAL( edit2->get_value(), 44);

        edit2.reset();
        db.dump();

        DB::Access<My_element> access( tag, transaction.get());
        db.dump();
        MI_CHECK( access);
        MI_CHECK_EQUAL( access->get_value(), 44);

        access.reset();
        db.dump();
    }

    transaction->commit();
    db.dump();
}

void test_transaction_two_edits_parallel()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    {
        DB::Edit<My_element> edit1( tag, transaction.get());
        db.dump();
        MI_CHECK( edit1);
        MI_CHECK_EQUAL( edit1->get_value(), 42);

        DB::Edit<My_element> edit2( tag, transaction.get());
        db.dump();
        MI_CHECK( edit2);
        MI_CHECK_EQUAL( edit2->get_value(), 42);

        edit1->set_value( 43);
        MI_CHECK_EQUAL( edit1->get_value(), 43);

        edit2->set_value( 44);
        MI_CHECK_EQUAL( edit2->get_value(), 44);

        edit1.reset();
        db.dump();

        edit2.reset();
        db.dump();

        // Edit started last survives
        DB::Access<My_element> access( tag, transaction.get());
        db.dump();
        MI_CHECK( access);
        MI_CHECK_EQUAL( access->get_value(), 44);

        access.reset();
        db.dump();
    }

    transaction->commit();
    db.dump();
}

void test_transaction_two_edits_parallel_reverse_finish()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    {
        DB::Edit<My_element> edit1( tag, transaction.get());
        db.dump();
        MI_CHECK( edit1);
        MI_CHECK_EQUAL( edit1->get_value(), 42);

        DB::Edit<My_element> edit2( tag, transaction.get());
        db.dump();
        MI_CHECK( edit2);
        MI_CHECK_EQUAL( edit2->get_value(), 42);

        edit1->set_value( 43);
        MI_CHECK_EQUAL( edit1->get_value(), 43);

        edit2->set_value( 44);
        MI_CHECK_EQUAL( edit2->get_value(), 44);

        edit2.reset();
        db.dump();

        edit1.reset();
        db.dump();

        // Edit started last survives
        DB::Access<My_element> access( tag, transaction.get());
        db.dump();
        MI_CHECK( access);
        MI_CHECK_EQUAL( access->get_value(), 44);

        access.reset();
        db.dump();
    }

    transaction->commit();
    db.dump();
}

void test_two_transactions_edits_parallel()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    My_element* element = new My_element( 42);
    DB::Tag tag = transaction0->store( element, "foo");
    transaction0->commit();

    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    DB::Transaction_ptr transaction2 = db.m_scope->start_transaction();

    {
        DB::Edit<My_element> edit2( tag, transaction1.get());
        edit2->set_value( 43);
        DB::Edit<My_element> edit3( tag, transaction2.get());
        edit3->set_value( 44);
    }

    transaction1->commit();
    transaction2->commit();
    db.dump();

    DB::Transaction_ptr transaction3 = db.m_scope->start_transaction();
    DB::Access<My_element> access4( tag, transaction3.get());
    MI_CHECK_EQUAL( access4->get_value(), 44);
    transaction3->commit();
}

void test_two_transactions_edits_parallel_reverse_commit()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    My_element* element = new My_element( 42);
    DB::Tag tag = transaction0->store( element, "foo");
    transaction0->commit();

    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    DB::Transaction_ptr transaction2 = db.m_scope->start_transaction();

    {
        DB::Edit<My_element> edit2( tag, transaction1.get());
        edit2->set_value( 43);
        DB::Edit<My_element> edit3( tag, transaction2.get());
        edit3->set_value( 44);
    }

    transaction2->commit();
    transaction1->commit();
    db.dump();

    DB::Transaction_ptr transaction3 = db.m_scope->start_transaction();
    DB::Access<My_element> access4( tag, transaction3.get());
    MI_CHECK_EQUAL( access4->get_value(), 44);
    transaction3->commit();
}

void test_visibility_single_transaction()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    {
        My_element* element = new My_element( 42);
        DB::Tag tag = transaction->store( element, "foo");
        db.dump();

        DB::Access<My_element> access1( tag, transaction.get());
        MI_CHECK_EQUAL( access1->get_value(), 42);

        DB::Edit<My_element> edit2( tag, transaction.get());
        MI_CHECK_EQUAL( edit2->get_value(), 42);

        // step 3: modify via edit2
        edit2->set_value( 43);
        db.dump();

        DB::Access<My_element> access4( tag, transaction.get());
        DB::Edit<My_element> edit5( tag, transaction.get());

        // modification visible everywhere but via access1
        MI_CHECK_EQUAL( access1->get_value(), 42);
        MI_CHECK_EQUAL( edit2->get_value(), 43);
        MI_CHECK_EQUAL( access4->get_value(), 43);
        MI_CHECK_EQUAL( edit5->get_value(), 43);

        // step 6: modify again via edit2
        edit2->set_value( 44);
        db.dump();

        DB::Access<My_element> access7( tag, transaction.get());
        DB::Edit<My_element> edit8( tag, transaction.get());

        // modification visible for edit2 and access4
        MI_CHECK_EQUAL( access1->get_value(), 42);
        MI_CHECK_EQUAL( edit2->get_value(), 44);
        MI_CHECK_EQUAL( access4->get_value(), 44);
        MI_CHECK_EQUAL( edit5->get_value(), 43);
        MI_CHECK_EQUAL( access7->get_value(), 43);
        MI_CHECK_EQUAL( edit8->get_value(), 43);

        // step 6: modify again via edit5
        edit5->set_value( 45);
        db.dump();

        // modification visible for edit5 and access7
        MI_CHECK_EQUAL( access1->get_value(), 42);
        MI_CHECK_EQUAL( edit2->get_value(), 44);
        MI_CHECK_EQUAL( access4->get_value(), 44);
        MI_CHECK_EQUAL( edit5->get_value(), 45);
        MI_CHECK_EQUAL( access7->get_value(), 45);
        MI_CHECK_EQUAL( edit8->get_value(), 43);
    }

    transaction->commit();
}

void test_visibility_multiple_transactions()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    My_element* element = new My_element( 42);
    DB::Tag tag = transaction0->store( element, "foo");
    transaction0->commit();

    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    DB::Transaction_ptr transaction2 = db.m_scope->start_transaction();
    {
        DB::Edit<My_element> edit3( tag, transaction2.get());
        edit3->set_value( 43);
    }
    DB::Transaction_ptr transaction3 = db.m_scope->start_transaction();
    db.dump();

    {
        // modification visible in transaction2
        DB::Access<My_element> access2( tag, transaction1.get());
        MI_CHECK_EQUAL( access2->get_value(), 42);
        DB::Access<My_element> access3( tag, transaction2.get());
        MI_CHECK_EQUAL( access3->get_value(), 43);
        DB::Access<My_element> access4( tag, transaction3.get());
        MI_CHECK_EQUAL( access4->get_value(), 42);
    }

    transaction1->commit();
    transaction2->commit();
    transaction3->commit();

    DB::Transaction_ptr transaction4 = db.m_scope->start_transaction();

    {
        // modification visible in transaction4
        DB::Access<My_element> access5( tag, transaction4.get());
        MI_CHECK_EQUAL( access5->get_value(), 43);
    }

    transaction4->commit();
}

void test_transaction_tag_to_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    My_element* element1 = new My_element( 42);
    transaction->store( tag, element1, "foo");
    My_element* element2 = new My_element( 43);
    transaction->store( tag, element2, "bar");
    db.dump();

    const char* name = transaction->tag_to_name( tag);
    MI_CHECK_EQUAL_CSTR( name, "bar");
    name = transaction->tag_to_name( DB::Tag( tag() + 1));
    MI_CHECK( !name);
    name = transaction->tag_to_name( DB::Tag());
    MI_CHECK( !name);

    transaction->commit();
}

void test_transaction_name_to_tag()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    My_element* element2 = new My_element( 42);
    DB::Tag tag2 = transaction->store( element2, "foo");
    db.dump();

    DB::Tag tag = transaction->name_to_tag( "foo");
    MI_CHECK_EQUAL( tag, tag2);
    MI_CHECK_NOT_EQUAL( tag, tag1);
    tag = transaction->name_to_tag( "bar");
    MI_CHECK( !tag);
    tag = transaction->name_to_tag( "");
    MI_CHECK( !tag);
    tag = transaction->name_to_tag( nullptr);
    MI_CHECK( !tag);

    transaction->commit();
}

void test_transaction_tag_to_name_and_back()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    My_element* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->store( element2, "foo");
    MI_CHECK_NOT_EQUAL( tag1, tag2);
    db.dump();

    const char* name = transaction->tag_to_name( tag1);
    MI_CHECK_EQUAL_CSTR( name, "foo");
    DB::Tag tag3 = transaction->name_to_tag( name);
    MI_CHECK_EQUAL( tag3, tag2);
    MI_CHECK_NOT_EQUAL( tag3, tag1);

    transaction->commit();
}

void test_transaction_name_to_tag_and_back()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    DB::Tag tag1 = transaction->reserve_tag();

    My_element* element1 = new My_element( 42);
    transaction->store( tag1, element1, "foo");
    My_element* element2 = new My_element( 42);
    transaction->store( tag1, element2, "bar");
    db.dump();

    DB::Tag tag3 = transaction->name_to_tag( "foo");
    MI_CHECK_EQUAL( tag3, tag1);
    const char* name = transaction->tag_to_name( tag3);
    MI_CHECK_EQUAL_CSTR( name, "bar");

    transaction->commit();
}

void test_transaction_get_class_id()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");

    SERIAL::Class_id class_id = transaction->get_class_id( tag);
    SERIAL::Class_id expected_class_id = My_element::id;
    MI_CHECK_EQUAL( class_id, expected_class_id);

    transaction->commit();
}

void test_transaction_get_tag_reference_count()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");

    mi::Uint32 pin_count = transaction->get_tag_reference_count( tag);
    MI_CHECK_EQUAL( pin_count, 1);
    pin_count = transaction->get_tag_reference_count( DB::Tag( tag() + 1));
    MI_CHECK_EQUAL( pin_count, 0);

    transaction->commit();
}

void test_transaction_get_tag_version()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump
    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction0->store( element1, "foo");
    My_element* element2 = new My_element( 43);
    DB::Tag tag2 = transaction1->store( element2, "bar");

    DB::Tag_version version = transaction0->get_tag_version( tag1);
    MI_CHECK( version == DB::Tag_version( tag1, transaction0->get_id(), 0));
    version = transaction1->get_tag_version( tag2);
    MI_CHECK( version == DB::Tag_version( tag2, transaction1->get_id(), 0));

    {
        DB::Edit<My_element> edit( tag2, transaction1.get());
        edit->set_value( 44);
    }

    version = transaction1->get_tag_version( tag2);
    MI_CHECK( version == DB::Tag_version( tag2, transaction1->get_id(), 1));
    version = transaction1->get_tag_version( DB::Tag( tag2() + 1));
    MI_CHECK( version == DB::Tag_version());

    transaction0->commit();
    transaction1->commit();
}

void test_transaction_remove()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    MI_CHECK( transaction->remove( tag));
    db.dump();

    // Repeated calls succeed, too.
    MI_CHECK( transaction->remove( tag));
    MI_CHECK( !transaction->remove( DB::Tag()));

    transaction->commit();
    db.dump();
}

void test_transaction_get_tag_is_removed()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    MI_CHECK( !transaction->get_tag_is_removed( tag));
    MI_CHECK( !transaction->get_tag_is_removed( DB::Tag()));

    MI_CHECK( transaction->remove( tag));
    db.dump();

    MI_CHECK( transaction->get_tag_is_removed( tag));

    transaction->commit();
    db.dump();
}

void test_transaction_can_reference()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    My_element* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->store( element2, "bar");

    MI_CHECK( transaction->can_reference_tag( tag1, tag2));
    MI_CHECK( transaction->can_reference_tag( 0, tag2));
    MI_CHECK( transaction->can_reference_tag( 255, tag2));

    transaction->commit();
}

void test_transaction_access_after_remove()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store_for_reference_counting( element1, "foo");
    My_element* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->reserve_tag();
    transaction->store_for_reference_counting( tag2, element2, "bar");
    db.dump();

    {
        // Look up the store info with the element, not the removal info.
        DB::Access<My_element> access1( tag1, transaction.get());
        MI_CHECK_EQUAL( access1->get_value(), 42);
        DB::Access<My_element> access2( tag2, transaction.get());
        MI_CHECK_EQUAL( access2->get_value(), 43);
    }

    transaction->commit();
    db.dump();
}

void test_transaction_access_after_abort()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    transaction->commit();
    transaction = db.m_scope->start_transaction();

    {
        DB::Edit<My_element> edit( tag, transaction.get());
        edit->set_value( 43);
    }

    transaction->abort();
    transaction = db.m_scope->start_transaction();

    {
        // Look up the 1st info, not the 2nd from the aborted transaction.
        DB::Access<My_element> access( tag, transaction.get());
        MI_CHECK_EQUAL( access->get_value(), 42);
    }

    transaction->commit();
}

void test_gc_simple_reference()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    transaction->remove( tag1);
    My_element* element2 = new My_element( 43, {tag1});
    DB::Tag tag2 = transaction->store( element2, "bar");
    db.dump();

    transaction->commit();
    db.dump();

    transaction = db.m_scope->start_transaction();

    transaction->remove( tag2);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_gc_simple_reference_abort()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    transaction->remove( tag1);
    My_element* element2 = new My_element( 43, {tag1});
    transaction->store( element2, "bar");
    db.dump();

    transaction->abort();
    db.dump();
}

void test_gc_multiple_references()
{
    // Same as before, but multiple references, and set references in an edit to test that.
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    My_element* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->store( element2, "bar");
    My_element* element3 = new My_element( 44);
    DB::Tag tag3 = transaction->store( element3, "baz");
    db.dump();

    transaction->commit();
    transaction = db.m_scope->start_transaction();

    // Remove request before adding the references works if done in the same transaction.
    transaction->remove( tag1);
    transaction->remove( tag2);

    {
        DB::Edit<My_element> edit2( tag2, transaction.get());
        edit2->set_tag_set( {tag1});
        DB::Edit<My_element> edit3( tag3, transaction.get());
        edit3->set_tag_set( {tag1, tag2});
    }

    transaction->commit();
    db.dump();

    transaction = db.m_scope->start_transaction();

    transaction->remove( tag3);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_gc_explicit_call()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");

    {
        DB::Edit<My_element> edit( tag, transaction.get());
        edit->set_value( 43);
    }

    db.dump();
    db.m_db->garbage_collection( /*priority*/ 0);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_gc_pin_count_zero()
{
    Test_db db( __func__);

    // Create element in transaction 0
    DB::Transaction_ptr transaction0 = db.m_scope->start_transaction();
    My_element* element = new My_element( 42);
    DB::Tag tag = transaction0->store( element, "foo");
    transaction0->commit();
    transaction0.reset();

    // Remove element in transaction 1. The open transaction 2 avoids that the removal becomes
    // globally visible when transaction 1 is committed.
    DB::Transaction_ptr transaction1 = db.m_scope->start_transaction();
    transaction1->remove( tag);
    DB::Transaction_ptr transaction2 = db.m_scope->start_transaction();
    transaction1->commit();
    transaction1.reset();
    db.dump();

    // Access the element in transaction 3.
    DB::Transaction_ptr transaction3 = db.m_scope->start_transaction();
    DB::Access<My_element> access( tag, transaction3.get());
    MI_CHECK_EQUAL( access->get_value(), 42);
    db.dump();

    // Commit transaction 2. This makes the removal visible for all open transactions. The pin count
    // of the Infos_per_tag set drops to 0, but the pin count of the contained info is larger than 0
    // due to the access above.
    transaction2->commit();
    transaction2.reset();
    db.dump();

    // Giving up the access finally allows the GC to remove the element (but the GC is not triggered
    // here).
    access.reset();
    db.dump();

    // Commit transaction 3 which triggers the GC.
    transaction3->commit();
    transaction3.reset();
    db.dump();
}

void test_use_of_closed_transaction()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    transaction->remove( tag);

    transaction->commit();

    DB::Info* info;
    info = transaction->access_element( tag);
    MI_CHECK( !info);
    info = transaction->edit_element( tag);
    MI_CHECK( !info);
    // artificial test arguments
    transaction->finish_edit( nullptr, DB::Journal_type());
    // artificial test arguments
    transaction->store(
        tag,
        static_cast<DB::Element_base*>( nullptr),
        /*name*/ nullptr,
        /*privacy_level*/ {},
        DB::JOURNAL_NONE,
        /*store_level*/ {});
    bool result = transaction->remove( tag);
    MI_CHECK( !result);
    const char* name = transaction->tag_to_name( tag);
    MI_CHECK( !name);
    DB::Tag tag2 = transaction->name_to_tag( "foo");
    MI_CHECK( !tag2);
    SERIAL::Class_id class_id = transaction->get_class_id( tag);
    MI_CHECK_EQUAL( class_id, SERIAL::class_id_unknown);
    mi::Uint32 ref_count = transaction->get_tag_reference_count( tag);
    MI_CHECK_EQUAL( 0, ref_count);
    DB::Tag_version tag_version = transaction->get_tag_version( tag);
    MI_CHECK( tag_version == DB::Tag_version());
    result = transaction->get_tag_is_removed( tag);
    MI_CHECK( !result);
}

void test_dump_with_pointers()
{
    Test_db db( __func__, /*compare*/ false); // Pointers are non-deterministic
    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump( /*mask_pointer_values*/ false);

    {
        DB::Access<My_element> access( tag, transaction.get());
        db.dump( /*mask_pointer_values*/ false);
        MI_CHECK( access);
        MI_CHECK_EQUAL( access->get_value(), 42);

        access.reset();
        db.dump( /*mask_pointer_values*/ false);
    }

    transaction->commit();
    db.dump( /*mask_pointer_values*/ false);
}

void test_not_implemented_with_assertions()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    db.m_db->lock( 42);
    bool result = db.m_db->unlock( 42);
    MI_CHECK( !result);
    db.m_db->check_is_locked( 42);

    MI_CHECK_EQUAL( db.m_db->set_memory_limits( 1000, 2000), -1);
    size_t low_water = 1;
    size_t high_water = 1;
    db.m_db->get_memory_limits( low_water, high_water);
    MI_CHECK_EQUAL( low_water, 0);
    MI_CHECK_EQUAL( high_water, 0);

    // artificial test arguments
    db.m_db->register_status_listener( nullptr);
    db.m_db->unregister_status_listener( nullptr);
    db.m_db->register_transaction_listener( nullptr);
    db.m_db->unregister_transaction_listener( nullptr);
    db.m_db->register_scope_listener( nullptr);
    db.m_db->unregister_scope_listener( nullptr);

    DB::Transaction_ptr transaction = db.m_scope->start_transaction();

    My_element* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");

    result = transaction->block_commit_or_abort();
    MI_CHECK( !result);
    result = transaction->unblock_commit_or_abort();
    MI_CHECK( !result);
    // artificial test arguments
    DB::Tag tag_job = transaction->store( static_cast<SCHED::Job*>( nullptr));
    MI_CHECK( !tag_job);
    // artificial test arguments
    transaction->store( DB::Tag(), static_cast<SCHED::Job*>( nullptr));
    // artificial test arguments
    tag_job = transaction->store_for_reference_counting( static_cast<SCHED::Job*>( nullptr));
    MI_CHECK( !tag_job);
    // artificial test arguments
    transaction->store_for_reference_counting( DB::Tag(), static_cast<SCHED::Job*>( nullptr));
    transaction->localize( tag, DB::Privacy_level( 0), DB::JOURNAL_NONE);
    auto journal = transaction->get_journal(
        DB::Transaction_id( 0), 0, DB::JOURNAL_ALL, /*lookup_parents*/ false);
    MI_CHECK( !journal);
    transaction->cancel_fragmented_jobs();
    transaction->invalidate_job_results( tag_job);
    transaction->advise( tag_job);
    DB::Element_base* base = transaction->construct_empty_element( My_element::id);
    MI_CHECK( !base);

    transaction->commit();
}

void test( const char* explicit_gc_method)
{
    if( explicit_gc_method) {
        SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
        CONFIG::Config_registry& registry = config_module->get_configuration();
        registry.overwrite_value( "dblight_gc_method", std::string( explicit_gc_method));
    }

    test_empty_db();

    test_transaction_create_commit();
    test_transaction_create_abort();
    test_two_transactions_create_commit();
    test_two_transactions_create_reverse_commit();

    test_transaction_store_with_name();
    test_transaction_store_without_name();
    test_transaction_store_multiple_versions_with_name();
    test_transaction_store_multiple_versions_without_name();
    test_transaction_store_multiple_names_per_tag();
    test_transaction_store_multiple_tags_per_name();
    test_two_transactions_store_parallel();
    test_two_transactions_store_parallel_reverse_commit();
    test_two_transactions_store_parallel_not_globally_visible();
    test_transaction_create_store_abort();

    test_transaction_access();
    test_transaction_edit();
    test_transaction_two_edits_sequential();
    test_transaction_two_edits_parallel();
    test_transaction_two_edits_parallel_reverse_finish();
    test_two_transactions_edits_parallel();
    test_two_transactions_edits_parallel_reverse_commit();

    test_visibility_single_transaction();
    test_visibility_multiple_transactions();

    test_transaction_tag_to_name();
    test_transaction_name_to_tag();
    test_transaction_tag_to_name_and_back();
    test_transaction_name_to_tag_and_back();

    test_transaction_get_class_id();
    test_transaction_get_tag_reference_count();
    test_transaction_get_tag_version();

    test_transaction_remove();
    test_transaction_get_tag_is_removed();
    test_transaction_can_reference();

    test_transaction_access_after_remove();
    test_transaction_access_after_abort();

    test_gc_simple_reference();
    test_gc_simple_reference_abort();
    test_gc_multiple_references();
    test_gc_explicit_call();
    test_gc_pin_count_zero();

    test_use_of_closed_transaction();
    test_dump_with_pointers();
#ifdef NDEBUG
    test_not_implemented_with_assertions();
#endif // NDEBUG
}

MI_TEST_AUTO_FUNCTION( test_dblight )
{
    SYSTEM::Access_module<LOG::Log_module> log_module( false);
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);

    config_module->override( "check_serializer_store=1");
    config_module->override( "check_serializer_edit=1");

    test_info_impl();

    // Note that the reference files are independent of the GC method. Each run below overwrites
    // the output files of the previous run.
    test( /*use the default*/ nullptr);
    test( "full_sweeps_only");
    test( "full_sweep_then_pin_count_zero");
    test( "general_candidates_then_pin_count_zero");
    test( "invalid_method");
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
