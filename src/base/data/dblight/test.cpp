/******************************************************************************
 * Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <utility>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "i_dblight.h"
#include "dblight_database.h"
#include "dblight_info.h"
#include "dblight_scope.h"

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_database.h>
#include <base/data/db/i_db_element.h>
#include <base/data/db/i_db_fragmented_job.h>
#include <base/data/db/i_db_scope.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/db/i_db_transaction_ptr.h>

#include <base/data/serial/i_serializer.h>
#include <base/data/serial/serial.h>
#include <base/hal/thread/i_thread_condition.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_module.h>
#include <base/system/main/access_module.h>
#include <base/system/main/i_assert.h>
#include <base/util/registry/i_config_registry.h>

using namespace MI;
namespace fs = std::filesystem;

const DB::Journal_type journal_a( 0x1u);
const DB::Journal_type journal_b( 0x2u);
const DB::Journal_type journal_c( 0x4u);
const DB::Journal_type journal_not_c( ~0x4u);

class My_element : public DB::Element<My_element, 0x12345678>
{
public:
    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const final
    { serializer->write( m_value); serializer->write( m_tag_set); return this + 1; }
    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer) final
    { deserializer->read( &m_value);  deserializer->read( &m_tag_set); return this + 1; }
    DB::Element_base* copy() const final { return new My_element( *this); }
    std::string get_class_name() const final { return "My_element"; }
    void get_references( DB::Tag_set* result) const final
    { result->insert( m_tag_set.begin(), m_tag_set.end()); }
    DB::Journal_type get_journal_flags() const { return journal_not_c; }

    My_element() : m_value( 0) { }
    explicit My_element( int value, DB::Tag_set tag_set = {})
      : m_value( value), m_tag_set( std::move( tag_set)) { }
    void set_value( int value) { m_value = value; }
    int get_value() const { return m_value; }
    void set_tag_set( const DB::Tag_set& tag_set) { m_tag_set = tag_set; }
    const DB::Tag_set& get_tag_set() const { return m_tag_set; }

private:
    int m_value;
    DB::Tag_set m_tag_set;
};

class My_fragmented_job : public DB::Fragmented_job
{
public:
    void execute_fragment(
        DB::Transaction* transaction,
        size_t index,
        size_t count,
        const mi::neuraylib::IJob_execution_context* context) final
    {
        m_db->yield();
        m_db->suspend_current_job();
        m_db->resume_current_job();
        ++m_fragments_done;
    }

    Scheduling_mode get_scheduling_mode() const final { return m_scheduling_mode; }

    mi::Sint8 get_priority() const final { return m_priority; }

    void assign_fragments_to_hosts( mi::Uint32* slots, size_t nr_slots)
    {
        for( size_t i = 0; i < nr_slots; ++i)
            slots[i] = 1;
    }

    My_fragmented_job( DB::Database* db, Scheduling_mode scheduling_mode, mi::Sint8 priority)
      : m_db( db),
        m_scheduling_mode( scheduling_mode),
        m_priority( priority) { }

    std::atomic_uint32_t m_fragments_done = 0;

private:
    DB::Database* m_db;
    Scheduling_mode m_scheduling_mode;
    mi::Sint8 m_priority = 0;
};

/// A fragmented job where the first fragment blocks execution until the job is cancelled.
class Cancel_job : public DB::Fragmented_job
{
public:
    void execute_fragment(
        DB::Transaction* transaction,
        size_t index,
        size_t count,
        const mi::neuraylib::IJob_execution_context* context) final
    {
        if( index == 0)
            m_condition.wait();
    }

    Scheduling_mode get_scheduling_mode() const final { return m_scheduling_mode; }

    mi::Sint8 get_priority() const final { return m_priority; }

    void cancel() final { m_condition.signal(); }

    Cancel_job( DB::Database* db, Scheduling_mode scheduling_mode, mi::Sint8 priority)
      : m_db( db),
        m_scheduling_mode( scheduling_mode),
        m_priority( priority) { }

    std::atomic_uint32_t m_fragments_done = 0;

private:
    DB::Database* m_db;
    Scheduling_mode m_scheduling_mode;
    mi::Sint8 m_priority = 0;
    THREAD::Condition m_condition;
};

class My_execution_listener : public DB::IExecution_listener
{
public:
    void job_finished() final { m_finished = true; }
    std::atomic_bool m_finished = false;
};

class My_transaction_listener : public mi::base::Interface_implement<DB::ITransaction_listener>
{
public:
    void transaction_created( DB::Transaction* transaction) final
    { m_state << "created " << transaction->get_id()() << "\n"; }
    void transaction_pre_commit( DB::Transaction* transaction) final
    { m_state << "pre_commit " << transaction->get_id()() << "\n"; }
    void transaction_pre_abort( DB::Transaction* transaction) final
    { m_state << "pre_abort " << transaction->get_id()() << "\n"; }
    void transaction_committed( DB::Transaction* transaction) final
    { m_state << "committed " << transaction->get_id()() << "\n"; }
    void transaction_aborted( DB::Transaction* transaction) final
    { m_state << "aborted " << transaction->get_id()() << "\n"; }
    void check_empty()
    { MI_CHECK( m_state.str().empty()); }
    void check_and_clear( const char* event, mi::Uint32 id)
    {
        std::ostringstream expected;
        expected << event << " " << id << "\n";
        MI_CHECK_EQUAL( expected.str(), m_state.str());
        m_state.str("");
    }
    void check_and_clear2( const char* event1, const char* event2, mi::Uint32 id)
    {
        std::ostringstream expected;
        expected << event1 << " " << id << "\n";
        expected << event2 << " " << id << "\n";
        MI_CHECK_EQUAL( expected.str(), m_state.str());
        m_state.str("");
    }
private:
    std::ostringstream m_state;
};

class My_scope_listener : public mi::base::Interface_implement<DB::IScope_listener>
{
public:
    void scope_created( DB::Scope* scope) final
    { m_state << "created " << scope->get_id() << "\n"; }
    void scope_removed( DB::Scope* scope) final
    { m_state << "removed " << scope->get_id() << "\n"; }
    void check_empty()
    { MI_CHECK( m_state.str().empty()); }
    void check_and_clear( const char* event, mi::Uint32 id)
    {
        std::ostringstream expected;
        expected << event << " " << id << "\n";
        MI_CHECK_EQUAL( expected.str(), m_state.str());
        m_state.str("");
    }
private:
    std::ostringstream m_state;
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
                R"(Difference in line %zu: "%s" vs "%s".)",
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
            R"(Difference in line %zu: "%s" vs "%s".)",
            line_number, line1.c_str(), line2.c_str());
        return false;
    }

    return true;
}

struct Test_db
{
public:
    Test_db( const char* test_name, bool compare = true, bool enable_journal = false)
      : m_test_name( test_name), m_compare( compare)
    {
        std::cerr << "Begin test: " << m_test_name << std::endl;
        std::cerr << std::endl;

        m_db = DBLIGHT::factory(
            /*thread_pool*/ nullptr, /*deserialization_manager*/ nullptr, enable_journal);
        m_db_impl = static_cast<DBLIGHT::Database_impl*>( m_db);
        m_global_scope = m_db->get_global_scope();

        SERIAL::Deserialization_manager* manager = m_db_impl->get_deserialization_manager();
        manager->register_class<My_element>();
    }

    ~Test_db()
    {
        m_db->prepare_close();
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
    DB::Scope* m_global_scope;
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
            R"(Reference data "%s" for generated data "%s" not found.)",
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

void test_info_base()
{
    DBLIGHT::Info_base a1( DB::Scope_id( 20), DB::Transaction_id( 30), 40);
    DBLIGHT::Info_base a2( DB::Scope_id( 21), DB::Transaction_id( 30), 40);

    DBLIGHT::Info_base b1( DB::Scope_id( 20), DB::Transaction_id( 30), 40);
    DBLIGHT::Info_base b2( DB::Scope_id( 20), DB::Transaction_id( 31), 40);

    DBLIGHT::Info_base c1( DB::Scope_id( 20), DB::Transaction_id( 30), 40);
    DBLIGHT::Info_base c2( DB::Scope_id( 20), DB::Transaction_id( 30), 41);

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

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction);
    MI_CHECK( transaction->is_open());

    bool success = transaction->commit();
    db.dump();
    MI_CHECK( success);
    MI_CHECK( !transaction->is_open());

    success = transaction->commit();
    MI_CHECK( !success);

    MI_CHECK_EQUAL( transaction->get_scope(), db.m_global_scope);

    transaction.reset();
    db.dump();
}

void test_transaction_create_abort()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction);
    MI_CHECK( transaction->is_open());

    transaction->abort();
    db.dump();
    MI_CHECK( !transaction->is_open());

    transaction->abort();

    MI_CHECK_EQUAL( transaction->get_scope(), db.m_global_scope);

    transaction.reset();
    db.dump();
}

void test_two_transactions_create_commit()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction0);

    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
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

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    db.dump();
    MI_CHECK( transaction0);

    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    transaction->store( element1, "foo");
    db.dump();

    // Triggers an error message.
    auto* element2 = new My_element( 43);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    auto* element = new My_element( 42);
    transaction->store( tag, element);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_multiple_versions_with_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    auto* element1 = new My_element( 42);
    transaction->store( tag, element1, "foo");
    db.dump();

    auto* element2 = new My_element( 43);
    transaction->store( tag, element2, "foo");
    db.dump();

    // Triggers an error message.
    auto* element3 = new My_element( 44);
    transaction->store( DB::Tag(), element3, "foo");
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_multiple_versions_without_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    auto* element1 = new My_element( 42);
    transaction->store( tag, element1);
    db.dump();

    auto* element2 = new My_element( 43);
    transaction->store( tag, element2);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_multiple_names_per_tag()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    auto* element1 = new My_element( 42);
    transaction->store( tag, element1, "foo");
    db.dump();

    auto* element2 = new My_element( 43);
    transaction->store( tag, element2, "bar");
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_store_multiple_tags_per_name()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->reserve_tag();
    transaction->store( tag1, element1, "foo");
    db.dump();

    auto* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->reserve_tag();
    transaction->store( tag2, element2, "foo");
    db.dump();

    transaction->commit();
    db.dump();
}

void test_two_transactions_store_parallel()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
    DB::Tag tag = transaction0->reserve_tag();

    auto* element1 = new My_element( 42);
    transaction0->store( tag, element1, "foo");

    auto* element2 = new My_element( 43);
    transaction1->store( tag, element2, "foo");
    transaction0->commit();
    transaction1->commit();
    db.dump();

    DB::Transaction_ptr transaction2 = db.m_global_scope->start_transaction();

    {
        DB::Access<My_element> access( tag, transaction2.get());
        MI_CHECK_EQUAL( access->get_value(), 43);
    }

    transaction2->commit();
}

void test_two_transactions_store_parallel_reverse_commit()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
    DB::Tag tag = transaction0->reserve_tag();

    auto* element1 = new My_element( 42);
    transaction0->store( tag, element1, "foo");

    auto* element2 = new My_element( 43);
    transaction1->store( tag, element2, "foo");

    transaction1->commit();
    transaction0->commit();
    db.dump();

    DB::Transaction_ptr transaction2 = db.m_global_scope->start_transaction();

    {
        DB::Access<My_element> access( tag, transaction2.get());
        MI_CHECK_EQUAL( access->get_value(), 43);
    }

    transaction2->commit();
}

void test_two_transactions_store_parallel_not_globally_visible()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction2 = db.m_global_scope->start_transaction();
    DB::Tag tag = transaction0->reserve_tag();

    auto* element1 = new My_element( 42);
    transaction1->store( tag, element1, "foo");

    auto* element2 = new My_element( 43);
    transaction2->store( tag, element2, "foo");
    transaction1->commit();
    transaction2->commit();
    // Transaction 0 prevents that changes from transaction 1 and transaction 2 become globally
    // visible. Still the version with value 42 should be collected by the GC as in
    // test_two_transactions_store_parallel() without transaction 0.
    db.dump();

    transaction0->commit();
    db.dump();

    DB::Transaction_ptr transaction3 = db.m_global_scope->start_transaction();

    {
        DB::Access<My_element> access( tag, transaction3.get());
        MI_CHECK_EQUAL( access->get_value(), 43);
    }

    transaction3->commit();
}

void test_transaction_create_store_abort()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    db.dump();

    auto* element = new My_element( 42);
    transaction->store( element, "foo");
    db.dump();

    transaction->abort();
    db.dump();
}

void test_transaction_access()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    auto* element = new My_element( 42);
    DB::Tag tag = transaction0->store( element, "foo");
    transaction0->commit();

    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction2 = db.m_global_scope->start_transaction();

    {
        DB::Edit<My_element> edit2( tag, transaction1.get());
        edit2->set_value( 43);
        DB::Edit<My_element> edit3( tag, transaction2.get());
        edit3->set_value( 44);
    }

    transaction1->commit();
    transaction2->commit();
    db.dump();

    DB::Transaction_ptr transaction3 = db.m_global_scope->start_transaction();
    DB::Access<My_element> access4( tag, transaction3.get());
    MI_CHECK_EQUAL( access4->get_value(), 44);
    transaction3->commit();
}

void test_two_transactions_edits_parallel_reverse_commit()
{
    Test_db db( __func__);

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    auto* element = new My_element( 42);
    DB::Tag tag = transaction0->store( element, "foo");
    transaction0->commit();

    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction2 = db.m_global_scope->start_transaction();

    {
        DB::Edit<My_element> edit2( tag, transaction1.get());
        edit2->set_value( 43);
        DB::Edit<My_element> edit3( tag, transaction2.get());
        edit3->set_value( 44);
    }

    transaction2->commit();
    transaction1->commit();
    db.dump();

    DB::Transaction_ptr transaction3 = db.m_global_scope->start_transaction();
    DB::Access<My_element> access4( tag, transaction3.get());
    MI_CHECK_EQUAL( access4->get_value(), 44);
    transaction3->commit();
}

void test_visibility_single_transaction()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    {
        auto* element = new My_element( 42);
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

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    auto* element = new My_element( 42);
    DB::Tag tag = transaction0->store( element, "foo");
    transaction0->commit();

    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction2 = db.m_global_scope->start_transaction();
    {
        DB::Edit<My_element> edit3( tag, transaction2.get());
        edit3->set_value( 43);
    }
    DB::Transaction_ptr transaction3 = db.m_global_scope->start_transaction();
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

    DB::Transaction_ptr transaction4 = db.m_global_scope->start_transaction();

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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    DB::Tag tag = transaction->reserve_tag();

    auto* element1 = new My_element( 42);
    transaction->store( tag, element1, "foo");
    auto* element2 = new My_element( 43);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    auto* element2 = new My_element( 42);
    DB::Tag tag2 = transaction->store( element2, "foo");
    db.dump();

    DB::Tag tag = transaction->name_to_tag( "foo");
    MI_CHECK_EQUAL( tag, tag2);
    MI_CHECK_NOT_EQUAL( tag, tag1);

    tag = transaction->name_to_tag( "baz");
    MI_CHECK( !tag);
    tag = transaction->name_to_tag( "");
    MI_CHECK( !tag);
    tag = transaction->name_to_tag( nullptr);
    MI_CHECK( !tag);

    transaction->commit();
}

void test_transaction_name_to_tag_modified()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    auto* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->store( element2, "bar");
    transaction->remove( tag2);
    db.dump();

    DB::Tag tag1a = transaction->name_to_tag_modified( "foo");
    DB::Tag tag2a = transaction->name_to_tag_modified( "bar");
    db.dump();
    MI_CHECK_EQUAL( tag1a, tag1);
    MI_CHECK_EQUAL( tag2a, tag2);

    DB::Tag tag = transaction->name_to_tag( "baz");
    MI_CHECK( !tag);
    tag = transaction->name_to_tag( "");
    MI_CHECK( !tag);
    tag = transaction->name_to_tag( nullptr);
    MI_CHECK( !tag);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_transaction_tag_to_name_and_back()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    auto* element2 = new My_element( 43);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    DB::Tag tag1 = transaction->reserve_tag();

    auto* element1 = new My_element( 42);
    transaction->store( tag1, element1, "foo");
    auto* element2 = new My_element( 42);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");

    SERIAL::Class_id class_id = transaction->get_class_id( tag);
    SERIAL::Class_id expected_class_id = My_element::id;
    MI_CHECK_EQUAL( class_id, expected_class_id);

    class_id = transaction->get_class_id( DB::Tag( tag() + 1));
    MI_CHECK_EQUAL( class_id, SERIAL::class_id_unknown);

    transaction->commit();
}

void test_transaction_get_tag_reference_count()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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
    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction0->store( element1, "foo");
    auto* element2 = new My_element( 43);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    auto* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->store( element2, "bar");
    db.dump();

    MI_CHECK( transaction->remove( tag1, /*remove_local_copy*/ false));
    MI_CHECK( transaction->remove( tag2, /*remove_local_copy*/ true));
    db.dump();

    // Repeated calls succeed, too.
    MI_CHECK( transaction->remove( tag1, /*remove_local_copy*/ false));
    MI_CHECK( transaction->remove( tag2, /*remove_local_copy*/ true));

    MI_CHECK( !transaction->remove( DB::Tag(), /*remove_local_copy*/ false));
    MI_CHECK( !transaction->remove( DB::Tag(), /*remove_local_copy*/ true));

    transaction->commit();
    db.dump();
}

void test_transaction_get_tag_is_removed()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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

void test_transaction_access_after_remove()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store_for_reference_counting( element1, "foo");
    auto* element2 = new My_element( 43);
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    db.dump();

    transaction->commit();
    transaction = db.m_global_scope->start_transaction();

    {
        DB::Edit<My_element> edit( tag, transaction.get());
        edit->set_value( 43);
    }

    transaction->abort();
    transaction = db.m_global_scope->start_transaction();

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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    transaction->remove( tag1);
    auto* element2 = new My_element( 43, {tag1});
    DB::Tag tag2 = transaction->store( element2, "bar");
    db.dump();

    transaction->commit();
    db.dump();

    transaction = db.m_global_scope->start_transaction();

    transaction->remove( tag2);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_gc_simple_reference_abort()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    transaction->remove( tag1);
    auto* element2 = new My_element( 43, {tag1});
    transaction->store( element2, "bar");
    db.dump();

    transaction->abort();
    db.dump();
}

void test_gc_multiple_references()
{
    // Same as before, but multiple references, and set references in an edit to test that.
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo");
    auto* element2 = new My_element( 43);
    DB::Tag tag2 = transaction->store( element2, "bar");
    auto* element3 = new My_element( 44);
    DB::Tag tag3 = transaction->store( element3, "baz");
    db.dump();

    transaction->commit();
    transaction = db.m_global_scope->start_transaction();

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

    transaction = db.m_global_scope->start_transaction();

    transaction->remove( tag3);
    db.dump();

    transaction->commit();
    db.dump();
}

void test_gc_explicit_call()
{
    Test_db db( __func__);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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

    // Create element in transaction 0.
    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    auto* element = new My_element( 42);
    DB::Tag tag = transaction0->store( element, "foo");
    transaction0->commit();
    transaction0.reset();

    // Remove element in transaction 1. The open transaction 2 avoids that the removal becomes
    // globally visible when transaction 1 is committed.
    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();
    transaction1->remove( tag);
    DB::Transaction_ptr transaction2 = db.m_global_scope->start_transaction();
    transaction1->commit();
    transaction1.reset();
    db.dump();

    // Access the element in transaction 3.
    DB::Transaction_ptr transaction3 = db.m_global_scope->start_transaction();
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
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
    DB::Tag tag = transaction->store( element, "foo");
    DB::Info* open_edit_info = transaction->edit_element( tag);
    MI_CHECK( open_edit_info);
    // Keeping open_edit_info while the transaction is committed is wrong, but we want to test
    // exactly that scenario with finish_edit() below.
    transaction->commit();

    DB::Info* info;
    info = transaction->access_element( tag);
    MI_CHECK( !info);
    info = transaction->edit_element( tag);
    MI_CHECK( !info);
    transaction->finish_edit( open_edit_info, DB::Journal_type());
    auto* element2 = new My_element( 43);
    transaction->store(
        tag,
        element2,
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
    mi::Uint32 level = transaction->get_tag_privacy_level( tag);
    MI_CHECK_EQUAL( 0, level);
    level = transaction->get_tag_store_level( tag);
    MI_CHECK_EQUAL( 0, level);

    // Explicitly run the garbage collection to clean up the transaction that was pinned by the
    // info. Otherwise, it will trigger an assertion on shutdown.
    open_edit_info->unpin();
    db.m_db->garbage_collection( /*priority*/ 0);
}

void test_scope_creation()
{
    Test_db db( __func__);

    // Create unnamed scope.
    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    MI_CHECK( scope1);
    db.dump();

    // Create unnamed scope with wrong privacy level.
    DB::Scope* scope2 = scope1->create_child( 1);
    MI_CHECK( !scope2);

    // Create unnamed scope with wrong privacy level.
    DB::Scope* scope3 = db.m_global_scope->create_child( 0);
    MI_CHECK( !scope3);

    // Create named scope.
    DB::Scope* scope4 = db.m_global_scope->create_child( 1, /*is_temporary*/ false, "foo");
    MI_CHECK( scope4);
    db.dump();

    // Retrieve named scope.
    DB::Scope* scope5 = db.m_global_scope->create_child( 1, /*is_temporary*/ false, "foo");
    MI_CHECK( scope5);
    MI_CHECK_EQUAL( scope5, scope4);
    db.dump();

    // Retrieve named scope with wrong privacy level.
    DB::Scope* scope6 = db.m_global_scope->create_child( 2, /*is_temporary*/ false, "foo");
    MI_CHECK( !scope6);

    // Retrieve named scope with wrong parent scope.
    DB::Scope* scope7 = scope4->create_child( 1, /*is_temporary*/ false, "foo");
    MI_CHECK( !scope7);

    db.dump();
}

void test_scope_lookup()
{
    Test_db db( __func__);

    // Create unnamed scope.
    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    MI_CHECK( scope1);
    db.dump();

    // Look up unnamed scope by scope ID.
    DB::Scope* scope2 = db.m_db->lookup_scope( scope1->get_id());
    MI_CHECK( scope2);
    MI_CHECK_EQUAL( scope2, scope1);

    // Look up global scope by scope ID.
    DB::Scope* scope3 = db.m_db->lookup_scope( 0);
    MI_CHECK( scope3);
    MI_CHECK_EQUAL( scope3, db.m_global_scope);

    // Create named scope.
    DB::Scope* scope4 = db.m_global_scope->create_child( 1, /*is_temporary*/ false, "foo");
    MI_CHECK( scope4);
    db.dump();

    // Look up named scope by name.
    DB::Scope* scope5 = db.m_db->lookup_scope( scope4->get_name());
    MI_CHECK( scope5);
    MI_CHECK_EQUAL( scope5, scope4);

    // Look up named scope by scope ID.
    DB::Scope* scope6 = db.m_db->lookup_scope( scope4->get_id());
    MI_CHECK( scope6);
    MI_CHECK_EQUAL( scope6, scope4);

    // No lookup of global scope by name.
    DB::Scope* scope7 = db.m_db->lookup_scope( "");
    MI_CHECK( !scope7);
}

void test_scope_removal()
{
    Test_db db( __func__);

    // Removal of global scope.
    bool success = db.m_db->remove_scope( 0);
    MI_CHECK( !success);

    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    DB::Scope* scope2 = scope1->create_child( 2);
    DB::Scope* scope3 = scope1->create_child( 2);
    DB::Scope_id scope1_id = scope1->get_id();
    DB::Scope_id scope2_id = scope2->get_id();
    DB::Scope_id scope3_id = scope3->get_id();
    db.dump();

    // Removal of leaf scope (imminent removal).
    success = db.m_db->remove_scope( scope3_id);
    MI_CHECK( success);
    db.dump();

    // Removal of no-longer existing scope.
    success = db.m_db->remove_scope( scope3_id);
    MI_CHECK( !success);

    // Removal of non-leaf scope (only marked for removal).
    success = db.m_db->remove_scope( scope1_id);
    MI_CHECK( success);
    db.dump();

    // Repeated removal of non-leaf scope.
    success = db.m_db->remove_scope( scope1_id);
    MI_CHECK( success);

    // Removal of scope that keeps scope1 around.
    success = db.m_db->remove_scope( scope2_id);
    MI_CHECK( success);
    db.dump();
}

void test_transaction_store_with_scopes()
{
    Test_db db( __func__);

    DB::Scope* scope1 = db.m_global_scope->create_child( 10);
    DB::Scope* scope2 = scope1->create_child( 20);
    db.dump();

    DB::Transaction_ptr transaction = scope2->start_transaction();

    // Stored in scope2 due to store level.
    auto* element1 = new My_element( 41);
    transaction->store( element1, "foo1", 255, 255);

    // Stored in scope2 due to store level.
    auto* element2 = new My_element( 42);
    transaction->store( element2, "foo2", 255, 20);

    // Stored in scope1 due to store level.
    auto* element3 = new My_element( 43);
    transaction->store( element3, "foo3", 255, 15);

    // Stored in scope1 due to store level.
    auto* element4 = new My_element( 44);
    transaction->store( element4, "foo4", 255, 10);

    // Stored in global scope due to store level.
    auto* element5 = new My_element( 45);
    transaction->store( element5, "foo5", 255, 0);

    // Stored in scope2 due to privacy level.
    auto* element6 = new My_element( 46);
    transaction->store( element6, "foo6", 20, 255);

    // Stored in scope2 due to privacy level.
    auto* element7 = new My_element( 47);
    transaction->store( element7, "foo7", 15, 255);

    // Stored in scope1 due to privacy level.
    auto* element8 = new My_element( 48);
    transaction->store( element8, "foo8", 10, 255);

    // Stored in global scope due to privacy level.
    auto* element9 = new My_element( 49);
    transaction->store( element9, "foo9", 0, 255);

    transaction->commit();
    db.dump();

    // Test again scope removal, now with content.

    // Reset transaction such it does no longer pin its scope.
    transaction.reset();

    // Removal of non-leaf scope (only marked for removal).
    bool success = db.m_db->remove_scope( scope1->get_id());
    MI_CHECK( success);
    db.dump();

    // Removal of scope that keeps scope1 around.
    success = db.m_db->remove_scope( scope2->get_id());
    MI_CHECK( success);
    db.dump();
}

void test_transaction_access_with_scopes()
{
    Test_db db( __func__);

    DB::Scope* scope1 = db.m_global_scope->create_child( 10);
    DB::Scope* scope2 = scope1->create_child( 20);

    DB::Transaction_ptr transaction = scope2->start_transaction();
    DB::Tag tag = transaction->reserve_tag();

    auto* element1 = new My_element( 41);
    transaction->store( tag, element1, "foo", 255, DB::JOURNAL_NONE, 0);
    auto* element2 = new My_element( 42);
    transaction->store( tag, element2, "foo", 255, DB::JOURNAL_NONE, 10);
    auto* element3 = new My_element( 43);
    transaction->store( tag, element3, "foo", 255, DB::JOURNAL_NONE, 20);
    transaction->commit();
    db.dump();

    transaction = db.m_global_scope->start_transaction();
    {
        DB::Access<My_element> access( tag, transaction.get());
        MI_CHECK_EQUAL( access->get_value(), 41);
    }
    transaction->commit();

    transaction = scope1->start_transaction();
    {
        DB::Access<My_element> access( tag, transaction.get());
        MI_CHECK_EQUAL( access->get_value(), 42);
    }
    transaction->commit();

    transaction = scope2->start_transaction();
    {
        DB::Access<My_element> access( tag, transaction.get());
        MI_CHECK_EQUAL( access->get_value(), 43);
    }
    transaction->commit();
}

void test_transaction_edit_with_scopes()
{
    Test_db db( __func__);

    DB::Scope* scope1 = db.m_global_scope->create_child( 10);
    DB::Scope* scope2 = scope1->create_child( 20);
    db.dump();

    // Store five elements in the global scope with different privacy levels.
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    auto* element1 = new My_element( 410);
    DB::Tag tag1 = transaction->store( element1, "foo1", 0, 0);
    auto* element2 = new My_element( 420);
    DB::Tag tag2 = transaction->store( element2, "foo2", 5, 0);
    auto* element3 = new My_element( 430);
    DB::Tag tag3 = transaction->store( element3, "foo3", 10, 0);
    auto* element4 = new My_element( 440);
    DB::Tag tag4 = transaction->store( element4, "foo4", 15, 0);
    auto* element5 = new My_element( 450);
    DB::Tag tag5 = transaction->store( element5, "foo5", 20, 0);
    transaction->commit();
    db.dump();

    // All edits end up in the global scope with a transaction from the global scope
    transaction = db.m_global_scope->start_transaction();
    {
        DB::Edit<My_element> edit1( tag1, transaction.get());
        edit1->set_value( 411);
        DB::Edit<My_element> edit2( tag2, transaction.get());
        edit2->set_value( 421);
        DB::Edit<My_element> edit3( tag3, transaction.get());
        edit3->set_value( 431);
        DB::Edit<My_element> edit4( tag4, transaction.get());
        edit4->set_value( 441);
        DB::Edit<My_element> edit5( tag5, transaction.get());
        edit5->set_value( 451);
    }
    transaction->commit();
    db.dump();

    // Some edits (edit3/4/5) end up in scope1 with a transaction from scope1.
    transaction = scope1->start_transaction();
    {
        DB::Edit<My_element> edit1( tag1, transaction.get());
        edit1->set_value( 412);
        DB::Edit<My_element> edit2( tag2, transaction.get());
        edit2->set_value( 422);
        DB::Edit<My_element> edit3( tag3, transaction.get());
        edit3->set_value( 432);
        DB::Edit<My_element> edit4( tag4, transaction.get());
        edit4->set_value( 442);
        DB::Edit<My_element> edit5( tag5, transaction.get());
        edit5->set_value( 452);
    }
    transaction->commit();
    db.dump();

    // Edit5 ends up in scope2 with a transaction from scope2.
    transaction = scope2->start_transaction();
    {
        DB::Edit<My_element> edit1( tag1, transaction.get());
        edit1->set_value( 413);
        DB::Edit<My_element> edit2( tag2, transaction.get());
        edit2->set_value( 423);
        DB::Edit<My_element> edit3( tag3, transaction.get());
        edit3->set_value( 433);
        DB::Edit<My_element> edit4( tag4, transaction.get());
        edit4->set_value( 443);
        DB::Edit<My_element> edit5( tag5, transaction.get());
        edit5->set_value( 453);
    }
    transaction->commit();
    db.dump();

    // Check from the global scope.
    transaction = db.m_global_scope->start_transaction();
    {
        DB::Access<My_element> access1( tag1, transaction.get());
        MI_CHECK_EQUAL( access1->get_value(), 413);
        DB::Access<My_element> access2( tag2, transaction.get());
        MI_CHECK_EQUAL( access2->get_value(), 423);
        DB::Access<My_element> access3( tag3, transaction.get());
        MI_CHECK_EQUAL( access3->get_value(), 431);
        DB::Access<My_element> access4( tag4, transaction.get());
        MI_CHECK_EQUAL( access4->get_value(), 441);
        DB::Access<My_element> access5( tag5, transaction.get());
        MI_CHECK_EQUAL( access5->get_value(), 451);
    }
    transaction->commit();

    // Check from scope1.
    transaction = scope1->start_transaction();
    {
        DB::Access<My_element> access1( tag1, transaction.get());
        MI_CHECK_EQUAL( access1->get_value(), 413);
        DB::Access<My_element> access2( tag2, transaction.get());
        MI_CHECK_EQUAL( access2->get_value(), 423);
        DB::Access<My_element> access3( tag3, transaction.get());
        MI_CHECK_EQUAL( access3->get_value(), 433);
        DB::Access<My_element> access4( tag4, transaction.get());
        MI_CHECK_EQUAL( access4->get_value(), 443);
        DB::Access<My_element> access5( tag5, transaction.get());
        MI_CHECK_EQUAL( access5->get_value(), 452);
    }
    transaction->commit();

    // Check from scope2.
    transaction = scope2->start_transaction();
    {
        DB::Access<My_element> access1( tag1, transaction.get());
        MI_CHECK_EQUAL( access1->get_value(), 413);
        DB::Access<My_element> access2( tag2, transaction.get());
        MI_CHECK_EQUAL( access2->get_value(), 423);
        DB::Access<My_element> access3( tag3, transaction.get());
        MI_CHECK_EQUAL( access3->get_value(), 433);
        DB::Access<My_element> access4( tag4, transaction.get());
        MI_CHECK_EQUAL( access4->get_value(), 443);
        DB::Access<My_element> access5( tag5, transaction.get());
        MI_CHECK_EQUAL( access5->get_value(), 453);
    }
    transaction->commit();
}

void test_transaction_get_store_and_privacy_level()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    DB::Scope* scope1 = db.m_global_scope->create_child( 10);
    DB::Scope* scope2 = scope1->create_child( 20);

    DB::Transaction_ptr transaction = scope2->start_transaction();

    auto* element1 = new My_element( 41);
    DB::Tag tag1 = transaction->store( element1, "foo", 5, 0);
    auto* element2 = new My_element( 42);
    DB::Tag tag2 = transaction->store( element2, "foo", 15, 10);
    auto* element3 = new My_element( 43);
    DB::Tag tag3 = transaction->store( element3, "foo", 25, 20);
    transaction->commit();

    transaction = scope2->start_transaction();
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag1),  0);
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag2), 10);
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag3), 20);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag1),  5);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag2), 15);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag3), 25);
    transaction->commit();

    transaction = scope1->start_transaction();
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag1),  0);
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag2), 10);
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag3),  0);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag1),  5);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag2), 15);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag3),  0);
    transaction->commit();

    transaction = db.m_global_scope->start_transaction();
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag1),  0);
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag2),  0);
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag3),  0);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag1),  5);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag2),  0);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag3),  0);
    transaction->commit();
}

void test_transaction_can_reference()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    DB::Scope* scope1 = db.m_global_scope->create_child( 10);
    DB::Scope* scope2 = scope1->create_child( 20);

    DB::Transaction_ptr transaction = scope2->start_transaction();

    auto* element1 = new My_element( 41);
    DB::Tag tag1 = transaction->store( element1, "foo", 255, 0);
    auto* element2 = new My_element( 42);
    DB::Tag tag2 = transaction->store( element2, "foo", 255, 10);
    auto* element3 = new My_element( 43);
    DB::Tag tag3 = transaction->store( element3, "foo", 255, 20);
    transaction->commit();

    transaction = scope2->start_transaction();
    MI_CHECK(  transaction->can_reference_tag(  0, tag1));
    MI_CHECK( !transaction->can_reference_tag(  0, tag2));
    MI_CHECK( !transaction->can_reference_tag(  0, tag3));
    MI_CHECK(  transaction->can_reference_tag( 10, tag1));
    MI_CHECK(  transaction->can_reference_tag( 10, tag2));
    MI_CHECK( !transaction->can_reference_tag( 10, tag3));
    MI_CHECK(  transaction->can_reference_tag( 20, tag1));
    MI_CHECK(  transaction->can_reference_tag( 20, tag2));
    MI_CHECK(  transaction->can_reference_tag( 20, tag3));

    MI_CHECK(  transaction->can_reference_tag(  tag1, tag1));
    MI_CHECK( !transaction->can_reference_tag(  tag1, tag2));
    MI_CHECK( !transaction->can_reference_tag(  tag1, tag3));
    MI_CHECK(  transaction->can_reference_tag(  tag2, tag1));
    MI_CHECK(  transaction->can_reference_tag(  tag2, tag2));
    MI_CHECK( !transaction->can_reference_tag(  tag2, tag3));
    MI_CHECK(  transaction->can_reference_tag(  tag3, tag1));
    MI_CHECK(  transaction->can_reference_tag(  tag3, tag2));
    MI_CHECK(  transaction->can_reference_tag(  tag3, tag3));
    transaction->commit();

    transaction = scope1->start_transaction();
    MI_CHECK(  transaction->can_reference_tag(  0, tag1));
    MI_CHECK( !transaction->can_reference_tag(  0, tag2));
    MI_CHECK( !transaction->can_reference_tag(  0, tag3));
    MI_CHECK(  transaction->can_reference_tag( 10, tag1));
    MI_CHECK(  transaction->can_reference_tag( 10, tag2));
    MI_CHECK( !transaction->can_reference_tag( 10, tag3));
    MI_CHECK(  transaction->can_reference_tag( 20, tag1));
    MI_CHECK(  transaction->can_reference_tag( 20, tag2));
    MI_CHECK( !transaction->can_reference_tag( 20, tag3));

    MI_CHECK(  transaction->can_reference_tag(  tag1, tag1));
    MI_CHECK( !transaction->can_reference_tag(  tag1, tag2));
    MI_CHECK( !transaction->can_reference_tag(  tag1, tag3));
    MI_CHECK(  transaction->can_reference_tag(  tag2, tag1));
    MI_CHECK(  transaction->can_reference_tag(  tag2, tag2));
    MI_CHECK( !transaction->can_reference_tag(  tag2, tag3));
    MI_CHECK( !transaction->can_reference_tag(  tag3, tag1));
    MI_CHECK( !transaction->can_reference_tag(  tag3, tag2));
    MI_CHECK( !transaction->can_reference_tag(  tag3, tag3));
    transaction->commit();

    transaction = db.m_global_scope->start_transaction();
    MI_CHECK(  transaction->can_reference_tag(  0, tag1));
    MI_CHECK( !transaction->can_reference_tag(  0, tag2));
    MI_CHECK( !transaction->can_reference_tag(  0, tag3));
    MI_CHECK(  transaction->can_reference_tag( 10, tag1));
    MI_CHECK( !transaction->can_reference_tag( 10, tag2));
    MI_CHECK( !transaction->can_reference_tag( 10, tag3));
    MI_CHECK(  transaction->can_reference_tag( 20, tag1));
    MI_CHECK( !transaction->can_reference_tag( 20, tag2));
    MI_CHECK( !transaction->can_reference_tag( 20, tag3));

    MI_CHECK(  transaction->can_reference_tag(  tag1, tag1));
    MI_CHECK( !transaction->can_reference_tag(  tag1, tag2));
    MI_CHECK( !transaction->can_reference_tag(  tag1, tag3));
    MI_CHECK( !transaction->can_reference_tag(  tag2, tag1));
    MI_CHECK( !transaction->can_reference_tag(  tag2, tag2));
    MI_CHECK( !transaction->can_reference_tag(  tag2, tag3));
    MI_CHECK( !transaction->can_reference_tag(  tag3, tag1));
    MI_CHECK( !transaction->can_reference_tag(  tag3, tag2));
    MI_CHECK( !transaction->can_reference_tag(  tag3, tag3));
    transaction->commit();
}

void test_transaction_localize()
{
    Test_db db( __func__);

    DB::Scope* scope1 = db.m_global_scope->create_child( 10);

    DB::Transaction_ptr transaction = scope1->start_transaction();

    auto* element = new My_element( 41);
    DB::Tag tag = transaction->store( element, "foo", 255, 0);
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag), 0);
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag), 255);

    transaction->localize( tag, 15, DB::JOURNAL_NONE);
    MI_CHECK_EQUAL( transaction->get_tag_store_level( tag), 10);
    // Explicit localization overrides earlier privacy level.
    MI_CHECK_EQUAL( transaction->get_tag_privacy_level( tag), 15);
    transaction->commit();
    db.dump();
}

void test_transaction_remove_with_scopes()
{
    Test_db db( __func__);

    {
        // Check that the scope of the transaction is ignored for global removals.
        DB::Scope* scope1 = db.m_global_scope->create_child( 10);
        DB::Transaction_ptr transaction = scope1->start_transaction();
        auto* element = new My_element( 42);
        DB::Tag tag = transaction->store( element, "foo", 0, 0);
        transaction->localize( tag, 10, DB::JOURNAL_NONE);
        MI_CHECK( transaction->remove( tag, /*remove_local_copy*/ false));
        MI_CHECK( transaction->remove( tag, /*remove_local_copy*/ false));
        transaction->commit();
        transaction.reset();
        db.dump();
        db.m_db->remove_scope( scope1->get_id());
    }

    DB::Tag tag;

    {
        // Prepare global scope (same for all remaining subtests).
        DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
        auto* element = new My_element( 42);
        tag = transaction->store( element, "foo", 0, 0);
        transaction->commit();
        db.dump();
    }

    {
        // Local removals fail if there is no version in the scope of the transaction.
        DB::Scope* scope1 = db.m_global_scope->create_child( 10);
        DB::Transaction_ptr transaction = scope1->start_transaction();
        MI_CHECK( !transaction->remove( tag, /*remove_local_copy*/ true));
        transaction->commit();
        transaction.reset();
        db.dump();
        db.m_db->remove_scope( scope1->get_id());
    }

    {
        // Local removals fail if there is no version in a more global scope.
        DB::Scope* scope1 = db.m_global_scope->create_child( 10);
        DB::Transaction_ptr transaction = scope1->start_transaction();
        auto* element = new My_element( 42);
        DB::Tag tag2 = transaction->store( element, "bar", 10, 10);
        MI_CHECK( !transaction->remove( tag2, /*remove_local_copy*/ true));
        transaction->commit();
        transaction.reset();
        db.dump();
        db.m_db->remove_scope( scope1->get_id());
    }

    {
        // Local removals hide older version from the same scope.
        DB::Scope* scope1 = db.m_global_scope->create_child( 10);
        DB::Transaction_ptr transaction = scope1->start_transaction();
        transaction->localize( tag, 10, DB::JOURNAL_NONE);
        {
            DB::Edit<My_element> edit( tag, transaction.get());
            edit->set_value( 43);
        }
        MI_CHECK( transaction->remove( tag, /*remove_local_copy*/ true));
        {
            DB::Access<My_element> access( tag, transaction.get());
            MI_CHECK_EQUAL( access->get_value(), 42);
            MI_CHECK_EQUAL( transaction->get_tag_store_level( tag), 0);
        }
        transaction->commit();
        transaction.reset();
        db.dump();
        db.m_db->remove_scope( scope1->get_id());
    }

    {
        // Local removals do not hide newer versions from the same scope.
        DB::Scope* scope1 = db.m_global_scope->create_child( 10);
        DB::Transaction_ptr transaction = scope1->start_transaction();
        transaction->localize( tag, 10, DB::JOURNAL_NONE);
        {
            DB::Edit<My_element> edit( tag, transaction.get());
            edit->set_value( 43);
        }
        MI_CHECK( transaction->remove( tag, /*remove_local_copy*/ true));
        transaction->localize( tag, 10, DB::JOURNAL_NONE);
        {
            DB::Access<My_element> access( tag, transaction.get());
            MI_CHECK_EQUAL( access->get_value(), 42);
            MI_CHECK_EQUAL( transaction->get_tag_store_level( tag), 10);
        }
        transaction->commit();
        transaction.reset();
        db.dump();
        db.m_db->remove_scope( scope1->get_id());
    }

    db.dump();
}

void test_scope_removal_possibly_delayed()
{
    Test_db db( __func__);

    // Adapted from prod/lib/neuray/example_db_scope_removal.cpp
    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    DB::Scope* scope2 = db.m_global_scope->create_child( 1);

    DB::Transaction_ptr transaction1 = scope1->start_transaction();
    auto* element = new My_element( 42);
    DB::Tag tag = transaction1->store( element, "foo", 0, 0);
    MI_CHECK( transaction1->remove( tag, /*remove_local_copy*/ false));

    DB::Transaction_ptr transaction2 = scope2->start_transaction();
    transaction2->commit();
    transaction2.reset();

    transaction1->commit();
    transaction1.reset();
    db.dump();

    DB::Scope_id scope1_id = scope1->get_id();
    db.m_db->remove_scope( scope1_id);
    db.dump();
    // Scope 1 has been removed.
    MI_CHECK( !db.m_db->lookup_scope( scope1_id));
}

void test_scope_delayed_gc()
{
    Test_db db( __func__);

    // See MDL-1353.
    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    DB::Scope* scope2 = db.m_global_scope->create_child( 1);

    // Prepare element in scope1.
    DB::Transaction_ptr transaction1a = scope1->start_transaction();
    auto* element = new My_element( 42);
    DB::Tag tag = transaction1a->store( element, "foo", 1, 1);
    transaction1a->commit();
    transaction1a.reset();

    // Long-running transaction in scope2.
    DB::Transaction_ptr transaction2 = scope2->start_transaction();

    // Short-running transaction in scope1 that edits the element.
    DB::Transaction_ptr transaction1b = scope1->start_transaction();
    {
        DB::Edit<My_element> edit( tag, transaction1b.get());
        edit->set_value( 43);
    }
    transaction1b->commit();
    transaction1b.reset();
    // The transaction is scope2 prevents the GC from collecting the old version of the element in
    // scope1.
    db.dump();

    // Committing the transaction in scope2 finally allows the GC to collect the old version of the
    // element in scope1.
    transaction2->commit();
    transaction2.reset();
    db.dump();
}

void test_fragmented_jobs_without_transaction()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        mi::Sint32 result = db.m_db->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( job.m_fragments_done, 10);
    }

    {
        mi::Sint32 result = db.m_db->execute_fragmented( nullptr, 10);
        MI_CHECK_EQUAL( result, -1);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        mi::Sint32 result = db.m_db->execute_fragmented( &job, 0);
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::CLUSTER, 0);
        mi::Sint32 result = db.m_db->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::ONCE_PER_HOST, 0);
        mi::Sint32 result = db.m_db->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::USER_DEFINED, 0);
        mi::Sint32 result = db.m_db->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, -1);
        mi::Sint32 result = db.m_db->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, -3);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }
}

void test_fragmented_jobs_without_transaction_async()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        My_execution_listener listener;
        mi::Sint32 result = db.m_db->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, 0);
        using namespace std::chrono_literals;
        while( !listener.m_finished)
            std::this_thread::sleep_for( 1ms);
        MI_CHECK_EQUAL( job.m_fragments_done, 10);
    }

    {
        My_execution_listener listener;
        mi::Sint32 result = db.m_db->execute_fragmented_async( nullptr, 10, &listener);
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK( !listener.m_finished);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        My_execution_listener listener;
        mi::Sint32 result = db.m_db->execute_fragmented_async( &job, 0, &listener);
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::CLUSTER, 0);
        My_execution_listener listener;
        mi::Sint32 result = db.m_db->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::ONCE_PER_HOST, 0);
        My_execution_listener listener;
        mi::Sint32 result = db.m_db->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::USER_DEFINED, 0);
        My_execution_listener listener;
        mi::Sint32 result = db.m_db->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, -1);
        My_execution_listener listener;
        mi::Sint32 result = db.m_db->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, -3);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }
}

void test_fragmented_jobs_with_transaction()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        mi::Sint32 result = transaction->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( job.m_fragments_done, 10);
    }

    {
        mi::Sint32 result = transaction->execute_fragmented( nullptr, 10);
        MI_CHECK_EQUAL( result, -1);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        mi::Sint32 result = transaction->execute_fragmented( &job, 0);
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::CLUSTER, 0);
        mi::Sint32 result = transaction->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( job.m_fragments_done, 10);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::ONCE_PER_HOST, 0);
        mi::Sint32 result = transaction->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( job.m_fragments_done, 1);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::USER_DEFINED, 0);
        mi::Sint32 result = transaction->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, 0);
        MI_CHECK_EQUAL( job.m_fragments_done, 10);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, -1);
        mi::Sint32 result = transaction->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, -3);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    transaction->commit();

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        mi::Sint32 result = transaction->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, -4);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }
}

void test_fragmented_jobs_with_transaction_async()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, 0);
        using namespace std::chrono_literals;
        while( !listener.m_finished)
            std::this_thread::sleep_for( 1ms);
        MI_CHECK_EQUAL( job.m_fragments_done, 10);
    }

    {
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( nullptr, 10, &listener);
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK( !listener.m_finished);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 0, &listener);
        MI_CHECK_EQUAL( result, -1);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::CLUSTER, 0);
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::ONCE_PER_HOST, 0);
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::USER_DEFINED, 0);
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, -2);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, -1);
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, -3);
        MI_CHECK( !listener.m_finished);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }

    transaction->commit();

    {
        My_fragmented_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        mi::Sint32 result = transaction->execute_fragmented( &job, 10);
        MI_CHECK_EQUAL( result, -4);
        MI_CHECK_EQUAL( job.m_fragments_done, 0);
    }
}

void test_fragmented_jobs_cancelling()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    {
        // Explicit cancelling.
        DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
        Cancel_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, 0);
        transaction->cancel_fragmented_jobs();
        using namespace std::chrono_literals;
        while( !listener.m_finished)
            std::this_thread::sleep_for( 1ms);
        transaction->commit();
    }
    {
        // Implicit cancelling via committing the transaction.
        DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
        Cancel_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        My_execution_listener listener;
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, 0);
        transaction->commit();
        // The transaction is unblocked just before the finished notification is passed on to the
        // job, i.e., there is a short period of time in which listener.m_finished is still false.
        using namespace std::chrono_literals;
        while( !listener.m_finished)
            std::this_thread::sleep_for( 1ms);
    }

    {
        // Implicit cancelling via aborting the transaction.
        DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
        My_execution_listener listener;
        Cancel_job job( db.m_db, DB::Fragmented_job::LOCAL, 0);
        mi::Sint32 result = transaction->execute_fragmented_async( &job, 10, &listener);
        MI_CHECK_EQUAL( result, 0);
        transaction->abort();
        // The transaction is unblocked just before the finished notification is passed on to the
        // job, i.e., there is a short period of time in which listener.m_finished is still false.
        using namespace std::chrono_literals;
        while( !listener.m_finished)
            std::this_thread::sleep_for( 1ms);
    }
}

void test_transaction_listeners()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    mi::base::Handle listener( new My_transaction_listener);
    listener->check_empty();
    db.m_db->register_transaction_listener( listener.get());
    db.m_db->register_transaction_listener( nullptr);
    db.m_db->unregister_transaction_listener( listener.get());
    db.m_db->unregister_transaction_listener( nullptr);

    db.m_db->register_transaction_listener( listener.get());
    db.m_db->register_transaction_listener( nullptr);
    listener->check_empty();

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    listener->check_and_clear( "created", 0);

    transaction->commit();
    listener->check_and_clear2( "pre_commit", "committed", 0);

    transaction = db.m_global_scope->start_transaction();
    listener->check_and_clear( "created", 1);

    transaction->abort();
    listener->check_and_clear2( "pre_abort", "aborted", 1);

    // test operation/destruction with the client giving up its reference of the listener
    listener.reset();

    transaction = db.m_global_scope->start_transaction();
    transaction->commit();
    transaction = db.m_global_scope->start_transaction();
    transaction->abort();
}

void test_scope_listeners()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    mi::base::Handle listener( new My_scope_listener);
    listener->check_empty();
    db.m_db->register_scope_listener( listener.get());
    db.m_db->register_scope_listener( nullptr);
    db.m_db->unregister_scope_listener( listener.get());
    db.m_db->unregister_scope_listener( nullptr);

    db.m_db->register_scope_listener( listener.get());
    db.m_db->register_scope_listener( nullptr);
    listener->check_empty();

    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    DB::Scope_id scope1_id = scope1->get_id();
    listener->check_and_clear( "created", scope1_id);

    // Removal of global scope.
    bool success = db.m_db->remove_scope( 0);
    MI_CHECK( !success);
    listener->check_empty();

    DB::Scope* scope2 = scope1->create_child( 2);
    DB::Scope_id scope2_id = scope2->get_id();
    listener->check_and_clear( "created", scope2_id);

    DB::Scope* scope3 = scope1->create_child( 2);
    DB::Scope_id scope3_id = scope3->get_id();
    listener->check_and_clear( "created", scope3_id);

    // Removal of leaf scope (imminent removal).
    success = db.m_db->remove_scope( scope3_id);
    MI_CHECK( success);
    listener->check_and_clear( "removed", scope3_id);

    // Removal of no-longer existing scope.
    success = db.m_db->remove_scope( scope3_id);
    MI_CHECK( !success);
    listener->check_empty();

    // Removal of non-leaf scope (only marked for removal).
    success = db.m_db->remove_scope( scope1_id);
    MI_CHECK( success);
    listener->check_and_clear( "removed", scope1_id);

    // Repeated removal of non-leaf scope.
    success = db.m_db->remove_scope( scope1_id);
    MI_CHECK( success);
    listener->check_empty();

    // Removal of scope that keeps scope1 around.
    success = db.m_db->remove_scope( scope2_id);
    MI_CHECK( success);
    listener->check_and_clear( "removed", scope2_id);

    // test operation/destruction with the client giving up its reference of the listener
    listener.reset();

    DB::Scope* scope4 = db.m_global_scope->create_child( 1);
    DB::Scope_id scope4_id = scope4->get_id();
    success = db.m_db->remove_scope( scope4_id);
    MI_CHECK( success);
}

void test_memory_limits()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    {
        MI_CHECK_EQUAL( db.m_db->set_memory_limits( 1000, 2000), 0);
        size_t low_water  = 1;
        size_t high_water = 1;
        db.m_db->get_memory_limits( low_water, high_water);
        MI_CHECK_EQUAL( low_water,  1000);
        MI_CHECK_EQUAL( high_water, 2000);
    }
    {
        MI_CHECK_EQUAL( db.m_db->set_memory_limits( 1000, 0), 0);
        size_t low_water  = 1;
        size_t high_water = 1;
        db.m_db->get_memory_limits( low_water, high_water);
        MI_CHECK_EQUAL( low_water,  1000);
        MI_CHECK_EQUAL( high_water,    0);
    }
    {
        MI_CHECK_EQUAL( db.m_db->set_memory_limits( 0, 0), 0);
        size_t low_water  = 1;
        size_t high_water = 1;
        db.m_db->get_memory_limits( low_water, high_water);
        MI_CHECK_EQUAL( low_water,     0);
        MI_CHECK_EQUAL( high_water,    0);
    }
    {
        MI_CHECK_EQUAL( db.m_db->set_memory_limits( 2000, 1000), -1);
        size_t low_water  = 1;
        size_t high_water = 1;
        db.m_db->get_memory_limits( low_water, high_water);
        MI_CHECK_EQUAL( low_water,     0);
        MI_CHECK_EQUAL( high_water,    0);
    }
    {
        MI_CHECK_EQUAL( db.m_db->set_memory_limits( 2000, 2000), -1);
        size_t low_water  = 1;
        size_t high_water = 1;
        db.m_db->get_memory_limits( low_water, high_water);
        MI_CHECK_EQUAL( low_water,     0);
        MI_CHECK_EQUAL( high_water,    0);
    }
}

void test_journal_recording_store_with_tag()
{
    Test_db db( __func__, /*compare*/ true, /*enable_journal*/ true);

    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    DB::Transaction_ptr transaction = scope1->start_transaction();

    DB::Tag tag1 = transaction->reserve_tag();
    DB::Tag tag2 = transaction->reserve_tag();
    DB::Tag tag3 = transaction->reserve_tag();
    DB::Tag tag4 = transaction->reserve_tag();
    DB::Tag tag5 = transaction->reserve_tag();
    DB::Tag tag6 = transaction->reserve_tag();

    {
        // Create six elements with different journal flags in different scopes.
        // Note that My_element uses journal_not_c as journal mask.
        auto* element1 = new My_element( 41);
        transaction->store( tag1, element1, "foo1", 0, journal_a, 0);
        auto* element2 = new My_element( 42);
        transaction->store( tag2, element2, "foo2", 0, journal_b, 0);
        auto* element3 = new My_element( 43);
        transaction->store( tag3, element3, "foo3", 0, journal_c, 0);

        auto* element4 = new My_element( 44);
        transaction->store( tag4, element4, "foo4", 1, journal_a, 1);
        auto* element5 = new My_element( 45);
        transaction->store( tag5, element5, "foo5", 1, journal_b, 1);
        auto* element6 = new My_element( 46);
        transaction->store( tag6, element6, "foo6", 1, journal_c, 1);
    }
    db.dump();

    // Committing the transaction propagates the journal entries to the scopes.
    transaction->commit();
    db.dump();

    transaction = scope1->start_transaction();
    {
        // Modify the six elements, adding different journal flags.
        DB::Edit<My_element> edit1( tag1, transaction.get(), journal_b);
        edit1->set_value( 51);
        edit1.reset();
        DB::Edit<My_element> edit2( tag2, transaction.get(), journal_c);
        edit2->set_value( 52);
        edit2.reset();
        DB::Edit<My_element> edit3( tag3, transaction.get(), journal_a);
        edit3->set_value( 53);
        edit3.reset();

        DB::Edit<My_element> edit4( tag4, transaction.get(), journal_b);
        edit4->set_value( 54);
        edit4.reset();
        DB::Edit<My_element> edit5( tag5, transaction.get(), journal_c);
        edit5->set_value( 55);
        edit5.reset();
        DB::Edit<My_element> edit6( tag6, transaction.get(), journal_a);
        edit6->set_value( 56);
        edit6.reset();
    }
    db.dump();

    transaction->commit();
    db.dump();
}

void test_journal_recording_store_without_tag()
{
    Test_db db( __func__, /*compare*/ true, /*enable_journal*/ true);

    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    DB::Transaction_ptr transaction = scope1->start_transaction();

    {
        // Create six elements without explicit journal flags.
        // Default journal mask is JOURNAL_NONE.
        auto* element1 = new My_element( 41);
        transaction->store( element1, "foo1", 0, 0);
        auto* element2 = new My_element( 42);
        transaction->store( element2, "foo2", 0, 0);
        auto* element3 = new My_element( 43);
        transaction->store( element3, "foo3", 0, 0);

        auto* element4 = new My_element( 44);
        transaction->store( element4, "foo4", 1, 1);
        auto* element5 = new My_element( 45);
        transaction->store( element5, "foo5", 1, 1);
        auto* element6 = new My_element( 46);
        transaction->store( element6, "foo6", 1, 1);
    }
    db.dump();

    // Nothing to propagate
    transaction->commit();
    db.dump();

}

void test_journal_recording_edit()
{
    Test_db db( __func__, /*compare*/ true, /*enable_journal*/ true);

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    DB::Tag tag1 = transaction->reserve_tag();
    DB::Tag tag2 = transaction->reserve_tag();

    {
        // Create two elements.
        auto* element1 = new My_element( 41);
        transaction->store( tag1, element1, "foo1", 0, journal_a, 0);
        auto* element2 = new My_element( 42);
        transaction->store( tag2, element2, "foo2", 0, journal_a, 0);

        // Edit the first element in this transaction.
        DB::Edit<My_element> edit( tag1, transaction.get(), DB::JOURNAL_NONE);
        edit->set_value( 43);
        edit.add_journal_flags( journal_b);
    }
    db.dump();

    transaction->commit();
    db.dump();

    transaction = db.m_global_scope->start_transaction();
    {
        // Edit the second element in another transaction.
        DB::Edit<My_element> edit( tag2, transaction.get(), DB::JOURNAL_NONE);
        edit->set_value( 44);
        edit.add_journal_flags( journal_b);
    }
    db.dump();

    transaction->commit();
    db.dump();
}

void test_journal_recording_abort()
{
    Test_db db( __func__, /*compare*/ true, /*enable_journal*/ true);

    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    DB::Transaction_ptr transaction = scope1->start_transaction();

    DB::Tag tag1 = transaction->reserve_tag();
    DB::Tag tag2 = transaction->reserve_tag();
    DB::Tag tag3 = transaction->reserve_tag();
    DB::Tag tag4 = transaction->reserve_tag();
    DB::Tag tag5 = transaction->reserve_tag();
    DB::Tag tag6 = transaction->reserve_tag();

    {
        // Create six elements with different journal flags in different scopes.
        // Note that My_element uses journal_not_c as journal mask.
        auto* element1 = new My_element( 41);
        transaction->store( tag1, element1, "foo1", 0, journal_a, 0);
        auto* element2 = new My_element( 42);
        transaction->store( tag2, element2, "foo2", 0, journal_b, 0);
        auto* element3 = new My_element( 43);
        transaction->store( tag3, element3, "foo3", 0, journal_c, 0);

        auto* element4 = new My_element( 44);
        transaction->store( tag4, element4, "foo4", 1, journal_a, 1);
        auto* element5 = new My_element( 45);
        transaction->store( tag5, element5, "foo5", 1, journal_b, 1);
        auto* element6 = new My_element( 46);
        transaction->store( tag6, element6, "foo6", 1, journal_c, 1);
    }
    db.dump();

    // Aborting the transaction clears all journal entries.
    transaction->abort();
    db.dump();
}

// The order of results is implementation-defined. Sort them by tag and journal type.
auto Journal_query_result_cmp = [](
    const std::pair<DB::Tag, DB::Journal_type>& lhs,
    const std::pair<DB::Tag, DB::Journal_type>& rhs)
{
    if( lhs.first < rhs.first)
        return true;
    if( (lhs.first == rhs.first) && (lhs.second.get_type() < rhs.second.get_type()))
        return true;
    return false;
};

void test_journal_query()
{
    Test_db db( __func__, /*compare*/ true, /*enable_journal*/ true);

    DB::Scope* scope1 = db.m_global_scope->create_child( 1);
    DB::Transaction_ptr transaction = scope1->start_transaction();

    DB::Tag tag1 = transaction->reserve_tag();
    DB::Tag tag2 = transaction->reserve_tag();
    DB::Tag tag3 = transaction->reserve_tag();
    DB::Tag tag4 = transaction->reserve_tag();
    DB::Tag tag5 = transaction->reserve_tag();
    DB::Tag tag6 = transaction->reserve_tag();
    DB::Tag tag7 = transaction->reserve_tag();
    DB::Tag tag8 = transaction->reserve_tag();
    DB::Transaction_id last_changed_transaction_id = transaction->get_id();
    DB::Transaction_id current_transaction_id = transaction->get_id();
    mi::Uint32 last_changed_version;

    {
        // Create four elements with different journal flags.
        auto* element1 = new My_element( 41);
        transaction->store( tag1, element1, "foo1", 0, journal_a, 0);
        auto* element2 = new My_element( 42);
        transaction->store( tag2, element2, "foo2", 0, journal_b, 0);
        auto* element3 = new My_element( 43);
        transaction->store( tag3, element3, "foo3", 1, journal_a, 1);
        auto* element4 = new My_element( 44);
        transaction->store( tag4, element4, "foo4", 1, journal_b, 1);
        last_changed_version = transaction->get_next_sequence_number();
        auto* element5 = new My_element( 45);
        transaction->store( tag5, element5, "foo5", 0, journal_a, 0);
        auto* element6 = new My_element( 46);
        transaction->store( tag6, element6, "foo6", 0, journal_b, 0);
        auto* element7 = new My_element( 47);
        transaction->store( tag7, element7, "foo7", 1, journal_a, 1);
        auto* element8 = new My_element( 48);
        transaction->store( tag8, element8, "foo8", 1, journal_b, 1);
    }
    {
        // From transaction (excluding parents).
        auto result = transaction->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            journal_a,
            /*lookup_parents*/ false);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 1);
        DB::Journal_query_result expected { { tag7, journal_a } };
        MI_CHECK( *result.get() == expected);
    }
    {
        // From transaction (including parents).
        auto result = transaction->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            journal_a,
            /*lookup_parents*/ true);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 2);
        std::sort( result->begin(), result->end(), Journal_query_result_cmp);
        DB::Journal_query_result expected { { tag5, journal_a }, { tag7, journal_a } };
        MI_CHECK( *result.get() == expected);
    }
    {
        // From local scope (transaction not yet committed, excluding parents).
        auto result = scope1->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            current_transaction_id,
            journal_a,
            /*lookup_parents*/ false);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 0);
    }
    {
        // From local scope (transaction not yet committed, including parents).
        auto result = scope1->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            current_transaction_id,
            journal_a,
            /*lookup_parents*/ true);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 0);
    }
    {
        // From global scope (transaction not yet committed).
        auto result = db.m_global_scope->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            current_transaction_id,
            journal_a,
            /*lookup_parents*/ false);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 0);
    }
    {
        // From transaction (huge query range, i.e., pruned scope journal).
        auto result = transaction->get_journal(
            last_changed_transaction_id-1,
            0,
            journal_a,
            /*lookup_parents*/ false);
        MI_CHECK( !result);
    }

    // Commit the transaction and query again.
    transaction->commit();
    transaction = scope1->start_transaction();
    current_transaction_id = transaction->get_id();

    {
        // From transaction (excluding parents).
        auto result = transaction->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            journal_a,
            /*lookup_parents*/ false);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 1);
        DB::Journal_query_result expected { { tag7, journal_a } };
        MI_CHECK( *result.get() == expected);
    }
    {
        // From transaction (including parents).
        auto result = transaction->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            journal_a,
            /*lookup_parents*/ true);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 2);
        std::sort( result->begin(), result->end(), Journal_query_result_cmp);
        DB::Journal_query_result expected { { tag5, journal_a }, { tag7, journal_a } };
        MI_CHECK( *result.get() == expected);
    }
    {
        // From local scope (transaction committed, excluding parents).
        auto result = scope1->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            current_transaction_id,
            journal_a,
            /*lookup_parents*/ false);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 1);
        DB::Journal_query_result expected { { tag7, journal_a } };
        MI_CHECK( *result.get() == expected);
    }
    {
        // From local scope (transaction committed, including parents).
        auto result = scope1->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            current_transaction_id,
            journal_a,
            /*lookup_parents*/ true);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 2);
        std::sort( result->begin(), result->end(), Journal_query_result_cmp);
        DB::Journal_query_result expected { { tag5, journal_a }, { tag7, journal_a } };
        MI_CHECK( *result.get() == expected);
    }
    {
        // From global scope (transaction committed).
        auto result = db.m_global_scope->get_journal(
            last_changed_transaction_id,
            last_changed_version,
            current_transaction_id,
            journal_a,
            /*lookup_parents*/ false);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 1);
        DB::Journal_query_result expected { { tag5, journal_a } };
        MI_CHECK( *result.get() == expected);
    }
    {
        // From transaction (huge query range, i.e., pruned scope journal).
        auto result = transaction->get_journal(
            last_changed_transaction_id-1,
            0,
            journal_a,
            /*lookup_parents*/ false);
        MI_CHECK( !result);
    }

    transaction->commit();
}

void test_journal_query_parallel()
{
    Test_db db( __func__, /*compare*/ true, /*enable_journal*/ true);

    DB::Transaction_ptr transaction0 = db.m_global_scope->start_transaction();
    DB::Transaction_ptr transaction1 = db.m_global_scope->start_transaction();

    DB::Tag tag1 = transaction0->reserve_tag();
    DB::Tag tag2 = transaction1->reserve_tag();
    DB::Tag tag3 = transaction0->reserve_tag();
    DB::Tag tag4 = transaction1->reserve_tag();
    DB::Transaction_id last_changed_transaction_id;
    mi::Uint32 last_changed_version;

    {
        // Create four elements with different journal flags, two before/after (in time) getting
        // the last_changed_version.
        auto* element1 = new My_element( 41);
        transaction0->store( tag1, element1, "foo1", 0, journal_a, 0);
        auto* element2 = new My_element( 42);
        transaction1->store( tag2, element2, "foo2", 0, journal_a, 0);
        last_changed_transaction_id = transaction1->get_id();
        last_changed_version = transaction1->get_next_sequence_number();
        auto* element3 = new My_element( 43);
        transaction0->store( tag3, element3, "foo3", 0, journal_a, 0);
        auto* element4 = new My_element( 44);
        transaction1->store( tag4, element4, "foo4", 0, journal_a, 0);
    }
    db.dump();

    {
        // Query from transaction 1. The change for tag1 is not yet visible to this transaction.
        auto result = transaction1->get_journal(
            last_changed_transaction_id, last_changed_version, journal_a, /*lookup_parents*/ false);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 1);
        DB::Journal_query_result expected { { tag4, journal_a } };
        MI_CHECK( *result.get() == expected);
    }

    transaction0->commit();
    transaction1->commit();
    db.dump();
    DB::Transaction_ptr transaction2 = db.m_global_scope->start_transaction();

    {
        // Query from transaction 2. This includes *both* changes from transaction 0. The first one
        // happened "in time" before the query begin, but became only visible afterwards.
        auto result = transaction2->get_journal(
            last_changed_transaction_id, last_changed_version, journal_a, /*lookup_parents*/ false);
        MI_CHECK( result);
        MI_CHECK_EQUAL( result->size(), 3);
        std::sort( result->begin(), result->end(), Journal_query_result_cmp);
        DB::Journal_query_result expected {
            { tag1, journal_a }, { tag3, journal_a }, { tag4, journal_a } };
        MI_CHECK( *result.get() == expected);
    }

    transaction2->commit();
}

void test_journal_query_disabled()
{
    Test_db db( __func__, /*compare*/ false, /*enable_journal*/ false);

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    auto result = transaction->get_journal(
        transaction->get_id(), 0, DB::JOURNAL_ALL, /*lookup_parents*/ false);
    transaction->commit();
}

void test_journal_pruning()
{
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    config_module->override( "dblight_journal_max_size=2");

    Test_db db( __func__, /*compare*/ true, /*enable_journal*/ true);
    const auto* scope_impl = static_cast<DBLIGHT::Scope_impl*>( db.m_global_scope);

    // Creates n elements in the global scope.
    auto create_elements = [&db]( size_t n) {
        DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
        for( size_t i = 0; i < n; ++i) {
            DB::Tag tag = transaction->reserve_tag();
            auto* element = new My_element( 42);
            transaction->store( tag, element, nullptr, 0, journal_a, 0);
        }
        transaction->commit();
    };

    // Fill the scope journal with two elements.
    create_elements( 2);
    db.dump();
    MI_CHECK_EQUAL( scope_impl->get_journal().size(), 2);

    // Add another element, pruning the first one.
    create_elements( 1);
    db.dump();
    MI_CHECK_EQUAL( scope_impl->get_journal().size(), 2);

    // Add two more elements, pruning all others, scope journal still non-empty.
    create_elements( 2);
    db.dump();
    MI_CHECK_EQUAL( scope_impl->get_journal().size(), 2);

    // Add three more elements, exceeding max size, scope journal cleared.
    create_elements( 3);
    db.dump();
    MI_CHECK_EQUAL( scope_impl->get_journal().size(), 0);

    config_module->override( "dblight_journal_max_size=10000000");
}

void test_commit_abort_blocking()
{
    Test_db db( __func__, /*compare*/ false);
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    bool success = false;

    {
        // block/unblock with open transaction
        bool success = transaction->block_commit_or_abort();
        MI_CHECK( success);
        success = transaction->block_commit_or_abort();
        MI_CHECK( success);
        success = transaction->unblock_commit_or_abort();
        MI_CHECK( success);
        success = transaction->unblock_commit_or_abort();
        MI_CHECK( success);
#ifdef NDEBUG
        // Triggers assertion in front of the check.
        success = transaction->unblock_commit_or_abort();
        MI_CHECK( !success);
#endif
    }
    {
        // commit while unblocked
        transaction->commit();
    }
    {
        // block/unblock with closed transaction
        success = transaction->block_commit_or_abort();
        MI_CHECK( !success);
        success = transaction->unblock_commit_or_abort();
        MI_CHECK( !success);
    }
    {
        // abort while unblocked
        transaction = db.m_global_scope->start_transaction();
        transaction->abort();
    }
    {
        // block/unblock with aborted transaction
        success = transaction->block_commit_or_abort();
        MI_CHECK( !success);
        success = transaction->unblock_commit_or_abort();
        MI_CHECK( !success);
    }
    {
        transaction = db.m_global_scope->start_transaction();

        // block twice
        success = transaction->block_commit_or_abort();
        MI_CHECK( success);
        success = transaction->block_commit_or_abort();
        MI_CHECK( success);

        // second thread that unblocks twice when CLOSING state has been reached
        using namespace std::chrono_literals;
        std::thread thread([&transaction]() {
            // sleep until commit() actually started, i.e., the CLOSING state has been reached
            while( transaction->is_open( /*closing_is_open*/ false))
                std::this_thread::sleep_for( 1ms);
            // try to block again in CLOSING state (fails)
            bool success2 = transaction->block_commit_or_abort();
            MI_CHECK( !success2);
            // unblock twice
            success2 = transaction->unblock_commit_or_abort();
            MI_CHECK( success2);
            success2 = transaction->unblock_commit_or_abort();
            MI_CHECK( success2);
        });

        // commit while blocked
        transaction->commit();
        thread.join();
    }
    {
        transaction = db.m_global_scope->start_transaction();

        // block twice
        success = transaction->block_commit_or_abort();
        MI_CHECK( success);
        success = transaction->block_commit_or_abort();
        MI_CHECK( success);

        // second thread that unblocks twice when CLOSING state has been reached
        using namespace std::chrono_literals;
        std::thread thread([&transaction]() {
            // sleep until commit() actually started, i.e., the CLOSING state has been reached
            while( transaction->is_open( /*closing_is_open*/ false))
                std::this_thread::sleep_for( 1ms);
            // try to block again in CLOSING state (fails)
            bool success2 = transaction->block_commit_or_abort();
            MI_CHECK( !success2);
            // unblock twice
            success2 = transaction->unblock_commit_or_abort();
            MI_CHECK( success2);
            success2 = transaction->unblock_commit_or_abort();
            MI_CHECK( success2);
        });

        // abort while blocked
        transaction->abort();
        thread.join();
    }
}

void test_dump_with_pointers()
{
    Test_db db( __func__, /*compare*/ false); // Pointers are non-deterministic
    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    auto* element = new My_element( 42);
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

void test_wrong_privacy_levels()
{
    // Note that the expected error messages are not part of the dump file.

    Test_db db( __func__);

    DB::Scope* scope1 = db.m_global_scope->create_child( 10);

    DB::Transaction_ptr transaction = scope1->start_transaction();
    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo1", 10, 10);
    db.dump();

    // Storing element2 in the global scope referencing element1 in scope1 triggers an error
    // message if the privacy level check is enabled.
    auto* element2 = new My_element( 43, {tag1});
    DB::Tag tag2 = transaction->store( element2, "foo2", 0,  0);
    db.dump();

    // Storing element3 in the global scope referencing element1 in scope1 triggers an error
    // message if the privacy level check is enabled.
    auto* element3 = new My_element( 43, {tag1});
    DB::Tag tag3 = transaction->store( element3, nullptr, 0,  0);
    db.dump();

    {
        // Editing element2 in the global scope referencing element1 in scope1 triggers an error
        // message if the privacy level check is enabled.
        DB::Edit<My_element> edit( tag2, transaction.get());
        edit->set_value( 44);
        db.dump();
    }

    {
        // Editing element3 in the global scope referencing element1 in scope1 triggers an error
        // message if the privacy level check is enabled.
        DB::Edit<My_element> edit( tag3, transaction.get());
        edit->set_value( 44);
        db.dump();
    }

    transaction->commit();
}

void test_reference_cycles1()
{
#ifdef NDEBUG
    // Note that the expected error messages are not part of the dump file.

    Test_db db( __func__);

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo1");
    db.dump();

    // Create a reference cycle within tag1. Triggers an error message if the
    // reference cycle check is enabled.
    auto* element2 = new My_element( 43, {tag1});
    transaction->store( tag1, element2, "foo1");
    db.dump();

    // Remove reference cycle again and commit transaction to get rid of the problematic version.
    auto* element3 = new My_element( 44);
    transaction->store( tag1, element3, "foo1");
    transaction->commit();
    db.dump();

    // Reference cycles via edits only tested with 2 elements.
#else
    // Self-references trigger an assertion in debug mode.
#endif
}

void test_reference_cycles2()
{
    // Note that the expected error messages are not part of the dump file.

    Test_db db( __func__);

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo1");
    auto* element2 = new My_element( 43, {tag1});
    DB::Tag tag2 = transaction->store( element2, "foo2");
    db.dump();

    // Create a reference cycle between tag1 and tag2. Triggers an error message if the
    // reference cycle check is enabled.
    auto* element3 = new My_element( 44, {tag2});
    transaction->store( tag1, element3, "foo1");
    db.dump();

    // Remove reference cycle again and commit transaction to get rid of the problematic version.
    auto* element4 = new My_element( 45);
    transaction->store( tag1, element4, "foo1");
    transaction->commit();
    db.dump();

    transaction = db.m_global_scope->start_transaction();

    {
        // Create a reference cycle between tag1 and tag2. Triggers an error message if the
        // reference cycle check is enabled.
        DB::Edit<My_element> edit( tag1, transaction.get());
        edit->set_tag_set( {tag2});
    }
    db.dump();

    {
        // Remove reference cycle again.
        DB::Edit<My_element> edit( tag1, transaction.get());
        edit->set_tag_set( {});
    }

    // Commit transaction to get rid of the problematic version.
    transaction->commit();
    db.dump();
}

void test_reference_cycles3()
{
    // Note that the expected error messages are not part of the dump file.

    Test_db db( __func__);

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();
    auto* element1 = new My_element( 42);
    DB::Tag tag1 = transaction->store( element1, "foo1");
    auto* element2 = new My_element( 43, {tag1});
    DB::Tag tag2 = transaction->store( element2, "foo2");
    auto* element3 = new My_element( 44, {tag2});
    DB::Tag tag3 = transaction->store( element3, "foo3");
    db.dump();

    // Create a reference cycle between tag1, tag2, and tag3. Triggers an error message if the
    // reference cycle check is enabled.
    auto* element4 = new My_element( 45, {tag3});
    transaction->store( tag1, element4, "foo1");
    db.dump();

    // Remove reference cycle again and commit transaction to get rid of the problematic version.
    auto* element5 = new My_element( 46);
    transaction->store( tag1, element5, "foo1");
    transaction->commit();
    db.dump();

    // Reference cycles via edits only tested with 2 elements.
}

void test_not_implemented_with_assertions()
{
    Test_db db( __func__, /*compare*/ false); // Empty dump

    db.m_db->lock( 42);
    bool result = db.m_db->unlock( 42);
    MI_CHECK( !result);
    db.m_db->check_is_locked( 42);

    DB::Transaction_ptr transaction = db.m_global_scope->start_transaction();

    // artificial test arguments
    DB::Tag tag_job = transaction->store( static_cast<SCHED::Job_base*>( nullptr));
    MI_CHECK( !tag_job);
    // artificial test arguments
    transaction->store( DB::Tag(), static_cast<SCHED::Job_base*>( nullptr));
    // artificial test arguments
    tag_job = transaction->store_for_reference_counting( static_cast<SCHED::Job_base*>( nullptr));
    MI_CHECK( !tag_job);
    // artificial test arguments
    transaction->store_for_reference_counting( DB::Tag(), static_cast<SCHED::Job_base*>( nullptr));
    transaction->invalidate_job_results( tag_job);
    transaction->advise( tag_job);

    transaction->cancel_fragmented_jobs();

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
    test_transaction_name_to_tag_modified();
    test_transaction_tag_to_name_and_back();
    test_transaction_name_to_tag_and_back();

    test_transaction_get_class_id();
    test_transaction_get_tag_reference_count();
    test_transaction_get_tag_version();

    test_transaction_remove();
    test_transaction_get_tag_is_removed();

    test_transaction_access_after_remove();
    test_transaction_access_after_abort();

    test_gc_simple_reference();
    test_gc_simple_reference_abort();
    test_gc_multiple_references();
    test_gc_explicit_call();
    test_gc_pin_count_zero();

    test_use_of_closed_transaction();

    test_scope_creation();
    test_scope_lookup();
    test_scope_removal();
    test_transaction_store_with_scopes();
    test_transaction_access_with_scopes();
    test_transaction_edit_with_scopes();
    test_transaction_get_store_and_privacy_level();
    test_transaction_can_reference();
    test_transaction_localize();
    test_transaction_remove_with_scopes();
    test_scope_removal_possibly_delayed();
    test_scope_delayed_gc();

    test_fragmented_jobs_without_transaction();
    test_fragmented_jobs_without_transaction_async();
    test_fragmented_jobs_with_transaction();
    test_fragmented_jobs_with_transaction_async();
    test_fragmented_jobs_cancelling();

    test_transaction_listeners();
    test_scope_listeners();
    test_memory_limits();

    test_journal_recording_store_with_tag();
    test_journal_recording_store_without_tag();
    test_journal_recording_edit();
    test_journal_recording_abort();
    test_journal_query();
    test_journal_query_parallel();
    test_journal_query_disabled();
    test_journal_pruning();

    test_commit_abort_blocking();

    test_dump_with_pointers();
    test_wrong_privacy_levels();
    test_reference_cycles1();
    test_reference_cycles2();
    test_reference_cycles3();
#ifdef NDEBUG
    test_not_implemented_with_assertions();
#endif // NDEBUG
}

MI_TEST_AUTO_FUNCTION( test_dblight )
{
    SYSTEM::Access_module<LOG::Log_module> log_module( false);
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);

    // For the debug messages emitted by test_fragmented_jobs_cancelling() and
    // test_commit_abort_blocking().
    log_module->set_severity_limit( LOG::ILogger::S_ALL);
    log_module->set_severity_by_category( LOG::ILogger::C_DATABASE, LOG::ILogger::S_ALL);

    config_module->override( "check_serializer_store=1");
    config_module->override( "check_serializer_edit=1");
    config_module->override( "check_privacy_levels=1");
    config_module->override( "check_reference_cycles_store=1");
    config_module->override( "check_reference_cycles_edit=1");

    test_info_base();

    // Note that the reference files are independent of the GC method. Each run below overwrites
    // the output files of the previous run.
    test( /*use the default*/ nullptr);
    test( "full_sweeps_only");
    test( "full_sweep_then_pin_count_zero");
    test( "general_candidates_then_pin_count_zero");
    test( "invalid_method");
}

MI_TEST_MAIN_CALLING_TEST_MAIN();

