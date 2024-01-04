/**************************************************************************************************
 * Copyright (c) 2013-2023, NVIDIA CORPORATION. All rights reserved.
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
 *************************************************************************************************/

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for base/data/thread_pool"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <atomic>

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/interface_implement.h>
#include <base/system/main/access_module.h>
#include <base/hal/thread/i_thread_condition.h>
#include <base/hal/time/i_time.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>

#include "i_thread_pool_ijob.h"
#include "i_thread_pool_thread_pool.h"

using namespace MI;
using namespace THREAD_POOL;

/// A simple test job with configurable loads and delay.
class Test_job : public mi::base::Interface_implement<IJob>
{
public:
    Test_job(
        Thread_pool* thread_pool,
        mi::Size id,
        mi::Float32 cpu_load,
        mi::Float32 gpu_load,
        mi::Float32 delay)
      : m_thread_pool( thread_pool),
        m_id( id),
        m_cpu_load( cpu_load),
        m_gpu_load( gpu_load),
        m_delay( delay)
    {
    }

    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return 0; }
    bool is_remaining_work_splittable() { return false; }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context) { }

    void execute( const mi::neuraylib::IJob_execution_context* context)
    {
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Started test job %llu (CPU load %.1f, GPU load %.1f)",
            m_id, m_cpu_load, m_gpu_load);
        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();

        TIME::sleep( m_delay);

        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Finished test job %llu (CPU load %.1f, GPU load %.1f)",
            m_id, m_cpu_load, m_gpu_load);
    }

private:
    Thread_pool* m_thread_pool;
    mi::Size m_id;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    mi::Float32 m_delay;
};

/// A simple test job that blocks its execution until continue_job() is called.
class Block_job : public mi::base::Interface_implement<IJob>
{
public:
    Block_job(
        Thread_pool* thread_pool,
        mi::Uint32 id,
        mi::Float32 cpu_load,
        mi::Float32 gpu_load)
      : m_thread_pool( thread_pool),
        m_id( id),
        m_cpu_load( cpu_load),
        m_gpu_load( gpu_load)
    {
    }

    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return 0; }
    bool is_remaining_work_splittable() { return false; }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context) { }

    void execute( const mi::neuraylib::IJob_execution_context* context)
    {
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Started block job %d (CPU load %.1f, GPU load %.1f)",
            m_id, m_cpu_load, m_gpu_load);
        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();

        m_condition.wait();

        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Finished block job %d (CPU load %.1f, GPU load %.1f)",
            m_id, m_cpu_load, m_gpu_load);
    }

    void continue_job() { m_condition.signal(); }

private:
    Thread_pool* m_thread_pool;
    mi::Uint32 m_id;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    mi::base::Condition m_condition;
};

/// A job that recursively submits one child jobs up to a certain nesting level.
class Recursive_job : public mi::base::Interface_implement<IJob>
{
public:
    Recursive_job(
        Thread_pool* thread_pool,
        mi::Uint32 level,
        mi::Uint32 max_levels,
        mi::Float32 cpu_load,
        mi::Float32 gpu_load,
        mi::Float32 delay)
      : m_thread_pool( thread_pool),
        m_level( level),
        m_max_levels( max_levels),
        m_cpu_load( cpu_load),
        m_gpu_load( gpu_load),
        m_delay( delay)
    {
        for( mi::Uint32 i = 0; i < level+1; ++i)
            m_prefix += "    ";
    }

    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return 0; }
    bool is_remaining_work_splittable() { return false; }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context) { }

    void execute( const mi::neuraylib::IJob_execution_context* context)
    {
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "%sStarted recursive job level %d (CPU load %.1f, GPU load %.1f)",
            m_prefix.c_str(), m_level, m_cpu_load, m_gpu_load);
        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();

        TIME::sleep( m_delay/2);
        if( m_level+1 < m_max_levels) {

#ifndef ENABLE_ASSERT
            // not needed, just to test suspension of suspended worker threads (but triggers
            // assertion)
            if( m_level % 2 == 0) m_thread_pool->suspend_current_job();
#endif // ENABLE_ASSERT

            mi::base::Handle<Recursive_job> child_job( new Recursive_job(
                m_thread_pool, m_level+1, m_max_levels, m_cpu_load, m_gpu_load, m_delay));
            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "%sSubmitting recursive job level %d (CPU load %.1f, GPU load %.1f), waiting for "
                "completion", m_prefix.c_str(), m_level+1, m_cpu_load, m_gpu_load);
            m_thread_pool->submit_job_and_wait( child_job.get());

#ifndef ENABLE_ASSERT
            // not needed, just to test resuming of resumed worker threads (might trigger
            // assertion)
            if( m_level % 2 == 0) m_thread_pool->resume_current_job();
#endif // ENABLE_ASSERT

        }
        TIME::sleep( m_delay/2);

        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "%sFinished recursive job level %d (CPU load %.1f, GPU load %.1f)",
            m_prefix.c_str(), m_level, m_cpu_load, m_gpu_load);
    }

private:
    Thread_pool* m_thread_pool;
    mi::Uint32 m_level;
    mi::Uint32 m_max_levels;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    mi::Float32 m_delay;
    std::string m_prefix;
};

/// A job that recursively submits two child jobs up to a certain nesting level.
class Tree_job : public mi::base::Interface_implement<IJob>
{
public:
    Tree_job(
        Thread_pool* thread_pool,
        mi::Uint32 level,
        mi::Uint32 max_levels,
        mi::Float32 cpu_load,
        mi::Float32 gpu_load,
        mi::Float32 delay,
        mi::base::Condition* condition = 0)
      : m_thread_pool( thread_pool),
        m_level( level),
        m_max_levels( max_levels),
        m_cpu_load( cpu_load),
        m_gpu_load( gpu_load),
        m_delay( delay),
        m_condition( condition)
    {
        for( mi::Uint32 i = 0; i < level+1; ++i)
            m_prefix += "    ";
    }

    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return 0; }
    bool is_remaining_work_splittable() { return false; }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context) { }

    void execute( const mi::neuraylib::IJob_execution_context* context)
    {
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "%sStarted tree job level %d (CPU load %.1f, GPU load %.1f)",
            m_prefix.c_str(), m_level, m_cpu_load, m_gpu_load);
        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();

        TIME::sleep( m_delay/2);
        if( m_level+1 < m_max_levels) {

            mi::base::Condition m_condition0;
            mi::base::Condition m_condition1;
            mi::base::Handle<Tree_job> job0( new Tree_job(
                m_thread_pool, m_level+1, m_max_levels, m_cpu_load, m_gpu_load, m_delay,
                &m_condition0));
            mi::base::Handle<Tree_job> job1( new Tree_job(
                m_thread_pool, m_level+1, m_max_levels, m_cpu_load, m_gpu_load, m_delay,
                &m_condition1));

            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "%sSubmitting tree job level %d (CPU load %.1f, GPU load %.1f)",
                m_prefix.c_str(), m_level+1, m_cpu_load, m_gpu_load);
            m_thread_pool->submit_job( job0.get());
            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "%sSubmitting tree job level %d (CPU load %.1f, GPU load %.1f)",
                m_prefix.c_str(), m_level+1, m_cpu_load, m_gpu_load);
            m_thread_pool->submit_job( job1.get());

            m_thread_pool->suspend_current_job();
            m_condition0.wait();
            m_condition1.wait();
            m_thread_pool->resume_current_job();
        }
        TIME::sleep( m_delay/2);

        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "%sFinished tree job level %d (CPU load %.1f, GPU load %.1f)",
            m_prefix.c_str(), m_level, m_cpu_load, m_gpu_load);

        if( m_condition)
            m_condition->signal();
    }

private:
    Thread_pool* m_thread_pool;
    mi::Uint32 m_level;
    mi::Uint32 m_max_levels;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    mi::Float32 m_delay;
    mi::base::Condition* m_condition;
    std::string m_prefix;
};

/// A test for fragmented jobs not based on the mixin THREAD_POOL::Fragmented_job, but directly
/// implementing similar functionality.
class Fragmented_job_without_mixin : public mi::base::Interface_implement<IJob>
{
public:
    Fragmented_job_without_mixin(
        Thread_pool* thread_pool,
        mi::Float32 cpu_load,
        mi::Float32 gpu_load,
        mi::Float32 delay,
        mi::Uint32 count,
        mi::Uint32 thread_limit)
      : m_thread_pool( thread_pool),
        m_cpu_load( cpu_load),
        m_gpu_load( gpu_load),
        m_delay( delay),
        m_count( count),
        m_thread_limit( thread_limit),
        m_next_fragment( 0),
        m_threads( 0)
    {
    }

    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return 0; }
    size_t get_thread_limit() const { return m_thread_limit; }
    bool is_remaining_work_splittable()
    {
        size_t thread_limit = get_thread_limit();
        return (m_next_fragment+1 < m_count) && (thread_limit == 0 || m_threads < thread_limit);
    }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context)
    { ++m_threads; }

    void execute( const mi::neuraylib::IJob_execution_context* context)
    {
        mi::Uint32 index = m_next_fragment++;
        while( index < m_count) {
            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "    Started fragment %d (CPU load %.1f, GPU load %.1f)",
                index, m_cpu_load, m_gpu_load);
            m_thread_pool->dump_load();
            m_thread_pool->dump_thread_state_counters();
            TIME::sleep( m_delay);
            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "    Finished fragment %d (CPU load %.1f, GPU load %.1f)",
                index, m_cpu_load, m_gpu_load);
            index = m_next_fragment++;
        }
    }

private:
    Thread_pool* m_thread_pool;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    mi::Float32 m_delay;
    mi::Uint32 m_count;
    mi::Uint32 m_thread_limit;
    std::atomic_uint32_t m_next_fragment;
    std::atomic_uint32_t m_threads;
};

/// A test for fragmented jobs based on the mixin THREAD_POOL::Fragmented_job.
class Fragmented_job_using_mixin : public Fragmented_job
{
public:
    Fragmented_job_using_mixin(
        Thread_pool* thread_pool,
        mi::Float32 cpu_load,
        mi::Float32 gpu_load,
        mi::Float32 delay,
        mi::Uint32 count,
        mi::Uint32 thread_limit)
      : Fragmented_job( 0, count),
        m_thread_pool( thread_pool),
        m_cpu_load( cpu_load),
        m_gpu_load( gpu_load),
        m_thread_limit( thread_limit),
        m_delay( delay),
        m_next_fragment( 0)
    {
    }

    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return 0; }
    size_t get_thread_limit() const { return m_thread_limit; }

    void execute_fragment(
        DB::Transaction* transaction,
        size_t index,
        size_t count,
        const mi::neuraylib::IJob_execution_context* context)
    {
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Started fragment %" FMT_SIZE_T " (CPU load %.1f, GPU load %.1f)",
            index, m_cpu_load, m_gpu_load);
        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();
        TIME::sleep( m_delay);
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Finished fragment %" FMT_SIZE_T " (CPU load %.1f, GPU load %.1f)",
            index, m_cpu_load, m_gpu_load);
    }

private:
    Thread_pool* m_thread_pool;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    size_t m_thread_limit;
    mi::Float32 m_delay;
    std::atomic_uint32_t m_next_fragment;
};

/// A simple test job with configurable priority. Compares its priority during execution against
/// s_expected_priority. Increments s_expected_priority at the end of its execution.
class Priority_job
: public mi::base::Interface_implement<IJob> { public:
    Priority_job(
        Thread_pool* thread_pool,
        mi::Size id,
        mi::Float32 cpu_load,
        mi::Float32 gpu_load,
        mi::Float32 delay,
        mi::Sint8 priority)
      : m_thread_pool( thread_pool),
        m_id( id),
        m_cpu_load( cpu_load),
        m_gpu_load( gpu_load),
        m_delay( delay),
        m_priority( priority)
    {
    }

    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return m_priority; }
    bool is_remaining_work_splittable() { return false; }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context) { }

    void execute( const mi::neuraylib::IJob_execution_context* context)
    {
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Started priority job %llu (CPU load %.1f, GPU load %.1f, priority %d)",
            m_id, m_cpu_load, m_gpu_load, (int) m_priority);
        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();

        ASSERT( M_THREAD_POOL, m_priority == s_expected_priority || m_priority == 127);
        TIME::sleep( m_delay/2);
        ASSERT( M_THREAD_POOL, m_priority == s_expected_priority || m_priority == 127);
        m_thread_pool->yield();
        ASSERT( M_THREAD_POOL, m_priority == s_expected_priority || m_priority == 127);
        TIME::sleep( m_delay/2);
        ASSERT( M_THREAD_POOL, m_priority == s_expected_priority || m_priority == 127);

        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Finished priority job %llu (CPU load %.1f, GPU load %.1f, priority %d)",
            m_id, m_cpu_load, m_gpu_load, (int) m_priority);

        ++s_expected_priority;
    }

private:
    Thread_pool* m_thread_pool;
    mi::Size m_id;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    mi::Float32 m_delay;
    mi::Sint8 m_priority;

public:
    static mi::Sint8 s_expected_priority;
};

mi::Sint8 Priority_job::s_expected_priority = 0;

/// A test job for yield() that can act as parent or child job. Parent jobs launch the child jobs
/// with higher/same/lower priority and check some flag after calling yield(). Child jobs just set
/// the flag during execution.
class Yield_job : public mi::base::Interface_implement<IJob>
{
public:
    Yield_job(
        Thread_pool* thread_pool,
        mi::Uint32 id,
        mi::Float32 cpu_load,
        mi::Float32 gpu_load,
        mi::Float32 delay,
        mi::Sint8 priority,
        bool parent)
      : m_thread_pool( thread_pool),
        m_id( id),
        m_cpu_load( cpu_load),
        m_gpu_load( gpu_load),
        m_delay( delay),
        m_priority( priority),
        m_parent( parent)
    {
    }

    mi::Float32 get_cpu_load() const { return m_cpu_load; }
    mi::Float32 get_gpu_load() const { return m_gpu_load; }
    mi::Sint8 get_priority() const { return m_priority; }
    bool is_remaining_work_splittable() { return false; }
    void pre_execute( const mi::neuraylib::IJob_execution_context* context) { }

    void execute( const mi::neuraylib::IJob_execution_context* context)
    {
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Started yield %s job %d (CPU load %.1f, GPU load %.1f, priority %d)",
            m_parent ? "parent" : "child", m_id, m_cpu_load, m_gpu_load, (int) m_priority);
        m_thread_pool->dump_load();
        m_thread_pool->dump_thread_state_counters();

        if( !m_parent) {
            TIME::sleep( m_delay);
            s_completed_yield_child_job = true;
        } else {
            mi::base::Handle<Yield_job> job;

            // check that yielding for higher priority job works
            s_completed_yield_child_job = false;
            job = new Yield_job(
                m_thread_pool, m_id+1, m_cpu_load, m_gpu_load, m_delay, m_priority-1, false);
            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "Submitting yield child job %u (CPU load %.1f, GPU load %.1f, priority %d)",
                m_id+1, m_cpu_load, m_gpu_load, m_priority-1);
            m_thread_pool->submit_job( job.get());
            m_thread_pool->yield();
            ASSERT( M_THREAD_POOL, s_completed_yield_child_job);

            // check that yielding for same priority job has no effect
            s_completed_yield_child_job = false;
            job = new Yield_job(
                m_thread_pool, m_id+2, m_cpu_load, m_gpu_load, m_delay, m_priority  , false);
            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "Submitting yield child job %u (CPU load %.1f, GPU load %.1f, priority %d)",
                m_id+2, m_cpu_load, m_gpu_load, m_priority  );
            m_thread_pool->submit_job( job.get());
            m_thread_pool->yield();
            ASSERT( M_THREAD_POOL, !s_completed_yield_child_job);

            // check that yielding for lower priority job has no effect
            s_completed_yield_child_job = false;
            job = new Yield_job(
                m_thread_pool, m_id+3, m_cpu_load, m_gpu_load, m_delay, m_priority+1, false);
            LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
                "Submitting yield child job %u (CPU load %.1f, GPU load %.1f, priority %d)",
                m_id+3, m_cpu_load, m_gpu_load, m_priority+1);
            m_thread_pool->submit_job( job.get());
            m_thread_pool->yield();
            ASSERT( M_THREAD_POOL, !s_completed_yield_child_job);
        }

        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "    Finished yield %s job %d (CPU load %.1f, GPU load %.1f, priority %d)",
            m_parent ? "parent" : "child", m_id, m_cpu_load, m_gpu_load, (int) m_priority);
    }

private:
    Thread_pool* m_thread_pool;
    mi::Uint32 m_id;
    mi::Float32 m_cpu_load;
    mi::Float32 m_gpu_load;
    mi::Float32 m_delay;
    mi::Sint8 m_priority;
    bool m_parent;
    static bool s_completed_yield_child_job;
};

bool Yield_job::s_completed_yield_child_job = false;

/// Submits a high number of simple jobs with different loads and delays.
void test_many_jobs()
{
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Testing many jobs ...\n ");

    Thread_pool thread_pool( 5.0, 5.0, 1);
    thread_pool.dump_thread_state_counters();
    thread_pool.dump_load();

    for( mi::Size i = 0; i < 100; ++i) {
        // Note: actual loads might be higher due to clipping against the global minimum.
        mi::Float32 cpu_load = ((i%10)+1)  * 0.1f;
        mi::Float32 gpu_load = (10-(i%10)) * 0.1f;
        mi::Float32 delay    = ((i%10)+1)  * 0.01f;
        mi::base::Handle<Test_job> job( new Test_job( &thread_pool, i, cpu_load, gpu_load, delay));
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "Submitting test job %llu (CPU load %.1f, GPU load %.1f)", i, cpu_load, gpu_load);
        thread_pool.submit_job( job.get());
    }

    // Wait for low priority job before shutting down.
    mi::Float32 cpu_load = 5.0f;
    mi::Float32 gpu_load = 5.0f;
    mi::Float32 delay    = 0.01f;
    mi::base::Handle<Priority_job> job( new Priority_job(
        &thread_pool, 100, cpu_load, gpu_load, delay, 127));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting priority job 100 (CPU load %.1f, GPU load %.1f, priority 127)",
        cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job.get());

    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, " ");
}

/// Submits a job that recursively submits child jobs.
void test_child_jobs()
{
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Testing child jobs ...\n ");

    mi::Float32 cpu_load = 1.0f;
    mi::Float32 gpu_load = 1.0f;
    mi::Float32 delay    = 0.1f;
    mi::Uint32  levels   = 8;

    Thread_pool thread_pool( cpu_load, gpu_load, 1);
    thread_pool.dump_thread_state_counters();
    thread_pool.dump_load();

    mi::base::Handle<Recursive_job> job0(
        new Recursive_job( &thread_pool, 0, levels, cpu_load, gpu_load, delay));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting recursive job level %d (CPU load %.1f, GPU load %.1f), waiting for completion",
        0, cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job0.get());

    cpu_load = 0.1f;
    gpu_load = 0.1f;
    mi::base::Handle<Tree_job> job1(
        new Tree_job( &thread_pool, 0, levels, cpu_load, gpu_load, delay));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting tree job level %d (CPU load %.1f, GPU load %.1f), waiting for completion",
        0, cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job1.get());

    thread_pool.dump_thread_state_counters();
    thread_pool.dump_load();

    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, " ");
}

/// Submits a job that can be split into several fragments.
void test_fragmented_jobs()
{
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Testing fragmented jobs ...\n ");

    mi::Float32 cpu_load = 1.0f;
    mi::Float32 gpu_load = 1.0f;
    mi::Float32 delay    = 0.1f;
    mi::Uint32  count    = 16;

    Thread_pool thread_pool( 5.0, 5.0, 1);
    thread_pool.dump_thread_state_counters();
    thread_pool.dump_load();

    mi::base::Handle<Fragmented_job_without_mixin> job0(
        new Fragmented_job_without_mixin( &thread_pool, cpu_load, gpu_load, delay, count, 0));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting fragmented job w/o mixin with %d fragments (CPU load %.1f, GPU load %.1f, "
        "no thread limit), waiting for completion", count, cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job0.get());

    mi::base::Handle<Fragmented_job_using_mixin> job1(
        new Fragmented_job_using_mixin( &thread_pool, cpu_load, gpu_load, delay, count, 0));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting fragmented job using mixin with %d fragments (CPU load %.1f, GPU load %.1f, "
        "no thread limit), waiting for completion", count, cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job1.get());

    mi::base::Handle<Fragmented_job_without_mixin> job2(
        new Fragmented_job_without_mixin( &thread_pool, cpu_load, gpu_load, delay, count, 1));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting fragmented job w/o mixin with %d fragments (CPU load %.1f, GPU load %.1f, "
        "thread limit 1), waiting for completion", count, cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job2.get());

    mi::base::Handle<Fragmented_job_using_mixin> job3(
        new Fragmented_job_using_mixin( &thread_pool, cpu_load, gpu_load, delay, count, 1));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting fragmented job using mixin with %d fragments (CPU load %.1f, GPU load %.1f, "
        "thread limit 1), waiting for completion", count, cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job3.get());

    thread_pool.dump_thread_state_counters();
    thread_pool.dump_load();

    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, " ");
}

/// Submits a second job that never fits the limits. It will be executed eventually when the load
/// drops to 0 after the first job has been finished.
void test_expensive_jobs()
{
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Testing expensive jobs ...\n ");

    mi::Float32 cpu_load = 1.0f;
    mi::Float32 gpu_load = 1.0f;
    mi::Float32 delay    = 0.1f;

    Thread_pool thread_pool( cpu_load, gpu_load, 1);
    thread_pool.dump_thread_state_counters();
    thread_pool.dump_load();

    mi::Uint32 id = 0;
    mi::base::Handle<Test_job> job0( new Test_job( &thread_pool, id, cpu_load, gpu_load, delay));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting expensive job %d (CPU load %.1f, GPU load %.1f)", id, cpu_load, gpu_load);
    thread_pool.submit_job( job0.get());

    // job1 will be executed even though it exceeds the limits but not before the current load is
    // 0.0, i.e., job0 is finished
    cpu_load *= 10;
    gpu_load *= 20;
    id = 1;
    mi::base::Handle<Test_job> job1( new Test_job( &thread_pool, id, cpu_load, gpu_load, delay));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting expensive job %d (CPU load %.1f, GPU load %.1f)", id, cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job1.get());

    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, " ");
}

/// Submits jobs with various priorities. Checks that they are executed in reverse order.
void test_priorities()
{
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Testing priorities ...\n ");

    mi::Float32 cpu_load = 1.0f;
    mi::Float32 gpu_load = 1.0f;
    mi::Float32 delay    = 0.1f;

    Thread_pool thread_pool( cpu_load, gpu_load, 1);
    thread_pool.dump_thread_state_counters();
    thread_pool.dump_load();

    // Keep thread pool busy to be able to submit the priority jobs without executing them during
    // submission. Otherwise a low priority job might get executed out of order.
    mi::base::Handle<Block_job> job0( new Block_job( &thread_pool, 0, cpu_load, gpu_load));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting block job %u (CPU load %.1f, GPU load %.1f)", 0, cpu_load, gpu_load);
    thread_pool.submit_job( job0.get());

    Priority_job::s_expected_priority = 1;

    // Submit priority jobs with priorities from 10 (lowest) to 1 (highest), executed in reverse
    // order.
    for( mi::Size i = 1; i < 11; ++i) {
        mi::Sint8 priority = static_cast<mi::Sint8>( 11-i);
        mi::base::Handle<Priority_job> job( new Priority_job(
            &thread_pool, i, cpu_load, gpu_load, delay, priority));
        LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
            "Submitting priority job %llu (CPU load %.1f, GPU load %.1f, priority %d)",
            i, cpu_load, gpu_load, (int) priority);
        thread_pool.submit_job( job.get());
    }

    job0->continue_job();

    // Wait for low priority job before shutting down.
    mi::base::Handle<Priority_job> job1( new Priority_job(
        &thread_pool, 11, cpu_load, gpu_load, delay, 127));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting priority job 11 (CPU load %.1f, GPU load %.1f, priority 127)",
        cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job1.get());

    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, " ");
}

/// Tests the yield() functionality using child jobs of higher/same/lower priority.
void test_yield()
{
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, "Testing yield() ...\n ");

    mi::Float32 cpu_load = 1.0f;
    mi::Float32 gpu_load = 1.0f;
    mi::Float32 delay    = 0.1f;

    Thread_pool thread_pool( cpu_load, gpu_load, 1);
    thread_pool.dump_thread_state_counters();
    thread_pool.dump_load();

    mi::base::Handle<Yield_job> job0( new Yield_job(
        &thread_pool, 0, cpu_load, gpu_load, delay, 42, true));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting yield parent job %u (CPU load %.1f, GPU load %.1f, priority 42)",
        0, cpu_load, gpu_load);
    thread_pool.submit_job( job0.get());

    // Wait for low priority job before shutting down.
    mi::base::Handle<Priority_job> job4( new Priority_job(
        &thread_pool, 3, cpu_load, gpu_load, delay, 127));
    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC,
        "Submitting priority job 3 (CPU load %.1f, GPU load %.1f, priority 127)",
        cpu_load, gpu_load);
    thread_pool.submit_job_and_wait( job4.get());

    LOG::mod_log->info( M_THREAD_POOL, LOG::Mod_log::C_MISC, " ");
}

MI_TEST_AUTO_FUNCTION( test_thread_pool )
{
    SYSTEM::Access_module<MEM::Mem_module> mem_module( false);
    SYSTEM::Access_module<LOG::Log_module> log_module( false);
    // log_module->set_severity_limit( LOG::ILogger::S_ALL);
    // log_module->set_severity_by_category( LOG::ILogger::C_MISC, LOG::ILogger::S_ALL);

    test_many_jobs();
    test_child_jobs();
    test_fragmented_jobs();
    test_expensive_jobs();
    test_priorities();
    test_yield();
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
