/******************************************************************************
 * Copyright (c) 2004-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The tests manipulate the global the_counter, using various subclasses of Counter.

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "base/hal/thread Test Suite"
#include <base/system/test/i_test_auto_driver.h>

#include <cstdio>

#include <mi/base/config.h>

#ifndef MI_PLATFORM_WINDOWS
#include <unistd.h>    // _SC_CLK_TCK
#include <sys/times.h> // struct tms
#endif

#include <boost/bind/bind.hpp>

#include <base/system/main/access_module.h>
#include <base/system/test/i_test_case.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_module.h>

#include "i_thread_thread.h"
#include "i_thread_lock.h"
#include "i_thread_condition.h"
#include "i_thread_rw_lock.h"

using namespace MI;
using namespace MI::CONFIG;

using MI::THREAD::Thread;
using MI::THREAD::Lock;
using MI::THREAD::Recursive_lock;
using MI::THREAD::Shared_lock;
using MI::THREAD::Condition;

#ifndef MI_PLATFORM_WINDOWS

const int max_counter = 10;
const int max_steps = 10000;

int the_count = 0;

Lock the_count_lock;
Lock the_print_lock;

class Counter;

Counter *counter[max_counter];
Thread *thread[max_counter];
Condition *condition[max_counter];

class Timer
{

private:

    int clk_tck;

    struct tms begin;

    struct tms now;

    double utime;

    double stime;

protected:

    void init_time()
    {
        clk_tck = sysconf(_SC_CLK_TCK);
    }

    void start_time()
    {
        times(&begin);
    }

    void stop_time()
    {
        times(&now);
        utime = (double) (now.tms_utime - begin.tms_utime) / (double) clk_tck;
        stime = (double) (now.tms_stime - begin.tms_stime) / (double) clk_tck;
    }

    void print_time()
    {
#ifdef MI_TEST_VERBOSE
        printf("user %f system %f total %f\n",
                get_utime(),
                get_stime(),
                get_ttime());
#endif
    }

public:

    double get_utime()
    {
        return utime;
    }

    double get_stime()
    {
        return stime;
    }

    double get_ttime()
    {
        return utime + stime;
    }

};

class Counter : public Thread, public Timer
{
};

class Counter_lock : public Counter
{

protected:

    void run()
    {
        init_time();
        start_time();
        for(int i = 0; i < max_steps; i++) {
            Lock::Block synchronized(&the_count_lock);

            ++the_count;
        }
        stop_time();
        {
            Lock::Block synchronized(&the_print_lock);

#ifdef MI_TEST_VERBOSE
            printf("Counter_lock ");
#endif
            print_time();
        }
    }

};

class Counter_condition : public Counter
{

private:

    int m_id;

public:

    Counter_condition(int id)
    {
        m_id = id;
    }

protected:

    void run()
    {
        init_time();
        start_time();
        for(int i = 0; i < max_steps; i++) {
            condition[m_id]->wait();
            ++the_count;
            condition[(m_id + 1) % max_counter]->signal();
        }
        stop_time();
        {
            Lock::Block synchronized(&the_print_lock);

#ifdef MI_TEST_VERBOSE
            printf("Counter_condition ");
#endif
            print_time();
        }
    }

};

bool test_counter_lock()
{
    double time = 0.0;
    int i;

#ifdef MI_TEST_VERBOSE
    printf("testing lock\n");
#endif
    the_count = 0;
    for(i = 0; i < max_counter; i++)
        counter[i] = new Counter_lock();
    for(i = 0; i < max_counter; i++)
        counter[i]->start();
    for(i = 0; i < max_counter; i++)
        counter[i]->join();
    for(i = 0; i < max_counter; i++)
        time += counter[i]->get_ttime();
#ifdef MI_TEST_VERBOSE
    printf("the_count == %d\n",the_count);
    printf("%f counts / second\n",the_count / time);
    printf("%f milliseconds / count\n",1000 * time / the_count);
#endif
    for(i = 0; i < max_counter; i++)
        delete counter[i];
    return true;
}

bool test_condition()
{
    double time = 0.0;
    int i;

#ifdef MI_TEST_VERBOSE
    printf("testing condition\n");
#endif
    the_count = 0;
    for(i = 0; i < max_counter; i++)
        condition[i] = new Condition();
    for(i = 0; i < max_counter; i++)
        counter[i] = new Counter_condition(i);
    for(i = 0; i < max_counter; i++)
        counter[i]->start();
    condition[0]->signal();
    for(i = 0; i < max_counter; i++)
        counter[i]->join();
    for(i = 0; i < max_counter; i++)
        time += counter[i]->get_ttime();
#ifdef MI_TEST_VERBOSE
    printf("the_count == %d\n",the_count);
    printf("%f counts / second\n",the_count / time);
    printf("%f milliseconds / count\n",1000 * time / the_count);
#endif
    for(i = 0; i < max_counter; i++)
        delete counter[i];
    for(i = 0; i < max_counter; i++)
        delete condition[i];
    return true;
}

class Server : public Timer
{

protected:

    int served;
    int count;

public:

    Server() : served(0), count(0) {}

    virtual ~Server() = default;

    virtual void serve(int i) = 0;

    int get_count()
    {
        return count;
    }

};

class Passive_server : public Server
{

private:

    Lock m_lock;

public:

    virtual ~Passive_server() = default;

    void serve(int i)
    {
        {
            Lock::Block synchronized(&m_lock);

            ++served;
            count += i;
            while(i--)
                /* skip */;
        }
    }

};

class Active_server : public Server, public Thread
{

private:

    Condition m_server_ready;
    Condition m_params_ready;

    int m_params;

public:

    virtual ~Active_server() = default;

    void serve(int i)
    {
        m_server_ready.wait();
        m_params = i;
        m_params_ready.signal();
    }

    void run()
    {
        init_time();
        start_time();

        m_server_ready.signal();
        for(;;) {
            m_params_ready.wait();
            ++served;
            count += m_params;
            while(m_params--)
                /* skip */;
            if(100000 <= served)
                break;
            m_server_ready.signal();
        }

        stop_time();
        {
            Lock::Block synchronized(&the_print_lock);

#ifdef MI_TEST_VERBOSE
            printf("Server ");
#endif
            print_time();
        }
    }

};

class Client : public Thread, public Timer
{

private:

    Server *m_server;

public:

    Client(Server *server) : m_server(server)
    {}

    void run()
    {
        init_time();
        start_time();
        for(int i = 0; i < 50000; i++) {
            int n = 1000 + (int) (1000 * ((double) random()
                            / (double) INT_MAX));
            int m = 0;
            if(!((1000 <= n) && (n <= 2000))) {
                fprintf(stderr,"random value out of range %d\n",n);
                ::exit(1);
            }
            for(int k = 0; k < n; k++)
                ++m;
            m_server->serve(m);
        }
        stop_time();
        {
            Lock::Block synchronized(&the_print_lock);

#ifdef MI_TEST_VERBOSE
            printf("Client ");
#endif
            print_time();
        }
    }

};

bool test_passive_server()
{
    double time = 0.0;

#ifdef MI_TEST_VERBOSE
    printf("testing passive server\n");
#endif
    srandom(1);
    Passive_server passive_server;
    Client client_1(&passive_server);
    Client client_2(&passive_server);
    client_1.start();
    client_2.start();
    client_1.join();
    client_2.join();
    time += client_1.get_ttime();
    time += client_2.get_ttime();
#ifdef MI_TEST_VERBOSE
    printf("count == %d\n",passive_server.get_count());
    printf("%f services / second\n",1000000 / time);
    printf("%f milliseconds / service\n",time / 1000);
#endif
    return true;
}

bool test_active_server()
{
    double time = 0.0;

#ifdef MI_TEST_VERBOSE
    printf("testing active server\n");
#endif
    srandom(1);
    Active_server active_server;
    Client client_1(&active_server);
    Client client_2(&active_server);
    active_server.start();
    client_1.start();
    client_2.start();
    client_1.join();
    client_2.join();
    active_server.join();
    time += client_1.get_ttime();
    time += client_2.get_ttime();
    time += active_server.get_ttime();
#ifdef MI_TEST_VERBOSE
    printf("count == %d\n",active_server.get_count());
    printf("%f services / second\n",1000000 / time);
    printf("%f milliseconds / service\n",time / 1000);
#endif
    return true;
}

class PrioritizedThread : public Thread
{

private:

    int m_prio;

public:

    PrioritizedThread(const int prio) : m_prio(prio)
    {
    }

    void run()
    {
        int s = 0;

#ifdef MI_TEST_VERBOSE
        printf("initial thread priority is %d\n",get_priority());
#endif
        set_priority(m_prio);
#ifdef MI_TEST_VERBOSE
        printf("thread priority set to %d\n",get_priority());
#endif
        for(int i = 0; i < 1000000; i++)
            ++s;
#ifdef MI_TEST_VERBOSE
        printf("priority %d thread terminating\n",m_prio);
#endif
    }

};

bool test_priorities()
{
    PrioritizedThread h(5);
    PrioritizedThread m(0);
    PrioritizedThread l(-5);

    h.start();
    m.start();
    l.start();
    h.join();
    m.join();
    l.join();
    return true;
}

class Pin_thread_to_cpu : public Thread
{
public:
    void run()
    {
        int s = 0;

        for(int i = 0; i < 1000000; i++)
            ++s;
    }
};

bool test_pin_cpu()
{
    Pin_thread_to_cpu thr;
    int cpu;

    // Ignore this test if Thread::get_cpu is unavailable (i.e. not supported)
    if ((cpu = thr.get_cpu()) < 0)
        return true;

    // Ignore this test if Thread::pin_cpu is unavailable (i.e. not supported)
    if (!thr.pin_cpu())
        return true;

    double f = 2.0f;
    for (int i = 0; i < max_steps; i++)
    {
        // Some meaningless computations to burn CPU cycles
        f += sqrt(static_cast<double>(i)) + 2.0f;

        if (cpu != thr.get_cpu())
            return false;
    }

    thr.unpin_cpu();

    return true;
}


MI_TEST_AUTO_FUNCTION( test_thread_module )
{
    SYSTEM::Access_module<LOG::Log_module> log_module( false);

    MI_REQUIRE(test_counter_lock());

    MI_REQUIRE(test_condition());

    MI_REQUIRE(test_passive_server());

    MI_REQUIRE(test_active_server());

    MI_REQUIRE(test_priorities());

    MI_REQUIRE(test_pin_cpu());
}
#endif // MI_PLATFORM_WINDOWS

MI_TEST_AUTO_FUNCTION( test_thread_class )
{
    class My_thread : public THREAD::Thread
    {
      public:
        My_thread(bool detached=false)
            : THREAD::Thread(detached), m_detached(detached) {}
        void run()
        {
            if (m_detached)
                m_cond.signal();
        }
        bool m_detached;
        THREAD::Condition m_cond;
    };

    My_thread t;
    MI_CHECK(t.start());
    MI_CHECK(!t.start());
    t.join();

#if 0
    // "true" means create it as a detached thread
    My_thread t2(true);
    MI_CHECK(t2.start());
    t2.join(); // does nothing on a detached thread
    t2.m_cond.wait();
#endif
}

MI_TEST_AUTO_FUNCTION( test_lock )
{
    {
        // lock() on unlocked lock
        Lock lock;
        lock.lock();
        lock.unlock();
    }

#if 0 // causes dead lock / aborts the process
    {
        // lock() on locked lock
        Lock lock;
        lock.lock();
        lock.lock();
    }
#endif

    {
        // try_lock() on unlocked lock
        Lock lock;
        bool success = lock.try_lock();
        MI_CHECK( success);
        lock.unlock();
    }

    {
        // try_lock() on locked lock
        Lock lock;
        lock.lock();
        bool success = lock.try_lock();
        MI_CHECK( !success);
        lock.unlock();
    }
}

MI_TEST_AUTO_FUNCTION( test_lock_block )
{
    {
        // lock() on unlocked lock
        Lock lock;
        Lock::Block block( &lock);
    }

#if 0 // causes dead lock / aborts the process
    {
        // lock() on locked lock
        Lock lock;
        Lock::Block block1( &lock);
        Lock::Block block2( &lock);
    }
#endif

    {
        // try_lock() on unlocked lock
        Lock lock;
        Lock::Block block;
        bool success = block.try_set( &lock);
        MI_CHECK( success);
    }

    {
        // try_lock() on locked lock
        Lock lock;
        Lock::Block block1( &lock);
        Lock::Block block2;
        bool success = block2.try_set( &lock);
        MI_CHECK( !success);
    }
}

MI_TEST_AUTO_FUNCTION( test_recursive_lock )
{
    {
        // lock() on unlocked lock
        Recursive_lock lock;
        lock.lock();
        lock.unlock();
    }

    {
        // lock() on locked lock
        Recursive_lock lock;
        lock.lock();
        lock.lock();
        lock.unlock();
        lock.unlock();
    }

    {
        // try_lock() on unlocked lock
        Recursive_lock lock;
        bool success = lock.try_lock();
        MI_CHECK( success);
        lock.unlock();
    }

    {
        // try_lock() on locked lock
        Recursive_lock lock;
        lock.lock();
        bool success = lock.try_lock();
        MI_CHECK( success);
        lock.unlock();
        lock.unlock();
    }
}

MI_TEST_AUTO_FUNCTION( test_recursive_lock_block )
{
    {
        // lock() on unlocked lock
        Recursive_lock lock;
        Recursive_lock::Block block( &lock);
    }

    {
        // lock() on locked lock
        Recursive_lock lock;
        Recursive_lock::Block block1( &lock);
        Recursive_lock::Block block2( &lock);
    }

    {
        // try_lock() on unlocked lock
        Recursive_lock lock;
        Recursive_lock::Block block;
        bool success = block.try_set( &lock);
        MI_CHECK( success);
    }

    {
        // try_lock() on locked lock
        Recursive_lock lock;
        Recursive_lock::Block block1( &lock);
        Recursive_lock::Block block2;
        bool success = block2.try_set( &lock);
        MI_CHECK( success);
    }
}

MI_TEST_AUTO_FUNCTION( test_rw_lock_block )
{
    {
        // lock_shared() on unlocked lock
        Shared_lock lock;
        Shared_lock::Block_shared block( &lock);
        lock.check_is_owned_shared();
    }

    {
        // lock_shared() on shared locked lock
        Shared_lock lock;
        Shared_lock::Block_shared block1( &lock);
        Shared_lock::Block_shared block2( &lock);
        lock.check_is_owned_shared();
    }

    {
        // try_lock_shared() on unlocked lock
        Shared_lock lock;
        Shared_lock::Block_shared block;
        bool success = block.try_set( &lock);
        MI_CHECK( success);
        lock.check_is_owned_shared();
    }

    {
        // try_lock_shared() on shared locked lock
        Shared_lock lock;
        Shared_lock::Block_shared block1;
        bool success = block1.try_set( &lock);
        MI_CHECK( success);
        lock.check_is_owned_shared();
        Shared_lock::Block_shared block2;
        success = block2.try_set( &lock);
        MI_CHECK( success);
        lock.check_is_owned_shared();
    }

    {
        // lock_exclusive() on unlocked lock
        Shared_lock lock;
        Shared_lock::Block_exclusive block( &lock);
        lock.check_is_owned();
    }

    {
        // try_lock_exclusive() on unlocked lock
        Shared_lock lock;
        Shared_lock::Block_exclusive block;
        bool success = block.try_set( &lock);
        MI_CHECK( success);
        lock.check_is_owned();
    }

    {
        // try_lock_exclusive() on exclusively locked lock
        Shared_lock lock;
        Shared_lock::Block_exclusive block1;
        bool success = block1.try_set( &lock);
        MI_CHECK( success);
        lock.check_is_owned();
        Shared_lock::Block_exclusive block2;
        success = block2.try_set( &lock);
        MI_CHECK( !success);
        lock.check_is_owned();
    }

    {
        // try_lock_shared() on exclusively locked lock
        Shared_lock lock;
        Shared_lock::Block_exclusive block1;
        bool success = block1.try_set( &lock);
        MI_CHECK( success);
        lock.check_is_owned();
        Shared_lock::Block_shared block2;
        success = block2.try_set( &lock);
        MI_CHECK( !success);
        lock.check_is_owned();
    }

    {
        // try_lock_exclusive() on shared locked lock
        Shared_lock lock;
        Shared_lock::Block_shared block1;
        bool success = block1.try_set( &lock);
        MI_CHECK( success);
        lock.check_is_owned_shared();
        Shared_lock::Block_exclusive block2;
        success = block2.try_set( &lock);
        MI_CHECK( !success);
    }
}

