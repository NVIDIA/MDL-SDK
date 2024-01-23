/******************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief regression test public/mi/base components
///
/// See \ref mi_base_threads.
///

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/lock.h>

using mi::base::Lock;
using mi::base::Recursive_lock;

#if 0 // methods on Lock are not public
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
#endif

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

#if 0 // methods on Recursive_lock are not public
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
#endif

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
