/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief Store a thread's identity and mark it's type
///
/// We need to assign a unique identifier to a thread rather than
/// using a 32bit arbitrary number like pthread id's or a handle on
/// NT. Therefore, pthread_self() is insufficient for neuray.
/// The id is kept as thread specific data (TSD) that is not to be
/// used for more than that. It is not intended for passing around
/// scope pointers and the like. Extending this needs explicit
/// approval! We don't want thread local storage in neuray.

#ifndef BASE_HAL_THREAD_I_THREAD_ATTR_H
#define BASE_HAL_THREAD_I_THREAD_ATTR_H

#ifndef WIN_NT
#include <pthread.h>
#else
#include <mi/base/miwindows.h>
#endif

#include <base/lib/log/i_log_assert.h>

namespace MI {
namespace THREAD {


/// thread specific information used for self-identification :)
typedef struct Thread_data {
    unsigned int id;
} Thr_data;

/// this class implements access to thread specific information using
/// a key. there's no other way to access this data.

class Thread_attr {

  public:
    /// constructor
    Thread_attr();

    /// constructor. version allocating memory for thread specific data.
    Thread_attr(
        bool flag);

    /// set the id
    /// \param thread_id set thread id to this
    void set_id(
        int thread_id);

    /// get the id
    /// \return the thread id
    int get_id();

  private:
    /// initialize key that all threads use to access tsd
    bool create();

    /// access the thread private data
    /// \return thread private data
    Thr_data* access_data();

    /// key to access thread specific data. needs to be static as it
    /// must be the same for all instances of the class. it is initialized
    /// during module initialization and is used by threads to access
    /// thread specific data (TSD).
#ifndef WIN_NT
    static pthread_key_t m_key;
#else
    static DWORD m_key;
#endif

    friend class Thread;                ///< needs access to the key
};


// constructor

inline Thread_attr::Thread_attr(
    bool flag)
{
#ifdef ENABLE_ASSERT
    bool key_created =
#endif
    create();
    ASSERT(M_THREAD, key_created == true);
}


// constructor

inline Thread_attr::Thread_attr()
{}


// used to create the key that all thread uses to access data

inline bool Thread_attr::create()
{
    static Thr_data thr_data;
#ifndef WIN_NT
    // note: added testing the return value here to narrow down
    // a problem sometimes occuring with the test program. it
    // actually shouldn't fail.
    if (pthread_key_create(&m_key, NULL) == 0)
        pthread_setspecific(m_key, (void*)&thr_data);
    else
        return false;
#else
    m_key = TlsAlloc();
    TlsSetValue(m_key, (LPVOID)&thr_data);
#endif
    return true;
}


// access the private thread data

inline Thr_data* Thread_attr::access_data()
{
#ifndef WIN_NT
    return (Thr_data*)pthread_getspecific(m_key);
#else
    return (Thr_data*)TlsGetValue(m_key);
#endif
}


// set the id

inline void Thread_attr::set_id(
    int id)                             // set thread id to this
{
    Thr_data* thr_data = access_data();
    ASSERT(M_THREAD, thr_data != 0);

    thr_data->id = id;
}


// get the id
//
// will return id 0 if there is no thread data since the main
// thread of a program is not a thread and therefore have not
// initialized the thread attributes.

inline int Thread_attr::get_id()
{
    Thr_data* thr_data = access_data();
    if (thr_data == 0)
        return 0; // main or an unknown thread (not derived from THREAD::Thread)

    return thr_data->id;
}

}
}

#endif
