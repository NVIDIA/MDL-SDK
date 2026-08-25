/***************************************************************************************************
 * Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_UTIL_H
#define BASE_DATA_DBLIGHT_DBLIGHT_UTIL_H

#include <chrono>
#include <iosfwd>
#include <functional>
#include <string>

#include <boost/core/noncopyable.hpp>
#include <boost/intrusive_ptr.hpp>

#include <mi/base/types.h>

#include <base/hal/thread/i_thread_lock.h>
#include <base/hal/thread/i_thread_rw_lock.h>

/// Enable this macro to collect some statistics.
///
/// The statistics are dumped when the database is destroyed.
// #define DBLIGHT_ENABLE_STATISTICS

/// Enable this macro to run the admin server.
// #define DBLIGHT_ENABLE_ADMIN_SERVER

namespace MI {

namespace DBLIGHT {


/// Creates a boost::intrusive_ptr \em without increasing the reference count.
///
/// Use the regular constructor to create intrusive pointers with increased reference count
/// (\p add_ref defaults to \c true).
template <class T>
boost::intrusive_ptr<T> make_ptr_no_add_ref( T* ptr)
{ return boost::intrusive_ptr<T>( ptr, /*add_ref*/ false); }


/// A flexible has the same interface as a shared lock, but can be can be configured at runtime to
/// use different lock implementation/strategies.
class Flexible_lock
{
public:
    /// The different lock implementations/strategies.
    enum Lock_implementation {
        /// Use a shared lock.
        SHARED_LOCK,
        /// Use a shared lock, but map all shared requests to exclusive requests.
        SHARED_LOCK_BUT_USED_EXCLUSIVELY,
        /// Use an exclusive lock.
        EXCLUSIVE_LOCK
    };

    /// Constructor.
    Flexible_lock( Lock_implementation impl = Flexible_lock::SHARED_LOCK);

    Flexible_lock( const Flexible_lock&) = delete;
    Flexible_lock& operator=( const Flexible_lock&) = delete;

    /// Returns a string representation of the lock implementation being used.
    std::string get_lock_impl_str() const;

    /// %Locks the lock in shared mode.
    void lock_shared();

    /// Tries to lock the lock in shared mode.
    bool try_lock_shared();

    /// Unlocks the lock in shared mode.
    void unlock_shared();

    /// Some sanity check.
    ///
    /// - This method does nothing if assertions are disabled.
    /// - Otherwise, the method checks that the lock is held in shared mode by \em some thread, not
    ///   necessarily by this thread.
    void check_is_owned_shared();

    /// %Locks the lock in exclusive mode.
    void lock();

    /// Tries to lock the lock in exclusive mode.
    bool try_lock();

    /// Unlocks the lock in exclusive mode.
    void unlock();

    /// Some sanity check.
    ///
    /// - This method does nothing if assertions are disabled.
    /// - Otherwise, the method checks that the lock is held in exclusive mode by \em some thread,
    ///   not necessarily by this thread.
    void check_is_owned();

    /// Some sanity check.
    ///
    /// - This method does nothing if assertions are disabled.
    /// - Otherwise, the method checks that the lock is held in shared \em or exclusive mode by
    ///   \em some thread, not necessarily by this thread.
    void check_is_owned_shared_or_exclusive();

private:
    /// The configured lock implementation/strategy.
    Lock_implementation m_lock_implementation;

    /// The exclusive lock (used if #m_lock_implementation == #EXCLUSIVE_LOCK).
    THREAD::Lock m_lock;

    /// The shared lock (used if #m_lock_implementation != #EXCLUSIVE_LOCK).
    THREAD::Shared_lock m_shared_lock;
};

/// Callbacks and static information needed by methods generating the HTML pages.
struct Html_context
{
    /// Function that encodes a string for HTML.
    std::function<std::string(const std::string&)> m_html_encoder;
    /// Function that encodes a string for URLs (percent encoding).
    std::function<std::string(const std::string&)> m_url_encoder;
    /// URL prefix for links to a specific tag.
    std::string m_tag_url_prefix;
    /// URL prefix for links to a specific name.
    std::string m_name_url_prefix;
};


/// Returns \c "yes" if \p value is \c true, and \c "no" otherwise.
const char* to_yes_no( bool value);

/// Dumps \p name \and \p value as two columns of an HTML table to \p s.
void dump_html_bool_settings( std::ostream& s, const char* name, bool value);

/// Dumps \p name \and \p value as two columns of an HTML table to \p s.
void dump_html_size_t_setting( std::ostream& s, const char* name, size_t value);

/// Dumps \p name \and \p value as two columns of an HTML table to \p s.
void dump_html_double_setting( std::ostream& s, const char* name, double value, const char* suffix);

/// Dumps \p name \and \p value as two columns of an HTML table to \p s.
void dump_html_string_setting( std::ostream& s, const char* name, const std::string& value);

/// Dumps the global accumulated statistics and the given tag to the stream.
void dump_statistics( std::ostream& s, mi::Uint32 next_tag);


struct Statistics_data
{
    size_t m_count = 0;
    double m_time  = 0.0;
};

class Statistics_helper : private boost::noncopyable
{
#ifdef DBLIGHT_ENABLE_STATISTICS
public:
    Statistics_helper( Statistics_data& data);
    ~Statistics_helper();

private:
    Statistics_data& m_data;
    std::chrono::time_point<std::chrono::system_clock> m_start_time;
#else // DBLIGHT_ENABLE_STATISTICS
public:
    Statistics_helper( Statistics_data& /*data*/) { }
#endif // DBLIGHT_ENABLE_STATISTICS
};

extern Statistics_data g_commit;
extern Statistics_data g_abort;
extern Statistics_data g_access_by_name;
extern Statistics_data g_access_by_tag;
extern Statistics_data g_edit_by_name;
extern Statistics_data g_edit_by_tag;
extern Statistics_data g_finish_edit;
extern Statistics_data g_store;
extern Statistics_data g_localize;
extern Statistics_data g_remove;
extern Statistics_data g_name_to_tag;
extern Statistics_data g_tag_to_name;
extern Statistics_data g_get_class_id;
extern Statistics_data g_get_tag_is_job;
extern Statistics_data g_get_tag_is_removed;
extern Statistics_data g_get_tag_privacy_level;
extern Statistics_data g_get_tag_store_level;
extern Statistics_data g_get_tag_reference_count;
extern Statistics_data g_get_tag_version;
extern Statistics_data g_invalidate_job_results;
extern Statistics_data g_advise;
extern Statistics_data g_can_reference_tag;
extern Statistics_data g_block_commit_or_abort;
extern Statistics_data g_unblock_commit_or_abort;
extern Statistics_data g_transaction_get_journal;

extern Statistics_data g_scope_get_journal;
extern Statistics_data g_scope_destructor;

extern Statistics_data g_lookup_info_by_name;
extern Statistics_data g_lookup_info_by_tag;
extern Statistics_data g_garbage_collection;

extern Statistics_data g_element_job_destructors;

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_UTIL_H
