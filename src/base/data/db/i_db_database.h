/***************************************************************************************************
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
 **************************************************************************************************/

#ifndef BASE_DATA_DB_I_DB_DATABASE_H
#define BASE_DATA_DB_I_DB_DATABASE_H

#include <string>

#include <boost/core/noncopyable.hpp>

#include <mi/base/interface_declare.h>

#include "i_db_scope.h"
#include "i_db_tag.h"

/// Reference counting strategy in base/data/db
///
/// The interfaces in base/data/db and its implementation in based/data/dbnr predate
/// mi::base::IInterface and the reference counting strategy used in the public APIs. The following
/// abbreviations are used to denote the convention for each pointer in parameters and return
/// values.
///
/// - RCS:NEU neutral, for parameters and return values
/// - RCS:ICE incremented by callee, for return values
/// - RCS:ICR incremented by caller, for parameters, rare
/// - RCS:TRO transfers ownership, for parameters and return values, rare, e.g.,
///           Transaction::store()
///
/// Pointers of classes derived from mi::base::IInterface are not annotated since the standard
/// reference counting strategy as in the public APIs is assumed (RCS:NEU for parameters and
/// RCS:ICE for return values).

namespace MI {

namespace DB {

class Elememt_base;
class Fragmented_job;
class IExecution_listener;
class IScope_listener;
class ITransaction_listener;
class Status_listener;
class Transaction;

/// The public database interface.
///
/// Provides access to scopes (which provide access to transactions and DB elements/jobs).
class Database : private boost::noncopyable
{
public:
    /// \name Scopes
    //@{

    /// Returns the global scope.
    ///
    /// The global scope is the root of a tree of scopes.
    ///
    /// \return       The global scope. Never \c NULL. RCS:NEU
    virtual Scope* get_global_scope() = 0;

    /// Looks up and returns a scope with a given ID.
    ///
    /// \param id     The ID of the scope as returned by #DB::Scope::get_id(). The global scope has
    ///               ID 0.
    /// \return       The found scope or \c NULL if no such scope exists. RCS:NEU
    virtual Scope* lookup_scope( Scope_id id) = 0;

    /// Looks up a named scope.
    ///
    /// \param name   The name of the scope as returned by #DB::Scope::get_name(). The global scope
    ///               has the empty string as name.
    /// \return       The found scope or \c NULL if no such scope exists. RCS:NEU
    virtual Scope* lookup_scope( const std::string& name) = 0;

    /// Removes a scope from the database.
    ///
    /// \note Without explicit pinning the corresponding scope must no longer be used in any way
    ///       after a call to this method without.
    ///
    /// \param id     The ID of the scope to remove
    /// \return       \c true in case of success, \c false otherwise (invalid scope ID, or already
    ///               marked for removal).
    virtual bool remove( Scope_id id) = 0;

    //@}
    /// \name Closing
    //@{

    /// Prepares closing the database.
    ///
    /// This call stops all background activity of the database, e.g. the garbage collection. This
    /// is necessary because the exit() method of higher level modules often assume that no other
    /// thread is using the module anymore.
    virtual void prepare_close() = 0;

    /// Closes the database.
    ///
    /// Calls the destructor of this instance.
    virtual void close() = 0;

    //@}
    /// \name Garbage collection
    //@{

    /// Triggers a synchronous garbage collection run.
    ///
    /// The method sweeps through the entire database and removes all database elements which have
    /// been marked for removal and are no longer referenced. Note that it is not possible to remove
    /// database elements if there are open transactions in which such an element is still
    /// referenced.
    ///
    /// \param priority   The priority (0/1/2 for low/medium/high priority).
    virtual void garbage_collection( int priority) = 0;

    //@}
    /// \name Locks
    //@{

    /// Acquires a DB lock.
    ///
    /// The method blocks until the requested lock has been obtained. Recursively locking the
    /// same lock from within the same thread on the same host is supported.
    ///
    /// If the host holding a lock leaves the cluster, the lock is automatically released.
    ///
    /// \param lock_id   The lock to acquire.
    ///
    /// \note The locking mechanism is kind of a co-operative locking mechanism: The lock does not
    ///       prevent other threads from accessing or editing the DB. It only prevents other threads
    ///       from obtaining the same lock.
    ///
    /// \note DB locks are not restricted to threads on a single host, they apply to all threads on
    ///       all hosts in the cluster.
    ///
    /// \note DB locks are an expensive operation and should only be used when absolutely necessary.
    virtual void lock( mi::Uint32 lock_id) = 0;

    /// Releases a previously obtained DB lock.
    ///
    /// If the lock has been locked several times from within the same thread on the same host,
    /// it simply decrements the lock count. If the lock count reaches zero, the lock is released.
    ///
    /// \param lock_id   The lock to release.
    /// \return          0, in case of success, -1 in case of failure, i.e, the lock is not held
    ///                  by this thread on this host
    virtual bool unlock( mi::Uint32 lock_id) = 0;

    /// Checks whether a DB lock is locked.
    ///
    /// In the debug mode, abort if not. In release mode, just log an error.
    ///
    /// \param lock_id   The lock to check.
    virtual void check_is_locked( mi::Uint32 lock_id) = 0;

    //@}
    /// \name Memory limits
    //@{

    /// Sets the limits for memory usage of the database.
    ///
    /// \see #mi::neuraylib::IDatabase_configuration::set_memory_limits().
    virtual mi::Sint32 set_memory_limits( size_t low_water, size_t high_water) = 0;

    /// Returns the limits for memory usage of the database.
    ///
    /// \see #mi::neuraylib::IDatabase_configuration::get_memory_limits().
    virtual void get_memory_limits( size_t& low_water, size_t& high_water) const = 0;

    //@}
    /// \name Listeners
    //@{

    /// Registers a listener to be notified when the status changes.
    ///
    /// \note Must not be called from within the callback (deadlock).
    ///
    /// \param listener   RCS:NEU
    virtual void register_status_listener( Status_listener* listener) = 0;

    /// Unregisters a previously registered status listener.
    ///
    /// \note Must not be called from within the callback (deadlock).
    ///
    /// \param listener   RCS:NEU
    virtual void unregister_status_listener( Status_listener* listener) = 0;

    /// Registers a listener to be notified for transaction events.
    ///
    /// \note Must not be called from within the callback (deadlock).
    virtual void register_transaction_listener( ITransaction_listener* listener) = 0;

    /// Unregisters a previously registered transaction listener.
    ///
    /// \note Must not be called from within the callback (deadlock).
    virtual void unregister_transaction_listener( ITransaction_listener* listener) = 0;

    /// Registers a listener to be notified for scope events.
    ///
    /// \note Must not be called from within the callback (deadlock).
    virtual void register_scope_listener( IScope_listener* listener) = 0;

    /// Unregisters a previously registered Scope listener.
    ///
    /// \note Must not be called from within the callback (deadlock).
    virtual void unregister_scope_listener( IScope_listener* listener) = 0;

    //@}
    /// \name Thread pool and fragmented jobs
    //@{

    /// Executes a job, splitting it into the given number of fragments (synchronous).
    ///
    /// This method is for fragmented jobs without transaction context. See
    /// #Transaction::execute_fragmented() for executing fragmented jobs with transaction context.
    ///
    /// This method will not return before all fragments have been executed. The fragments may be
    /// executed in any number of threads.
    ///
    /// \note This method is restricted to localhost-only jobs.
    ///
    /// \param job                      The fragmented job to be executed. RCS:NEU
    /// \param count                    The number of fragments this job should be split into. This
    ///                                 number must be greater than zero.
    /// \return
    ///                                 -  0: Success.
    ///                                 - -1: Invalid parameters (\p job is \c NULL or \c count is
    ///                                       zero).
    ///                                 - -2: Invalid scheduling mode (transaction-less or
    ///                                       asynchronous execution is restricted to local jobs).
    virtual mi::Sint32 execute_fragmented( Fragmented_job* job, size_t count) = 0;

    /// Executes a job, splitting it into the given number of fragments (asynchronous).
    ///
    /// This method is for fragmented jobs without transaction context. See
    /// #Transaction::execute_fragmented_async() for executing fragmented jobs with transaction
    /// context.
    ///
    /// This will return immediately, typically before all fragments have been executed. The
    /// fragments may be executed in any number of threads.
    ///
    /// \note This method is restricted to localhost-only jobs.
    ///
    /// \param job                      The fragmented job to be executed. RCS:NEU
    /// \param count                    The number of fragments this job should be split into. This
    ///                                 number must be greater than zero.
    /// \param listener                 Provides a callback to be called when the job is done.
    /// \return
    ///                                 -  0: Success.
    ///                                 - -1: Invalid parameters (\p job is \c NULL or \c count is
    ///                                       zero).
    ///                                 - -2: Invalid scheduling mode (transaction-less or
    ///                                       asynchronous execution is restricted to local jobs).
    virtual mi::Sint32 execute_fragmented_async(
        Fragmented_job* job, size_t count, IExecution_listener* listener) = 0;

    /// See THREAD_POOL::Thread_pool::suspend_current_job().
    virtual void suspend_current_job() = 0;

    /// See THREAD_POOL::Thread_pool::resume_current_job().
    virtual void resume_current_job() = 0;

    /// See THREAD_POOL::Thread_pool::yield().
    virtual void yield() = 0;

    //@}
};

/// Status that the database can be in.
enum Db_status
{
    DB_OK,        ///< The database if fully operational.
    DB_NOT_READY, ///< The database is not yet ready. This state may occur initially upon creation.
    DB_RECOVERING ///< The database is recovering from a network split.
};

/// Abstract interface for status listeners.
class Status_listener
{
public:
    /// Invoked when the status of the database changed.
    virtual void status_changed( Db_status new_status) = 0;
};

/// Abstract interface for listeners for major transaction events.
class ITransaction_listener : public
    mi::base::Interface_declare<0xf33f94ac,0xc43f,0x4ea1,0x97,0xd5,0xbc,0x8b,0xd1,0xae,0x6d,0x83>
{
public:
    /// Invoked when the provided transaction has been created.
    ///
    /// \param transaction   RCS:NEU
    virtual void transaction_created( Transaction* transaction) = 0;

    /// Invoked when the provided transaction is about to be committed.
    ///
    /// The transaction is still open at this point. Note that the transaction can be aborted
    /// instead of committed in case of an error, even if the pre-commit callback has been made.
    /// In such a case, this call will be followed by a #transaction_aborted() call instead of a
    /// #transaction_committed() call.
    ///
    /// \param transaction   RCS:NEU
    virtual void transaction_pre_commit( Transaction* transaction) = 0;

    /// Invoked when the provided transaction is about to be aborted.
    ///
    /// The transaction is still open at this point.
    ///
    /// \param transaction   RCS:NEU
    virtual void transaction_pre_abort( Transaction* transaction) = 0;

    /// Invoked when the provided transaction has been committed.
    ///
    /// \param transaction   RCS:NEU
    virtual void transaction_committed( Transaction* transaction) = 0;

    /// Invoked when the provided transaction has been aborted.
    ///
    /// \param transaction   RCS:NEU
    virtual void transaction_aborted( Transaction* transaction) = 0;
};

/// Abstract interface for listeners for major scope events.
class IScope_listener : public
    mi::base::Interface_declare<0x7d8320ce,0xd104,0x4465,0xa0,0xb8,0x9f,0x3e,0xc8,0x6c,0x30,0xac>
{
public:
    /// Invoked when a scope has been created.
    ///
    /// The scope can be used as soon as the callback is received. Note that there is no callback
    /// for the global scope.
    ///
    /// \param scope   RCS:NEU
    virtual void scope_created( Scope* scope) = 0;

    /// Invoked when a scope has been removed.
    ///
    /// The scope must not be used anymore when the callback is received. Note that there is no
    /// callback for the global scope.
    ///
    /// \param scope   RCS:NEU
    virtual void scope_removed( Scope* scope) = 0;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_DATABASE_H
