/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file i_db_database.h
 ** \brief This declares the database class which owns all transactions etc.
 **
 ** This file contains the pure virtual base class for the database class which owns all
 ** transactions, database elements, etc.
 **/

#ifndef BASE_DATA_DB_I_DB_DATABASE_H
#define BASE_DATA_DB_I_DB_DATABASE_H

#include "i_db_scope.h"
#include "i_db_transaction.h"

#include <base/hal/time/i_time.h>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <mi/base/interface_declare.h>

namespace MI
{

namespace CLUSTER     { class Cluster_manager; }
namespace SERIAL      { class Deserialization_manager; }
namespace HTTP        { class Server; class Ssi_handler; }
namespace MSG         { class Selector; }
namespace SCHED       { class IScheduler; }
namespace EVENT       { class Event0_base; }
namespace NET         { class Message_logger; }
namespace RDMA        { class IRDMA_group; }
namespace THREAD_POOL { class Thread_pool; }

namespace DB
{

class Elememt_base;
class Fragmented_job;
class IExecution_listener;

/// Status that the database can be in.
enum Db_status
{
    DB_NOT_READY,
    DB_RECOVERING,
    DB_OK
};

/// This is an abstract class which can be used to register a status listener with the database.
/// The listener will be called whenever the status of the database changes.
struct IStatus_listener
{
    /// This is called when the status of the database changed
    ///
    /// \param status           The new status
    virtual void status_changed(Db_status status) { }
};

/// This is an abstract class which can be used to register a transaction listener with the 
/// database. The listener will be called whenever a transaction is created, just before
/// it is committed or aborted, and when it has been committed/aborted.
struct ITransaction_listener : public
    mi::base::Interface_declare<0xf33f94ac,0xc43f,0x4ea1,0x97,0xd5,0xbc,0x8b,0xd1,0xae,0x6d,0x83>
{
    /// This method is called when the provided transaction has been created.
    ///
    /// \param trans            The new transaction
    virtual void transaction_created(Transaction* trans) = 0;

    /// This method is called when the provided transaction is about to be committed. The 
    /// transaction is still valid at this point. Note that the transaction can be aborted
    /// in case of an error even if the pre-commit callback has been made, in which case
    /// this call will be followed by a transaction_aborted call.
    ///
    /// \param trans            The transaction
    virtual void transaction_pre_commit(Transaction* trans) = 0;

    /// This method is called when the provided transaction is about to be aborted. The 
    /// transaction is still valid at this point.
    ///
    /// \param trans            The transaction
    virtual void transaction_pre_abort(Transaction* trans) = 0;

    /// This method is called when the provided transaction has been committed.
    ///
    /// \param trans            The transaction
    virtual void transaction_committed(Transaction* trans) = 0;
    
    /// This method is called when the provided transaction has been aborted.
    ///
    /// \param trans            The transaction
    virtual void transaction_aborted(Transaction* trans) = 0;
};

/// This is an abstract class which can be used to register a scope listener with the 
/// database. The listener will be called whenever a scope is created or destroyed.
struct IScope_listener : public
    mi::base::Interface_declare<0x7d8320ce,0xd104,0x4465,0xa0,0xb8,0x9f,0x3e,0xc8,0x6c,0x30,0xac>
{
    /// Called when the provided scope is created. Note that no callback is made for the 
    /// global scope. The scope may be used at this point.
    virtual void scope_created(Scope* scope) = 0;

    /// Called when the provided scope is removed. 
    virtual void scope_removed(Scope* scope) = 0;
};

/// Statistics for the database. Used to allow the application to query some interesting statistical
/// values for performance checking and tuning.
struct Database_statistics
{
    // constructor
    Database_statistics();

    /// the number of tags currently stored
    Uint m_nr_of_stored_tags;
    /// number of update messages
    Uint m_nr_of_received_updates;
    /// number of objects received so far
    Uint m_nr_of_received_objects;
    /// number of transactions received
    Uint m_nr_of_received_transactions;
    /// number of self-created transactions
    Uint m_nr_of_created_transactions;
    /// number of hosts we know about
    Uint m_nr_of_known_hosts;
};

/// The database class manages the whole database. It holds the caches for the database elements and
/// manages all the communication with other hosts. The Database class is the interface class which
/// is visible from the outside.
class Database
{
  public:
    /// Create a database instance.
    /// \param selector                         For event delivery
    /// \param cluster_manager                  For communication
    /// \param deserialization_manager          For deserialization
    /// \param redundancy_level                 Nr of tag copies
    /// \param web_interface_url_prefix         For web interface
    /// \param http_server                      For web interface
    /// \param logger                           For logging messages
    /// \param send_elements_only_to_owners     Send elements to all host or only to owners?
    /// \param disk_cache_path                  When a disk cache should be created the path or NULL
    /// \param max_journal_size
    /// \param track_memory_usage               Track memory usage of DB elements?
    /// \return                                 The new database
    static Database* create(
        MSG::Selector* selector,
        CLUSTER::Cluster_manager* cluster_manager,
        SERIAL::Deserialization_manager* deserialization_manager,
        Uint redundancy_level = 1,
        const char* web_interface_url_prefix = NULL,
        HTTP::Server* http_server = NULL,
        NET::Message_logger* logger = NULL,
        bool is_allowed_to_own_data = true,
        bool send_elements_only_to_owners = false,
        const char* disk_cache_path = NULL,
        int max_journal_size = 10000,
        bool track_memory_usage = false);

    /// Prepare closing the database
    ///
    /// This call stops the garbage collection and offloading callbacks. This is necessary because
    /// the exit() method of higher level modules often assume that no other thread is using the
    /// module anymore.
    virtual void prepare_close() = 0;

    /// Close the database
    virtual void close() = 0;

    /// Do a synchronous garbage collection sweep
    virtual void garbage_collection() = 0;

    /// The database always contains a global scope. The global scope is the root of a tree of
    /// scopes. This function is used to get the global scope so that child scopes can be created
    /// etc.
    ///
    /// \return                                 The global scope
    virtual Scope *get_global_scope() = 0;

    /// In some applications the application needs to lookup a certain scope. This has to be
    /// possible from every host in the whole system. Consider an application where a scope
    /// addresses an arena (e.g. a chess game). The system presents a page to the user listing all
    /// available arenas. The user selects one of the arenas. Now the application will create a new
    /// scope which is a child of the arena. Thus it must be possible for each host to address the
    /// arena scope. The application would store the scope id in the database and later it would use
    /// the id to lookup the scope.
    ///
    /// \param id                               The id of the scope to lookup.
    /// \return                                 The found scope
    virtual Scope *lookup_scope(
        Scope_id id) = 0;

    /// Lookup a named scope.
    ///
    /// \param name                             The name of the scope to lookup.
    /// \return                                 The found scope or NULL if it was not found.
    virtual Scope *lookup_scope(
        const std::string& name) = 0;

    /// Remove a scope from the database. Return true, if succeeded, otherwise false.
    ///
    /// \param id                       The id of the scope to remove
    /// \return                         A bool indicating success, true, or failure, false
    virtual bool remove(Scope_id id) = 0;

    /// Lock a tag making it inaccessible to others. Note that this is a kind of cooperative lock:
    /// It will not stop others from editing the tag. It will only prevent them from obtaining a
    /// lock on the same tag.
    ///
    /// \param tag                              Lock this tag
    virtual void lock(
        Tag tag) = 0;

    /// Unlock a tag previously locked from the same context.
    ///
    /// \param tag                              Unlock this tag
    /// \return                                 A bool indicating success, true, or failure, false
    virtual bool unlock(
        Tag tag) = 0;

    /// Check, if a tag is locked. In the debug version, abort if not. In the release version, just
    /// print an error.
    ///
    /// \param tag                              Check this tag
    virtual void check_is_locked(
        Tag tag) = 0;

    /// Wait for a notification for a certain tag.
    /// Note that this can return although the tag was not actually notified. This will be the case,
    /// when a host leaves to avoid lost notifications blocking threads forever.
    /// The return value true means, that the notification arrived, false means, that a timeout
    /// occurred.
    ///
    /// \param tag                              Tag to wait for
    /// \param timeout                          Return after the timeout, -1 means never
    /// \return                                 See description.
    virtual bool wait_for_notify(
        Tag tag,
        TIME::Time timeout = TIME::Time(-1)) = 0;

    /// Send a notification for a certain tag. This will wakeup all threads in the network waiting
    /// for a notification on that tag.
    ///
    /// \param tag                              The tag to send a notification for
    virtual void notify(
        Tag tag) = 0;

    /// Get the next name following the given name. This can be used to iterate throw the names
    /// stored in the database. This is relatively slow, since it requires a map lookup for each
    /// call.
    ///
    /// \param pattern                          The pattern for searching the names
    virtual std::string get_next_name(
        const std::string& pattern) = 0;

    /// Get the database statistics
    ///
    /// \return                                 The gathered statistics
    virtual Database_statistics get_statistics() = 0;

    /// Get the database status
    ///
    /// \return status
    virtual Db_status get_database_status() = 0;

    /// Register a listener to be notified when the status changes.
    ///
    /// Note that calls to register and unregister a listener of this type must not 
    /// be made from the callback since this will result in a deadlock.
    ///
    /// \param listener                         The new listener.
    virtual void register_status_listener(IStatus_listener* listener) = 0;

    /// Unregister a previously registered status listener
    ///
    /// Note that calls to register and unregister a listener of this type must not 
    /// be made from the callback since this will result in a deadlock.
    ///
    /// \param listener                         The old listener.
    virtual void unregister_status_listener(IStatus_listener* listener) = 0;

    /// Register a listener to be notified for transaction events.
    ///
    /// Note that calls to register and unregister a listener of this type must not 
    /// be made from the callback since this will result in a deadlock.
    ///
    /// \param listener                         The new listener.
    virtual void register_transaction_listener(ITransaction_listener* listener) = 0;

    /// Unregister a previously registered transaction listener
    ///
    /// Note that calls to register and unregister a listener of this type must not 
    /// be made from the callback since this will result in a deadlock.
    ///
    /// \param listener                         The old listener.
    virtual void unregister_transaction_listener(ITransaction_listener* listener) = 0;

    /// Register a listener to be notified for transaction events.
    ///
    /// Note that calls to register and unregister a listener of this type must not 
    /// be made from the callback since this will result in a deadlock.
    ///
    /// \param listener                         The new listener.
    virtual void register_scope_listener(IScope_listener* listener) = 0;

    /// Unregister a previously registered transaction listener
    ///
    /// Note that calls to register and unregister a listener of this type must not 
    /// be made from the callback since this will result in a deadlock.
    ///
    /// \param listener                         The old listener.
    virtual void unregister_scope_listener(IScope_listener* listener) = 0;
    

    /// Sets the limits for memory usage of the database.
    virtual Sint32 set_memory_limits(size_t low_water, size_t high_water) = 0;
        
    /// Returns the limits for memory usage of the database.
    virtual void get_memory_limits(size_t& low_water, size_t& high_water) const = 0;

  //
  // The functions below may only be used by DATA!!!!
  //

    /// Set an event to be signalled when the database is completely ready.
    ///
    /// \param event                            The even to signal when the database is ready
    virtual void set_ready_event(
        EVENT::Event0_base* event) = 0;

  //
  // The functions below may only be used by SCHED!!!!
  //

    /// This is to be used by the sched module only. It will return the requested transaction. It
    /// may return NULL, if the transaction is already committed.
    ///
    /// \param id                               The id of the transaction to be retrieved.
    /// \return                                 The transaction or NULL
    virtual Transaction* get_transaction(
        Transaction_id id) = 0;

// currently unused, prevent accidential use
private:
  //
  // The function below are experimental and should not be used in production code
  //

    // Cancel all fragmented jobs in the datbase
    virtual void cancel_all_fragmented_jobs() = 0;

public:
    /// Execute a job splitting it in a given number of fragments. This will not return before all
    /// fragments have been executed. The fragments may be executed in any number of threads and if
    /// the job allows on any number of hosts.
    ///
    /// \param job                      The fragmented job to be executed.
    /// \param count                    The number of fragments this job should be split into. This
    ///                                 number must be greater than zero.
    /// \return
    ///                                 -  0: Success.
    ///                                 - -1: Invalid parameters (\p job is \c NULL or \c count is
    ///                                       zero).
    ///                                 - -2: Invalid scheduling mode (transaction-less or
    ///                                       asynchronous execution is retricted to local jobs).
    virtual Sint32 execute_fragmented(
        Fragmented_job* job,
        size_t count) = 0;  

    /// Execute a job splitting it in a given number of fragments. This will return immediately,
    /// typically before all fragments have been executed. The fragments may be executed in any
    /// number of threads.
    /// NOTE: Currently this is restricted to local host only jobs! This restriction might be
    ///       lifted in the future.
    ///
    /// \param job                      The fragmented job to be executed.
    /// \param count                    The number of fragments this job should be split into. This
    ///                                 number must be greater than zero.
    /// \param listener                 Provides a callback to be called when the job is done.
    /// \return
    ///                                 -  0: Success.
    ///                                 - -1: Invalid parameters (\p job is \c NULL or \c count is
    ///                                       zero).
    ///                                 - -2: Invalid scheduling mode (transaction-less or
    ///                                       asynchronous execution is retricted to local jobs).
    virtual Sint32 execute_fragmented_async(
        Fragmented_job* job,
        size_t count,
        IExecution_listener* listener) = 0;

    /// See THREAD_POOL::Thread_pool::suspend_current_job().
    virtual void suspend_current_job() = 0;

    /// See THREAD_POOL::Thread_pool::resume_current_job().
    virtual void resume_current_job() = 0;

    /// See THREAD_POOL::Thread_pool::yield().
    virtual void yield() = 0;

  protected:
    /// The destructor is protected because destroying the database is not part
    /// of the public interface.
    virtual ~Database() { }
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_DATABASE_H

