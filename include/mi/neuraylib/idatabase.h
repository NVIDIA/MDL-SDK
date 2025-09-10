/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief API component that provides access to the database.

#ifndef MI_NEURAYLIB_IDATABASE_H
#define MI_NEURAYLIB_IDATABASE_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/version.h> // for MI_NEURAYLIB_DEPRECATED_ENUM_VALUE

namespace mi {

namespace neuraylib {

class IScope;

/** \defgroup mi_neuray_database_access Database Access
    \ingroup mi_neuray

    Provides access to the database using concepts like scopes and transactions.
*/

/** \addtogroup mi_neuray_database_access
@{

\section mi_neuray_database_limitations Database limitations

The database does not support certain usage patterns. These patterns can not be rejected
programmatically, but need to be ensured by the user for proper operation.

\ifnot DICE_API
\note This section mentions some DB internals, e.g. \c tags, which are not further explained here.
\endif


\subsection mi_neuray_database_scopes Identical names due to different scopes

See #mi::neuraylib::IScope for general documentation about scopes.

Be careful when \em storing DB elements of the same name in different scopes. If the scopes are not
in a parent-child relation, then the following limitation does not apply since no transaction will
ever see both elements. But if the scopes are in a parent-child relation, then it is required that
the store operation happens first in the parent scope and that this element is visible in the child
scope before the store operation in that child scope occurs (in the same transaction, or with
different transactions where the first one is committed before the second one is started).
Otherwise, this results in different DB elements of the same name (and not just different versions
of the same DB elements as it would happen when the correct order is observed).

\ifnot DICE_API
Example (creation in wrong order):
\code
mi::base::Handle<mi::neuraylib::IDatabase> database(
    neuray->get_api_component<mi::neuraylib::IDatabase>());

mi::base::Handle<mi::neuraylib::IScope> parent_scope( database->get_global_scope());

mi::base::Handle<mi::neuraylib::IScope> child_scope(
    database->create_scope( parent_scope.get()));

// DB element with name "texture" is created first in the child scope ...
mi::base::Handle<mi::neuraylib::ITransaction> child_transaction(
    child_scope->create_transaction());
{
    mi::base::Handle<mi::neuraylib::ITexture> texture(
        child_transaction->create<mi::neuraylib::ITexture>( "Texture"));
    child_transaction->store( texture.get(), "texture");
}
child_transaction->commit();

mi::base::Handle<mi::neuraylib::ITransaction> parent_transaction(
    parent_scope->create_transaction());
{
    mi::base::Handle<mi::neuraylib::ITexture> texture(
        parent_transaction->create<mi::neuraylib::ITexture>( "Texture"));
    // ... and is not visible here.
    parent_transaction->store( texture.get(), "texture");
}
parent_transaction->commit();

// The name "texture" maps now to two different tags. Both tags are visible from the child scope via
// tag references (not part of this example), but only one of them via name.
\endcode
\endif

While such a situation is not necessarily a problem for the database itself, it leads to unexpected
behavior on the user side, as accesses might return different instances, depending on the details
of the access method.

This limitation does not apply if there is already a DB element with that name accessible at the
time of the store operation, i.e., the method does not create a new DB element, but essentially
overwrites/edits an existing one.


\subsection mi_neuray_database_transactions Identical names due to parallel transactions

See \if DICE_API #mi::neuraylib::IDice_transaction \else #mi::neuraylib::ITransaction \endif for
general documentation about transactions.

Be careful when \em storing DB elements of the same name in parallel transactions, unless the
corresponding scopes are different and not in a parent-child relation to each other. Failure to
observe this limitation results in different DB elements of the same name (and not just different
versions of the same DB elements as it would happen with serialized transactions).

\ifnot DICE_API
Example (wrong creation in parallel transactions):
\code
mi::base::Handle<mi::neuraylib::IDatabase> database(
     neuray->get_api_component<mi::neuraylib::IDatabase>());

mi::base::Handle<mi::neuraylib::IScope> scope( database->get_global_scope());

mi::base::Handle<mi::neuraylib::ITransaction> transaction1(
    scope->create_transaction());
mi::base::Handle<mi::neuraylib::ITransaction> transaction2(
    scope->create_transaction());

{
    mi::base::Handle<mi::neuraylib::ITexture> texture(
        transaction1->create<mi::neuraylib::ITexture>( "Texture"));
    transaction1->store( texture.get(), "texture");
}

{
    mi::base::Handle<mi::neuraylib::ITexture> texture(
        transaction2->create<mi::neuraylib::ITexture>( "Texture"));
    transaction2->store( texture.get(), "texture");
}

transaction1->commit();
transaction2->commit();

// The name "texture" maps now to two different tags. Both tags are visible via tag references from
// the global scope (not part of this example), but only one of them via name. \endcode
\endif

While such a situation is not necessarily a problem for the database itself, it leads to unexpected
behavior on the user side, as accesses might return different instances, depending on the details
of the access method.

This limitation does not apply if there is already a DB element with that name accessible at the
time of the store operations, i.e., the method does not create new DB elements, but essentially
overwrites/edits an existing one.

Note that \em editing (as opposed to storing) the very same DB element in parallel transactions is
supported by the database, but it is discouraged, since the semantics might not be as desired (see
\if DICE_API #mi::neuraylib::IDice_transaction \else #mi::neuraylib::ITransaction \endif).


\subsection mi_neuray_database_references References to elements in more private scopes

See #mi::neuraylib::IScope for general documentation about scopes.

Be careful when creating references to DB elements that exist in a different scope than the
referencing element. References to elements in parent scopes (or the the same scope) are perfectly
fine. But you must not create references to DB elements that exist only in a more private scope.
Typically, this happens when using \if DICE_API #mi::neuraylib::IDice_transaction::store() \else
#mi::neuraylib::ITransaction::store() \endif with an (explicit) wrong privacy level.

\ifnot DICE_API
Example (invalid reference to element in more private scope):
\code
mi::base::Handle<mi::neuraylib::IDatabase> database(
    neuray->get_api_component<mi::neuraylib::IDatabase>());

mi::base::Handle<mi::neuraylib::IScope> parent_scope( database->get_global_scope());

mi::base::Handle<mi::neuraylib::IScope> child_scope(
    database->create_scope( parent_scope.get()));

mi::base::Handle<mi::neuraylib::ITransaction> child_transaction(
    child_scope->create_transaction());
{
    mi::base::Handle<mi::neuraylib::IImage> image(
        child_transaction->create<mi::neuraylib::IImage>( "Image"));
    check_success( 0 == child_transaction->store( image.get(), "image"));
    mi::base::Handle<mi::neuraylib::ITexture> texture(
        child_transaction->create<mi::neuraylib::ITexture>( "Texture"));
    check_success( 0 == texture->set_image( "image"));
    // Triggers an error message since "texture" is to be stored in the parent scope (due to
    // explicit privacy level 0), but references "image" in the child scope.
    check_success( 0 == child_transaction->store( texture.get(), "texture", 0));
}
child_transaction->commit();

mi::base::Handle<mi::neuraylib::ITransaction> parent_transaction(
    parent_scope->create_transaction());
{
    mi::base::Handle<const mi::neuraylib::ITexture> texture(
        parent_transaction->access<mi::neuraylib::ITexture>( "texture"));
    // Triggers a fatal message about an invalid tag access.
    const char name = texture->get_image();
}
parent_transaction->commit();
\endcode
\endif

A reference to an element in a more private scope triggers an error message when the referencing
element is stored in the DB, but does not prevent the operation from being completed nor is it
signaled via a return code. As soon as the incorrect reference is used, this triggers a fatal error
message (and the process aborts if it returns from the logger callback). Even if the incorrect
reference is never used, its existence hints at a conceptual error in the way the application uses
scopes.

*/

/// This interface is used to interact with the distributed database.
class IDatabase : public
    mi::base::Interface_declare<0x814ae637,0xde35,0x4870,0x8e,0x5b,0x7e,0x28,0x9d,0x30,0xfb,0x82>
{
public:
    /// Returns the global scope which is the root of a tree of scopes
    ///
    /// \return  The global scope which is guaranteed to exist after startup of the
    ///          system.
    virtual IScope* get_global_scope() const = 0;

    /// Creates a new unnamed scope.
    ///
    /// \note A scope continues to exist if the pointer returned by this method is released. Use
    ///       #remove_scope() to remove a scope.
    ///
    /// \param parent         The parent scope for this scope. The value \c nullptr represents the
    ///                       global scope.
    /// \param privacy_level  The privacy level of the scope, which must be higher than the privacy
    ///                       level of the parent scope. The privacy level of the global scope is 0
    ///                       (and the global scope is the only scope with privacy level 0). The
    ///                       default value of 0 indicates the privacy level of the parent scope
    ///                       plus 1.
    /// \param temp           \ifnot MDL_SDK_API A flag indicating whether the scope is temporary.
    ///                       A non-temporary scope needs to be explicitly removed via
    ///                       #remove_scope(). A temporary scope is automatically removed when the
    ///                       host creating it leaves the cluster. \else Unused. \endif
    /// \return               The created scope, or \c nullptr in case of errors.
    virtual IScope* create_scope( IScope* parent, Uint8 privacy_level = 0, bool temp = false) = 0;

    /// Creates a new named scope (or retrieves an existing one).
    ///
    /// \param name           The name of the new scope. If there is no scope with that name, it
    ///                       will be created with the given parent scope and privacy level.
    ///                       Otherwise, if the given parent scope and privacy level match the
    ///                       properties of an existing scope with that name, then that scope will
    ///                       be returned. If there is a mismatch, neither a scope will be created
    ///                       nor returned, and the method returns \c nullptr.
    /// \param parent         The parent scope for this scope. The value \c nullptr represents the
    ///                       global scope.
    /// \param privacy_level  The privacy level of the scope, which must be higher than the privacy
    ///                       level of the parent scope. The privacy level of the global scope is 0
    ///                       (and the global scope is the only scope with privacy level 0). The
    ///                       default value of 0 indicates the privacy level of the parent scope
    ///                       plus 1.
    /// \return               The created scope, or \c nullptr in case of errors.
    virtual IScope* create_or_get_named_scope(
        const char* name, IScope* parent = nullptr, Uint8 privacy_level = 0) = 0;

    /// Looks up a scope by ID.
    ///
    /// \param id             The ID of the scope as returned by #mi::neuraylib::IScope::get_id().
    /// \return               The found scope or \c nullptr if no such scope exists.
    virtual IScope* get_scope( const char* id) const = 0;

    /// Looks up a scope by name.
    ///
    /// \param name           The name of the scope
    /// \return               The found scope or \c nullptr if no such scope exists.
    virtual IScope* get_named_scope( const char* name) const = 0;

    /// Removes a scope with the specified ID.
    ///
    /// Note that scopes are reference counted. The actual removal will not happen before all
    /// elements referencing the scope have been released, e.g., child scopes, transactions,
    /// database elements, including handles to the scope itself. Even when all these conditions
    /// are met, scope removal might actually happen at a later point in time, depending on the
    /// timing of past and current transactions, even in unrelated scopes.
    ///
    /// It is not possible to remove the global scope.
    ///
    /// \param id             The ID of the scope as returned by #mi::neuraylib::IScope::get_id().
    /// \return               0, in case of success, -1 in case of failure.
    virtual Sint32 remove_scope( const char* id) const = 0;

    /// Priorities for synchronous garbage collection runs.
    enum Garbage_collection_priority : Uint32 {

        /// Low priority for synchronous garbage collection runs. Use this priority if the
        /// performance of other concurrent DB operations is more important than a fast synchronous
        /// garbage collection.
        PRIORITY_LOW = 0,

        /// Medium priority for synchronous garbage collection runs. This priority attempts to
        /// maintain a balance between the synchronous garbage collection and other concurrent DB
        /// operations.
        PRIORITY_MEDIUM = 1,

        /// High priority for synchronous garbage collection runs. Other concurrent DB operations
        /// will experience a large performance drop. Therefore, this priority should not be used
        /// in multi-user settings.
        PRIORITY_HIGH = 2
        MI_NEURAYLIB_DEPRECATED_ENUM_VALUE(PRIORITY_FORCE_32_BIT, 0xffffffffU)
    };

    /// Triggers a synchronous garbage collection run.
    ///
    /// The method sweeps through the entire database and removes all database elements which have
    /// been marked for removal and are no longer referenced. Note that it is not possible to remove
    /// database elements if there are open transactions in which such an element is still
    /// referenced.
    ///
    /// To mark an element for removal use \ifnot DICE_API #mi::neuraylib::ITransaction::remove().
    /// \else #mi::neuraylib::IDice_transaction::remove() or
    /// #mi::neuraylib::IDice_transaction::store_for_reference_counting(). \endif
    ///
    /// \param priority   The intended priority of the synchronous garbage collection run.
    ///                   \if MDL_SDK_API The MDL SDK does not support different priorities, and
    ///                   the synchronous garbage collection always runs at highest priority.
    ///                   \endif
    virtual void garbage_collection( Garbage_collection_priority priority = PRIORITY_MEDIUM) = 0;

    /// Acquires a DB lock.
    ///
    /// The method blocks until the requested lock has been obtained. Recursively locking the
    /// same lock from within the same thread \ifnot MDL_SDK_API on the same host \endif is
    /// supported.
    ///
    /// \ifnot MDL_SDK_API
    /// If the host holding a lock leaves the cluster, the lock is automatically released.
    /// \endif
    ///
    /// \param lock_id   The lock to acquire.
    ///
    /// \note The locking mechanism is kind of a co-operative locking mechanism: The lock does not
    ///       prevent other threads from accessing or editing the DB. It only prevents other threads
    ///       from obtaining the same lock.
    ///
    /// \ifnot MDL_SDK_API
    /// \note DB locks are not restricted to threads on a single host, they apply to all threads on
    ///       all hosts in the cluster.
    /// \endif
    ///
    /// \note DB locks are an expensive operation and should only be used when absolutely necessary.
    virtual void lock( Uint32 lock_id) = 0;

    /// Releases a previously obtained DB lock.
    ///
    /// If the lock has been locked several times from within the same \ifnot MDL_SDK_API thread on
    /// the same host, \else thread, \endif it simply decrements the lock count. If the lock count
    /// reaches zero, the lock is released.
    ///
    /// \param lock_id   The lock to release.
    /// \return          0, in case of success, -1 in case of failure, i.e., the lock is not held
    ///                  by this \ifnot MDL_SDK_API thread on this host. \else thread. \endif
    virtual Sint32 unlock( Uint32 lock_id) = 0;
};

/**@}*/ // end group mi_neuray_database_access

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IDATABASE_H
