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
/// \brief Database transactions.

#ifndef MI_NEURAYLIB_ITRANSACTION_H
#define MI_NEURAYLIB_ITRANSACTION_H

#include <mi/base/interface_declare.h>
#include <mi/neuraylib/type_traits.h>
#include <mi/neuraylib/version.h>

namespace mi {

class IArray;

namespace neuraylib {

class IScope;

/** \addtogroup mi_neuray_database_access
@{
*/

/// A transaction provides a consistent view on the database.
///
/// This view on the database is isolated from changes by other (parallel) transactions. Eventually,
/// each transaction must be either committed or aborted, i.e., all changes become either atomically
/// visible to transactions started afterwards, or not at all.
///
/// Transactions are associated with a scope of the database and can be created with
/// #mi::neuraylib::IScope::create_transaction().
///
/// Transactions are not thread-safe. If you use a particular transaction from multiple threads,
/// then you have to serialize all transaction uses. This does not only apply to methods of
/// #mi::neuraylib::ITransaction, but all methods that implicitly use the transaction. For example,
/// such a use can happen by methods of DB elements returned from #access() or #edit() calls, or by
/// objects returned from factories taking the transaction as an argument, like
/// #mi::neuraylib::IMdl_factory::create_module_builder().
///
/// \par Concurrent accesses to database elements within a transaction
///
/// Access to database elements is provided by #access() (read-only) and #edit() (for modification).
/// The interface pointers returned by these methods must be released when you are done, in
/// particular before the transaction is committed or aborted.
/// \par
/// Note that it is possible to access the same database element concurrently in the same
/// transaction. Concurrently means that the interface pointer obtained from an earlier #access()
/// or #edit() call has not yet been released and the same database element is accessed once more
/// using #access() or #edit(). It is advisable to avoid such concurrent accesses since it can
/// easily lead to difficult to understand effects. The semantics are as follows:
/// <ul>
/// <li> multiple #access() calls: Since all obtained interface pointers are const there is no
///      way to modify the database elements.</li>
/// <li> #access() call after #edit() calls: The interface pointer returned from #access() reflects
///      the changes as they are done via the interface pointer returned from the last #edit() call.
///      \if IRAY_API Note that this use case is not supported for user-defined classes (classes
///      derived from #mi::neuraylib::IUser_class).\endif </li>
/// <li> #edit() call after #access() calls: The changes done via the interface pointer returned
///      from #edit() are not observable through any interface pointer returned from the #access()
///      calls.</li>
/// <li> multiple #edit() calls: The changes done via the individual interface pointers are not
///      observable through the other interface pointers, except for the changes done via the
///      interface pointer obtained first at the time the second interface pointer is obtained. The
///      changes from the interface pointer from the last #edit() call survive, independent of the
///      order in which the pointers are released.</li>
/// </ul>
/// \par
/// Note that these semantics do not only apply to #access() and #edit() calls. They also apply
/// to other API methods that access other database elements, e.g., #mi::IRef::get_reference(),
/// which internally calls #access().
///
/// \par Concurrent transactions
///
/// If the same database element is edited in multiple overlapping transactions, the changes from
/// the transaction created last survive, independent of the order in which the transactions are
/// committed. \if IRAY_API If needed, the lifetime of transactions can be serialized across
/// hosts (see #mi::neuraylib::IDatabase::lock() for details). \endif
class ITransaction : public
    mi::base::Interface_declare<0x6ca1f0c2,0xb262,0x4f09,0xa6,0xa5,0x05,0xae,0x14,0x45,0xed,0xfa>
{
public:
    /// Symbolic privacy level for the privacy level of the scope of this transaction.
    ///
    /// This symbolic constant can be passed to #store() and #copy() to indicate the privacy level
    /// of the scope of this transaction. It has the same affect as passing the result of
    /// #mi::neuraylib::IScope::get_privacy_level(), but is more convenient.
    static const mi::Uint8 LOCAL_SCOPE = 255;

    /// \name Static properties
    //@{

    /// Returns the ID of this transaction.
    ///
    /// The transaction ID is of most use when debugging an application as the value returned allows
    /// one to correlate log messages and admin HTTP server output with the API actions.
    ///
    /// \return            The ID of the transaction.
    virtual const char* get_id() const = 0;

    /// Returns the scope of this transaction.
    virtual IScope* get_scope() const = 0;

    //@}
    /// \name Committing and aborting
    //@{

    /// Indicates whether the transaction is open.
    ///
    /// \return   \c true if the transaction is still open, or \c false if the transaction is
    ///           closed, i.e., it has been committed or aborted.
    virtual bool is_open() const = 0;

    /// Commits the transaction.
    ///
    /// Note that a commit() implicitly closes the transaction.
    /// A closed transaction does not allow any future operations and needs to be released.
    ///
    /// \return
    ///                     -  0: Success.
    ///                     - -1: Unspecified failure.
    ///                     - -3: The transaction is not open.
    virtual Sint32 commit() = 0;

    /// Aborts the transaction.
    ///
    /// Note that an abort() implicitly closes the transaction.
    /// A closed transaction does not allow any future operations and needs to be released.
    virtual void abort() = 0;

    //@}
    /// \name Creation of DB elements
    //@{

    /// Creates an object of the type \p type_name.
    ///
    /// Objects created with this method are typically \if IRAY_API \ref mi_neuray_types,
    /// \ref mi_neuray_scene_element and \ref mi_neuray_functors. It is also possible to create
    /// instances of user-defined classes. \else \ref mi_neuray_types and
    /// \ref mi_neuray_scene_element. \endif Note that most types can also be created via the API
    /// component #mi::neuraylib::IFactory which does not require the context of a transaction.
    ///
    /// This method can not be used to create MDL modules, definitions, or function calls. To
    /// create instances of #mi::neuraylib::IFunction_call, use the method
    /// #mi::neuraylib::IFunction_definition::create_function_call().
    ///
    /// The created object will be initialized in a manner dependent upon the passed type
    /// name. Each class has its own policy on initialization. So, one should not make any
    /// assumptions on the values of the various class members.
    ///
    /// \param type_name    The type name of the object to create. See \ref mi_neuray_types for
    ///                     possible type names. In addition, \ref mi_neuray_scene_element
    ///                     \if IRAY_API and \ref mi_neuray_functors \endif can be created by
    ///                     passing the name of the requested interfaces without namespace
    ///                     qualifiers and the leading \c "I", e.g., \c "Image" for
    ///                     #mi::neuraylib::IImage. \if IRAY_API Names of user-defined classes are
    ///                     also valid arguments. \endif Note that you can not create instances of
    ///                     #mi::neuraylib::IAttribute_set or #mi::neuraylib::IScene_element, only
    ///                     instances of the derived interfaces. \if IRAY_API (See also
    ///                     #mi::neuraylib::IAttribute_container). \endif
    /// \param argc         The number of elements in \p argv. Passed to the constructor of factory
    ///                     of the object to create.
    /// \param argv         The array of arguments passed to the constructor. Passed to the
    ///                     constructor of factory of the object to create.
    /// \return             A pointer to the created object, or \c nullptr if \p type_name is
    ///                     invalid (\c nullptr) or not a valid type name.
    virtual base::IInterface* create(
        const char* type_name,
        Uint32 argc = 0,
        const base::IInterface* argv[] = nullptr) = 0;

    /// Creates an object of the type \p type_name.
    ///
    /// Objects created with this method are typically \if IRAY_API \ref mi_neuray_types,
    /// \ref mi_neuray_scene_element and \ref mi_neuray_functors. It is also possible to create
    /// instances of user-defined classes. \else \ref mi_neuray_types and
    /// \ref mi_neuray_scene_element. \endif Note that most types can also be created via the API
    /// component #mi::neuraylib::IFactory which does not require the context of a transaction.
    ///
    /// This method can not be used to create MDL modules, definitions, or function calls. To
    /// create instances of #mi::neuraylib::IFunction_call, use the method
    /// #mi::neuraylib::IFunction_definition::create_function_call().
    ///
    /// The created object will be initialized in a manner dependent upon the passed type name. Each
    /// class has its own policy on initialization. So, one should not make any assumptions on the
    /// values of the various class members.
    ///
    /// Note that there are two versions of this templated member function, one that takes no
    /// arguments, and another one that takes one or three arguments (the type name, and two
    /// optional arguments passed to the constructor or factory). The version with no arguments can
    /// only be used to create a subset of supported types derived from #mi::IData: it supports only
    /// those types where the type name can be deduced from the template parameter, i.e., it does
    /// not support arrays, structures, maps, and pointers. The version with one or three arguments
    /// can be used to create any type (but requires the type name as parameter, which is redundant
    /// for many types). Attempts to use the version with no arguments with a template parameter
    /// where the type name can not be deduced results in compiler errors.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid&)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \param type_name    The type name of the object to create. See \ref mi_neuray_types for
    ///                     possible type names. In addition, \ref mi_neuray_scene_element
    ///                     \if IRAY_API and \ref mi_neuray_functors \endif can be created by
    ///                     passing the name of the requested interfaces without namespace
    ///                     qualifiers and the leading \c "I", e.g., \c "Image" for
    ///                     #mi::neuraylib::IImage. \if IRAY_API Names of user-defined classes are
    ///                     also valid arguments. \endif Note that you can not create instances of
    ///                     #mi::neuraylib::IAttribute_set or #mi::neuraylib::IScene_element, only
    ///                     instances of the derived interfaces. \if IRAY_API (See also
    ///                     #mi::neuraylib::IAttribute_container). \endif
    /// \param argc         The number of elements in \p argv. Passed to the constructor of factory
    ///                     of the object to create.
    /// \param argv         The array of arguments passed to the constructor. Passed to the
    ///                     constructor of factory of the object to create.
    /// \tparam T           The interface type of the class to create.
    /// \return             A pointer to the created object, or \c nullptr if \p type_name is
    ///                     invalid (\c nullptr), not a valid type name, or does not create an
    ///                     object of type \c T.
    template<class T>
    T* create(
        const char* type_name,
        Uint32 argc = 0,
        const base::IInterface* argv[] = nullptr)
    {
        base::IInterface* ptr_iinterface = create( type_name, argc, argv);
        if( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Creates an object of the type \p T.
    ///
    /// Objects created with this method are typically \if IRAY_API \ref mi_neuray_types,
    /// \ref mi_neuray_scene_element and \ref mi_neuray_functors. It is also possible to create
    /// instances of user-defined classes. \else \ref mi_neuray_types and
    /// \ref mi_neuray_scene_element. \endif Note that most types can also be created via the API
    /// component #mi::neuraylib::IFactory which does not require the context of a transaction.
    ///
    /// This method can not be used to create MDL modules, definitions, or function calls. To
    /// create instances of #mi::neuraylib::IFunction_call, use the method
    /// #mi::neuraylib::IFunction_definition::create_function_call().
    ///
    /// The created object will be initialized in a manner dependent upon the passed type name. Each
    /// class has its own policy on initialization. So, one should not make any assumptions on the
    /// values of the various class members.
    ///
    /// Note that there are two versions of this templated member function, one that takes no
    /// arguments, and another one that takes one or three arguments (the type name, and two
    /// optional arguments passed to the constructor or factory). The version with no arguments can
    /// only be used to create a subset of supported types derived from #mi::IData: it supports only
    /// those types where the type name can be deduced from the template parameter, i.e., it does
    /// not support arrays, structures, maps, and pointers. The version with one or three arguments
    /// can be used to create any type (but requires the type name as parameter, which is redundant
    /// for many types). Attempts to use the version with no arguments with a template parameter
    /// where the type name can not be deduced results in compiler errors.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid&)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T           The interface type of the class to create.
    /// \return             A pointer to the created object.
    template<class T>
    T* create()
    {
        return create<T>( Type_traits<T>::get_type_name());
    }

    //@}

    /// \name Storing of DB elements
    //@{

    /// Stores the element \p db_element in the database under the name \p name and with the privacy
    /// level \p privacy.
    ///
    /// After a successful store operation the passed interface pointer must no longer be used,
    /// except for releasing it. This is due to the fact that after a #store() the database
    /// retains ownership of the stored data. You can obtain the stored version from the database
    /// using the #access() or #edit() methods.
    ///
    /// \note <b>Overwriting vs editing of existing DB elements</b> \n
    ///       While it is possible to overwrite existing DB elements, for performance reasons it is
    ///       often better to edit the already existing DB element instead. Editing a DB element
    ///       allows the DB to keep track of the type of changes which might help render modes to
    ///       update their data structures more efficiently. When overwriting an existing DB element
    ///       such information is not available and pessimistic assumptions have to be made which
    ///       may result in lower performance.
    ///
    /// \param db_element The #mi::base::IInterface to store.
    /// \param name       The name under which to store \p db_element. If there exists already a DB
    ///                   element with that name then it will be overwritten (but see also return
    ///                   code -9 below),
    /// \param privacy    The privacy level under which to store \p db_element (in the range from 0
    ///                   to the privacy level of the scope of this transaction). In addition, the
    ///                   constant #LOCAL_SCOPE can be used as a shortcut to indicate the privacy
    ///                   level of the scope of this transaction without supplying the actual value
    ///                   itself.
    /// \return
    ///        -  0: Success.
    ///        - -1: Unspecified failure.
    ///        - -2: Invalid parameters (\c nullptr).
    ///        - -3: The transaction is not open.
    ///        - -4: \p db_element is not a DB element.
    ///        - -5: Invalid privacy level.
    ///        - -6: \p db_element has already been stored previously.
    ///        - -7: The element is to be stored in a transaction different from the one that was
    ///              used to create it.
    ///        - -8: The element is a user-defined class that has not been \if IRAY_API
    ///              registered (see #mi::neuraylib::IExtension_api::register_class()). \else
    ///              registered. \endif
    ///
    ///        - -9: There is already an element of name \p name and overwriting elements of that
    ///              type is not supported. This applies to elements of type
    ///              #mi::neuraylib::IModule and #mi::neuraylib::IFunction_definition. It also
    ///              applies to elements of type #mi::neuraylib::IFunction_call that are used as
    ///              defaults in an #mi::neuraylib::IFunction_definition.
    virtual Sint32 store(
        base::IInterface* db_element, const char* name, Uint8 privacy = LOCAL_SCOPE) = 0;

    //@}
    /// \name Accessing and editing DB elements
    //@{

    /// Retrieves an element from the database.
    ///
    /// The database searches for the most recent version of the named DB element visible for the
    /// current transaction. That version will be returned.
    ///
    /// \param name   The name of the element to retrieve.
    /// \return       The requested element from the database, or \c nullptr if \p name is invalid,
    ///               no DB element with that name exists, or the transaction is already closed.
    virtual const base::IInterface* access( const char* name) = 0;

    /// Retrieves an element from the database.
    ///
    /// The database searches for the most recent version of the named DB element visible for the
    /// current transaction. That version will be returned.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid&)
    /// on the returned pointer, since the return type already is a const pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \param name   The name of the element to retrieve.
    /// \tparam T     The interface type of the element to retrieve.
    /// \return       The requested element from the database, or \c nullptr if \p name is invalid,
    ///               no DB element with that name exists, the transaction is already closed, or
    ///               the element is not of type \c T.
    template<class T>
    const T* access( const char* name)
    {
        const base::IInterface* ptr_iinterface = access( name);
        if( !ptr_iinterface)
            return nullptr;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Retrieves an element from the database and returns it ready for editing.
    ///
    /// The database searches for the most recent version of the named DB element visible for the
    /// current transaction. It will then make a copy of that version and return the copy. The
    /// edited DB element will be committed to the database automatically, when the obtained
    /// interface is released. It is neither necessary nor possible to store the edited element
    /// manually in the database using the #store() method.
    ///
    /// \param name   The name of the element to retrieve.
    /// \return       The requested element from the database, or \c nullptr if \p name is invalid,
    ///               no DB element with that name exists, or the transaction is already closed.
    virtual base::IInterface* edit( const char* name) = 0;

    /// Retrieves an element from the database and returns it ready for editing.
    ///
    /// The database searches for the most recent version of the named database element visible for
    /// the current transaction. It will then make a copy of that version and return the copy. The
    /// edited DB element will be committed to the database automatically, when the obtained
    /// interface is released. It is neither necessary nor possible to store the edited element
    /// manually in the database using the #store() method.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid&)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \param name   The name of the element to retrieve.
    /// \tparam T     The interface type of the element to retrieve.
    ///               no DB element with that name exists, the transaction is already closed, or
    ///               the element is not of type \c T.
    template<class T>
    T* edit( const char* name)
    {
        base::IInterface* ptr_iinterface = edit( name);
        if( !ptr_iinterface)
            return nullptr;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    //@}
    /// \name Copies, removal, and properties of DB elements
    //@{

    /// Creates a copy of a database element.
    ///
    /// Note that DB elements of type #mi::neuraylib::IModule and
    /// #mi::neuraylib::IFunction_definition can not be copied.
    ///
    /// \param source    The name of the element to be copied.
    /// \param target    The desired name of the copy.
    /// \param privacy   The desired privacy level of the copy (in the range from
    ///                  0 to the privacy level of the scope of this transaction). In addition, the
    ///                  constant #LOCAL_SCOPE can be used as a shortcut to indicate the privacy
    ///                  level of the scope of this transaction without supplying the actual value
    ///                  itself.
    /// \return
    ///                  -  0: Success.
    ///                  - -2: Invalid parameters (\c nullptr).
    ///                  - -3: The transaction is not open.
    ///                  - -4: There is no DB element named \p source visible in this transaction.
    ///                  - -5: Invalid privacy level.
    ///                  - -6: DB elements of this type cannot be copied.
    ///                  - -9: There is already an element of name \p name and overwriting elements
    ///                        of that type is not supported. This applies to elements of type
    ///                        #mi::neuraylib::IModule and #mi::neuraylib::IFunction_definition.
    ///                        It also applies to elements of type #mi::neuraylib::IFunction_call
    ///                        that are used as defaults in an #mi::neuraylib::IFunction_definition.
#ifdef MI_NEURAYLIB_DEPRECATED_ITRANSACTION_COPY_DEFAULT_PRIVACY_LEVEL_ZERO
    virtual Sint32 copy( const char* source, const char* target, Uint8 privacy = 0) = 0;
#else
    virtual Sint32 copy( const char* source, const char* target, Uint8 privacy = LOCAL_SCOPE) = 0;
#endif

    /// Marks the element with the name \p name for removal from the database.
    ///
    /// \par Global removals
    ///
    /// The purpose of global removals is to mark all versions of a database element for garbage
    /// collection. Such a marker has no effect while the element is still referenced (in any scope)
    /// by other elements or while the transaction where the removal request was made is still open.
    /// When these conditions do no longer apply, the element becomes eligible for garbage
    /// collection and must no longer be used in any way. There is no guarantee when the garbage
    /// collection will actually remove the element.
    ///
    /// This implies that a #remove() call might actually remove an element that was stored later
    /// under the same name. This can potentially lead to invalid tag accesses. Those cases can be
    /// avoided by using #mi::neuraylib::IDatabase::garbage_collection() after a transaction was
    /// committed and before starting the next one to force garbage collection of all possible
    /// elements.
    ///
    /// See also \ref mi_neuray_database_reuse_of_names for more details and correct usage patterns.
    ///
    /// \par Local removals
    ///
    /// The purpose of local removals is to undo the effects of an earlier localization via
    /// #mi::neuraylib::ITransaction::copy(). A local removal request requires that the element
    /// exists in the scope of the transaction, and at least one more version of that element
    /// exists in one of the parent scopes. The effect of a local removal request is to immediately
    /// hide the version the scope of the transaction (the \em local copy), and to make the next
    /// version in one of the parent scopes accessible from the very same transaction. The hidden
    /// local copy will be lazily removed by the garbage collection of the DB. There is no
    /// guarantee when this will happen.
    ///
    /// \param name                The name of the element in the database to mark for removal.
    /// \param only_localized      \c false for global removals (the default) or \c true for local
    ///                            removals. The flag is ignored in favor of global removals if the
    ///                            transaction belongs to the global scope.
    /// \return                    -  0: Success (including subsequent global removals on elements
    ///                                  already marked for global removal).
    ///                            - -1: For global removals: there is no DB element named \p name
    ///                                  visible in this transaction. For local removals: there is
    ///                                  no DB element named \p name in the scope of this
    ///                                  transaction or there is no version of that DB element in
    ///                                  one of the parent scopes.
    ///                            - -2: Invalid parameters (\c nullptr).
    ///                            - -3: The transaction is not open.
    virtual Sint32 remove( const char* name, bool only_localized = false) = 0;

    /// Returns the name of a database element.
    ///
    /// \param db_element   The DB element.
    /// \return             The name of the DB element, or \c nullptr if \p db_element is invalid
    ///                     (\c nullptr), the object is not in the database, or the
    ///                     transaction is already closed.
    virtual const char* name_of( const base::IInterface* db_element) const = 0;

    /// Returns the privacy level of the element with the name \p name.
    ///
    /// \param name          The name of the element.
    /// \return
    ///                      - >= 0: Success. The privacy level of the element (in the range 0-255).
    ///                      -   -2: Invalid parameters (\c nullptr).
    ///                      -   -3: The transaction is not open.
    ///                      -   -4: There is no DB element named \p name visible in this
    ///                              transaction.
    virtual Sint32 get_privacy_level( const char* name) const = 0;

    //@}
    /// \name Time stamps
    //@{

    /// Returns the time stamp describing the current "time".
    ///
    /// \note The time stamp is not related to time in the classical meaning. It rather relates to
    ///       the current transaction and the number of database changes since the start of the
    ///       transaction.
    ///
    /// \note The time stamp is only meaningful for the current \neurayAdjectiveName instance. It
    ///       should not be put into external storage and re-used for different or later
    ///       \neurayAdjectiveName instances.
    ///
    /// \note The return value of this method is only valid until the next call of this method
    ///       (or one of its overloads) on this instance.
    ///
    /// \see has_changed_since_time_stamp(), #get_time_stamp(const char*)const
    virtual const char* get_time_stamp() const = 0;

    /// Returns the time stamp of the last change of a database element.
    ///
    /// \note The time stamp is not related to time in the classical meaning. It rather relates to
    ///       the transaction and the number of database changes since the start of the transaction
    ///       when the database element was changed last.
    ///
    /// \note The time stamp is only meaningful for the current \neurayAdjectiveName instance. It
    ///       should not be put into external storage and re-used for different or later
    ///       \neurayAdjectiveName instances.
    ///
    /// \note The return value of this method is only valid until the next call of this method
    ///       (or one of its overloads) on this instance.
    ///
    /// \see has_changed_since_time_stamp(), #get_time_stamp()
    virtual const char* get_time_stamp( const char* element) const = 0;

    /// Checks whether an element has been stored or changed in the database since a given time
    /// stamp.
    ///
    /// \note \p time_stamp should not stem from another concurrent transaction. Such changes will
    ///       never be visible in this transaction, but the method might still return \c true
    ///       depending on the start order of the two transactions.
    ///
    /// \note In case of multiple overlapping transactions the returned answer may not list
    ///       all changes due to the isolation of the transactions. If accurate results are
    ///       required, transactions changing elements should be committed before transactions
    ///       querying the journal for such changes are started.
    ///
    /// \see #get_time_stamp(), #get_time_stamp(const char*)const
    ///
    /// \param element     The name of the element.
    /// \param time_stamp  The time stamp obtained from #get_time_stamp() or
    ///                    #get_time_stamp(const char*)const.
    /// \return            \c true if the element has been stored or changed since the time stamp
    ///                    (or if \p element or \p time_stamp is invalid, or there is no element
    ///                    with that name), \c false otherwise.
    virtual bool has_changed_since_time_stamp(
        const char* element, const char* time_stamp) const = 0;

    //@}
    /// \name Traversal
    //@{

    /// Returns scene elements of a subgraph originating at a given scene element.
    ///
    /// The method iterates over all elements of a subgraph originating at the given scene element
    /// and returns their names. Optionally, the results can be filtered by a regular expression
    /// for the element names and a list for type names.
    ///
    /// Note that the runtime of the method depends on the number of elements in the subgraph. It
    /// might be expensive to call this method for large subgraphs.
    ///
    /// The returned scene elements are in such an order that all elements referenced by a given
    /// element are listed before that element ("before" in the sense of "using smaller array
    /// indices").
    ///
    /// \param root_element   The root of the subgraph to traverse.
    /// \param name_pattern   An extended regular expression that acts as filter on the names of
    ///                       returned scene elements (see [\ref OGBS7]). The regular expression
    ///                       is matched to \em any \em part of the scene element name, not just to
    ///                       the \em entire scene element name. The value \c nullptr is handled as
    ///                       \c ".*".
    /// \param type_names     A list of type names that acts as filter on the names of returned
    ///                       scene elements. Only scene elements with a matching type name pass
    ///                       the filter. The value \c nullptr lets all scene elements pass the
    ///                       filter irrespective of their type name.
    /// \return               A list of name of scene elements in the subgraph matching the given
    ///                       regular expression and type name filter, or \c nullptr in case of an
    ///                       invalid root element name or an invalid regular expression.
    virtual IArray* list_elements(
        const char* root_element,
        const char* name_pattern = nullptr,
        const IArray* type_names = nullptr) const = 0;

    //@}
};

/**@}*/ // end group mi_neuray_database_access

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ITRANSACTION_H
