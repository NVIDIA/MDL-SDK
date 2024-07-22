/**************************************************
 * Copyright (c) 2010-2024, NVIDIA CORPORATION. All rights reserved.
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
 *************************************************/

/* \file
 * \brief Header for the factory.
 */

#ifndef BASE_DATA_IDATA_IDATA_FACTORY_H
#define BASE_DATA_IDATA_IDATA_FACTORY_H

#include <sstream>
#include <string>
#include <vector>

#include <boost/core/noncopyable.hpp>

#include <mi/base/handle.h>
#include <mi/base/iinterface.h>
#include <mi/base/lock.h>
#include <mi/base/uuid.h>
#include <mi/neuraylib/idata.h>
#include <mi/neuraylib/type_traits.h>

#include <base/lib/robin_hood/robin_hood.h>
#include <base/data/db/i_db_tag.h>

namespace mi {
class IArray;
class ICompound;
class IConst_pointer;
class IData;
class IData_collection;
class IData_simple;
class IDynamic_array;
class IEnum;
class IEnum_decl;
class IMap;
class INumber;
class IPointer;
class ISint32;
class IStructure;
class IStructure_decl;
}

namespace MI {

namespace DB { class Transaction; }

namespace IDATA {

class ITag_handler;

/// The factory for implementation of #mi::IData.
///
/// This factory allows to create instances of all interfaces derived from mi::IData. For most
/// interfaces there are two implementations, the so-called default implementation owning the
/// memory to store the actual values, and the so-called proxy implementation that merely points to
/// memory owned by something else. Typical use cases for the proxy implementation are attributes,
/// and elements of compounds.
///
/// Instances of #mi::IStructure and #mi::IEnum depend on a corresponding declaration which has
/// to be registered upfront. Instances of #mi::IRef are only supported when the factory is
/// constructed with a tag handler implementing the low-level support.
///
/// The #create() method supports a transaction, which is optional in most cases. It is required
/// for
/// - instances of the default implementation of #mi::IRef, as well as pointers to and collections
///   of such instances, and
/// - instances of the proxy implementations of #mi::IRef, #mi::IArray, #mi::IDynamic_array, and
///   #mi::IStructure, as well as pointers to and collections of such instances.
class Factory : public boost::noncopyable
{
public:

    /// Constructor.
    ///
    /// \param tag_handler    Optional tag handler. Required for support of #mi::IRef.
    Factory( ITag_handler* tag_handler = nullptr);

    /// Destructor.
    ~Factory();

    /// The type of factory methods.
    ///
    /// Each factory method is free to decide which values and types of \p argc and \p argv are
    /// valid.
    ///
    /// \param factory              The factory.
    /// \param transaction          The transaction, see #Factory.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional constructor arguments.
    /// \return                     An instance of the class, or \c NULL on failure (incl.
    ///                             invalid arguments).
    using Factory_function = mi::base::IInterface*(*)(
        const Factory* factory, DB::Transaction*, mi::Uint32, const mi::base::IInterface**);

    /// \name Instance creation
    //@{

    /// Creates an instance of an #mi::IData type.
    ///
    /// \param transaction          The transaction, see #Factory.
    /// \param type_name            The name of the type to create.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that gets passed to the factory
    ///                             methods.
    /// \return                     An instance of the requested type, or \c NULL on failure.
    mi::base::IInterface* create(
        DB::Transaction* transaction,
        const char* type_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr) const;

    /// Creates an instance of an #mi::IData type.
    ///
    /// \param transaction          The transaction, see #Factory.
    /// \param type_name            The name of the type to create.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that gets passed to the factory
    ///                             methods.
    /// \return                     An instance of the requested type, or \c NULL on failure.
    template<class T>
    T* create(
        DB::Transaction* transaction,
        const char* type_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr) const
    {
        mi::base::IInterface* ptr_iinterface = create(
            transaction, type_name, argc, argv);
        if( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Creates an instance of an #mi::IData type.
    ///
    /// Creates an instance of a type (not (yet) connected with DB).
    ///
    /// This wrapper only supports types supported by #mi::Type_traits.
    ///
    /// \param transaction          The transaction, see #Factory.
    /// \return                     An instance of the requested type, or \c NULL on failure.
    template<class T>
    T* create( DB::Transaction* transaction = nullptr)
    {
        return create<T>( transaction, mi::Type_traits<T>::get_type_name());
    }

    //@}
    /// \name Implementation of methods from #mi::neuraylib::IFactory.
    //@{

    /// Implementation of #mi::neuraylib::IFactory::assign_from_to().
    mi::Uint32 assign_from_to(
        const mi::IData* source, mi::IData* target, mi::Uint32 options = 0) const;

    /// Implementation of #mi::neuraylib::IFactory::clone().
    mi::IData* clone( const mi::IData* source, mi::Uint32 options = 0);

    /// Implementation of #mi::neuraylib::IFactory::clone<T>().
    template<class T>
    T* clone( const mi::IData* source, mi::Uint32 options = 0)
    {
        mi::IData* ptr_idata = clone( source, options);
        if( !ptr_idata)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_idata->get_interface( typename T::IID()));
        ptr_idata->release();
        return ptr_T;
    }

    /// Implementation of #mi::neuraylib::IFactory::compare().
    mi::Sint32 compare( const mi::IData* lhs, const mi::IData* rhs);

    /// Implementation of #mi::neuraylib::IFactory::dump().
    void dump( const char* name, const mi::IData* data, mi::Size depth, std::ostringstream& s);

    //@}
    /// \name Registration of enums
    //@{

    /// Registers an enum declaration with the \neurayApiName.
    ///
    /// \param enum_name        The name to be used to refer to this enum declaration.
    ///                         The name must consist only of alphanumeric characters or
    ///                         underscores, must not start with an underscore, and must not be the
    ///                         empty string.
    /// \param decl             The enum declaration. The declaration is internally cloned such
    ///                         that subsequent changes have no effect on the registered
    ///                         declaration.
    /// \return                 -  0: Success.
    ///                         - -1: There is already a class, structure or enum declaration
    ///                               registered under the name \p enum_name.
    ///                         - -5: A registration under the name \p enum_name would cause an
    ///                               infinite cycle of nested enum types.
    mi::Sint32 register_enum_decl( const char* enum_name, const mi::IEnum_decl* decl);

    /// Unregisters an enum declaration with the \neurayApiName.
    ///
    /// \param enum_name        The name of the enum declaration to be unregistered .
    /// \return                 -  0: Success.
    ///                         - -1: There is no enum declaration registered under the name
    ///                               \p enum_name.
    mi::Sint32 unregister_enum_decl( const char* enum_name);

    /// Unregister all enum declarations.
    void unregister_enum_decls();

    /// Returns a registered enum declaration.
    ///
    /// \param enum_name        The name of the enum declaration to return.
    /// \return                 The enum declaration for \p enum_name, or \c NULL if there is no
    ///                         enum declaration for that name.
    const mi::IEnum_decl* get_enum_decl( const char* enum_name) const;

    //@}
    /// \name Registration of structures
    //@{

    /// Registers a structure declaration with the \neurayApiName.
    ///
    /// \param structure_name   The name to be used to refer to this structure declaration.
    ///                         The name must consist only of alphanumeric characters or
    ///                         underscores, must not start with an underscore, and must not be the
    ///                         empty string.
    /// \param decl             The structure declaration. The declaration is internally cloned such
    ///                         that subsequent changes have no effect on the registered
    ///                         declaration.
    /// \return                 -  0: Success.
    ///                         - -1: There is already a class, structure or enum declaration
    ///                               registered under the name \p structure_name.
    ///                         - -5: A registration under the name \p structure_name would cause an
    ///                               infinite cycle of nested structure types.
    mi::Sint32 register_structure_decl(
        const char* structure_name, const mi::IStructure_decl* decl);

    /// Unregisters a structure declaration with the \neurayApiName.
    ///
    /// \param structure_name   The name of the structure declaration to be unregistered .
    /// \return                 -  0: Success.
    ///                         - -1: There is no structure declaration registered under the name
    ///                               \p structure_name.
    mi::Sint32 unregister_structure_decl( const char* structure_name);

    /// Unregister all structure declarations.
    void unregister_structure_decls();

    /// Returns a registered structure declaration.
    ///
    /// \param structure_name   The name of the structure declaration to return.
    /// \return                 The structure declaration for \p structure_name, or \c NULL if there
    ///                         is no structure declaration for that name.
    const mi::IStructure_decl* get_structure_decl( const char* structure_name) const;

    //@}
    /// \name Various utility methods.
    //@{

    /// Converts \p uuid into a string.
    static std::string uuid_to_string( const mi::base::Uuid& uuid);

    /// Returns the element type name of an array type name, or the empty string in case of failure.
    ///
    /// The length of the array is stored in \p length (only in case of success, zero for dynamic
    /// arrays).
    static std::string strip_array( const std::string& type_name, mi::Size& length);

    /// Returns the value type name of a map type name, or the empty string in case of failure.
    static std::string strip_map( const std::string& type_name);

    /// Returns the nested type name of a pointer type name, or the empty string in case of failure.
    static std::string strip_pointer( const std::string& type_name);

    /// Returns the nested type name of a const pointer type name, or the empty string in case of
    /// failure.
    static std::string strip_const_pointer( const std::string& type_name);

    /// Returns the transaction that is embedded in references, arrays, or calls (or in collections
    /// containing such types), or \c NULL in case of failure.
    static DB::Transaction* get_transaction( const mi::IData* data);

    /// Returns the tag handler (or \c NULL if none was registered in the constructor).
    ITag_handler* get_tag_handler() const;

    //@}

private:
    /// \name Registration
    //@{

    /// Registers a factory method for a class name.
    ///
    /// Used to register API classes without corresponding DB class.
    ///
    /// \param class_name           The class name to register.
    /// \param factory              The factory method that creates an instance of the
    ///                             corresponding API class.
    /// \return                     -  0: Success.
    ///                             - -1: There is already a class registered under the name
    ///                                   \p class_name.
    mi::Sint32 register_class( const char* class_name, Factory_function factory);

    /// Checks whether any member of a structure declaration contains a blacklisted type name.
    ///
    /// Used to reject recursive structure declarations (including indirect recursion).
    ///
    /// \note The caller must lock #m_structure_decls_lock.
    ///
    /// \param type_name   The type name to check.
    /// \param blacklist   List of blacklisted type names. Modified during execution,
    ///                    restored before the method returns.
    /// \return            \c true if the call declaration contains any of the blacklisted type
    ///                    names, \c false otherwise.
    bool contains_blacklisted_type_names(
        const std::string& type_name, std::vector<std::string>& blacklist) const;

    /// Checks whether any member of a structure declaration contains a blacklisted type name.
    ///
    /// Used to reject recursive structure declarations (including indirect recursion).
    ///
    /// \note The caller must lock #m_structure_decls_lock.
    ///
    /// \param decl        The structure declaration to check.
    /// \param blacklist   List of blacklisted type names. Modified during execution,
    ///                    restored before the method returns.
    /// \return            \c true if the structure declaration contains any of the blacklisted type
    ///                    names, \c false otherwise.
    bool contains_blacklisted_type_names(
        const mi::IStructure_decl* decl, std::vector<std::string>& blacklist) const;

    //@}
    /// \name Instance creation
    //@{

    /// Creates an instance of an #mi::IData type (base case).
    ///
    /// This method handles the base cases, i.e., those that are registered in
    /// #m_factory_functions
    ///
    /// \param transaction          The transaction, see #Factory.
    /// \param class_name           The name of the class to create.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that gets passed to the factory
    ///                             methods.
    /// \return                     An instance of the requested class, or \c NULL on failure.
    mi::base::IInterface* create_registered(
        DB::Transaction* transaction,
        const char* class_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr) const;

    /// Creates an instance of an #mi::IData type (base case).
    ///
    /// This method handles the base cases, i.e., those that are registered in
    /// #m_factory_functions
    ///
    /// \param transaction          The transaction, see #Factory.
    /// \param class_name           The name of the class to create.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that gets passed to the factory
    ///                             methods.
    /// \return                     An instance of the requested class, or \c NULL on failure.
    template<class T>
    T* create_registered(
        DB::Transaction* transaction,
        const char* class_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr) const
    {
        mi::base::IInterface* ptr_iinterface
            = create_registered( transaction, class_name, argc, argv);
        if( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Creates an instance of an array class.
    mi::base::IInterface* create_array(
        DB::Transaction* transaction, const char* type_name) const;

    /// Creates an instance of a map class.
    mi::base::IInterface* create_map(
        DB::Transaction* transaction, const char* type_name) const;

    /// Creates an instance of a pointer or const pointer class.
    mi::base::IInterface* create_pointer(
        DB::Transaction* transaction, const char* type_name) const;

    /// Creates an instance of a structure declaration.
    mi::base::IInterface* create_structure(
        DB::Transaction* transaction, const char* type_name, const mi::IStructure_decl* decl) const;

    /// Creates an instance of an enum declaration.
    mi::base::IInterface* create_enum(
        DB::Transaction* transaction, const char* type_name, const mi::IEnum_decl* decl) const;

    /// Convenience wrapper around create() for interfaces that need a transaction.
    ///
    /// First calls #create(). If that fails, extracts the transaction from \p prototype, and calls
    /// #mi::neuraylib::ITransaction::create() (if it succeeded to extract the transaction).
    mi::base::IInterface* create_with_transaction(
        const char* type_name, const mi::IData* prototype);

    /// Convenience wrapper around create() for interfaces that need a transaction.
    ///
    /// First calls #create(). If that fails, extracts the transaction from \p prototype, and calls
    /// #mi::neuraylib::ITransaction::create() (if it succeeded to extract the transaction).
    template<class T>
    T* create_with_transaction(
        const char* type_name, const mi::IData* prototype)
    {
        mi::base::IInterface* ptr_iinterface = create_with_transaction( type_name, prototype);
        if( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    //@}
    /// \name Assignment
    //@{

    mi::Uint32 assign_from_to(
        const mi::IData_simple* source, mi::IData_simple* target, mi::Uint32 options) const;

    mi::Uint32 assign_from_to(
        const mi::IData_collection* source, mi::IData_collection* target, mi::Uint32 options) const;

    mi::Uint32 assign_from_to( const mi::INumber* source, mi::INumber* target) const;

    mi::Uint32 assign_from_to( const mi::IString* source, mi::IString* target) const;

    mi::Uint32 assign_from_to( const mi::IRef* source, mi::IRef* target) const;

    mi::Uint32 assign_from_to(
        const mi::IEnum* source, mi::IEnum* target, mi::Uint32 options) const;

    mi::Uint32 assign_from_to( const mi::IEnum* source, mi::ISint32* target) const;

    mi::Uint32 assign_from_to( const mi::IUuid* source, mi::IUuid* target) const;

    mi::Uint32 assign_from_to(
        const mi::IPointer* source, mi::IPointer* target, mi::Uint32 options) const;

    mi::Uint32 assign_from_to(
        const mi::IConst_pointer* source, mi::IPointer* target, mi::Uint32 options) const;

    mi::Uint32 assign_from_to(
        const mi::IConst_pointer* source, mi::IConst_pointer* target, mi::Uint32 options) const;

    mi::Uint32 assign_from_to(
        const mi::IPointer* source, mi::IConst_pointer* target, mi::Uint32 options) const;

    //@}
    /// \name Cloning
    //@{

    mi::IData_simple* clone( const mi::IData_simple* source, mi::Uint32 options);

    mi::IData_collection* clone( const mi::IData_collection* source, mi::Uint32 options);

    mi::IRef* clone( const mi::IRef* source, mi::Uint32 options);

    mi::IPointer* clone( const mi::IPointer* source, mi::Uint32 options);

    mi::IConst_pointer* clone( const mi::IConst_pointer* source, mi::Uint32 options);

    mi::ICompound* clone( const mi::ICompound* source, mi::Uint32 options);

    mi::IDynamic_array* clone( const mi::IDynamic_array* source, mi::Uint32 options);

    mi::IArray* clone( const mi::IArray* source, mi::Uint32 options);

    mi::IStructure* clone( const mi::IStructure* source, mi::Uint32 options);

    mi::IMap* clone( const mi::IMap* source, mi::Uint32 options);

    //@}
    /// \name Comparison
    //@{

    mi::Sint32 compare( const mi::IData_simple* lhs, const mi::IData_simple* rhs);

    mi::Sint32 compare( const mi::IData_collection* lhs, const mi::IData_collection* rhs);

    mi::Sint32 compare( const mi::INumber* lhs, const mi::INumber* target);

    mi::Sint32 compare( const mi::IString* lhs, const mi::IString* target);

    mi::Sint32 compare( const mi::IRef* lhs, const mi::IRef* target);

    mi::Sint32 compare( const mi::IEnum* lhs, const mi::IEnum* rhs);

    mi::Sint32 compare( const mi::IUuid* lhs, const mi::IUuid* target);

    mi::Sint32 compare( const mi::IPointer* lhs, const mi::IPointer* rhs);

    mi::Sint32 compare( const mi::IConst_pointer* lhs, const mi::IConst_pointer* rhs);

    //@}

    /// Maps type names to factory methods.
    ///
    /// Not locked since it is modified only before startup. No reference counting.
    robin_hood::unordered_map<std::string, Factory_function> m_factory_functions;

    /// Maps type names to structure declarations.
    ///
    /// \note Any access needs to be protected by #m_structure_decls_lock.
    robin_hood::unordered_map<std::string, mi::base::Handle<const mi::IStructure_decl>>
        m_structure_decls;

    /// Maps type names to enum declarations.
    ///
    /// \note Any access needs to be protected by #m_enum_decls_lock.
    robin_hood::unordered_map<std::string, mi::base::Handle<const mi::IEnum_decl>> m_enum_decls;

    /// The lock that protects the map #m_structure_decls.
    mutable mi::base::Lock m_structure_decls_lock;

    /// The lock that protects the map #m_enum_decls.
    mutable mi::base::Lock m_enum_decls_lock;

    /// The tag handler (or invalid if none was registered in the constructor).
    mi::base::Handle<ITag_handler> m_tag_handler;
};

} // namespace IDATA

} // namespace MI

#endif // BASE_DATA_IDATA_IDATA_FACTORY_H
