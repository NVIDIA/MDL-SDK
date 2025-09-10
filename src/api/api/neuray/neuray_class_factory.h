/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Header for the class factory.
 **/

#ifndef API_API_NEURAY_NEURAY_CLASS_FACTORY_H
#define API_API_NEURAY_NEURAY_CLASS_FACTORY_H

#include <mi/base/lock.h>
#include <mi/base/handle.h>
#include <mi/base/iinterface.h>
#include <mi/neuraylib/iuser_class_factory.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <boost/core/noncopyable.hpp>
#include <base/data/serial/i_serial_classid.h>
#include <base/data/db/i_db_tag.h>

namespace mi {

class IEnum_decl;
class IStructure_decl;

namespace neuraylib { class ISerializable; class ITransaction; }

}

namespace MI {

namespace DB { class Element_base; class Info; }
namespace IDATA { class Factory; }

namespace NEURAY {

class Class_factory;
class Expression_factory;
class IDb_element;
class Tag_handler;
class Transaction_impl;
class Type_factory;
class Value_factory;

extern Class_factory* s_class_factory;

/// The class factory.
///
/// The class factory is mainly called by ITransaction::create()/access()/edit() and by
/// IFactory::create(). It is also called directly by the global pointer s_class_factory in a few
/// places where the API components are not easily accessible.
///
/// There are three different kinds of registered API classes:
/// - API classes that have a DB element as backing implementation, e.g., Camera_impl
/// - API classes that have no corresponding DB element, e.g., String_impl
/// - User classes that have a DB element as wrapper (User_class_db_wrapper)
/// Consequently, there are three different methods to register API classes:
/// - Via class name and class ID, registering factories for the API class and the DB element
/// - Via class name, registering a factory for the API class
/// - Via class name and UUID, registering a factory for the user class
/// The second case is delegated to #MI::IDATA::Factory.
///
/// There are two different ways to create instances of API/user classes:
/// - an instance that wraps a DB element stored in the DB
///   (corresponds to STATE_ACCESS or STATE_EDIT in Db_element_impl_base)
/// - an instance that is not associated with the DB in any way (either because the instance has
///   not yet been stored in the DB, or the instance has no DB element as backing implementation)
///   (corresponds to STATE_POINTER in Db_element_impl_base)
class Class_factory : public boost::noncopyable
{
public:

    /// Constructor.
    Class_factory();

    /// Destructor.
    ~Class_factory();

    // class registration

    /// The type of factory methods for API classes.
    ///
    /// Each class factory is free to decide which values and types of \p argc and \p argv are
    /// valid.
    ///
    /// \param transaction          The transaction in which the instance is created.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional constructor arguments.
    /// \return                     An instance of the class, or \c nullptr on failure (incl.
    ///                             invalid arguments).
    using Api_class_factory = mi::base::IInterface*(*)(
        mi::neuraylib::ITransaction*, mi::Uint32, const mi::base::IInterface**);

    /// The type of factory methods for DB elements.
    ///
    /// Each class factory is free to decide which values and types of \p argc and \p argv are
    /// valid. However, it must handle the case \p transaction = \c nullptr, \p argc = 0, and
    /// \p argv = \c nullptr which is needed for deserialization.
    ///
    /// \param transaction          The transaction in which the instance is created.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional constructor arguments.
    /// \return                     An instance of the class, or \c nullptr on failure (incl.
    ///                             invalid arguments).
    using Db_element_factory = DB::Element_base*(*)(
        mi::neuraylib::ITransaction*, mi::Uint32, const mi::base::IInterface**);

    /// Registers class factories for a class ID and a class name.
    ///
    /// Used to register API classes with a corresponding DB class.
    ///
    /// \param class_name           The class name to register.
    /// \param class_id             The class ID to register.
    /// \param api_class_factory    The factory method that creates an instance of the
    ///                             corresponding API class.
    /// \param db_element_factory   The factory method that creates an instance of the
    ///                             corresponding DB element.
    /// \return                     -  0: Success.
    ///                             - -1: There is already a class registered under the name
    ///                                   \p class_name or the class ID \p class_id.
    mi::Sint32 register_class(
        const char* class_name,
        SERIAL::Class_id class_id,
        Api_class_factory api_class_factory,
        Db_element_factory db_element_factory);

    /// Registers a class factory for a class name.
    ///
    /// Used to register API classes without corresponding DB class.
    ///
    /// \param class_name           The class name to register.
    /// \param factory              The factory method that creates an instance of the
    ///                             corresponding API class.
    /// \return                     -  0: Success.
    ///                             - -1: There is already a class registered under the name
    ///                                   \p class_name.
    mi::Sint32 register_class(
        const char* class_name,
        Api_class_factory factory);

    /// Registers a class factory for a class name and class UUID.
    ///
    /// Used to register user-defined classes derived from #mi::neuraylib::IUser_class.
    ///
    /// \param class_name           The class name to register.
    /// \param uuid                 The UUID to register.
    /// \param factory              The factory method that creates an instance of the
    ///                             corresponding user-defined class.
    /// \return                     -  0: Success.
    ///                             - -1: There is already a class registered under the name
    ///                                   \p class_name or the UUID \p uuid.
    mi::Sint32 register_class(
        const char* class_name,
        const mi::base::Uuid& uuid,
        mi::neuraylib::IUser_class_factory* factory);

    /// Unregisters a class factory for a class name and class UUID.
    ///
    /// Used to unregister user-defined classes derived from #mi::neuraylib::IUser_class.
    ///
    /// \param class_name           The class name to unregister.
    /// \param uuid                 The UUID to register.
    /// \return                     -  0: Success.
    ///                             - -1: There is no registered under the name \p class_name and
    ///                                   the UUID \p uuid.
    mi::Sint32 unregister_class(
        const char* class_name,
        const mi::base::Uuid& uuid);

    /// Checks whether a DB class has been registered.
    ///
    /// \param class_id             The class ID to  check.
    /// \return                     \c true if the class ID is registered.
    bool is_class_registered( SERIAL::Class_id class_id) const;

    /// Checks whether a user-defined class has been registered.
    ///
    /// \param serializable         The serializable to check.
    /// \return                     \c true if the class ID of \p serializable is registered.
    bool is_class_registered( const mi::neuraylib::ISerializable* serializable) const;

    /// Checks whether a user-defined class has been registered.
    ///
    /// \param uuid                 The UUID to check.
    /// \return                     \c true if the UUID is registered.
    bool is_class_registered( const mi::base::Uuid& uuid) const;

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

    /// Returns a registered structure declaration.
    ///
    /// \param structure_name   The name of the structure declaration to return.
    /// \return                 The structure declaration for \p structure_name, or \c nullptr if
    ///                         there is no structure declaration for that name.
    const mi::IStructure_decl* get_structure_decl( const char* structure_name) const;

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
    mi::Sint32 register_enum_decl(
        const char* enum_name, const mi::IEnum_decl* decl);

    /// Unregisters an enum declaration with the \neurayApiName.
    ///
    /// \param enum_name        The name of the enum declaration to be unregistered .
    /// \return                 -  0: Success.
    ///                         - -1: There is no enum declaration registered under the name
    ///                               \p enum_name.
    mi::Sint32 unregister_enum_decl( const char* enum_name);

    /// Returns a registered enum declaration.
    ///
    /// \param enum_name        The name of the enum declaration to return.
    /// \return                 The enum declaration for \p enum_name, or \c nullptr if there is no
    ///                         enum declaration for that name.
    const mi::IEnum_decl* get_enum_decl( const char* enum_name) const;

    /// Unregister all user defined classes.
    void unregister_user_defined_classes();

    /// Unregister all structure declarations.
    void unregister_structure_decls();

    /// Unregister all enum declarations.
    void unregister_enum_decls();

    /// Returns the class ID for a registered class name.
    ///
    /// \param class_name   The class name. Note that some class names start with "__".
    /// \return             The class ID, or 0 if \p class_name is not a registered class name with
    ///                     a class ID.
    SERIAL::Class_id get_class_id( const char* class_name) const;

    /// Returns the class ID for a given tag.
    ///
    /// \param transaction  The transaction.
    /// \param tag          The tag.
    /// \return             The class ID for \p tag.
    SERIAL::Class_id get_class_id( const Transaction_impl* transaction, DB::Tag tag) const;

    // class instance creation

    /// Creates an instance of a class (access/edit of a DB element, by tag).
    ///
    /// Used by #ITransaction::access() and ITransaction::edit() to create instances of classes
    /// stored in the database.
    ///
    /// \param transaction          The transaction.
    /// \param tag                  The tag of the DB element which is connected with the to be
    ///                             created API class.
    /// \param is_edit              Indicates whether the DB element should get accessed via an
    ///                             Access or Edit. The caller is responsible not to hand out
    ///                             mutable pointers of the return value if \p is_edit is \c false.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    mi::base::IInterface* create_class_instance(
        Transaction_impl* transaction,
        DB::Tag tag,
        bool is_edit) const;

    /// Creates an instance of a class (access/edit of a DB element, by name).
    ///
    /// Used by #ITransaction::access() and ITransaction::edit() to create instances of classes
    /// stored in the database.
    ///
    /// \param transaction          The transaction.
    /// \param name                 The name of the DB element which is connected with the to be
    ///                             created API class.
    /// \param is_edit              Indicates whether the DB element should get accessed via an
    ///                             Access or Edit. The caller is responsible not to hand out
    ///                             mutable pointers of the return value if \p is_edit is \c false.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    mi::base::IInterface* create_class_instance(
        Transaction_impl* transaction,
        const char* name,
        bool is_edit) const;

    /// Creates an instance of a class (access/edit of a DB element, by info).
    ///
    /// Common code shared by both overloads of #create_class_instance() above.
    mi::base::IInterface* create_class_instance_shared(
        Transaction_impl* transaction,
        DB::Info* info,
        bool is_edit) const;

    /// Creates an instance of a type (not (yet) connected with DB).
    ///
    /// Used by #ITransaction::create() and #IFactory::create() to create instances of types not
    /// (yet) stored in the database.
    ///
    /// \param transaction          The transaction.
    /// \param type_name            The name of the type to create.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that is passed to the class
    ///                             factories.
    /// \return                     An instance of the requested type, or \c nullptr on failure.
    mi::base::IInterface* create_type_instance(
        Transaction_impl* transaction,
        const char* type_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr) const;

    /// Creates an instance of a type (not (yet) connected with DB).
    ///
    /// Used by #ITransaction::create() and #IFactory::create() to create instances of types not
    /// (yet) stored in the database.
    ///
    /// This templated member function is a wrapper of the function above for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid&)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \param transaction          The transaction.
    /// \param type_name            The name of the type to create.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that is passed to the class
    ///                             factories.
    /// \return                     An instance of the requested type, or \c nullptr on failure.
    template<class T>
    T* create_type_instance(
        Transaction_impl* transaction,
        const char* type_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr) const
    {
        mi::base::IInterface* ptr_iinterface = create_type_instance(
            transaction, type_name, argc, argv);
        if ( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Creates an instance of a user-defined class.
    ///
    /// \param uuid                 The UUID of the class to create.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    mi::base::IInterface* create_class_instance( const mi::base::Uuid& uuid) const;

    /// Creates an instance of a user-defined class.
    ///
    /// This templated member function is a wrapper of the function above for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid&)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \param uuid                 The UUID of the class to create.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    template<class T>
    T* create_class_instance( const mi::base::Uuid& uuid) const
    {
        mi::base::IInterface* ptr_iinterface = create_class_instance( uuid);
        if ( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Returns the corresponding factory for instances of #mi::neuraylib::IData.
    IDATA::Factory* get_idata_factory() const { return m_idata_factory.get(); }

    /// Converts \p uuid into a string.
    std::string uuid_to_string( const mi::base::Uuid& uuid);

    /// Returns the MDL type factory for a given transaction.
    Type_factory* create_type_factory( mi::neuraylib::ITransaction* transaction) const;

    /// Returns the MDL value factory for a given transaction.
    Value_factory* create_value_factory( mi::neuraylib::ITransaction* transaction) const;

    /// Returns the MDL expression factory for a given transaction.
    Expression_factory* create_expression_factory( mi::neuraylib::ITransaction* transaction) const;

private:
    /// Creates an instance of a class (not (yet) connected with DB).
    ///
    /// Used by #create_type_instance() and other members to create instances of classes not (yet)
    /// stored in the database. Note that #create_type_instance() and other members accept type
    /// names and use this method to handle the primitive class names.
    ///
    /// \param transaction          The transaction.
    /// \param class_name           The name of the class to create.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that gets passed to the class
    ///                             factories.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    mi::base::IInterface* create_class_instance(
        Transaction_impl* transaction,
        const char* class_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr) const;

    /// Creates an instance of a class (not (yet) connected with DB).
    ///
    /// Used by #create_type_instance() and other members to create instances of classes not (yet)
    /// stored in the database. Note that #create_type_instance() and other members accept type
    /// names and use this method to handle the primitive class names.
    ///
    /// This templated member function is a wrapper of the function above for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid&)
    /// on the returned pointer, since the return type already is a pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \param transaction          The transaction.
    /// \param class_name           The name of the class to create.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that gets passed to the class
    ///                             factories.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    template<class T>
    T* create_class_instance(
        Transaction_impl* transaction,
        const char* class_name,
        mi::Uint32 argc = 0,
        const mi::base::IInterface* argv[] = nullptr) const
    {
        mi::base::IInterface* ptr_iinterface
            = create_class_instance( transaction, class_name, argc, argv);
        if ( !ptr_iinterface)
            return 0;
        T* ptr_T = static_cast<T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Extracts the actual user class from the API wrapper and returns it.
    ///
    /// Also ensures that there is at most one API wrapper instance per DB wrapper instance.
    /// Clears the embedded transaction pointer if this access is a read-only access and there was
    /// no prior read-write access.
    ///
    /// \param db_element           The instance of the API class which wraps the user class.
    /// \param is_edit              Indicates whether the DB element should get accessed via an
    ///                             Access or Edit. The caller is responsible not to hand out
    ///                             mutable pointers of the return value if \p is_edit is \c false.
    /// \return                     The actual user class, or \c nullptr on failure.
    mi::base::IInterface* extract_user_class(
        IDb_element* idb_element,
        bool is_edit) const;

    /// Extracts the actual DiCE element from the API wrapper and returns it.
    ///
    /// Also ensures that there is at most one API wrapper instance per DB wrapper instance.
    /// Clears the embedded transaction pointer if this access is a read-only access and there was
    /// no prior read-write access.
    ///
    /// \param db_element           The instance of the API class which wraps the DiCE element.
    /// \param is_edit              Indicates whether the DB element should get accessed via an
    ///                             Access or Edit. The caller is responsible not to hand out
    ///                             mutable pointers of the return value if \p is_edit is \c false.
    /// \return                     The actual DiCE element, or \c nullptr on failure.
    mi::base::IInterface* extract_element(
        IDb_element* db_element,
        bool is_edit) const;

    /// Invokes the API class factory registered for the given class ID.
    ///
    /// No transaction or argc/argv arguments are passed to the factory since there are none in the
    /// calling context.
    ///
    /// \param transaction          The transaction.
    /// \param class_id             The class ID of the (DB element corresponding to the) API class.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    mi::base::IInterface* invoke_api_class_factory(
        Transaction_impl* transaction,
        SERIAL::Class_id class_id) const;

    /// Invokes the API or user class factory registered for the given class name.
    ///
    /// \param transaction          The transaction.
    /// \param class_name           The class name of the API class.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that is passed to the class
    ///                             factory.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    mi::base::IInterface* invoke_api_or_user_class_factory(
        Transaction_impl* transaction,
        const char* class_name,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]) const;

    /// Invokes the DB element factory registered for the given class name.
    ///
    /// \param transaction          The transaction.
    /// \param class_name           The class name of the API class.
    /// \param argc                 The size of the \p argv array.
    /// \param argv                 An array of optional arguments that is passed to the class
    ///                             factory.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    DB::Element_base* invoke_db_element_factory(
        Transaction_impl* transaction,
        const char* class_name,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]) const;

    /// Invokes the user class factory registered for the given class UUID.
    ///
    /// No transaction or argc/argv arguments are passed to the factory since there are none in the
    /// calling context.
    ///
    /// \param uuid                 The UUID of the class to create.
    /// \return                     An instance of the requested class, or \c nullptr on failure.
    mi::base::IInterface* invoke_user_class_factory( const mi::base::Uuid& uuid) const;

    /// A callback passed to #m_idata_factory to implement the support for #mi::IRef.
    mi::base::Handle<Tag_handler> m_tag_handler;

    /// The corresponding factory for instances of #mi::neuraylib::IData.
    std::unique_ptr<IDATA::Factory> m_idata_factory;

    /// Maps class names to class IDs.
    ///
    /// Not locked since it is modified only before startup.
    std::map<std::string, SERIAL::Class_id> m_map_name_id;

    /// Maps class IDs to API class factories.
    ///
    /// Not locked since it is modified only before startup. No reference counting.
    std::map<SERIAL::Class_id, Api_class_factory> m_map_id_api_class_factory;

    /// Maps class names to API class factories.
    ///
    /// Not locked since it is modified only before startup. No reference counting.
    std::map<std::string, Api_class_factory> m_map_name_api_class_factory;

    /// Maps class names to DB element factories.
    ///
    /// Not locked since it is modified only before startup. No reference counting.
    std::map<std::string, Db_element_factory> m_map_name_db_element_factory;

    /// Maps class names to user class factories.
    ///
    /// Not locked since it is modified only before startup/after shutdown.
    std::map<std::string, mi::base::Handle<mi::neuraylib::IUser_class_factory>>
        m_map_name_user_class_factory;

    /// Maps class UUIDs to user class factories.
    ///
    /// Not locked since it is modified only before startup/after shutdown.
    std::map<mi::base::Uuid, mi::base::Handle<mi::neuraylib::IUser_class_factory>>
        m_map_uuid_user_class_factory;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_CLASS_FACTORY_H

