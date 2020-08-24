/***************************************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Attribute system for database elements
///
/// An attribute is an extra piece of data that can be attached to a database
/// element. It is named (although names are converted to perfect hashes early
/// on, for faster lookup). All inheritance is based on attributes.
/// For example, a geometric object may have a boolean attribute "trace" that
/// controls whether the object is hit by trace rays; or an array attribute
/// containing a texture space (one vector per vertex); or a shader or instance
/// may have attributes like "ambient" that replace mental ray 3's parameters.
 
#ifndef BASE_DATA_ATTR_ATTR_H
#define BASE_DATA_ATTR_ATTR_H

#include <base/system/main/types.h>
#include <base/system/main/i_module.h>
#include <base/data/db/i_db_journal_type.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/serial/i_serial_serializable.h>
#include <base/system/stlext/i_stlext_any.h>
#include <boost/shared_ptr.hpp>
#include <mi/base/config.h>

#include "i_attr_types.h"
#include "i_attr_type.h"
#include "i_attr_type_code_traits.h"
#include "i_attr_registry.h"
#include "i_attr_attribute.h"
#include "i_attr_attribute_list.h"

#include <map>
#include <string>
#include <regex>
namespace MI {
namespace DB { class Transaction; }
namespace SERIAL {
class Deserialization_manager;
class Serializer;
class Deserializer;
}
namespace SYSTEM { class Module_registration_entry; }
namespace ATTR {

const Uint reserved_ids = 1 << 20;	///< reserved for fixed Attr IDs
const Uint reserved_flag_ids = 32;	///< reserved for fixed boolean Attr IDs


/// Represents the module. All modules must be registered this way.
///
/// Reserved attributes have name, type and journal-flags which the scene module
/// can register here. Other (user) attributes store their name and type.
/// For journal-flags, this is not necessary (yet) since all user attributes
/// have the same journal flags which are also registered by the scene-module.
/// The Journal-flags of an attribute need to be passed to an Edit pointer of a
/// db-element if this specific attribute is changed (see scene.h or db.h).
class Attr_module : public SYSTEM::IModule
{
  public:
    /// Fully initialize the module such that its members can register themselves for serialization.
    /// If this function was not called then the module still works except that its members
    /// cannot be serialized.
    /// \param dm a deserialization manager
    virtual void register_for_serialization(
        SERIAL::Deserialization_manager* dm) = 0;

    /// Give names to the reserved flag attributes (they are defined above BASE).
    /// \param id give this attribute ID a name
    /// \param name new attr name, literal, not copied
    /// \param tc simple type of attribute
    /// \param flags if attr changes, put these flags
    /// \param inh inheritable, may have GLOBAL flag
    /// \param def default value for this Attributes
    virtual void set_reserved_attr(
        Attribute_id		id,
        const char		*name,
        const Type_code		tc,
        const DB::Journal_type	flags,
        bool		  	inh,
        const STLEXT::Any&	def=STLEXT::Any()) = 0;

    /// Some attributes have deprecated names that should also work, and map to
    /// the same IDs. For example, "sample_max" is now "samples".
    /// \param id give this ID an alternate name
    /// \param name new attr name, literal, not copied
    virtual void set_deprecated_attr_name(
        Attribute_id		id,
        const char		*name) = 0;

    /// Return deprecated name, or 0 if there is none.
    /// \param id return alternate name of this ID
    /// \return deprecated name, or 0 if there is none.
    virtual const char *get_deprecated_attr_name(
        Attribute_id	   	id) = 0;

    /// Register journal-flags for all user attributes.
    /// For now, all user attributes have the same journal flags
    /// SCENE::JOURNAL_CHANGE_SHADER.
    /// \param flags if attr changes, put journal flags
    virtual void set_user_attr_journal_flags(
        const DB::Journal_type flags) = 0;

    /// Return previously set names of the reserved flag attributes.
    /// \param id return name of this attribute ID
    /// \param tc return simple type of attr if nz
    /// \param jf return journal flags of attr if nz
    /// \param inh false if attr is never inheritable
    /// \return previously set names of the reserved flag attributes.
    virtual const char *get_reserved_attr(
        Attribute_id	id,
        Type_code	*tc  = 0,
        DB::Journal_type*jf  = 0,
        bool		*inh = 0) const = 0;

    /// Retrieve the \c Attribute_spec for the given \p id.
    /// \param id the \c Attribute_id of the reserved \c Attribute
    /// \return the corresponding \c Attribute_spec or 0 else
    virtual const Attribute_spec* get_reserved_attr_spec(
        Attribute_id id) const = 0;

    /// Retrieve the default value for the reserved \c Attribute with the given name.
    /// \param name name of the reserved \c Attribute
    /// \param [out] default_value non-const ref to store the default into
    /// \return success
    template <typename T>
    bool get_reserved_attr_default(
        const std::string& name,
        T& default_value) const
    {
        STLEXT::Any any;
        bool result = false;
        retrieve_reserved_attr_default(name, any);
        if (!any.empty() && any.type() == typeid(T)) {
            default_value = *STLEXT::any_cast<T>(&any);
            result = true;
        }
        return result;
    }

    using Custom_attr_filters = std::vector<std::string>;

    virtual const Custom_attr_filters& get_custom_attr_filters() const = 0;

    virtual bool add_custom_attr_filter(const std::string& filter) = 0;

    virtual bool remove_custom_attr_filter(const std::string& filter) = 0;

    virtual void clear_custom_attr_filters() = 0;

    virtual const std::wregex& get_custom_attr_filter() const = 0;

    /// Look up a type code given a type name. This uses a hash map, so it should
    /// be pretty fast. Returns \c TYPE_UNDEF if the look up fails.
    /// \param type_name name of the type
    /// \return \c Type_code, \c TYPE_UNDEF if the look up fails.
    virtual ATTR::Type_code get_type_code(
        const std::string& type_name) const = 0;

    /// \name ModuleImpl
    /// Required functionality for implementing a \c SYSTEM::IModule.
    //@{
    /// Retrieve the name.
    static const char* get_name() { return "ATTR"; }
    /// Allow link time detection.
    static SYSTEM::Module_registration_entry* get_instance();
    //@}

  private:
    /// Internal helper. It allows us to hide its implementation, even though it is used inside
      /// the templated member \c get_reserved_attr_default(). But thanks to the \c Any...
    void retrieve_reserved_attr_default(
        const std::string& name,
        STLEXT::Any& any) const;
};


//--------------------------------------------- Iterators ---------------------
//
// Define iterators iterating over Type and Attribute_set.
//

class Attribute;

/// Typedef for convenience only
typedef std::map<Attribute_id, boost::shared_ptr<Attribute> > Attributes;


/// Another Type iterator.
///
/// It iterates from the given root down in a depth-first traversal, ie first
/// down the childrens hierarchy and then following the next link.
/// If you want to iterate just on the next link hierarchy then enable the
/// top-level-only iteration on construction.
///
/// \todo: The only two missing functions to the above iterator are
///     to_next()  going explicitly to next top-level node indep. of iter mode
///     at_end()   are we at the (child) end of an type traversal?
/// If needed it should be easy to add them - possibly another inheritance
/// level would keep such add-ons apart of the basic iterator.
class Type_iterator_rec
{
  public:
    /// Constructor.
    /// \param root starting point of iteration
    /// \param top traverse only top level
    explicit Type_iterator_rec(
        const Type 	*root,
        bool		top=false);

    /// Copy constructor.
    /// \param other the one to copy
    Type_iterator_rec(
        const Type_iterator_rec &other);

    // Assignment operator.
    /// \param other the one to assign from
    Type_iterator_rec& operator=(
        const Type_iterator_rec &other);

    /// Incrementing the iterator.
    Type_iterator_rec &operator++();
    /// Incrementing the iterator - the slow version.
    Type_iterator_rec operator++(int);

    /// Dereferencing the iterator.
    const Type &operator*() const;
    /// Dereferencing the iterator.
    const Type *operator->() const;

    /// Safe dereferencing - identical to operator->().
    const Type *get_type() const;

    /// \name Comparing iterators.
    /// @{
    bool operator==(
        const Type_iterator_rec &other) const;
    bool operator!=(
        const Type_iterator_rec &other) const;
    /// @}

  private:
    /// convenience typedef
    typedef std::vector<const Type*> Types;
    Types 		m_roots;	//</ stack of parent Types
    const Type		*m_iter;	///< current position
    bool		m_toplevel_only;///< traverse only the toplevel

    /// Retrieve the current (sub)root.
    const Type *get_root() const;
#ifdef DEBUG
    friend class Type_iter_checker;
#endif
};



/// A type iterator iterating over a given Attribute_set.
///
/// It starts with the given begin and type positions and iterates over
/// all children of the current base attribute's type first! *Then* it
/// continues parsing the the next base attribute('s type) in the set.
///
/// In addition to that the constructor is configurable for a 'base attributes
/// only' traversal in contrast to the default behaviour described above.
class Attribute_set_type_iterator
{
  public:
    typedef Attributes::const_iterator Const_iter;

    /// Constructor.
    explicit Attribute_set_type_iterator();
    /// Constructor.
    /// \param begin iterator over the Attributes
    /// \param attrs corresponding Attribute_set
    /// \param top traverse only top level
    Attribute_set_type_iterator(
        Const_iter	  begin,
        const Attributes &attrs,
        bool		  top=false);
    /// Copy constructor.
    Attribute_set_type_iterator(
        const Attribute_set_type_iterator &iter);
    /// Assignment operator.
    Attribute_set_type_iterator &operator=(
        const Attribute_set_type_iterator &other);

    /// Incrementing the iterator.
    Attribute_set_type_iterator &operator++();
    /// Incrementing the iterator - the slow version.
    Attribute_set_type_iterator operator++(int);

    /// Dereferencing the iterator.
    const Type &operator*() const;
    /// Dereferencing the iterator.
    const Type *operator->() const;

    /// \name Comparing iterators.
    /// @{
    bool operator==(
        const Attribute_set_type_iterator &iter) const;
    bool operator!=(
        const Attribute_set_type_iterator &iter) const;
    /// @}

    /// Is iteration over top-level types only?
    bool is_toplevel_only() const;

  protected:
    Const_iter			m_attr_iter;	///< Attribute iterator
    mutable Type_iterator_rec	m_type_iter;	///< Type iterator
    bool			m_toplevel_only;///< traverse only the toplevel
};


/// The manager of the \c Attributes attached for instance to any scene element.
/// Every Element contains such an attribute set that allows attributes to be
/// attached to it. This attribute set is also useful during inheritance, where it
/// can maintain the current list of attributes inherited so far.
/// There are 32 reserved bits for reserved attributes, which are predefined in
/// io/scene/scene/i_scene_attr_resv_id.h.
class Attribute_set : public SERIAL::Serializable
{
  public:
    explicit Attribute_set();
    ~Attribute_set();

    /// Copy constructor. Note that this constructor copies the underlying attribute set by
    /// a deep copy of every Attribute.
    /// \param other copy this Attribute_set
    Attribute_set(
        const Attribute_set& other);
    /// Assignment operator. Note that this member copies the underlying attribute set by
    /// a deep copy of every Attribute.
    /// \param attrset attribute set to copy
    Attribute_set &operator=(
        const Attribute_set& attrset);

    /// Attach a new attribute, which must have been previously created on the
    /// heap. Never attach an attribute to multiple attribute sets - otherwise an edit on one
    /// shared_ptr<Attribute> in one set would change all of the other shared ones in different
    /// sets. Return the success of insertion, ie false means that attr is already a member.
    /// \param attr attribute to attach
    /// \return success or failure
    bool attach(
        const boost::shared_ptr<Attribute>& attr);

    /// Detach an attribute, but don't destroy it. The boost::shared_ptr return value
    /// takes care of destroying the attribute in the end.
    /// \param id ID of attribute to detach
    boost::shared_ptr<Attribute> detach(
        Attribute_id id);

    /// Detach an attribute, but don't destroy it. The boost::shared_ptr return value
    /// takes care of destroying the attribute in the end.
    /// \param name name of attribute to detach
    boost::shared_ptr<Attribute> detach(
        const char* name);

    /// Clear the attribute set, i.e. detach and delete all attributes
    void clear();

    /// Fast data exchange of two Attribute_sets.
    /// \param other the one to swap with
    void swap(
        Attribute_set& other);

    /// Look up an attached attribute by ID. Return 0 on failure. O(log n)
    /// \param id ID of attribute to look up
    Attribute* lookup(
        Attribute_id id);
    /// Look up an attached attribute by ID. Return 0 on failure. O(log n)
    /// \param id ID of attribute to look up
    const Attribute* lookup(
        Attribute_id id) const;
    /// Look up an attached attribute by name. Return 0 on failure. O(n)
    /// \param name name of attribute to look up
    const Attribute* lookup(
        const char* name) const;
    /// Look up an attached attribute by name. Return 0 on failure. O(n)
    /// \param name name of attribute to look up
    Attribute* lookup(
        const char* name);

    /// Removing this requires changes in several files, but nevertheless this
    /// should soon be removed though
    typedef MI::ATTR::Attributes Attributes;

    /// Get access to Attributes set.
    const Attributes &get_attributes() const;
    Attributes &get_attributes();

    /// For convenience only
    typedef Attributes::const_iterator Const_iter;
    typedef Attributes::iterator Iter;

    /// Return number of attributes in the attribute_set
    size_t size() const;

    /// Return the approximate size in bytes of the element including all its
    /// substructures. This is used to make decisions about garbage collection.
    size_t get_size() const;

    /// Return a set of all tags in the attribute set
    /// \param[out] result all referenced tags
    void get_references(
        DB::Tag_set	*result) const;

    /// Unique class ID so that the receiving host knows which class to create
    SERIAL::Class_id get_class_id() const;

    /// Serialize the object to the given serializer including all sub elements.
    /// It must return a pointer behind itself (e.g. this + 1) to handle arrays.
    /// \param serializer functions for byte streams
    const SERIAL::Serializable *serialize(
        SERIAL::Serializer*serializer) const;

    /// Deserialize the object and all sub-objects from the given deserializer.
    /// It must return a pointer behind itself (e.g. this + 1) to handle arrays.
    /// \param deser useful functions for byte streams
    SERIAL::Serializable *deserialize(
        SERIAL::Deserializer *deser);

    /// Dump attribute set to info messages, for debugging only.
    void dump() const;

    /// Print the given attribute including its values.
    /// \param transaction the transaction to print attributes from
    void print(
        DB::Transaction *transaction) const;

    /// Factory function used for deserialization
    static SERIAL::Serializable *factory();

    static const SERIAL::Class_id id = ID_ATTRIBUTE_SET; ///< for serialization

  private:
    Attributes m_attrs;					///< the collected attributes

    /// deep copy of all Attributes
    /// \param other copy from this
    void deep_copy(
        const Attribute_set &other);
  public:
    /// lookup an attached attribute by ID - special internal version
    /// the only reason why it is public is that it is available to
    /// Attribute_set_impl_helper in api/api/neuray/neuray_attribute_set_impl_helper.cpp
    /// \param id ID of attribute to look up
    /// \return a shared pointer to the attached attribute
    boost::shared_ptr<Attribute> lookup_shared_ptr(
        Attribute_id	id) const;
};


// see base/lib/mem/i_mem_consumption.h

inline bool has_dynamic_memory_consumption (const Attribute_set&)
{
    return true;
}

inline size_t dynamic_memory_consumption (const Attribute_set& attribute_set)
{
    return attribute_set.get_size() - sizeof(Attribute_set);
}

/// \name Some convenience functions for attached attributes.
/// @{
///
/// Retrieve value of a bool attribute. Return false when not found.
/// \param id which attribute?
/// \return false when not found
bool get_bool_attrib(
    const Attribute_set& attr_set,
    Attribute_id id);
/// Set the value of a bool attribute. Note that this function creates an Attribute
/// if it cannot find one by the given id.
/// \param id which attribute
/// \param v new value
/// \param create create when missing?
void set_bool_attrib(
    Attribute_set& attr_set,
    Attribute_id id,
    bool v,
    bool create=true);
/// Retrieve the override value of the Attribute with the given \p id.
/// Every Attribute comes with an override specifier that during inheritance causes
/// parent values to override child values.
/// \param id which flag?
/// return the override value
Attribute_propagation get_override(
    const Attribute_set& attr_set,
    Attribute_id id);
/// Convenience method to retrieve values of single-value attributes. If the attribute is found
/// and type-correct the value is written to the second argument and the method returns true.
/// \param name      name of the attribute
/// \param value     write value to this reference
/// \return true on success.
template <typename T>
bool get_attribute_value(
    const Attribute_set& attr_set,
    const char* name,
    T& value);
/// @}

/// Iterate over a type. If a returned type is a struct or array (or both),
/// the caller must recurse using the second constructor.
class Type_iterator
{
  public:
    /// constructor: iterate over all toplevel elements of a type
    /// \param type current type
    /// \param values pointer to data of the type instance
    explicit Type_iterator(
        const Type	*type = 0,
        char		*values = 0);

    /// constructor: iterate over the elements of a struct parameter
    /// \param par struct to recurse into
    /// \param values pointer to data of the type instance
    explicit Type_iterator(
        Type_iterator	*par,
        char		*values);

    /// let the iterator pointer to the given type and value
    /// \param type current type
    /// \param values  pointer to data of the type instance
    void set(
        const Type	*type,
        char		*values);

    /// proceed to next element
    void to_next();

    /// at end of element chain?
    bool at_end() const;

    /// \name access functions for the current element.
    /// The caller needs to cast the return value from get_value to the type
    /// indicated by get_typecode because the type can be different for each
    /// step of the iteration. If get_arraysize==0, get_value returns a pointer
    /// to Dynamic_array.
    /// @{
    const char	*get_name()		const;
    /// \param array_elem_type if true, return array's elem type
    ATTR::Type_code  get_typecode(
        bool array_elem_type=false)	const;
    char	*get_value()		const;
    /// \return 0 if it is a dynamic array
    int		 get_arraysize()	const;
    const Type *operator->() const;
    /// @}

    /// get the size of the type currently pointed to. This will only return the
    /// size of one element in an array
    size_t	 sizeof_elem()		const;

  private:
    const ATTR::Type	*m_type;	///< name, type, and array status of parm
    char		*m_value;	///< value of parameter

    Type_iterator& operator=(const Type_iterator&);
#ifndef MI_PLATFORM_MACOSX
    Type_iterator(const Type_iterator&);
#else
  // This constructor should not be used - MacOS compiler somehow insists to do so nevertheless.
  // See comments in implementation.
  public:
    Type_iterator(const Type_iterator&);
#endif
};


/// Retrieve all tags from the given \c Attribute.
/// \note Implemented in attr_attrset.cpp due to existing helper function.
/// \param attr the attribute in question
/// \param [out] result store found tags here
/// \param type_comparison function to compare a given \c Type whether it contains a tag value
void get_references(
    const Attribute& attr,
    DB::Tag_set& result,
    Compare type_comparison);


/// Overload of the default swap() for \c Attribute_sets.
/// \sa Attribute_set::swap().
void swap(
    ATTR::Attribute_set& one,
    ATTR::Attribute_set& other);

}
}

#include "attr_inline_type_iterator.h"
#include "attr_inline_attrset.h"

#endif
