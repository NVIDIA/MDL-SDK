/***************************************************************************************************
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DBLIGHT_DBLIGHT_INFO_H
#define BASE_DATA_DBLIGHT_DBLIGHT_INFO_H

#include <base/data/db/i_db_info.h>

#include <atomic>
#include <functional>
#include <vector>

#include <boost/core/noncopyable.hpp>
#include <boost/intrusive/set.hpp>

#include <base/data/db/i_db_scope.h>
#include <base/lib/robin_hood/robin_hood.h>

#include "dblight_transaction.h"

namespace MI {

namespace DB { class Element_base; }

namespace DBLIGHT {

class Infos_per_name;
class Infos_per_tag;
class Transaction_impl;

namespace bi = boost::intrusive;

/// Infos are created with pin count 1. The pin count is decremented again in
/// #Info_manager::store(), finish_edit() and remove().
class Info_impl : public DB::Info
{
public:
    /// Regular constructor (used for store operations).
    ///
    /// m_references is initialized via element->get_references().
    Info_impl(
        DB::Element_base* element,
        DB::Scope_id scope_id,
        Transaction_impl* transaction,
        mi::Uint32 version,
        DB::Tag tag,
        const char* name);

    /// Regular constructor (used for edit operations).
    Info_impl(
        DB::Element_base* element,
        DB::Scope_id scope_id,
        Transaction_impl* transaction,
        mi::Uint32 version,
        DB::Tag tag,
        const char* name,
        const DB::Tag_set& references);

    /// Regular constructor (used for removal operations).
    Info_impl(
        DB::Scope_id scope_id,
        Transaction_impl* transaction,
        mi::Uint32 version,
        DB::Tag tag);

    /// Constructor for patterns for lookup operations.
    Info_impl(
        DB::Scope_id scope_id,
        DB::Transaction_id transaction_id,
        mi::Uint32 version);

    /// Destructor.
    ///
    /// Decrementing the pin counts of the referenced DB elements is supposed to be done by the
    /// caller.
    virtual ~Info_impl();

    // methods of DB::Info

    void pin() override;

    /// Does \em not invoke the destructor if the pin count drops to zero.
    void unpin() override;

    bool get_is_job() const override { return false; }

    DB::Element_base* get_element() const override { return m_element; }

    SCHED::Job* get_job() const override { return nullptr; }

    DB::Tag get_tag() const override { return m_tag; }

    DB::Scope_id get_scope_id() const override { return m_scope_id; }

    DB::Transaction_id get_transaction_id() const override { return m_transaction_id; }

    mi::Uint32 get_version() const override { return m_version; }

    DB::Privacy_level get_privacy_level() const override { return 0; }

    const char* get_name() const override { return m_name; }

    // internal methods

    /// Returns the pin count.
    mi::Uint32 get_pin_count() const { return m_pin_count; }

    /// Indicates whether this is a removal info.
    bool get_is_removal() const { return !m_element; }

    /// Returns the set of tags referenced by the DB element.
    const DB::Tag_set& get_references() const { return m_references; }

    /// Returns the creator transaction.
    ///
    /// Can be invalid if the info is visible for all open (and future) transactions. RCS:NEU
    const Transaction_impl_ptr& get_transaction() const { return m_transaction; }

    /// Clears the creator transaction.
    void clear_transaction() { m_transaction.reset(); }

    /// Updates #m_references to hold the set of references of the DB element.
    void update_references();

    /// Returns the pointer to containing name set.
    Infos_per_name* get_infos_per_name() const { return m_infos_per_name; }

    /// Sets the pointer to the containing name set.
    void set_infos_per_name( Infos_per_name* infos_per_name);

    /// Replaces the referenced element.
    ///
    /// Only to be used for serialization checks.
    void set_element( DB::Element_base* element);

private:
    // The only members that can change after construction are the element pointer (if serialization
    /// checks for edits are enabled), the pin count, the transaction pointer (reset only), the set
    /// of references (for edits), the name set pointer (if the info has a name) and the hooks for
    /// the intrusive sets.

    /// The DB element managed (and owned) by this instance, \c NULL for removals.
    DB::Element_base* /*almost const*/ m_element = nullptr;

    /// Pin count of this info.
    std::atomic_uint32_t m_pin_count = 1;

    /// \name Used by the comparison operator.
    //@{

    /// ID of the scope this info belongs to.
    const DB::Scope_id m_scope_id;
    /// ID of the creator transaction.
    const DB::Transaction_id m_transaction_id;
     /// Sequence number of the info in the creator transaction.
    const mi::Uint32 m_version;

    //@}
    /// \name Cached from/pointer to the corresponding containers.
    //@{

    /// The tag of this info (never invalid), same as in Infos_per_tag::m_tag.
    const DB::Tag m_tag;
    /// The name of this info. Either \c NULL or points to m_infos_per_name->m_name.c_str().
    const char* const m_name = nullptr;
    /// The corresponding name set that contains this info (or \c NULL iff \c m_name is \c NULL).
    Infos_per_name* m_infos_per_name = nullptr;

    //@}

    /// The set of tags references by m_element.
    ///
    /// \note When editing the referenced element, i.e., between DB::Transaction::edit_element()
    ///       and DB::Transaction::finish_edit(), this value here might be out of sync with the
    ///       return value of DB::Element::get_references(). At other times, this value here serves
    ///       as cache to avoid potential costly calls of DB::Element::get_references().
    DB::Tag_set m_references;

    /// The creator transaction.
    ///
    /// Can be invalid if the info is visible for all open (and future) transactions.
    Transaction_impl_ptr m_transaction;

public:
    /// Hook for Infos_per_name::m_infos.
    bi::set_member_hook<> m_infos_per_name_hook;
    /// Hook for Infos_per_tag::m_infos.
    bi::set_member_hook<> m_infos_per_tag_hook;
};

/// Comparison operators for Info_impl.
///
/// Sort by lexicographic comparison of the scope ID, transaction ID, and version.

bool operator==( const Info_impl& lhs, const Info_impl& rhs);
bool operator!=( const Info_impl& lhs, const Info_impl& rhs);
bool operator<( const Info_impl& lhs, const Info_impl& rhs);
bool operator<=( const Info_impl& lhs, const Info_impl& rhs);
bool operator>( const Info_impl& lhs, const Info_impl& rhs);
bool operator>=( const Info_impl& lhs, const Info_impl& rhs);

/// Set of all infos with a given name.
///
/// Infos_per_name have no explicit pin count.
class Infos_per_name : private boost::noncopyable
{
public:

    using Infos_per_name_hook = bi::member_hook<
        Info_impl, bi::set_member_hook<>, &Info_impl::m_infos_per_name_hook>;

    using Infos_per_name_set = bi::set<Info_impl, Infos_per_name_hook>;

    /// Constructor.
    Infos_per_name( const std::string& name) : m_name( name) { }

    /// Returns the name shared by all infos. Never empty.
    const std::string& get_name() const { return m_name; }

    /// Returns the set of infos sharing this name.
    const Infos_per_name_set& get_infos() const { return m_infos; }

    /// Inserts the info into this set.
    ///
    /// \pre info->get_name() equals m_name.c_str()
    /// \pre The set does not contain the triple (scope ID, transaction ID, version).
    ///
    /// \param info   The info to insert. RCS:NEU
    void insert_info( Info_impl* info);

    /// Removes the info from this set.
    ///
    /// \pre info->get_name() equals m_name.c_str()
    ///
    /// \param info   The info to remove. RCS:NEU
    /// \return       An iterator to the next info in the set.
    Infos_per_name_set::iterator erase_info( Info_impl* info);

    /// Looks up an info.
    ///
    /// \param scope            The scope where to start the look up. RCS:NEU
    /// \param transaction_id   The transaction ID looking up the info.
    /// \return                 The looked up info, or \c nullptr in case of failure. RCS:ICE
    Info_impl* lookup_info( DB::Scope* scope, DB::Transaction_id transaction_id);

private:
    /// Name shared by all infos in m_infos.
    std::string m_name;

    /// Set of infos sharing m_name.
    Infos_per_name_set m_infos;
};

/// Comparison operator for Infos_per_name.
///
/// Sorts by comparison of the name.
inline bool operator<( const Infos_per_name& lhs, const Infos_per_name& rhs)
{
    return lhs.get_name() < rhs.get_name();
}

/// Set of all infos with a given tag.
///
/// Infos_per_tag are created with pin count 1. The pin count is decremented again when removal
/// requests are processed in Info_manager::cleanup_tag(). The pin count is also incremented/
/// decremented by references from other DB elements.
class Infos_per_tag : private boost::noncopyable
{
public:

    using Infos_per_tag_hook = bi::member_hook<
        Info_impl, bi::set_member_hook<>, &Info_impl::m_infos_per_tag_hook>;

    using Infos_per_tag_set = bi::set<Info_impl, Infos_per_tag_hook>;

    /// Constructor.
    Infos_per_tag( DB::Tag tag) : m_tag( tag) { }

    /// Increments the pin count.
    mi::Uint32 pin() { return ++m_pin_count; }

    /// Decrements the pin count.
    ///
    /// Does \em not invoke the destructor if the pin count drops to zero.
    mi::Uint32 unpin() { return --m_pin_count; }

    /// Returns the pin count.
    mi::Uint32 get_pin_count() const { return m_pin_count; }

    /// Indicates whether the tag has been marked for removal in the global scope.
    bool get_is_removed() const { return m_is_removed; }

    /// Returns the tag shared by all infos. Never empty.
    DB::Tag get_tag() const { return m_tag; }

    /// Returns the set of infos sharing this tag.
    const Infos_per_tag_set& get_infos() const { return m_infos; }

    /// Returns the set of infos sharing this tag.
    ///
    /// This non-const variant is used by the cleanup methods and the destructor of Info_manager.
    Infos_per_tag_set& get_infos() { return m_infos; }

    /// Inserts the info into this set.
    ///
    /// \pre info->get_tag() equals m_tag
    /// \pre The set does not contain the triple (scope ID, transaction ID, version).
    ///
    /// \param info   The info to insert. RCS:NEU
    void insert_info( Info_impl* info);

    /// Removes the info from this set.
    ///
    /// \pre info->get_tag() equals m_tag
    ///
    /// \param info   The info to remove. RCS:NEU
    /// \return       An iterator to the next info in the set.
    Infos_per_tag_set::iterator erase_info( Info_impl* info);

    /// Looks up an info.
    ///
    /// \param scope            The scope where to start the look up. RCS:NEU
    /// \param transaction_id   The transaction ID looking up the info.
    /// \return                 The looked up info, or \c nullptr in case of failure. RCS:ICE
    Info_impl* lookup_info( DB::Scope* scope, DB::Transaction_id transaction_id);

    /// Marks the tag for removal in the global scope.
    void set_removed();

private:
    /// Pin count of this tag.
    std::atomic_uint32_t m_pin_count = 1;

    /// Indicates whether this tag was already marked for removal in the global scope.
    bool m_is_removed = false;

    /// Tag shared by all infos in m_infos.
    DB::Tag m_tag;

    /// Set of infos sharing m_tag.
    Infos_per_tag_set m_infos;
};

/// Comparison operator for Infos_per_tag.
///
/// Sorts by comparison of the tag.
inline bool operator<( const Infos_per_tag& lhs, const Infos_per_tag& rhs)
{
    return lhs.get_tag() < rhs.get_tag();
}

/// A minor page holds an array of Infos_per_tag pointers.
///
/// Does \em not own the Infos_per_tag instances.
class Minor_page
{
public:
    friend class Major_page;

    /// Number of index bits handled on this level.
    static const size_t K = 11;
    /// Array size of this level (number of Infos_per_tag).
    static const size_t N = 1U << K;
    /// Valid index range on this level.
    static const size_t L = N;

    /// Constructor.
    Minor_page();

    /// Returns the array element with index \p tag.
    Infos_per_tag* find( size_t index) const;

    /// Sets the array element with index \p tag to \p element.
    ///
    /// \pre The array element is currently \c NULL.
    void insert( size_t index, Infos_per_tag* element);

    /// Clears the array element with index  \p tag.
    ///
    /// \pre The array element is currently not \c NULL.
    void erase( size_t index);

    /// Applies \p f to all non-\c NULL array elements.
    void apply( std::function<void( Infos_per_tag*)> f) const;

    /// Returns all indices with non-\c NULL array elements.
    void get_tags( std::vector<DB::Tag>& tags) const;

private:
    /// Returns the number of non-\c NULL array elements.
    size_t get_local_size() const { return m_local_size; }

    /// The array of minor pages.
    Infos_per_tag* m_infos_per_tags[N];
    /// The number of non-\c NULL array elements.
    size_t m_local_size = 0;
};

/// A major page holds an array of minor page pointers.
///
/// Owns the minor pages, but does \em not own the Infos_per_tag instances.
class Major_page
{
public:
    friend class Tag_tree;

    /// Number of index bits handled on this level.
    static const size_t K = 11;
    /// Array size of this level (number of minor pages).
    static const size_t N = 1U << K;
    /// Valid index range on this level.
    static const size_t L = N * Minor_page::N;
    /// Number of bits to shift the index to obtain the corresponding minor page index.
    static const size_t S = Minor_page::K;
    /// Mask for the index to obtain the index for the next level.
    static const size_t M = (1U << S) - 1;

    /// Constructor.
    Major_page();
    /// Destructor.
    ~Major_page();

    /// Returns the array element with index \p tag.
    Infos_per_tag* find( size_t index) const;

    /// Sets the array element with index \p tag to \p element.
    ///
    /// \pre The array element is currently \c NULL.
    void insert( size_t index, Infos_per_tag* element);

    /// Clears the array element with index  \p tag.
    ///
    /// \pre The array element is currently not \c NULL.
    void erase( size_t index);

    /// Applies \p f to all non-\c NULL array elements.
    void apply( std::function<void( Infos_per_tag*)> f) const;

    /// Returns all indices with non-\c NULL array elements.
    void get_tags( std::vector<DB::Tag>& tags) const;

private:
    /// Returns the number of allocated minor pages.
    size_t get_local_size() const { return m_local_size; }

    /// The array of minor pages.
    Minor_page* m_minor_pages[N];
    /// The number of allocated minor pages.
    size_t m_local_size = 0;
};

/// Conceptually, the tag tree behaves like an array of 2^32 Infos_per_tag pointers.
///
/// The tag value serves as index into this array. Entries can be looked up with #find(), set with
/// #insert() and cleared with #erase().
///
/// Technically, the vector is split into a 3-level hierarchy where the intermediate levels are
/// the major and minor pages, which are allocated and deallocated on demand.
///
/// Owns the major and minor pages, but does \em not own the Infos_per_tag instances.
class Tag_tree
{
public:
    /// Number of index bits handled on this level.
    static const size_t K = 10;
    /// Array size of this level (number of major pages).
    static const size_t N = 1U << K;
    /// Valid index range on this level.
    static const size_t L = N * Major_page::N * Minor_page::N;
    /// Number of bits to shift the index to obtain the corresponding major page index.
    static const size_t S = Major_page::K + Minor_page::K;
    /// Mask for the index to obtain the index for the next level.
    static const size_t M = (1U << S) - 1;

    /// Constructor.
    Tag_tree();
    /// Destructor.
    ~Tag_tree();

    /// Returns the array element with index \p tag.
    Infos_per_tag* find( DB::Tag tag) const;

    /// Sets the array element with index \p tag to \p element.
    ///
    /// \pre The array element is currently \c NULL.
    void insert( DB::Tag tag, Infos_per_tag* element);

    /// Clears the array element with index  \p tag.
    ///
    /// \pre The array element is currently not \c NULL.
    void erase( DB::Tag tag);

    /// Applies \p f to all non-\c NULL array elements.
    void apply( std::function<void( Infos_per_tag*)> f) const;

    /// Returns all indices with non-\c NULL array elements.
    void get_tags( std::vector<DB::Tag>& tags) const;

    /// Returns the number of non-\c NULL array elements.
    size_t size() const { return m_total_size; }

    /// Indicates whether there no non-\c NULL array elements.
    bool empty() const { return m_total_size == 0; }

private:
    /// The array of major pages.
    Major_page* m_major_pages[N];
    /// The number of allocated major pages.
    size_t m_local_size = 0;
    /// The total number of non-\c NULL array elements.
    size_t m_total_size = 0;
};

/// The three-level hierarchy needs to cover the entire value range of tags.
static_assert( Minor_page::N + Major_page::N + Tag_tree::N >= 32);

/// The info manager owns all infos, and indirectly, all elements.
class Info_manager
{
public:
    /// Constructor.
    ///
    /// \param database   Instance of the database this manager belongs to.
    Info_manager( Database_impl* database);

    /// Destructor.
    ///
    /// Destroys all infos and elements.
    ~Info_manager();

    /// Creates an info for the given element and stores it under the given tag/name.
    ///
    /// \param element       The element to store. RCS:TRO
    /// \param scope_id      The ID of the scope the element belongs to.
    /// \param transaction   The transaction creating this info. RCS:NEU
    /// \param version       Sequence number of the operation within the creator transaction.
    /// \param tag           The tag to be used for lookup.
    /// \param name          The name to be used for lookup.
    void store(
        DB::Element_base* element,
        DB::Scope_id scope_id,
        Transaction_impl* transaction,
        mi::Uint32 version,
        DB::Tag tag,
        const char* name);

    /// Looks up an info for the given tag.
    ///
    /// \param tag              The tag to look up.
    /// \param scope            The scope where to start the look up. RCS:NEU
    /// \param transaction_id   The transaction ID looking up the info.
    /// \return                 The looked up info, or \c nullptr in case of failure. RCS:ICE
    Info_impl* lookup_info( DB::Tag tag, DB::Scope* scope, DB::Transaction_id transaction_id);

    /// Looks up an info for the given name.
    ///
    /// \param name             The name to look up.
    /// \param scope            The scope where to start the look up. RCS:NEU
    /// \param transaction_id   The transaction ID looking up the info.
    /// \return                 The looked up info, or \c nullptr in case of failure. RCS:ICE
    Info_impl* lookup_info( const char* name, DB::Scope* scope, DB::Transaction_id transaction_id);

    /// Creates an info for the given element and stores it under the given tag/name.
    ///
    /// Quite similar to store(), except that the implementation is slightly more efficient w.r.t.
    /// the references to other DB elements.
    ///
    /// \param element       The element to edit (this is already the copy). RCS:TRO
    /// \param scope_id      The ID of the scope the element belongs to.
    /// \param transaction   The transaction creating this info. RCS:NEU
    /// \param version       Sequence number of the operation within the creator transaction.
    /// \param tag           The tag to be used for lookup.
    /// \param name          The name to be used for lookup.
    /// \param references    The set of tags referenced by the DB element.
    Info_impl* start_edit(
        DB::Element_base* element,
        DB::Scope_id scope_id,
        Transaction_impl* transaction,
        mi::Uint32 version,
        DB::Tag tag,
        const char* name,
        const DB::Tag_set& references);

    /// Finishes an edit operation.
    ///
    /// Decrements the pin counts of tags referenced when the edit operation started, and increments
    /// the pin counts of the tags referenced now.
    ///
    /// \param info   The info of the edit to finish. RCS:ICR
    void finish_edit( Info_impl* info);

    /// Marks a tag for removal.
    ///
    /// \param scope_id      The ID of the scope where the removal should happen.
    /// \param transaction   The transaction creating this info. RCS:NEU
    /// \param version       Sequence number of the operation within the creator transaction.
    /// \param tag           The tag to mark for removal.
    /// \return              \c true in case of success, \c false otherwise (invalid tag).
    bool remove(
        DB::Scope_id scope_id,
        Transaction_impl* transaction,
        mi::Uint32 version,
        DB::Tag tag);

    /// Runs the garbage collection.
    ///
    /// \param lowest_open   The currently lowest ID of all open transactions (or the ID of the
    ///                      next transaction if there are no open transactions).
    void garbage_collection( DB::Transaction_id lowest_open);

    /// Returns the pin count of the corresponding Infos_per_tag set.
    mi::Uint32 get_tag_reference_count( DB::Tag tag);

    /// Indicates whether the tag has been marked for removal.
    bool get_tag_is_removed( DB::Tag tag);

    /// Dumps the state of the info manager to the stream.
    void dump( std::ostream& s, bool mask_pointer_values);

private:
    /// Runs the garbage collection for a particular tag.
    ///
    /// This includes all cleanup methods known, including handling of pin count zero via a call to
    /// #cleanup_tag_with_pin_count_zero().
    ///
    /// \param tag             The tag to run the garbage collection on.
    /// \param lowest_open     The currently lowest ID of all open transactions (or the ID of the
    ///                        next transaction if there are no open transactions).
    /// \param[out] progress   Indicates whether any progress was made.
    void cleanup_tag_general( DB::Tag, DB::Transaction_id lowest_open, bool& progress);

    /// Runs the garbage collection for a particular tag with pin count zero.
    ///
    /// \param tag             The tag to run the garbage collection on.
    /// \param[out] progress   Indicates whether any progress was made.
    void cleanup_tag_with_pin_count_zero( DB::Tag, bool& progress);

    /// Performs all required steps to destroy a particular info.
    ///
    /// - If the info has a name, then the info is removed from the corresponding Infos_per_name
    ///   set. If that name set becomes empty, it is removed from #m_infos_by_name.
    /// - Removes the info from \p infos_per_tag.
    /// - Decrements the pin counts of the DB elements referenced by \p it.
    /// - Destroys the info (which in turn destroys the DB element managed by the info).
    /// - Returns an iterator to the next info in \p infos_per_tag.
    ///
    /// Tag sets becoming empty are \em not handled by this method, but are supposed to be handled
    /// by the callers #cleanup_tag_general() and #cleanup_tag_with_pin_count_zero().
    ///
    /// \param infos_per_tag   The set that contains \p it.
    /// \param it              The info to clean up.
    /// \return                An iterator to next element in \p infos_per_tag.
    Infos_per_tag::Infos_per_tag_set::iterator cleanup_info(
        Infos_per_tag* infos_per_tag, Infos_per_tag::Infos_per_tag_set::iterator it);

    /// Increments the pin counts of the given tags.
    void increment_pin_counts( const DB::Tag_set& tag_set);

    /// Decrements the pin counts of the given tags.
    void decrement_pin_counts( const DB::Tag_set& tag_set, bool from_gc);

    /// Instance of the database this manager belongs to.
    Database_impl* const m_database;

    /// \name Data structures holding all the infos
    //@{

    using Infos_by_name = robin_hood::unordered_map<std::string, Infos_per_name*>;

    using Infos_by_tag = Tag_tree;

    /// All infos that have a name ordered by name.
    Infos_by_name m_infos_by_name;

    /// All infos ordered by tag.
    Infos_by_tag m_infos_by_tag;

    //@}
    /// \name Garbage collection
    //@{

    /// Different GC methods
    enum Gc_method {
        /// Full sweeps through the entire database until no more progress is made. Simple, but
        /// expensive.
        GC_FULL_SWEEPS_ONLY,
        /// Starts with a full sweep through the entire database. Afterwards, considers only those
        /// tags whose pin count just dropped to zero (or were skipped in previous runs). Much
        /// better for chains of referencing DB elements (in particular if an element references a
        /// tag smaller than its own), but still expensive for transactions with none or almost no
        /// changes.
        GC_FULL_SWEEP_THEN_PIN_COUNT_ZERO,
        /// Starts with tags previously identified for GC. Afterwards, considers only those tags
        /// whose pin count just dropped to zero (or were skipped in previous runs). Best
        /// performance, but also most complex.
        GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO
    };

    /// The default GC method.
    Gc_method m_gc_method = GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO;

    /// Instances of Infos_per_tag on which the GC should run all known cleanup methods.
    ///
    /// Only used when #m_gc_method is #GC_GENERAL_CANDIDATES_THEN_PIN_COUNT_ZERO.
    DB::Tag_set m_gc_candidates_general;

    /// Instances of Infos_per_tag whose pin count is zero.
    ///
    /// Only used when #m_gc_method is not #GC_FULL_SWEEPS_ONLY.
    DB::Tag_set m_gc_candidates_pin_count_zero;

    //@}
};

} // namespace DBLIGHT

} // namespace MI

#endif // BASE_DATA_DBLIGHT_DBLIGHT_INFO_H
