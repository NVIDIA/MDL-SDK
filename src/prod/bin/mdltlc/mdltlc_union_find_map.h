/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

#ifndef MDLTLC_UNION_FIND_MAP_H
#define MDLTLC_UNION_FIND_MAP_H 1

#include <iostream>
#include <ios>
#include <optional>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

/// Union-find data structure that performs efficient set union
/// operations and additionally maps sets to values.
///
/// Algorithms taken from Wikipedia:
/// <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>
/// retrieved on 2023-11-08.
///
/// This data structure does not take ownership of the keys, it only
/// works on pointers to keys. It does take ownership of values,
/// though. Values must be copyable and default constructable and
/// should be cheap to construct. The expected use case is to use
/// pointers as values.
template <class K, class V>
class Union_find_map {
    static_assert(std::is_default_constructible_v<V>,
                  "Parameter V must be default-constructible");

  public:
    typedef typename mi::mdl::Arena_ptr_hash_map<K, K *>::Type Parent_map;
    typedef typename mi::mdl::Arena_ptr_hash_map<K, size_t>::Type Rank_map;
    typedef typename mi::mdl::Arena_ptr_hash_map<K, V>::Type Value_map;
    typedef typename mi::mdl::Arena_ptr_hash_set<K>::Type Key_set;
    typedef typename mi::mdl::Arena_vector<K *>::Type Key_vector;
    typedef typename mi::mdl::Arena_vector<Key_set>::Type Key_set_vector;

  Union_find_map(mi::mdl::Memory_arena *arena)
      : m_arena(arena)
        , m_parent(arena)
        , m_rank(arena)
        , m_value(arena) {}

    /// Create a deep copy of the given Union_find_map. The keys are
    /// shared with the other map, because they are stored as
    /// pointers, values are copied.
    Union_find_map(Union_find_map const &other)
        : m_arena(other.m_arena)
        , m_parent(m_arena)
        , m_rank(m_arena)
        , m_value(m_arena) {
        for (auto &p : other.m_parent) {
            m_parent.insert(std::make_pair(p.first, p.second));
        }
        for (auto &p : other.m_rank) {
            m_rank.insert(std::make_pair(p.first, p.second));
        }
        for (auto &p : other.m_value) {
            m_value.insert(std::make_pair(p.first, p.second));
        }
    }

    Union_find_map<K, V> & operator=(Union_find_map<K, V>&& other) {
        m_arena = other.m_arena;
        std::swap(m_parent, other.m_parent);
        std::swap(m_rank, other.m_rank);
        std::swap(m_value, other.m_value);
        return *this;
    }

    /// Declare `k' to be a member of one of the sets in this
    /// union-find data structure. This must be called for all `k'
    /// that are to be used with one of the other methods of this
    /// class.
    void announce(K *k) {
        MDL_ASSERT(k);
        MDL_ASSERT(m_parent.find(k) == m_parent.end());
        m_parent[k] = k;
        m_rank[k] = size_t(0);
    }

    /// Map `k' to the representative of its set, which could be `k'
    /// itself.
    K *find(K *k) {
        MDL_ASSERT(k);
        MDL_ASSERT(m_parent[k]);
        // Perform path-halving, attributed to Tarjan and Van Leeuwen
        // on Wikipedia.
        while (m_parent[k] != k) {
            m_parent[k] = m_parent[m_parent[k]];
            k = m_parent[k];
        }
        return k;
    }

    /// Return a pointer to the value associated with the
    /// representative of `k'.
    V* get_value(K *k)  {
        MDL_ASSERT(k);
        MDL_ASSERT(m_parent[k]);
        k = find(k);
        typename Value_map::iterator it = m_value.find(k);
        if (it != m_value.end()) {
            return &it->second;
        } else {
            return nullptr;
        }
    }

    /// Create the union of the sets in which `x' and `y' are. The
    /// values associated with the sets of `x' and `y' are combined
    /// (if any) using the given `merge' function. If merge returns
    /// `false' in the boolean `success' parameter, the sets will not
    /// be unioned, and `false' is returned.  Return `true' on success
    /// (that is, if no conflicts are found for the values of the sets
    /// when merging them).
    bool union_(K *x, K *y, std::function<V(V &, V &, bool &)> merge) {
        MDL_ASSERT(x);
        MDL_ASSERT(y);
        MDL_ASSERT(m_parent[x]);
        MDL_ASSERT(m_parent[y]);

        // Replace nodes by roots
        x = find(x);
        y = find(y);

        if (x == y) {
            return true;
        }

        bool success = true;
        V new_val{};
        bool has_val = true;
        typename Value_map::iterator xit = m_value.find(x);
        typename Value_map::iterator yit = m_value.find(y);
        if (xit != m_value.end()) {
            if (yit != m_value.end()) {
                new_val = merge(xit->second, yit->second, success);
                if (!success) {
                    return false;
                }
            } else {
                new_val = xit->second;
            }
        } else {
            if (yit != m_value.end()) {
                new_val = yit->second;
            } else {
                has_val = false;
            }
        }

        // Make sure the new parent has a higher rank.
        if (m_rank[x] < m_rank[y]) {
            std::swap(x, y);
        }

        // Make x the parent of y.
        m_parent[y] = x;

        if (m_rank[x] == m_rank[y]) {
            m_rank[x] = m_rank[x] + 1;
        }

        // Assign merged value (if any) to parent node.
        if (has_val) {
            m_value[x] = new_val;
        }
        return success;
    }

    /// Set `value' as the value of the set of `x'. If there is
    /// already a value associated with that set, the merge function
    /// is called on the existing and the new value. The existing
    /// value of the set of `x' is not changed if the `merge' function
    /// returns `false' in its `success' parameter.
    bool set_value(K *x, V value, std::function<V(V &, V &, bool & success)> merge) {
        MDL_ASSERT(x);
        MDL_ASSERT(m_parent[x]);
        x = find(x);

        bool success = true;
        typename Value_map::iterator xit = m_value.find(x);
        if (xit != m_value.end()) {
            V new_val = merge(xit->second, value, success);
            if (success) {
                m_value[x] = new_val;
            }
        } else {
            m_value[x] = value;
        }
        return success;
    }

    /// Dump the state of the data structure to the given output
    /// stream. The output format is intentionally not stable.
    void dump(std::ostream &out) {
        out << std::string("Union-find-map:\n");
        for (typename Parent_map::const_iterator it = m_parent.begin();
             it != m_parent.end();
             ++it) {
            out << std::hex << it->first <<
                "(" << std::dec << m_rank[it->first] << ")" <<
                " rep: " << std::hex << it->second <<
                "(" << std::dec << m_rank[it->second] << ")";
            if (V *v = get_value(it->first)) {
                out << " val: " << std::hex << v;
            }
            out << "\n";
        }
    }

    /// Return the list of sets represented by this union-find map.
    Key_set_vector sets()  {
        typedef typename mi::mdl::Arena_ptr_hash_map<K, Key_set>::Type
            Key_collection_map;

        Key_set_vector result(m_arena);
        Key_collection_map coll_map(m_arena);
        for (typename Parent_map::const_iterator it = m_parent.begin();
             it != m_parent.end();
             ++it) {
            K* elem = it->first;
            K* root = find(it->first);
            typename Key_collection_map::iterator kci = coll_map.find(root);
            if (kci == coll_map.end()) {
                Key_set ks(m_arena);
                ks.insert(elem);
                coll_map.insert(std::move(std::make_pair(root, ks)));
            } else {
                kci->second.insert(elem);
            }
        }
        for (auto &coll_map_entry : coll_map) {
            Key_set &ks = coll_map_entry.second;
            result.push_back(std::move(ks));
        }
        return result;
    }

    Key_set keys() {
        Key_set ret(m_arena);
        for (auto [k, v] : m_parent) {
            ret.insert(k);
        }
        return ret;
    }

  private:
    mi::mdl::Memory_arena *m_arena;
    Parent_map m_parent;
    Rank_map m_rank;
    Value_map m_value;
};

template <class V>
class Dense_union_find_map {
    static_assert(std::is_default_constructible_v<V>,
                  "Parameter V must be default-constructible");

  public:
    typedef typename mi::mdl::Arena_vector<size_t>::Type Parent_map;
    typedef typename mi::mdl::Arena_vector<size_t>::Type Rank_map;
    typedef typename mi::mdl::Arena_vector<std::optional<V>>::Type Value_map;
    typedef typename mi::mdl::Arena_hash_set<size_t>::Type Key_set;
    typedef typename mi::mdl::Arena_vector<size_t>::Type Key_vector;
    typedef typename mi::mdl::Arena_vector<Key_set>::Type Key_set_vector;

  Dense_union_find_map(mi::mdl::Memory_arena *arena)
      : m_arena(arena)
        , m_parent(arena)
        , m_rank(arena)
        , m_value(arena) {}

    /// Create a deep copy of the given Dense_union_find_map. The keys are
    /// shared with the other map, because they are stored as
    /// pointers, values are copied.
    Dense_union_find_map(Dense_union_find_map const &other)
        : m_arena(other.m_arena)
        , m_parent(m_arena)
        , m_rank(m_arena)
        , m_value(m_arena) {
        m_parent = other.m_parent;
        m_rank = other.m_rank;
        m_value = other.m_value;
    }

    Dense_union_find_map<V> & operator=(Dense_union_find_map<V>&& other) {
        m_arena = other.m_arena;
        std::swap(m_parent, other.m_parent);
        std::swap(m_rank, other.m_rank);
        std::swap(m_value, other.m_value);
        return *this;
    }

    /// Declare `k' to be a member of one of the sets in this
    /// union-find data structure. This must be called for all `k'
    /// that are to be used with one of the other methods of this
    /// class.
    ///
    /// Note that for dense union-find maps, all keys from zero up to
    /// the largest one in the map exist. Keys that are smaller than
    /// the largest one that has been announced are announced
    /// implicitly and default-initialized without a value because of
    /// how the map works internally.
    void announce(size_t k) {
        if (k >= m_parent.size()) {
            m_parent.resize(k + 1, 0);
            m_rank.resize(k + 1, size_t(0));
            m_value.resize(k + 1, {});
        }
        m_parent[k] = k;
    }

    /// Map `k' to the representative of its set, which could be `k'
    /// itself.
    size_t find(size_t k) {
        // Perform path-halving, attributed to Tarjan and Van Leeuwen
        // on Wikipedia.
        while (m_parent[k] != k) {
            m_parent[k] = m_parent[m_parent[k]];
            k = m_parent[k];
        }
        return k;
    }

    /// Return a pointer to the value associated with the
    /// representative of `k'.
    std::optional<V> get_value(size_t k)  {
        k = find(k);
        std::optional<V> &val = m_value[k];
        if (val) {
            return std::make_optional(val.value());
        } else {
            return {};
        }
    }

    /// Create the union of the sets in which `x' and `y' are. The
    /// values associated with the sets of `x' and `y' are combined
    /// (if any) using the given `merge' function. If merge returns
    /// `false' in the boolean `success' parameter, the sets will not
    /// be unioned, and `false' is returned.  Return `true' on success
    /// (that is, if no conflicts are found for the values of the sets
    /// when merging them).
    bool union_(size_t x, size_t y, std::function<V(V &, V &, bool &)> merge) {
        // Replace nodes by roots
        x = find(x);
        y = find(y);

        if (x == y) {
            return true;
        }

        bool success = true;
        std::optional<V> new_val{};
        if (m_value[x]) {
            if (m_value[y]) {
                new_val = merge(m_value[x].value(), m_value[y].value(), success);
                if (!success) {
                    return false;
                }
            } else {
                new_val = m_value[x];
            }
        } else {
            if (m_value[y]) {
                new_val = m_value[y];
            }
        }

        // Make sure the new parent has a higher rank.
        if (m_rank[x] < m_rank[y]) {
            std::swap(x, y);
        }

        // Make x the parent of y.
        m_parent[y] = x;

        if (m_rank[x] == m_rank[y]) {
            m_rank[x] = m_rank[x] + 1;
        }

        // Assign merged value (if any) to parent node.
        if (new_val) {
            m_value[x] = new_val;
        }
        return success;
    }

    /// Set `value' as the value of the set of `x'. If there is
    /// already a value associated with that set, the merge function
    /// is called on the existing and the new value. The existing
    /// value of the set of `x' is not changed if the `merge' function
    /// returns `false' in its `success' parameter.
    bool set_value(size_t x, V value, std::function<V(V &, V &, bool & success)> merge) {
        x = find(x);

        bool success = true;
        if (m_value[x]) {
            V new_val = merge(m_value[x].value(), value, success);
            if (success) {
                m_value[x] = {new_val};
            }
        } else {
            m_value[x] = value;
        }
        return success;
    }

    /// Dump the state of the data structure to the given output
    /// stream. The output format is intentionally not stable.
    void dump(std::ostream &out) {
        out << std::string("Union-find-map:\n");
        for (size_t elem = 0; elem < m_parent.size(); elem++) {
            out << std::hex << elem <<
                "(" << std::dec << m_rank[elem] << ")" <<
                " rep: " << std::hex << m_parent[elem] <<
                "(" << std::dec << m_rank[m_parent[elem]] << ")";
            if (m_value[elem]) {
                out << " val: " << std::hex << m_value[elem].value();
            }
            out << "\n";
        }
    }

    /// Return the list of sets represented by this union-find map.
    Key_set_vector sets()  {
        typedef typename mi::mdl::Arena_hash_map<size_t, Key_set>::Type
            Key_collection_map;

        Key_set_vector result(m_arena);
        Key_collection_map coll_map(m_arena);
        for (size_t elem = 0; elem < m_parent.size(); elem++) {
            size_t root = find(elem);
            Key_collection_map::iterator it(coll_map.find(root));
            if (it != coll_map.end()) {
                Key_set &ks = it->second;
                ks.insert(elem);
            } else {
                Key_set ks(m_arena);
                ks.insert(elem);
                coll_map.insert(std::make_pair(root, std::move(ks)));
            }
        }
        for (auto &coll_map_entry : coll_map) {
            Key_set &ks = coll_map_entry.second;
            result.push_back(std::move(ks));
        }
        return result;
    }

    Key_set keys() {
        Key_set ret(m_arena);
        for (auto [k, v] : m_parent) {
            ret.insert(k);
        }
        return ret;
    }

  private:
    mi::mdl::Memory_arena *m_arena;
    Parent_map m_parent;
    Rank_map m_rank;
    Value_map m_value;
};

#endif
