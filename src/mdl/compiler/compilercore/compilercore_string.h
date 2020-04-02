/******************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_STRING_H
#define MDL_COMPILERCORE_STRING_H 1

#include <algorithm>
#include <cstring>
#include <cwchar>
#include <functional>
#include "compilercore_assert.h"

// Replacement for buggy basic_string implementation in the gcc STL

namespace mi {
namespace mdl {

template<typename CharT>
struct char_traits {

    // compare [s, s + count) with [t, t + count)
    static int compare(
        CharT const *s,
        CharT const *t,
        size_t     count)
    {
        for (; 0 < count; --count, ++s, ++t)
            if (!eq(*s, *t))
                return (lt(*s, *t) ? -1 : +1);
        return 0;
    }

    // find length of null-terminated sequence
    static size_t length(CharT const *s) {
        size_t count;
        for (count = 0; !eq(*s, CharT(0)); ++s)
            ++count;
        return count;
    }

    // test for element equality
    static bool eq(CharT const &s, CharT const &t) throw() {
        return s == t;
    }

    // test if s precedes t
    static bool lt(CharT const& s, CharT const &t) throw() {
        return s < t;
    }

    // copy data
    static void copy(CharT *dst, CharT const *src, size_t len) {
        std::memcpy(dst, src, sizeof(CharT) * len);
    }

    // move data
    static void move(CharT *dst, CharT const *src, size_t len) {
        std::memmove(dst, src, sizeof(CharT) * len);
    }

    static CharT const  *find(CharT const *s, size_t n, CharT c) {
        for ( ; n > 0 ; ++s, --n)
            if (eq(*s, c))
                return s;
        return 0;
    }
};

template<>
struct char_traits<wchar_t> {
    typedef wchar_t CharT;

    // compare [s, s + count) with [t, t + count)
    static int compare(
        CharT const *s,
        CharT const *t,
        size_t      count) {
        return count == 0 ? 0 : std::wmemcmp(s, t, count);
    }

    // find length of null-terminated sequence
    static size_t length(CharT const *s) {
        return *s == 0 ? 0 : std::wcslen(s);
    }

    // test for element equality
    static bool eq(CharT const &s, CharT const &t) throw() {
        return s == t;
    }

    // test if s precedes t
    static bool lt(CharT const& s, CharT const &t) throw() {
        return s < t;
    }

    // copy data
    static void copy(CharT *dst, CharT const *src, size_t len) {
        std::wmemcpy(dst, src, len);
    }

    // move data
    static void move(CharT *dst, CharT const *src, size_t len) {
        std::wmemmove(dst, src, len + 1);
    }

    static CharT const  *find(CharT const *s, size_t n, CharT c) {
        for ( ; n > 0 ; ++s, --n)
            if (eq(*s, c))
                return s;
        return 0;
    }
};

template<>
struct char_traits<char> {
    typedef char CharT;

    // compare [s, s + count) with [t, t + count)
    static int compare(
        CharT const *s,
        CharT const *t,
        size_t      count) {
            return count == 0 ? 0 : std::memcmp(s, t, count);
    }

    // find length of null-terminated sequence
    static size_t length(CharT const *s) {
        return *s == 0 ? 0 : std::strlen(s);
    }

    // test for element equality
    static bool eq(CharT const &s, CharT const &t) throw() {
        return s == t;
    }

    // test if s precedes t
    static bool lt(CharT const& s, CharT const &t) throw() {
        return s < t;
    }

    // copy data
    static void copy(CharT *dst, CharT const *src, size_t len) {
        std::memcpy(dst, src, len);
    }

    // move data
    static void move(CharT *dst, CharT const *src, size_t len) {
        std::memmove(dst, src, len);
    }

    static CharT const  *find(CharT const *s, size_t n, CharT c) {
        for ( ; n > 0 ; ++s, --n)
            if (eq(*s, c))
                return s;
        return 0;
    }
};

template<typename CharT, typename Traits, typename Alloc>
class simple_string
{
    static size_t const s_buf_size = 8;
public:
    typedef simple_string<CharT, Traits, Alloc> Self;
    typedef CharT                               value_type;
    typedef Traits                              traits_type;
    typedef Alloc                               allocator_type;
    typedef size_t                              size_type;
    typedef CharT                               &reference;
    typedef CharT const                         &const_reference;
    typedef CharT                               *pointer;
    typedef CharT const                         *const_pointer;

    static size_t const npos = ~(size_t)0;

    // Constructor.
    simple_string(Alloc alloc)
    : m_allocater(alloc)
    , m_buf(m_static_buf)
    , m_reserved_size(s_buf_size - 1)
    , m_size(0)
    {
        m_buf[0] = CharT(0);
    }

    simple_string(Self const &other)
    : m_allocater(other.get_allocator())
    , m_buf(m_static_buf)
    , m_reserved_size(s_buf_size - 1)
    , m_size(0)
    {
        this->assign(other, 0, npos);
    }

    simple_string(Self const &other, size_type off, size_type len = npos)
    : m_allocater(other.get_allocator())
    , m_buf(m_static_buf)
    , m_reserved_size(s_buf_size - 1)
    , m_size(0)
    {
        this->assign(other, off, len);
    }

    simple_string(CharT const *ptr, size_type len, allocator_type const &alloc)
    : m_allocater(alloc)
    , m_buf(m_static_buf)
    , m_reserved_size(s_buf_size - 1)
    , m_size(0)
    {
        this->assign(ptr, len);
    }

    simple_string(CharT const *begin, CharT const *end, allocator_type const &alloc)
    : m_allocater(alloc)
    , m_buf(m_static_buf)
    , m_reserved_size(s_buf_size - 1)
    , m_size(0)
    {
        this->assign(begin, end - begin);
    }

    simple_string(CharT const *ptr, allocator_type const &alloc)
    : m_allocater(alloc)
    , m_buf(m_static_buf)
    , m_reserved_size(s_buf_size - 1)
    , m_size(0)
    {
        this->assign(ptr);
    }

    ~simple_string() {
        // free the buffer
        if (this->m_buf != this->m_static_buf) {
            this->dropbuffer();
        }
    }

    // assign other [off, off + len)
    Self &assign(Self const &other, size_type off, size_type len)
    {
        MDL_ASSERT(off <= other.size());
        if (off > other.size()) {
            // STL fires exception here
            return *this;
        }
        size_type n = other.size() - off;
        if (len == npos)
            len = n;
        if (len < n)
            n = len;  // trim n to size

        if (this == &other) {
            Traits::move(this->m_buf, other.m_buf + off, n);
            this->m_size = n;
            this->m_buf[this->m_size] = CharT(0);
        } else {
            this->reserve(n);
            Traits::copy(this->m_buf, other.m_buf + off, n);
            this->m_size = n;
            this->m_buf[this->m_size] = CharT(0);
        }
        return *this;
    }

    Self &assign(Self const &other)
    {
        if (this == &other)
            return *this;
        return assign(other, 0, npos);
    }

    Self &assign(CharT const *ptr, size_type off, size_type len)
    {
        if (len == npos)
            len = Traits::length(ptr + off);

        if (m_reserved_size < len)
            this->bufalloc(len);

        Traits::move(m_buf, ptr + off, len);
        this->m_size = len;
        this->m_buf[this->m_size] = CharT(0);

        return *this;
    }

    Self &assign(CharT const *ptr, size_type len)
    {
        return this->assign(ptr, 0, len);
    }

    Self &assign(CharT const *ptr)
    {
        return this->assign(ptr, 0, npos);
    }

    Self &append(Self const &other)
    {
        return this->append(other, 0, npos);
    }

    Self &append(Self const &other, size_type off, size_type len)
    {
        if (len == npos) {
            len = other.size();
            if (off > len)
                len = 0;
            else
                len -= off;
        }

        return append(other.m_buf, off, len);
    }

    Self &append(CharT const *ptr, size_type off, size_type len)
    {
        if (len == npos) {
            len = Traits::length(ptr + off);
        }

        if (this->m_size + len > this->m_reserved_size)
            this->reserve(std::max(this->m_size + len, 2*this->m_size));

        Traits::copy(this->m_buf + this->m_size, ptr + off, len);
        this->m_size += len;
        this->m_buf[this->m_size] = CharT(0);

        return *this;
    }

    Self &append(CharT const *ptr, size_type len)
    {
        return append(ptr, 0, len);
    }

    Self &append(CharT const *ptr)
    {
        return append(ptr, 0, npos);
    }

    Self &append(CharT ch)
    {
        if (this->m_size + 1 > this->m_reserved_size)
            this->reserve(std::max(this->m_size + 1, 2*this->m_size));

        this->m_buf[this->m_size] = ch;
        ++this->m_size;
        this->m_buf[this->m_size] = CharT(0);

        return *this;
    }

    Self &append(size_t count, CharT ch)
    {
        if (this->m_size + count > this->m_reserved_size)
            this->reserve(std::max(this->m_size + count, 2*this->m_size));

        for (size_t i = this->m_size; i < this->m_size + count; ++i)
            this->m_buf[i] = ch;
        this->m_size += count;
        this->m_buf[this->m_size] = CharT(0);

        return *this;
    }

    Self &operator=(Self const &other)
    {
        if (this != &other) {
            if (this->get_allocator() != other.get_allocator()) {
                this->bufalloc(0);
                this->m_allocater = other.get_allocator();
            }

            this->assign(other);
        }
        return *this;
    }

    Self &operator=(CharT const *ptr)
    {
        return this->assign(ptr);
    }

    Self &operator+=(Self const &other)
    {
        return this->append(other);
    }

    Self &operator+=(CharT const *ptr)
    {
        return this->append(ptr);
    }

    Self &operator+=(CharT ch) {
        return this->append(ch);
    }

    reference operator[](size_type off)
    {
        return this->data()[off];
    }

    const_reference operator[](size_type off) const
    {
        return this->data()[off];
    }

    // look for ch before or at off
    size_type find_last_of(CharT ch, size_type off = npos) const
    {
        return rfind(const_pointer(&ch), off, 1);
    }

    // look for none of [ptr, ptr + len) before or at off
    size_type find_last_not_of(const_pointer ptr, size_type off, size_type len) const
    {
        if (0 < this->m_size) {
            const_pointer p = this->data() + (off < this->m_size ? off : this->m_size - 1);
            for (; ; --p) {
                if (Traits::find(ptr, len, *p) == 0)
                    return (p - this->data());
                else if (p == this->data())
                    break;
            }
        }
        return npos;  // not found
    }

    // look for non ch before or at off
    size_type find_last_not_of(CharT ch, size_type off = npos) const
    {
        return find_last_not_of(const_pointer(&ch), off, 1);
    }

    Self substr(size_type off = 0, size_type count = npos) const
    {
        return Self(*this, off, count);
    }

    // look for other beginning at or after off
    size_type find(Self const &other, size_type off = 0) const
    {
        return find(other.data(), off, other.size());
    }

    // look for [ptr, ptr + len) beginning at or after off
    size_type find(const_pointer ptr, size_type off, size_type len) const
    {
        if (len == 0 && off <= this->m_size)
            return off;     // empty string always matches (if inside string)

        size_type n = this->m_size - off;
        if (off < this->m_size && len <= n) {
            const_pointer p, q;
            for (n -= len - 1, p = this->data() + off;
                (q = Traits::find(p, n, *ptr)) != 0;
                n -= q - p + 1, p = q + 1)
                if (Traits::compare(q, ptr, len) == 0)
                    return (q - this->data());  // found
        }
        return npos;  // not found
    }

    // look for [ptr, <null>) beginning at or after off
    size_type find(const_pointer ptr, size_type off = 0) const
    {
        return find(ptr, off, Traits::length(ptr));
    }

    // look for ch at or after off
    size_type find(CharT ch, size_type off = 0) const
    {
        return find(const_pointer(&ch), off, 1);
    }

    // look for [ptr, ptr + len) beginning before or at off
    size_type rfind(CharT const *ptr, size_type off, size_type len) const
    {
        if (len == 0)
            return off < this->m_size ? off : this->m_size;  // empty string always matches
        if (len <= this->m_size) {
            CharT const *p = this->data() + (off < this->m_size - len ? off : this->m_size - len);
            for (; ; --p) {
                if (Traits::eq(*p, *ptr)  && Traits::compare(p, ptr, len) == 0)
                    return p - this->data();  // found
                else if (p == this->data())
                    break;  // reach start
            }
        }
        return npos;  // not found
    }

    // look for [ptr, <null>) beginning before or at off
    size_type rfind(CharT const *ptr, size_type off = npos) const
    {
        return rfind(ptr, off, Traits::length(ptr));
    }

    // look for ch before or at off
    size_type rfind(CharT ch, size_type off = npos) const
    {
        return rfind(const_pointer(&ch), off, 1);
    }

    // compare [off, off + len0) with [ptr, ptr + len1)
    int compare(size_type off, size_type len0, CharT const *ptr, size_type len1) const
    {
        if (this->m_size < off) {
            // STL will fire exception
            return 1;
        }
        if (this->m_size - off < len0)
            len0 = this->m_size - off;  // trim len0 to size

        size_type res = Traits::compare(this->data() + off, ptr, len0 < len1 ? len0 : len1);
        if (res != 0)
            return int(res);
        return len0 < len1 ? -1 : len0 == len1 ? 0 : +1;
    }

    // compare [off, off + len) with [ptr, <null>)
    int compare(size_type off, size_type len, CharT const *ptr) const
    {
        return compare(off, len, ptr, Traits::length(ptr));
    }

    // compare [0, size()) with [ptr, <null>)
    int compare(CharT const *ptr) const
    {
        return compare(0, this->m_size, ptr, Traits::length(ptr));
    }

    // compare [0, size()) with other
    int compare(Self const &other) const
    {
        return compare(0, this->m_size, other.data(), other.m_size);
    }

    void clear() throw()
    {
        this->m_size = 0;
        this->m_buf[0] = CharT(0);
    }

    pointer begin() throw() { return this->m_buf; }

    const_pointer begin() const throw() { return this->m_buf; }

    pointer end() throw() { return this->m_buf + this->m_size; }

    const_pointer end() const throw() { return this->m_buf + this->m_size; }

    pointer data() throw() { return this->m_buf; }

    const_pointer data() const throw() { return this->m_buf; }

    const_pointer c_str() const throw() { return data(); }

    bool empty() const throw() { return this->m_size == 0; }

    size_type size() const throw() { return this->m_size; }

    size_type length() const throw() { return this->m_size; }

    CharT *erase(CharT *first, CharT *last) {
        if (first != last) {
            // The move includes the terminating _CharT().
            Traits::move(first, last, end() - last + 1);
            this->m_size -= (last - first);
        }
        return first;
    }

    CharT *erase(CharT *first) {
        return this->erase(first, this->end());
    }

    void resize(size_type n, CharT ch)
    {
        if (n <= size())
            erase(data() + n, data() + size());
        else
            append(n - size(), ch);
    }

    void resize(size_type n) { resize(n, CharT()); }

    size_type capacity() const throw() { return this->m_reserved_size; }

    allocator_type &get_allocator() const { return this->m_allocater; }

    void reserve(size_type len)
    {
        if (len <= this->m_reserved_size)
            return;

        if (this->empty())
            return bufalloc(len);

        CharT *old = this->m_buf;

        this->m_reserved_size = len;
        this->m_buf = m_allocater.allocate(this->m_reserved_size + 1);

        Traits::copy(this->m_buf, old, this->m_size);
        this->m_buf[this->m_size] = CharT(0);

        if (old != this->m_static_buf) {
            m_allocater.deallocate(old, this->m_reserved_size + 1);
        }
    }

    // exchange contents with other
    void swap(Self &other)
    {
        if (this == &other)
            return;     // same object, do nothing
        
        if (this->get_allocator() == other.get_allocator()) {
            // same allocator, swap control information
            this->swap_buffers(other);
        } else {
            // different allocator, do multiple assigns
            Self t = *this;

            *this = other;
            other = t;
        }
    }

private:
    void swap_buffers(Self &other) {
        if (this->m_buf != this->m_static_buf && other.m_buf != other.m_static_buf) {
            // none are dynamic
            std::swap(this->m_buf, other.m_buf);
        } else if (this->m_buf == this->m_static_buf) {
            // our is static
            Traits::copy(other.m_static_buf, this->m_static_buf, s_buf_size);
            this->bufalloc(other.m_reserved_size);
            Traits::copy(this->m_buf, other.m_buf, other.m_size + 1);

            other.dropbuffer();
        } else if (other.m_buf == other.m_static_buf) {
            // other is static
            Traits::copy(this->m_static_buf, other.m_static_buf, s_buf_size);
            other.bufalloc(this->m_reserved_size);
            Traits::copy(other.m_buf, this->m_buf, this->m_size + 1);

            this->dropbuffer();
        } else {
            // both are static
            for (size_type i = 0; i < s_buf_size; ++i) {
                std::swap(this->m_static_buf[i], other.m_static_buf[i]);
            }
        }

        std::swap(this->m_size,          other.m_size);
        std::swap(this->m_reserved_size, other.m_reserved_size);
    }

    void bufalloc(size_type len)
    {
        if (len <= this->m_reserved_size)
            return;

        if (this->m_buf != this->m_static_buf) {
            dropbuffer();
        }
        this->m_reserved_size = len;
        this->m_buf = m_allocater.allocate(this->m_reserved_size + 1);

        this->m_size = 0;
    }

    void dropbuffer() {
        this->m_allocater.deallocate(this->m_buf, this->m_reserved_size + 1);
        this->m_buf = this->m_static_buf;
        this->m_reserved_size = s_buf_size - 1;
    }

private:
    CharT m_static_buf[s_buf_size];

    mutable Alloc m_allocater;

    CharT *m_buf;

    size_type m_reserved_size;

    size_t m_size;
};

// simple_string template operators

// return string + string
template<typename CharT, typename Traits, typename Alloc>
inline simple_string<CharT, Traits, Alloc> operator+(
    simple_string<CharT, Traits, Alloc> const &left,
    simple_string<CharT, Traits, Alloc> const &right)
{
    simple_string<CharT, Traits, Alloc> res(left.get_allocator());
    res.reserve(left.size() + right.size());
    res += left;
    res += right;
    return res;
}

// return char + string
template<typename CharT, typename Traits, typename Alloc>
inline simple_string<CharT, Traits, Alloc> operator+(
    CharT const                               *left,
    simple_string<CharT, Traits, Alloc> const &right)
{
    simple_string<CharT, Traits, Alloc> res(right.get_allocator());
    res.reserve(Traits::length(left) + right.size());
    res += left;
    res += right;
    return res;
}

// return character + string
template<typename CharT, typename Traits, typename Alloc>
inline simple_string<CharT, Traits, Alloc> operator+(
    CharT                                     left,
    simple_string<CharT, Traits, Alloc> const &right)
{
    simple_string<CharT, Traits, Alloc> res(right.get_allocator());
    res.reserve(1 + right.size());
    res += left;
    res += right;
    return res;
}

// return string + char
template<typename CharT, typename Traits, typename Alloc>
inline simple_string<CharT, Traits, Alloc> operator+(
    simple_string<CharT, Traits, Alloc> const &left,
    CharT const                               *right)
{
    simple_string<CharT, Traits, Alloc> res(left.get_allocator());
    res.reserve(left.size() + Traits::length(right));
    res += left;
    res += right;
    return res;
}

// return string + character
template<typename CharT, typename Traits, typename Alloc>
inline simple_string<CharT, Traits, Alloc> operator+(
    simple_string<CharT, Traits, Alloc> const &left,
    CharT                                     right)
{
    simple_string<CharT, Traits, Alloc> res(left.get_allocator());
    res.reserve(left.size() + 1);
    res += left;
    res += right;
    return (res);
}

// test for string equality
template<typename CharT, typename Traits, typename Alloc>
inline bool operator==(
    simple_string<CharT, Traits, Alloc> const &left,
    simple_string<CharT, Traits, Alloc> const &right) {
    return left.compare(right) == 0;
}

// test for char vs. string equality
template<typename CharT, typename Traits, typename Alloc>
inline bool operator==(
    CharT const                               *left,
    simple_string<CharT, Traits, Alloc> const &right)
{
    return right.compare(left) == 0;
}

// test for string vs. char equality
template<typename CharT, typename Traits, typename Alloc>
inline bool operator==(
    simple_string<CharT, Traits, Alloc> const &left,
    CharT const                               *right)
{
    return left.compare(right) == 0;
}

// test for string inequality
template<typename CharT, typename Traits, typename Alloc>
inline bool operator!=(
    simple_string<CharT, Traits, Alloc> const &left,
    simple_string<CharT, Traits, Alloc> const &right)
{
    return !(left == right);
}

// test for char vs. string inequality
template<typename CharT, typename Traits, typename Alloc>
inline bool operator!=(
    CharT const                               *left,
    simple_string<CharT, Traits, Alloc> const &right)
{
    return !(left == right);
}

// test for string vs. char inequality
template<typename CharT, typename Traits, typename Alloc>
inline bool operator!=(
    simple_string<CharT, Traits, Alloc> const &left,
    CharT const                               *right)
{
    return !(left == right);
}

// test for string less
template<typename CharT, typename Traits, typename Alloc>
inline bool operator<(
    simple_string<CharT, Traits, Alloc> const &left,
    simple_string<CharT, Traits, Alloc> const &right) {
    return left.compare(right) < 0;
}

namespace {
template<typename CharT>
struct ntcs_hash
{
    size_t operator()(CharT const *s) const
    {
        if (s == NULL)
            return 0;
        size_t offset_basis;
        size_t prime;
        if (sizeof(size_t) == 8) {
            offset_basis = 14695981039346656037ULL;
            prime = 1099511628211ULL;
        } else {
            offset_basis = 2166136261U;
            prime = 16777619U;
        }

        size_t value = offset_basis;
        while (*s != CharT(0)) {
            value ^= size_t(*s);
            value *= prime;
            ++s;
        }

        return value;
    }
};

}  // anonymous

// hash functor for simple_string
template<typename String>
struct string_hash
{
    size_t operator()(String const &key) const {
        ntcs_hash<typename String::value_type> hasher;
        return hasher(key.c_str());
    }
};

}  // mdl
}  // mi

#endif  // MDL_COMPILERCORE_STRING_H
