/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/** \file
 ** \brief      Fixed-size bitset similar to the STL bitset template.
 **             See \c Bitset for details.
 **
 **/

#ifndef BASE_LIB_CONT_BITSET_INLINE_H
#define BASE_LIB_CONT_BITSET_INLINE_H

namespace MI {
namespace CONT {

namespace DETAIL {

template <typename T, size_t count>
Bitset_storage_base<T,count>::Bitset_storage_base()
{
    memset(m_data,0,count*sizeof(T));
}


template <typename T, size_t count>
Bitset_storage_base<T,count>::Bitset_storage_base(T data)
{
    memset(m_data,0,count*sizeof(T));
    m_data[0] = data;
}


template <typename T, size_t count>
bool Bitset_storage_base<T,count>::equals(const Bitset_storage_base& rhs) const
{
    for (size_t i=0; i<count; ++i)
        if (m_data[i] != rhs.m_data[i])
            return false;
    return true;
}


template <typename T, size_t count>
inline void Bitset_storage_base<T,count>::set(size_t bit)
{
    m_data[index(bit)] |= (T(1) << subbit(bit));
}


template <typename T, size_t count>
inline void Bitset_storage_base<T,count>::unset(size_t bit)
{
    m_data[index(bit)] &= ~(T(1) << subbit(bit));
}


template <typename T, size_t count>
inline bool Bitset_storage_base<T,count>::is_set(size_t bit) const
{
    return 0 != (m_data[index(bit)] & (T(1) << subbit(bit)));
}


template <typename T, size_t count>
bool Bitset_storage_base<T,count>::any(const Bitset_storage_base& data) const
{
    for (size_t i=0; i<count; ++i)
        if (0 != (data.m_data[i] & m_data[i]))
            return true;
    return false;
}


template <typename T, size_t count>
bool Bitset_storage_base<T,count>::any() const
{
    for (size_t i=0; i<count; ++i)
        if (0 != m_data[i])
            return true;
    return false;
}


template <typename T, size_t count>
bool Bitset_storage_base<T,count>::all(const Bitset_storage_base& data) const
{
    for (size_t i=0; i<count; ++i)
        if (data.m_data[i] != (data.m_data[i] & m_data[i]))
            return false;
    return true;
}


template <typename T, size_t count>
void Bitset_storage_base<T,count>::do_and(const Bitset_storage_base& rhs)
{
    for (size_t i=0; i<count; ++i)
        m_data[i] &= rhs.m_data[i];
}


template <typename T, size_t count>
void Bitset_storage_base<T,count>::do_or(const Bitset_storage_base& rhs)
{
    for (size_t i=0; i<count; ++i)
        m_data[i] |= rhs.m_data[i];
}


template <typename T, size_t count>
void Bitset_storage_base<T,count>::do_xor(const Bitset_storage_base& rhs)
{
    for (size_t i=0; i<count; ++i)
        m_data[i] ^= rhs.m_data[i];
}


template <typename T, size_t count>
void Bitset_storage_base<T,count>::flip()
{
    for (size_t i=0; i<count; ++i)
        m_data[i] = ~m_data[i];
}


template <typename T, size_t count>
inline T& Bitset_storage_base<T,count>::data(size_t bit)
{
    return m_data[index(bit)];
}


template <typename T, size_t count>
inline const T* Bitset_storage_base<T,count>::begin() const
{
    return m_data;
}


template <typename T, size_t count>
inline const T* Bitset_storage_base<T,count>::end() const
{
    return m_data+count;
}


template <typename T, size_t count>
inline T* Bitset_storage_base<T,count>::begin()
{
    return m_data;
}


template <typename T, size_t count>
inline T* Bitset_storage_base<T,count>::end()
{
    return m_data+count;
}


template <typename T, size_t count>
inline T Bitset_storage_base<T,count>::mask(size_t bit)
{
    return T(1) << subbit(bit);
}


template <typename T, size_t count>
inline size_t Bitset_storage_base<T,count>::index(size_t bit)
{
    return bit / Bit_sizeof<T>::value;
}


template <typename T, size_t count>
inline size_t Bitset_storage_base<T,count>::subbit(size_t bit)
{
    return bit % Bit_sizeof<T>::value;
}


//-------------------------------------------------------------------------------------------------


template <typename T>
inline Bitset_storage_base<T,1>::Bitset_storage_base()
: m_data(0)
{
}


template <typename T>
inline Bitset_storage_base<T,1>::Bitset_storage_base(T data)
: m_data(data)
{
}


template <typename T>
inline bool Bitset_storage_base<T,1>::equals(const Bitset_storage_base& rhs) const
{
    return m_data == rhs.m_data;
}


template <typename T>
inline void Bitset_storage_base<T,1>::set(size_t bit)
{
    m_data |= (T(1) << bit);
}


template <typename T>
inline void Bitset_storage_base<T,1>::unset(size_t bit)
{
    m_data &= ~(T(1) << bit);
}


template <typename T>
inline bool Bitset_storage_base<T,1>::is_set(size_t bit) const
{
    return 0 != (m_data & (T(1) << bit));
}


template <typename T>
inline bool Bitset_storage_base<T,1>::any(T data) const
{
    return 0 != (data & m_data);
}


template <typename T>
inline bool Bitset_storage_base<T,1>::any() const
{
    return 0 != m_data;
}


template <typename T>
inline bool Bitset_storage_base<T,1>::all(T data) const
{
    return data == (data & m_data);
}


template <typename T>
inline void Bitset_storage_base<T,1>::do_and(const Bitset_storage_base& rhs)
{
    m_data &= rhs.m_data;
}


template <typename T>
inline void Bitset_storage_base<T,1>::do_or(const Bitset_storage_base& rhs)
{
    m_data |= rhs.m_data;
}


template <typename T>
inline void Bitset_storage_base<T,1>::do_xor(const Bitset_storage_base& rhs)
{
    m_data ^= rhs.m_data;
}


template <typename T>
inline void Bitset_storage_base<T,1>::flip()
{
    m_data = ~m_data;
}


template <typename T>
inline T& Bitset_storage_base<T,1>::data(size_t)
{
    return m_data;
}


template <typename T>
inline T Bitset_storage_base<T,1>::mask(size_t bit)
{
    return T(1) << bit;
}


template <typename T>
inline const T* Bitset_storage_base<T,1>::begin() const
{
    return &m_data;
}


template <typename T>
inline const T* Bitset_storage_base<T,1>::end() const
{
    return &m_data+1;
}


template <typename T>
inline T* Bitset_storage_base<T,1>::begin()
{
    return &m_data;
}


template <typename T>
inline T* Bitset_storage_base<T,1>::end()
{
    return &m_data+1;
}


template <typename T>
inline Bitset_storage_base<T,1>::operator T() const
{
    return m_data;
}


template <typename T>
inline Bitset_storage_base<T,1>::operator T&()
{
    return m_data;
}


//-------------------------------------------------------------------------------------------------


template <typename T, size_t used_bit_count>
struct Bit_cleaner
{
    static void clean(T& bits) { bits &= ~(~T(0) << used_bit_count); }
};


template <typename T>
struct Bit_cleaner<T,0> // all bits used
{
    static void clean(T&) {}
};


}


//-------------------------------------------------------------------------------------------------



template <size_t bit_count>
Bitset<bit_count>::Bitset() {}


template <size_t bit_count>
Bitset<bit_count>::Bitset(Data_t value)
: m_data(value)
{
    clean();
}


template <size_t bit_count>
inline bool Bitset<bit_count>::operator[](size_t bit_index) const
{
    ASSERT(M_CONT,bit_index<bit_count);
    return m_data.is_set(bit_index);
}


template <size_t bit_count>
inline typename Bitset<bit_count>::Bit Bitset<bit_count>::operator[](size_t bit_index)
{
    ASSERT(M_CONT,bit_index<bit_count);
    return Bit(m_data,bit_index);
}


template <size_t bit_count>
inline bool Bitset<bit_count>::is_set(size_t bit_index) const
{
    ASSERT(M_CONT,bit_index<bit_count);
    return m_data.is_set(bit_index);
}


template <size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::set(size_t bit_index)
{
    ASSERT(M_CONT,bit_index<bit_count);
    m_data.set(bit_index);
    return *this;
}


template <size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::unset(size_t bit_index)
{
    ASSERT(M_CONT,bit_index<bit_count);
    m_data.unset(bit_index);
    return *this;
}


template <size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::set(size_t bit_index, bool value)
{
    ASSERT(M_CONT,bit_index<bit_count);
    value ? set(bit_index) : unset(bit_index);
    return *this;
}


template <size_t bit_count>
inline bool Bitset<bit_count>::all(const Bitset& mask) const
{
    return m_data.all(mask.m_data);
}


template <size_t bit_count>
inline bool Bitset<bit_count>::all(const Data_t mask) const
{
    return all(Bitset{mask});
}


template <size_t bit_count>
inline bool Bitset<bit_count>::any(const Bitset& mask) const
{
    return m_data.any(mask.m_data);
}


template <size_t bit_count>
inline bool Bitset<bit_count>::any(const Data_t mask) const
{
    return any(Bitset{mask});
}


template <size_t bit_count>
inline bool Bitset<bit_count>::any() const
{
    return m_data.any();
}


template <size_t bit_count>
inline bool Bitset<bit_count>::none(const Bitset& mask) const
{
    return !m_data.any(mask.m_data);
}


template <size_t bit_count>
inline bool Bitset<bit_count>::none(const Data_t mask) const
{
    return none(Bitset{mask});
}


template <size_t bit_count>
inline bool Bitset<bit_count>::none() const
{
    return !m_data.any();
}


template <size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::flip()
{
    m_data.flip();
    clean();
    return *this;
}


template <size_t bit_count>
inline bool Bitset<bit_count>::operator==(const Bitset& rhs) const
{
    return m_data.equals(rhs.m_data);
}


template <size_t bit_count>
inline bool Bitset<bit_count>::operator!=(const Bitset& rhs) const
{
    return !m_data.equals(rhs.m_data);
}


template <size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::operator&=(const Bitset& rhs)
{
    m_data.do_and(rhs.m_data);
    return *this;
}


template <size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::operator|=(const Bitset& rhs)
{
    m_data.do_or(rhs.m_data);
    return *this;
}


template <size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::operator^=(const Bitset& rhs)
{
    m_data.do_xor(rhs.m_data);
    return *this;
}


template <size_t bit_count>
inline Bitset<bit_count> Bitset<bit_count>::operator~() const
{
    return Bitset(*this).flip();
}


template <size_t bit_count>
inline Bitset<bit_count>::operator bool() const
{
    return any();
}


template <size_t bit_count>
inline bool Bitset<bit_count>::operator!() const
{
    return none();
}


template <size_t bit_count>
inline const typename Bitset<bit_count>::Data_t* Bitset<bit_count>::begin_data() const
{
    return m_data.begin();
}


template <size_t bit_count>
inline const typename Bitset<bit_count>::Data_t* Bitset<bit_count>::end_data() const
{
    return m_data.end();
}


template <size_t bit_count>
inline typename Bitset<bit_count>::Data_t* Bitset<bit_count>::begin_data()
{
    return m_data.begin();
}


template <size_t bit_count>
inline typename Bitset<bit_count>::Data_t* Bitset<bit_count>::end_data()
{
    return m_data.end();
}


template <size_t bit_count>
void Bitset<bit_count>::clean()
{
    typedef DETAIL::Bit_cleaner<Data_t,bit_count%(sizeof(Data_t)*CHAR_BIT)> Cleaner;
    Cleaner::clean(m_data.data(bit_count-1));
}


//-------------------------------------------------


template <size_t bit_count>
inline Bitset<bit_count>::Bit::Bit(Storage& data, size_t index)
: m_mask(data.mask(index))
, m_data(data.data(index))
{
}


template <size_t bit_count>
inline typename Bitset<bit_count>::Bit& Bitset<bit_count>::Bit::operator=(const Bit& val)
{
    const bool b = val;
    return *this = b;
}


template <size_t bit_count>
inline typename Bitset<bit_count>::Bit& Bitset<bit_count>::Bit::operator=(bool val)
{
    if (val)
        m_data |= m_mask;
    else
        m_data &= ~m_mask;
    return *this;
}


template <size_t bit_count>
inline Bitset<bit_count>::Bit::operator bool() const
{
    return 0 != (m_data & m_mask);
}


template <size_t bit_count>
inline bool Bitset<bit_count>::Bit::operator~() const
{
    return 0 == (m_data & m_mask);
}


template <size_t bit_count>
inline typename Bitset<bit_count>::Bit& Bitset<bit_count>::Bit::flip()
{
    m_data ^= m_mask;
    return *this;
}


//-------------------------------------------------------------------------------------------------


template <size_t bit_count>
inline Bitset<bit_count> operator&(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs)
{
    lhs &= rhs;
    return lhs;
}


template <size_t bit_count>
inline Bitset<bit_count> operator|(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs)
{
    lhs |= rhs;
    return lhs;
}


template <size_t bit_count>
inline Bitset<bit_count> operator^(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs)
{
    lhs ^= rhs;
    return lhs;
}

}}

#endif //BASE_LIB_CONT_BITSET_INLINE_H
