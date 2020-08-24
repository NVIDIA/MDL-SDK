/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief 

#include <type_traits>

namespace MI {

namespace SERIAL {

template <typename T>
inline void Serializer::write(const DB::Typed_tag<T>& value)
{
    write(value.get_untyped());
}

template <typename T>
inline void Deserializer::read(DB::Typed_tag<T>* value_pointer)
{
    DB::Tag t;
    read(&t);
    *value_pointer = DB::Typed_tag<T>(t);
}

template <typename T>
inline void Serializer::write(const CONT::Array<T>& array)
{
    write(STLEXT::safe_cast<Uint32>(array.size()));
    write_range(*this, array.begin(), array.end());
}

template <typename T>
inline void Serializer::write(const CONT::Array<T*>& array)
{
    const Uint32 size( STLEXT::safe_cast<Uint32>(array.size()) );
    write(size);
    for (Uint32 i(0u); i != size; ++i)
        serialize(array[i]);
}

template <typename T, typename SWO>
inline void Serializer::write(const std::set<T, SWO>& set)
{
    write_size_t(set.size());
    write_range(*this, set.begin(), set.end());
}

template <typename T1, typename T2> 
inline void Serializer::write(const std::pair<T1, T2>& pair)
{
    write(pair.first);
    write(pair.second);
}

template <typename T>
inline void Deserializer::read(CONT::Array<T>* array)
{
    Uint32 size;
    read(&size);
    array->resize(size);
    read_range(*this, array->begin(), array->end());
}

template <typename T>
inline void Deserializer::read(CONT::Array<T*>* array)
{
    Uint32 size;
    read(&size);
    array->resize(size);
    for (Uint32 i(0u); i != size; ++i)
        (*array)[i] = reinterpret_cast<T*>( deserialize() );
}

template <typename T, typename A1, typename A2>
inline void Deserializer::read(std::vector< std::vector<T, A1>, A2>* array)
{
    size_t size;
    this->read_size_t(&size);
    array->resize(size);
    for (size_t i = 0u; i != size; ++i) {
        std::vector<T, A1> &inner = (*array)[i];

        size_t isize;
        this->read_size_t(&isize);
        inner.resize(isize);
        for (size_t j = 0u; j != isize; ++j) {
            T temp;
            this->read(&temp);
            inner[j] = temp;
        }
    }
}

template <typename T, typename SWO>
inline void Deserializer::read(std::set<T, SWO>* set)
{
    size_t size;
    set->clear();
    read_size_t(&size);
    T value;
    for (size_t i(0); i != size; ++i) {
        read(&value);
        // Values were serialized in sequence, so we can use end position as hint.
        // This should amount to an O(n) complexity for deserialization.
        set->insert(set->end(), value);
    }
}

template <typename T1, typename T2>
inline void Deserializer::read(std::pair<T1, T2>* pair)
{
    read(&pair->first);
    read(&pair->second);
}

template <typename T, typename A1, typename A2>
inline void Serializer::write(const std::vector< std::vector<T, A1>, A2>& array)
{
    size_t size = array.size();
    this->write_size_t(size);
    for (size_t i = 0; i < size; ++i) {
        const std::vector<T, A1> &inner = array[i];

        size_t isize = inner.size();
        this->write_size_t(isize);
        for (size_t j = 0; j < isize; ++j) {
            this->write(inner[j]);
        }
    }
}

inline void write(Serializer* serial, bool value)   { serial->write( value ); }
inline void write(Serializer* serial, Uint8 value)  { serial->write( value ); }
inline void write(Serializer* serial, Uint16 value) { serial->write( value ); }
inline void write(Serializer* serial, Uint32 value) { serial->write( value ); }
inline void write(Serializer* serial, Uint64 value) { serial->write( value ); }
inline void write(Serializer* serial, Sint8 value)  { serial->write( value ); }
inline void write(Serializer* serial, Sint16 value) { serial->write( value ); }
inline void write(Serializer* serial, Sint32 value) { serial->write( value ); }
inline void write(Serializer* serial, Sint64 value) { serial->write( value ); }
inline void write(Serializer* serial, float value)  { serial->write( value ); }
inline void write(Serializer* serial, double value) { serial->write( value ); }

inline void write(
    Serializer* serial,
    const DB::Tag& value)
{
    serial->write( value );
}

template <class T>
inline void write(Serializer* serial, const DB::Typed_tag<T>& value)
{
    write(serial, value.get_untyped());
}

inline void write(Serializer* serial, const DB::Tag_version& value )
{
    write( serial, value.m_tag );
    write( serial, value.m_version );
    write( serial, value.m_transaction_id );
}

template <typename T, Size R, Size C>
inline void write(Serializer* serial, const mi::math::Matrix<T,R,C>& value)
{
    serial->write(value.begin(),value.size());
}

inline void write(Serializer* serial, const char* value)
{
    serial->write( value );
}

inline void write(Serializer* serial,  const std::string& value)
{
    serial->write( value );
}

inline void write(Serializer* serial, const mi::base::Uuid& value)
{
    serial->write( value );
}

inline void write(Serializer* serial, const mi::math::Color& value)
{
    serial->write( value );
}

template <typename T, Size DIM>
inline void write(Serializer* serial, const mi::math::Vector<T,DIM>& value)
{
    serial->write(value.begin(),DIM);
}

inline void write(Serializer* serial, const CONT::Bitvector& value)
{
    serial->write( value );
}

inline void write(Serializer* serial, const CONT::Dictionary& value)
{
    serial->write( value );
}

inline void write(Serializer* serial, const DB::Transaction_id& value)
{
    serial->write(value());
}

inline void write(Serializer* serial, const Serializable& value)
{
    serial->write( value );
}

inline void read(Deserializer* deser, bool* value_pointer)   { deser->read( value_pointer ); }
inline void read(Deserializer* deser, Uint8* value_pointer)  { deser->read( value_pointer ); }
inline void read(Deserializer* deser, Uint16* value_pointer) { deser->read( value_pointer ); }
inline void read(Deserializer* deser, Uint32* value_pointer) { deser->read( value_pointer ); }
inline void read(Deserializer* deser, Uint64* value_pointer) { deser->read( value_pointer ); }
inline void read(Deserializer* deser, Sint8* value_pointer)  { deser->read( value_pointer ); }
inline void read(Deserializer* deser, Sint16* value_pointer) { deser->read( value_pointer ); }
inline void read(Deserializer* deser, Sint32* value_pointer) { deser->read( value_pointer ); }
inline void read(Deserializer* deser, Sint64* value_pointer) { deser->read( value_pointer ); }
inline void read(Deserializer* deser, float* value_pointer)  { deser->read( value_pointer ); }
inline void read(Deserializer* deser, double* value_pointer) { deser->read( value_pointer ); }

inline void read(Deserializer* deser, DB::Tag* value_pointer)
{
    deser->read( value_pointer );
}

template <class T>
inline void read(Deserializer* deser, DB::Typed_tag<T>* value_pointer)
{
    DB::Tag tag;
    read(deser, &tag );
    *value_pointer = tag;
}

inline void read(Deserializer* deser, DB::Tag_version* value_pointer)
{
    read( deser, &value_pointer->m_tag );
    read( deser, &value_pointer->m_version );
    read( deser, &value_pointer->m_transaction_id );
}


template <typename T, Size R, Size C>
inline void read(Deserializer* deser, mi::math::Matrix<T,R,C>* value_pointer)
{
    deser->read( value_pointer->begin(), value_pointer->size() );
}

inline void read(Deserializer* deser, char** value_pointer)
{
    deser->read( value_pointer );
}

inline void read(Deserializer* deser, std::string* value_pointer)
{
    deser->read( value_pointer );
}

inline void read(Deserializer* deser, mi::math::Color* value_pointer)
{
    deser->read( value_pointer );
}

inline void read(Deserializer* deser, mi::base::Uuid* value_pointer)
{
    deser->read( value_pointer );
}

template <typename T, Size DIM>
inline void read(Deserializer* deser, mi::math::Vector<T,DIM>* value)
{
    deser->read(value->begin(),DIM);
}

inline void read(Deserializer* deser, CONT::Bitvector* value_type)
{
    deser->read( value_type );
}

inline void read(Deserializer* deser, CONT::Dictionary* value_pointer)
{
    deser->read( value_pointer );
}

inline void read(Deserializer* deser, DB::Transaction_id* value_pointer)
{
    Uint32 value;
    deser->read(&value);
    *value_pointer = value;
}

inline void read(Deserializer* deser, Serializable* value_type)
{
    deser->read( value_type );
}

template <class Iterator>
inline void read_range(Deserializer& deserializer, Iterator begin, Iterator end)
{
    while (begin != end)
        read(&deserializer, &(*(begin++)));
}

template <typename T, size_t N>
inline void read_range(Deserializer& deserializer, T (&arr)[N])
{
    read_range(deserializer,arr+0,arr+N);
}

template <class Iterator>
inline void write_range(Serializer& serializer, Iterator begin, Iterator end)
{
    while (begin != end)
        write(&serializer, *begin++);
}

template <typename T, size_t N>
inline void write_range(Serializer& serializer, const T (&arr)[N])
{
    write_range(serializer,arr+0,arr+N);
}

template <typename T>
inline void write(Serializer* serializer, const std::vector<T>& array)
{
    serializer->write_size_t(array.size());
    write_range(*serializer, array.begin(), array.end());
}

template <typename T,typename A>
inline void write(Serializer* serializer, const std::vector<T,A>& array)
{
    serializer->write_size_t(array.size());
    write_range(*serializer, array.begin(), array.end());
}

template <typename T>
inline void write(Serializer* serializer, const std::vector<T*>& array)
{
    const size_t size(array.size());
    serializer->write_size_t(size);
    for (size_t i=0; i != size; ++i)
        write(serializer,array[i]);
}

template <typename T,typename A>
inline void write(Serializer* serializer, const std::vector<T*,A>& array)
{
    const size_t size(array.size());
    serializer->write_size_t(size);
    for (size_t i=0; i != size; ++i)
        write(serializer,array[i]);
}

template <typename T>
inline void read(Deserializer* deserializer, std::vector<T>* array)
{
    size_t size;
    deserializer->read_size_t(&size);
    array->resize(size);
    read_range(*deserializer, array->begin(), array->end());
}

inline void read(Deserializer* deserializer, std::vector<bool>* array)
{
    size_t size;
    deserializer->read_size_t(&size);
    array->resize(size);
    for (std::vector<bool>::iterator it = array->begin(), end = array->end(); it != end; ++it)
    {
        bool tmp;
        deserializer->read(&tmp);
        *it = tmp;
    }
}

template <typename T,typename A>
inline void read(Deserializer* deserializer, std::vector<T,A>* array)
{
    size_t size;
    deserializer->read_size_t(&size);
    array->resize(size);
    read_range(*deserializer, array->begin(), array->end());
}

template <typename T>
inline void read(Deserializer* deserializer, std::vector<T*>* array)
{
    size_t size;
    deserializer->read_size_t(&size);
    array->resize(size);
    for (size_t i=0; i != size; ++i)
        (*array)[i] = reinterpret_cast<T*>( deserializer->deserialize() );
}

template <typename T,typename A>
inline void read(Deserializer* deserializer, std::vector<T*,A>* array)
{
    size_t size;
    deserializer->read_size_t(&size);
    array->resize(size);
    for (size_t i=0; i != size; ++i)
        (*array)[i] = reinterpret_cast<T*>( deserializer->deserialize() );
}

template <typename T>
inline void write(Serializer* serializer, const std::list<T>& list)
{
    serializer->write_size_t(list.size());
    for (typename std::list<T>::const_iterator i=list.begin(),e=list.end(); i!=e; ++i)
        write(serializer,*i);
}

template <typename T>
inline void read(Deserializer* deserializer, std::list<T>* list)
{
    size_t size = 0;
    deserializer->read_size_t(&size);
    list->resize(size);
    for (typename std::list<T>::iterator i=list->begin(),e=list->end(); i!=e; ++i)
        read(deserializer,&*i);
}

template <typename T, typename U>
inline void write(Serializer* serializer, const std::pair<T, U>& pair)
{
    write(serializer, pair.first);
    write(serializer, pair.second);
}

template <typename T, typename U>
inline void read(Deserializer* deserializer, std::pair<T, U>* pair)
{
    read(deserializer, &(pair->first));
    read(deserializer, &(pair->second));
}

template <typename T, typename SWO>
void write(Serializer* serializer, const std::set<T,SWO>& set)
{
    const size_t size(set.size());
    serializer->write_size_t(size);
    write_range( *serializer, set.begin(), set.end() );
}

template <typename T, typename SWO>
void read(Deserializer* deserializer, std::set<T,SWO>* set)
{
    size_t size;
    set->clear();
    deserializer->read_size_t(&size);
    T value;
    for ( size_t i(0); i != size; ++i ) {
        read( deserializer, &value );
        // Values were serialized in sequence, so we can use end position as hint.
        // This should amount to an O(n) complexity for deserialization.
        set->insert( set->end(), value );
    }
}

template <typename T, typename U, typename SWO>
void write(Serializer* serializer, const std::map<T,U,SWO>& map)
{
    const size_t size(map.size());
    serializer->write_size_t(size);
    write_range( *serializer, map.begin(), map.end() );
}

template <typename T, typename U, typename SWO>
void read(Deserializer* deserializer, std::map<T,U,SWO>* map)
{
    size_t size;
    map->clear();
    deserializer->read_size_t(&size);
    std::pair<T,U> value;
    for ( size_t i(0); i != size; ++i ) {
        read( deserializer, &value );
        // Values were serialized in sequence, so we can use end position as hint.
        // This should amount to an O(n) complexity for deserialization.
        map->insert( map->end(), value );
    }
}

template<class K, class V, class C, class A>
void write(Serializer* ser, const std::multimap<K,V,C,A>& map)
{
    const size_t size(map.size());
    ser->write_size_t(size);
    write_range(*ser,map.begin(),map.end());
}

template<class K, class V, class C, class A>
void read(Deserializer* deser, std::multimap<K,V,C,A>* map)
{
    size_t size;
    map->clear();
    deser->read_size_t(&size);
    std::pair<K,V> value;
    for (size_t i(0); i != size; ++i) {
        read(deser,&value);
        // Values were serialized in sequence, so we can use end position as hint.
        // This should amount to an O(n) complexity for deserialization.
        map->insert(map->end(),value);
    }
}



template <typename Enum_type>
void write_enum(Serializer* serializer, Enum_type enum_value )
{
    write(serializer,static_cast<typename std::underlying_type<Enum_type>::type>(enum_value));
}


template <typename Enum_type>
void read_enum(Deserializer* deserializer, Enum_type* enum_value )
{
    typename std::underlying_type<Enum_type>::type v;
    read(deserializer,&v);
    *enum_value = static_cast<Enum_type>(v);
}

} // namespace SERIAL

} // namespace MI

