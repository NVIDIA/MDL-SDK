/***************************************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <string>
#include <mutex>
#include <map>
#include <cassert>

namespace mi
{
    namespace base
    {
        class IInterface;
    }
}

// Ref count tracking for debugging
class SmartPtrBase
{
public:
    static void _print_open_handle_statistic()
    {
        std::unique_lock<std::mutex> lock(s_lock_open_handles);
        printf("Open Handle Statistic:\n");
        for (auto&& entry : s_open_handles)
        {
            printf("- [%p] %d of type '%s'\n", entry.first, entry.second.ref_count_difference, entry.second.type_name.c_str());
        }
    }

    static void _enable_print_ref_counts(bool enabled)
    {
        s_print_ref_counts = enabled;
    }

    template<typename T>
    T* get_iinterface_weak()
    {
        mi::base::IInterface* b = get_iinterface();
        if (!b)
            return nullptr;

        T* i = b->get_interface<T>();
        b->release();

        if (!i)
            return nullptr;

        size_t ref_count = i->release();
        assert(ref_count > 0 && "get_iinterface_weak++: RefCounter is invalid");
        (void) ref_count;
        return i;
    }

protected:
    void _assign_open_handle_typename(void* address, const char* type_name)
    {
        if (!address)
            return;

        std::unique_lock<std::mutex> lock(s_lock_open_handles);
        s_open_handles[address].type_name = type_name;
    }

    void _increment_counter(void* address)
    {
        if (!address)
            return;

        std::unique_lock<std::mutex> lock(s_lock_open_handles);
        s_open_handles[address].ref_count_difference++;
    }

    void _decrement_counter(void* address)
    {
        if (!address)
            return;

        std::unique_lock<std::mutex> lock(s_lock_open_handles);
        s_open_handles[address].ref_count_difference--;
    }

    bool _print_ref_counts() const
    {
        return s_print_ref_counts;
    }

protected:
    virtual mi::base::IInterface* get_iinterface() = 0;

private:
    struct Open_handle
    {
        Open_handle()
            : type_name("not set")
            , ref_count_difference(0)
        {
        }

        std::string type_name;
        int ref_count_difference;
    };

    static std::mutex s_lock_open_handles;
    static std::map<void*, Open_handle> s_open_handles;
    static bool s_print_ref_counts;
};

 // Smart-pointer class
template<class T> class SmartPtr : public SmartPtrBase
{
public:
    SmartPtr(T* pointee = nullptr)
        : m_pointee(pointee)
        , m_dropped(!m_pointee)
    {
        std::unique_lock<std::recursive_mutex> lock(m_ref_mutex);
        _increment_counter(m_pointee); // incremented from the outside (get, new)
        keep_refcount("SmartPtr Constructor");
    }

    SmartPtr(const SmartPtr<T>& toCopy)
        : m_pointee(toCopy.m_pointee)
        , m_dropped(!m_pointee)
    {
        increase_refcount("SmartPtr Copy Constructor");
    }

    // Not picked up by SWIG but keep it as reference
    SmartPtr<T>& operator=(const SmartPtr<T>& toCopy)
    {
        std::unique_lock<std::recursive_mutex> lock(m_ref_mutex);
        if (this == &toCopy)
            return *this;

        decrease_refcount("Copy Assigned SmartPtr");
        m_pointee = toCopy.m_pointee;
        increase_refcount("Copy Assigned SmartPtr");
        return *this;
    }

    ~SmartPtr()
    {
        decrease_refcount("SmartPtr Destructor");
    }

    T* operator->()
    {
        return m_pointee;
    }

    T& operator*()
    {
        return *m_pointee;
    }

    void assign_open_handle_typename(const char* name)
    {
        m_typename = name;
        _assign_open_handle_typename((void*)m_pointee, name);
    }

    const char* get_debug_str() const
    {
        std::unique_lock<std::recursive_mutex> lock(m_ref_mutex);
        size_t counter = get_ref_count();

        int n = snprintf(nullptr, 0, "<%s [%s, RefCounter: %d] at %p>", m_typename.c_str(), (m_pointee ? "valid" : "invalid"), (int)counter, m_pointee);
        std::string str(n + 1, '\0');
        snprintf(&str[0], n + 1, "<%s [%s, RefCounter: %d] at %p>", m_typename.c_str(), (m_pointee ? "valid" : "invalid"), (int)counter, m_pointee);
        m_debug_string = str.substr(0, n);
        return m_debug_string.c_str();
    }

    void drop(bool on__exit__)
    {
        // Note, the retain and release functions are not exposed to python (renamed with underscore to mark for internal use only)
        std::unique_lock<std::recursive_mutex> lock(m_ref_mutex);
        decrease_refcount(on__exit__ ? "SmartPtr __exit__" : "SmartPtr release");
        m_pointee = nullptr;
        m_dropped = true;
    }

    T* get()
    {
        return m_pointee;
    }

    bool is_valid_interface() const
    {
        return m_pointee;
    }

protected:
    mi::base::IInterface* get_iinterface() final
    {
        if (!m_pointee)
            return nullptr;

        return m_pointee->template get_interface<mi::base::IInterface>();
    }

private:

    void increase_refcount(const char* action)
    {
        std::unique_lock<std::recursive_mutex> lock(m_ref_mutex);
        if (m_pointee)
        {
            size_t refCounter = m_pointee->retain();
            if(_print_ref_counts())
                printf("RefCount++ of %p: %d    (%s)\n", m_pointee, (int)refCounter, action);
            _increment_counter((void*)m_pointee);
        }
        else
        {
            assert(m_dropped && "RefCount++: RefCounter is invalid");
        }
    }

    void decrease_refcount(const char* action)
    {
        std::unique_lock<std::recursive_mutex> lock(m_ref_mutex);
        if (m_pointee)
        {
            _decrement_counter((void*)m_pointee);
            size_t refCounter = m_pointee->release();
            if (_print_ref_counts())
                printf("RefCount-- of %p: %d    (%s)\n", m_pointee, (int)refCounter, action);
            if (refCounter == 0)
                m_pointee = nullptr;
        }
        else
        {
            assert(m_dropped && "RefCount--: RefCounter is invalid");
        }
    }

    void keep_refcount(const char* action)
    {
        std::unique_lock<std::recursive_mutex> lock(m_ref_mutex);
        if (m_pointee)
        {
            if (_print_ref_counts())
                printf("RefCount   of %p: %d    (%s)\n", m_pointee, (int)get_ref_count(), action);
        }
        else
        {
            assert(m_dropped && "RefCount: RefCounter is invalid");
        }
    }

    size_t get_ref_count() const
    {
        size_t counter = 0;
        if (m_pointee)
        {
            counter = m_pointee->retain();
            assert(counter > 0 && "RefCounter is invalid");
            if (counter <= 0 && _print_ref_counts())
                printf("RefCounter is invalid");
            counter = m_pointee->release();
        }
        return counter;
    }

    T* m_pointee;
    bool m_dropped;
    std::string m_typename;
    mutable std::string m_debug_string;
    mutable std::recursive_mutex m_ref_mutex;
};
