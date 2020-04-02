/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief Class to realize persistent settings



#ifndef MDL_SDK_EXAMPLES_MDL_BROWSER_APPLICATION_SETTINGS_H
#define MDL_SDK_EXAMPLES_MDL_BROWSER_APPLICATION_SETTINGS_H

#include <map>
#include <string>
#include <sstream>

class Application_settings_base
{
public:
    class Application_settings_serializer_base
    {
        friend class Application_settings_base;

    protected:

        virtual ~Application_settings_serializer_base() = default;

        virtual bool serialize(const Application_settings_base& settings,
                                const std::string& file_path) const = 0;

        virtual bool deserialize(Application_settings_base& settings,
                                    const std::string& file_path) const = 0;

        std::map<std::string, std::string>& get_key_value_storage(
            Application_settings_base& settings) const
        {
            return settings.m_settings;
        }

        const std::map<std::string, std::string>& get_key_value_storage(
            const Application_settings_base& settings) const
        {
            return settings.m_settings;
        }
    };


    bool load(const std::string& path)
    {
        if (!m_serializer) return false;
        return m_serializer->deserialize(*this, path);
    }
    bool save(const std::string& path) const
    {
        if (!m_serializer) return false;
        return m_serializer->serialize(*this, path);
    }

    template<typename T>
    class Setting
    {
    public:
        Setting(Application_settings_base& settings,
                const std::string& name,
                const T& default_value)
            : m_settings(settings)
            , m_name(name)
            , m_default_value(default_value)
            , m_value(m_default_value)
            , m_changed(true)
        {
        }

        virtual ~Setting() = default;

        const T& get() const
        {
            if (m_changed) // load only the first time
            {
                m_settings.get_value(m_name, m_value);
                m_changed = false;
            }
            return m_value;
        }

        void set(const T& value)
        {
            if (value != get()) // update only if changed
            {
                m_value = value;
                m_settings.set_value(m_name, m_value);
            }
        }

        // assignment of a value
        const T& operator=(const T& rhs)
        {
            set(rhs);
            return get();
        }

        // implicit conversion to "get assigned"
        operator const T&() const { return get(); }

    private:
        Application_settings_base & m_settings;
        const std::string m_name;
        const T m_default_value;
        mutable T m_value;
        mutable bool m_changed;
    };


    typedef Setting<std::string> SettingString;
    typedef Setting<size_t> SettingSize;
    typedef Setting<bool> SettingBool;

protected:

    explicit Application_settings_base(
        Application_settings_serializer_base* serializer, const std::string& auto_file_path)
        : m_serializer(serializer)
        , m_auto_file_path(auto_file_path)
    {
        // try to load the settings file right away
        if (!m_auto_file_path.empty())
            load(m_auto_file_path);
    }

    virtual ~Application_settings_base()
    {
        delete m_serializer;
    }

    // --------------------------------------------------------------------------------------------
    // get value
    // --------------------------------------------------------------------------------------------

    template<typename T>
    bool get_value(const std::string& key, T& out_value) const;

    // --------------------------------------------------------------------------------------------
    // set value
    // --------------------------------------------------------------------------------------------

    template<typename T>
    bool set_value(const std::string& key, const T& value)
    {
        std::stringstream s;
        s << value;
        return set_value(key, s.str()); // this will also handle auto save, 
                                        // so use it for specialization
    }

private:
    Application_settings_serializer_base * m_serializer;
    std::map<std::string, std::string> m_settings;
    const std::string m_auto_file_path;
};


template<>
bool Application_settings_base::get_value(const std::string& key, std::string& out_value) const;
    
template<>
bool Application_settings_base::get_value(const std::string& key, size_t& out_value) const;

template<>
bool Application_settings_base::get_value(const std::string& key, bool& out_value) const;

template<>
bool Application_settings_base::set_value(const std::string& key, const std::string& value);

template<>
bool Application_settings_base::set_value(const std::string& key, const bool& value);

#endif 