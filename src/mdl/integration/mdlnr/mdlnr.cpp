/******************************************************************************
 * Copyright (c) 2012-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include <map>

#include <mi/base/ilogger.h>
#include <mi/mdl/mdl_code_generators.h>

#include <base/system/main/access_module.h>
#include <base/system/main/module_registration.h>
#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/config/config.h>
#include <base/lib/mem/mem.h>
#include <base/util/registry/i_config_registry.h>
#include <base/data/serial/i_serializer.h>
#include <base/system/stlext/i_stlext_no_unused_variable_warning.h>

#include "mdlnr.h"
#include "mdlnr_search_path.h"

#include <mdl/compiler/compilercore/compilercore_assert.h>
#include <mdl/compiler/compilercore/compilercore_fatal.h>
#include <mdl/compiler/compilercore/compilercore_debug_tools.h>
#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_file_utils.h>

namespace MI {
namespace MDLC {

class Assert_helper : public mi::mdl::IAsserter {
    virtual void assertfailed(
        char const   *exp,
        char const   *file,
        unsigned int line)
    {
        ::MI::LOG::report_assertion_failure(M_MDLC, exp, file, line);
    }
};

class Fatal_helper : public mi::mdl::IFatal_error {
    virtual void fatal(
        char const *msg)
    {
        ::MI::LOG::mod_log->fatal(M_MDLC, MI::LOG::ILogger::C_COMPILER, "%s", msg);
    }
};

class DebugOutput : public mi::mdl::IOutput_stream {
public:
    /// Write a character to the output stream.
    virtual void write_char(char c) {
        ::MI::LOG::mod_log->info(M_MDLC, LOG::ILogger::C_RENDER, "%c", c);
    }

    /// Write a string to the stream.
    virtual void write(const char *string) {
        ::MI::LOG::mod_log->info(M_MDLC, LOG::ILogger::C_RENDER, "%s", string);
    }

    /// Flush stream.
    virtual void flush() {}

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    virtual bool unput(char c) {
        // unsupported
        return false;
    }

    // from IInterface: This object is not reference counted.
    virtual Uint32 retain() const { return 1; }
    virtual Uint32 release() const { return 1; }
    virtual const mi::base::IInterface* get_interface(mi::base::Uuid const &interface_id) const {
        return NULL;
    }
    virtual mi::base::IInterface* get_interface(mi::base::Uuid const &interface_id) {
        return NULL;
    }
    virtual mi::base::Uuid get_iid() const { return IID(); }
};

static Assert_helper g_assert_helper;
static Fatal_helper  g_fatal_helper;
static DebugOutput   g_debug_output;

/// An allocator based on new/delete.
class Allocator : public mi::base::Interface_implement<mi::base::IAllocator>
{
public:
    void* malloc(mi::Size size) { return new char[size]; }
    void free(void* memory) { delete[] (char*) memory; }
};

/// MDL object deserializer that wraps a deserializer.
class MDL_deserializer : public mi::mdl::Base_deserializer
{
public:
    /// Read a byte.
    virtual Byte read() { Uint8 c; m_deserializer->read(&c); return c; }

    /// Reads a DB::Tag 32bit encoding.
    virtual unsigned read_db_tag() {
        DB::Tag t;
        m_deserializer->read(&t);
        return t.get_uint();
    }

    /// Constructor.
    MDL_deserializer(mi::mdl::IAllocator *alloc, SERIAL::Deserializer *deserializer)
    : mi::mdl::Base_deserializer(alloc)
    {
        m_deserializer = deserializer;
    }

private:
    SERIAL::Deserializer *m_deserializer;
};

/// MDL object serializer that wraps a serializer.
class MDL_serializer : public mi::mdl::Base_serializer
{
public:
    /// Write a byte.
    ///
    /// \param b  the byte to write
    virtual void write(Byte b) { m_serializer->write(Uint8(b)); }

    /// Write a DB::Tag.
    ///
    /// \param tag  the DB::Tag encoded as 32bit
    virtual void write_db_tag(unsigned tag) {
        DB::Tag t(tag);
        m_serializer->write(t);
    }

    /// Constructor.
    MDL_serializer(SERIAL::Serializer *serializer) { m_serializer = serializer; }

private:
    SERIAL::Serializer *m_serializer;
};

/// The code cache helper class.
class Code_cache : public mi::base::Interface_implement<mi::mdl::ICode_cache>
{
    class Key {
        friend class Code_cache;
    public:
        /// Constructor.
        Key(unsigned char const key[16])
        {
            // copy the key
            memcpy(m_key, key, sizeof(m_key));
        }

        bool operator <(Key const &other) const {
            return memcmp(m_key, other.m_key, sizeof(m_key)) < 0;
        }

        bool operator ==(Key const &other) const {
            return memcmp(m_key, other.m_key, sizeof(m_key)) == 0;
        }

    private:
        unsigned char m_key[16];
    };

    class Cache_entry : public mi::mdl::ICode_cache::Entry {
        typedef mi::mdl::ICode_cache::Entry Base;
        friend class Code_cache;
        friend class Cache_entry_less;
    public:
        /// Constructor.
        Cache_entry(Base const &entry, unsigned char const key[16])
        : Base(entry)
        , m_key(key)
        , m_prev(NULL)
        , m_next(NULL)
        {
            // copy all data
            size_t size = entry.get_cache_data_size();

            char *blob      = new char[size];
            char **mapped   = (char **)blob;
            char *data_area = blob + entry.mapped_string_size * sizeof(char *);
            Func_info *func_info_area = (Func_info *) (data_area + entry.mapped_string_data_size);
            char *func_info_data_area = (char *) (func_info_area + entry.func_info_size);

            blob = func_info_data_area + entry.func_info_string_data_size;

            char *seg    = blob + entry.code_size;
            char *layout = seg + entry.const_seg_size;

            if (entry.code_size > 0)
                memcpy(blob, entry.code, entry.code_size);

            if (entry.const_seg_size > 0)
                memcpy(seg, entry.const_seg, entry.const_seg_size);

            if (entry.arg_layout_size > 0)
                memcpy(layout, entry.arg_layout, entry.arg_layout_size);

            if (entry.mapped_string_data_size > 0) {
                char *p = data_area;
                for (size_t i = 0; i < entry.mapped_string_size; ++i) {
                    size_t l = strlen(entry.mapped_strings[i]);

                    memcpy(p, entry.mapped_strings[i], l + 1);
                    mapped[i] = p;
                    p += l + 1;
                }
            }

            if (entry.func_info_size > 0) {
                Func_info *cur_info = func_info_area;
                char *p = func_info_data_area;
                for (size_t i = 0; i < entry.func_info_size; ++i, ++cur_info) {
                    size_t len = strlen(entry.func_infos[i].name);
                    memcpy(p, entry.func_infos[i].name, len + 1);
                    cur_info->name = p;
                    p += len + 1;

                    cur_info->dist_kind = entry.func_infos[i].dist_kind;
                    cur_info->func_kind = entry.func_infos[i].func_kind;

                    for (int j = 0 ; j < int(mi::mdl::IGenerated_code_executable::PL_NUM_LANGUAGES);
                            ++j)
                    {
                        len = strlen(entry.func_infos[i].prototypes[j]);
                        memcpy(p, entry.func_infos[i].prototypes[j], len + 1);
                        cur_info->prototypes[j] = p;
                        p += len + 1;
                    }

                    cur_info->arg_block_index = entry.func_infos[i].arg_block_index;
                }
            }

            code           = blob;
            const_seg      = seg;
            arg_layout     = layout;
            mapped_strings = mapped;
            func_infos     = func_info_area;
        }

        /// Destructor.
        ~Cache_entry()
        {
            char const *blob = (char const *)mapped_strings;

            delete [] blob;
        }

    private:
        Key m_key;

        Cache_entry *m_prev;
        Cache_entry *m_next;
    };

    class Cache_entry_less {
    public:
        bool operator() (Cache_entry const *a, Cache_entry const *b)
        {
            return a->m_key < b->m_key;
        }
    };

public:
    // Lookup a data blob.
    virtual Entry const *lookup(unsigned char const key[16]) const
    {
        mi::base::Lock::Block block(&m_cache_lock);

        Search_map::const_iterator it = m_search_map.find(Key(key));
        if (it != m_search_map.end()) {
            // found
            Cache_entry *p = it->second;
            to_front(*p);
            return p;
        }
        return NULL;
    }

    // Enter a data blob.
    virtual bool enter(unsigned char const key[16], Entry const &entry)
    {
        mi::base::Lock::Block block(&m_cache_lock);

        // don't try to enter it if it doesn't fit into the cache at all
        if (entry.get_cache_data_size() > m_max_size)
            return false;

        m_curr_size += entry.get_cache_data_size();
        strip_size();

        Cache_entry *res = new_entry(entry, key);

        m_search_map.insert(Search_map::value_type(res->m_key, res));
        return true;
    }

private:
    /// Create a new entry and put it in front.
    /// Assumes that current size has already been updated.
    Cache_entry *new_entry(mi::mdl::ICode_cache::Entry const &entry, unsigned char const key[16])
    {
        Cache_entry *res = new Cache_entry(entry, key);

        to_front(*res);

        return res;
    }

    /// Remove an entry from the list.
    void remove_from_list(Cache_entry &entry) const
    {
        if (m_head == &entry)
            m_head = entry.m_next;
        if (m_tail == &entry)
            m_tail = entry.m_prev;

        if (entry.m_next != NULL)
            entry.m_next->m_prev = entry.m_prev;
        if (entry.m_prev != NULL)
            entry.m_prev->m_next = entry.m_next;

        entry.m_prev = entry.m_next = NULL;
    }

    /// Move an entry to front.
    void to_front(Cache_entry &entry) const
    {
        remove_from_list(entry);

        entry.m_next = m_head;

        if (m_head != NULL)
            m_head->m_prev = &entry;

        m_head = &entry;

        if (m_tail == NULL)
            m_tail = &entry;
    }

    /// Compare an entry with a key.
    static int cmp(Cache_entry const &entry, unsigned char const key[16])
    {
        return memcmp(entry.m_key.m_key, key, sizeof(entry.m_key));
    }

    /// Drop entries from the end until size is reached.
    void strip_size()
    {
        Cache_entry *next = NULL;
        for (Cache_entry *p = m_tail; p != NULL; p = next) {
            if (m_curr_size < m_max_size)
                break;

            next = p->m_prev;

            m_curr_size -= p->get_cache_data_size();
            m_search_map.erase(p->m_key);
            remove_from_list(*p);
            delete p;
        }
    }

public:
    /// Constructor.
    Code_cache(size_t max_size)
    : m_cache_lock()
    , m_head(NULL)
    , m_tail(NULL)
    , m_max_size(max_size)
    , m_curr_size(0)
    {
    }

    /// Destructor.
    virtual ~Code_cache()
    {
        m_search_map.clear();
        for (Cache_entry *n = NULL, *p = m_head; p != NULL; p = n) {
            n = p->m_next;
            delete p;
        }
    }

private:
    mutable mi::base::Lock m_cache_lock;

    mutable Cache_entry *m_head;
    mutable Cache_entry *m_tail;

    typedef std::map<Key, Cache_entry *> Search_map;

    /// The map of all cache entry to speed up searches.
    mutable Search_map m_search_map;

    /// Maximum size of this cache object.
    size_t m_max_size;

    /// Current size.
    size_t m_curr_size;
};

// Registration of the module.
static SYSTEM::Module_registration<Mdlc_module_impl> s_module(M_MDLC,"MDLC");

// Allow link time detection.
Module_registration_entry *Mdlc_module::get_instance()
{
    return s_module.init_module(s_module.get_name());
}

Mdlc_module_impl::Mdlc_module_impl()
  : m_mdl(0)
  , m_used_with_mdl_sdk(false) /*arbitrary*/
  , m_used_with_mdl_sdk_set(false)
  , m_code_cache(0)
  , m_implicit_cast_enabled(true)
{
}

#ifdef DEBUG
static mi::mdl::dbg::DebugMallocAllocator *g_dbg_allocator = NULL;

static void flush_dbg_allocator() {
    delete g_dbg_allocator;
    g_dbg_allocator = NULL;
}
#endif

bool Mdlc_module_impl::init()
{
    mi::mdl::iasserter   = &g_assert_helper;
    mi::mdl::ifatal      = &g_fatal_helper;
    mi::mdl::i_debug_log = &g_debug_output;

    m_allocator = new Allocator();

#ifdef DEBUG
    // use the debug allocator
    //
    // Note: the MEM module and the MDLC module have different lifetimes. In particular, the MDLC
    // might be initialized and shutdown several times while the MEM module is alive. Maybe it is
    // better to bind the debug allocator to the lifetime of the DATA module instead.
    delete g_dbg_allocator;
    g_dbg_allocator = new mi::mdl::dbg::DebugMallocAllocator(m_allocator.get());
    m_allocator = g_dbg_allocator;
    SYSTEM::Access_module<MEM::Mem_module> mem_module(/*deferred=*/false);
    mem_module->set_exit_cb(flush_dbg_allocator);
#endif

    m_mdl = mi::mdl::initialize(m_allocator.get());
    if (m_mdl != NULL) {
        m_mdl->install_search_path(MDL_search_path::create(m_allocator.get()));

        // retrieve MDL parameters
        SYSTEM::Access_module<CONFIG::Config_module> config_module(/*deferred=*/false);
        CONFIG::Config_registry const                &registry = config_module->get_configuration();
        mi::mdl::Options                             &options = m_mdl->access_options();

        int opt_level = 0;
        if (registry.get_value("mdl_opt_level", opt_level)) {
            options.set_option(
                mi::mdl::MDL::option_opt_level, std::to_string(opt_level).c_str());
        }

        // neuray always runs in "relaxed" mode for compatibility with old releases
        options.set_option(mi::mdl::MDL::option_strict, "false");


        // 1MB cache size by default
        size_t cache_size = 1*1024*1024;
        m_code_cache = new Code_cache(cache_size);

        return true;
    }
    return false;
}

void Mdlc_module_impl::exit()
{
    if(m_mdl) {
        m_mdl->release();

        if (m_code_cache) {
            m_code_cache->release();
            m_code_cache = NULL;
        }
    }
}

mi::mdl::IMDL *Mdlc_module_impl::get_mdl() const
{
    m_mdl->retain();
    return m_mdl;
}

void Mdlc_module_impl::serialize_module(
    SERIAL::Serializer *serializer, const mi::mdl::IModule *module)
{
    MDL_serializer mdl_serializer(serializer);

    m_mdl->serialize_module(module, &mdl_serializer, /*include_dependencies=*/false);
}

const mi::mdl::IModule *Mdlc_module_impl::deserialize_module(SERIAL::Deserializer *deserializer)
{
    MDL_deserializer mdl_deserializer(m_allocator.get(), deserializer);

    return m_mdl->deserialize_module(&mdl_deserializer);
}

void Mdlc_module_impl::serialize_code_dag(
    SERIAL::Serializer *serializer, const mi::mdl::IGenerated_code_dag *code)
{
    MDL_serializer mdl_serializer(serializer);

    m_mdl->serialize_code_dag(code, &mdl_serializer);
}

const mi::mdl::IGenerated_code_dag *Mdlc_module_impl::deserialize_code_dag(
    SERIAL::Deserializer *deserializer)
{
    MDL_deserializer mdl_deserializer(m_allocator.get(), deserializer);
    return m_mdl->deserialize_code_dag(&mdl_deserializer);
}

// Serializes the lambda function to the given serializer.
void Mdlc_module_impl::serialize_lambda_function(
    SERIAL::Serializer *serializer, const mi::mdl::ILambda_function *lambda)
{
    MDL_serializer mdl_serializer(serializer);
    m_mdl->serialize_lambda(lambda, &mdl_serializer);
}

// Deserializes the lambda function from the given deserializer.
mi::mdl::ILambda_function *Mdlc_module_impl::deserialize_lambda_function(
    SERIAL::Deserializer *deserializer)
{
    MDL_deserializer mdl_deserializer(m_allocator.get(), deserializer);
    return m_mdl->deserialize_lambda(&mdl_deserializer);
}

void Mdlc_module_impl::set_used_with_mdl_sdk(bool flag)
{
    ASSERT(M_MDLC, !m_used_with_mdl_sdk_set);
    m_used_with_mdl_sdk_set = true;
    m_used_with_mdl_sdk = flag;

}

bool Mdlc_module_impl::get_used_with_mdl_sdk() const
{
    ASSERT(M_MDLC, m_used_with_mdl_sdk_set);
    return m_used_with_mdl_sdk;
}


void Mdlc_module_impl::client_build_version(const char* build, const char* bridge_protocol) const
{

}

mi::mdl::ICode_cache *Mdlc_module_impl::get_code_cache() const
{
    if (m_code_cache)
        m_code_cache->retain();
    return m_code_cache;
}

bool Mdlc_module_impl::utf8_match(char const *file_mask, char const *file_name) const
{
    return mi::mdl::utf8_match(file_mask, file_name);
}

bool Mdlc_module_impl::get_implicit_cast_enabled() const
{
    return m_implicit_cast_enabled;
}

void Mdlc_module_impl::set_implicit_cast_enabled(bool v)
{
    m_implicit_cast_enabled = v;
}

} // namespace MDLC

} // namespace MI

