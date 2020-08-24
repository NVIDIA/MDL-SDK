/***************************************************************************************************
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
 **************************************************************************************************/

/** \file
 ** \brief Source for the IMdl_compatibility_api implementation.
 **/

#include "pch.h"

#include "neuray_mdl_compatibility_api_impl.h"

#include <mi/base/handle.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_comparator.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_thread_context.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/istring.h>

#include <base/lib/path/i_path.h>
#include <api/api/neuray/neuray_mdl_execution_context_impl.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

namespace MI {
namespace NEURAY {

Mdl_compatibility_api_impl::Mdl_compatibility_api_impl(mi::neuraylib::INeuray *neuray)
: m_neuray(neuray)
, m_mdlc_module( true)
{
}

Mdl_compatibility_api_impl::~Mdl_compatibility_api_impl()
{
    m_neuray = nullptr;
}

namespace {

void handle_messages(Mdl_execution_context_impl* ctx, const mi::mdl::Messages& messages)
{
    if(ctx)
        ctx->get_context().add_messages(messages);
    else
        MDL::report_messages(messages, nullptr);
}


/// MDL search path.
class Mdl_search_path : public mi::base::Interface_implement<mi::mdl::IMDL_search_path>
{
public:
    /// Get the number of search paths.
    ///
    /// \param set  the path set
    size_t get_search_path_count(Path_set set) const {

        return set == mi::mdl::IMDL_search_path::MDL_SEARCH_PATH ? m_roots.size() : 0;
    }

    /// Get the i'th search path.
    ///
    /// \param set  the path set
    /// \param i    index of the path
    char const *get_search_path(Path_set set, size_t i) const {

        if (set != mi::mdl::IMDL_search_path::MDL_SEARCH_PATH)
            return nullptr;
        if (i >= m_roots.size())
            return nullptr;
        return m_roots[i].c_str();
    }

    /// Add a search path.
    void add_path(const char *path) { m_roots.push_back(path); }

public:

    /// Constructor
    Mdl_search_path() {

        // Initialize with installed search paths
        SYSTEM::Access_module<PATH::Path_module> path_module(false);
        const std::vector<std::string>& mdl_paths
            = path_module->get_search_path(PATH::MDL);

        for (const auto& p : mdl_paths)
            add_path(p.c_str());
    }
   
private:

    /// The search path roots.
    std::vector<std::string> m_roots;
};

} // anonymous

mi::Sint32 Mdl_compatibility_api_impl::compare_modules(
    const char* module_name,
    const char* repl_file_name,
    const mi::IArray* search_paths,
    mi::neuraylib::IMdl_execution_context* context) const
{
    if (module_name == nullptr || repl_file_name == nullptr)
        return -1;

    Mdl_execution_context_impl* context_impl = static_cast<Mdl_execution_context_impl*>(context);

    mi::base::Handle<mi::mdl::IMDL> mdl(m_mdlc_module->get_mdl());

    mi::base::Handle<mi::mdl::IThread_context> ctx(mdl->create_thread_context());

    // get the comparator
    mi::base::Handle<mi::mdl::IMDL_comparator> comparator(mdl->create_mdl_comparator());

    if (search_paths) {

        mi::base::Handle<Mdl_search_path> replacement_search_path(new Mdl_search_path());
        for (mi::Size i = 0; i < search_paths->get_length(); ++i) {
            mi::base::Handle<mi::IString const> path(
                search_paths->get_element<mi::IString const>(i));
            if (path)
                replacement_search_path->add_path(path->get_c_str());
        } 
        comparator->install_replacement_search_path(replacement_search_path.get());
    }
   
    // load the first module
    mi::base::Handle<mi::mdl::IModule const> mod1(
        comparator->load_module(ctx.get(), module_name));
    {
        handle_messages(context_impl, ctx->access_messages());

        if (!mod1.is_valid_interface())
            return -2;
    }

    // load the second module
    mi::base::Handle<mi::mdl::IModule const> mod2(
        comparator->load_replacement_module(ctx.get(), module_name, repl_file_name));
    {
        handle_messages(context_impl, ctx->access_messages());

        if (!mod2.is_valid_interface())
            return -2;
    }

    comparator->compare_modules(ctx.get(), mod1.get(), mod2.get());

    mi::mdl::Messages const &msgs = ctx->access_messages();
    handle_messages(context_impl, msgs);

    return msgs.get_error_message_count() ? -2 : 0;
}

mi::Sint32 Mdl_compatibility_api_impl::compare_archives(
    const char* archive_fname1,
    const char* archive_fname2,
    const mi::IArray* search_paths,
    mi::neuraylib::IMdl_execution_context* context) const
{
    if (archive_fname1 == nullptr || archive_fname2 == nullptr)
        return -1;

    Mdl_execution_context_impl* context_impl = static_cast<Mdl_execution_context_impl*>(context);

    mi::base::Handle<mi::mdl::IMDL> mdl(m_mdlc_module->get_mdl());

    mi::base::Handle<mi::mdl::IThread_context> ctx(mdl->create_thread_context());

    // get the comparator
    mi::base::Handle<mi::mdl::IMDL_comparator> comparator(mdl->create_mdl_comparator());

    if (search_paths) {

        mi::base::Handle<Mdl_search_path> replacement_search_path(new Mdl_search_path());
        for (mi::Size i = 0; i < search_paths->get_length(); ++i) {
            mi::base::Handle<mi::IString const> path(
                search_paths->get_element<mi::IString const>(i));
            if(path)
                replacement_search_path->add_path(path->get_c_str());
        }
        comparator->install_replacement_search_path(replacement_search_path.get());
    }
    comparator->compare_archives(ctx.get(), archive_fname1, archive_fname2);

    mi::mdl::Messages const &msgs = ctx->access_messages();
    handle_messages(context_impl, msgs);

    return msgs.get_error_message_count() ? -2 : 0;
}

mi::Sint32 Mdl_compatibility_api_impl::start()
{
    m_mdlc_module.set();
    return 0;
}

mi::Sint32 Mdl_compatibility_api_impl::shutdown()
{
    m_mdlc_module.reset();
    return 0;
}

} // namespace NEURAY
} // namespace MI
