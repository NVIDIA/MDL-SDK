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
#include "xliff.h"
#include "util.h"
#include "application.h"
#include "errors.h"
#include "traversal.h"
#include "search_path.h"
#include <base/lib/tinyxml2/tinyxml2.h>
#include <stack>

using namespace i18n;
using mi::base::Handle;
using mi::neuraylib::INeuray;
using mi::neuraylib::IModule;
using mi::neuraylib::IAnnotation;
using mi::neuraylib::IAnnotation_list;
using mi::neuraylib::IAnnotation_block;
using mi::neuraylib::IExpression_list;
using mi::neuraylib::IExpression;
using mi::neuraylib::IFunction_definition;
using mi::neuraylib::IMaterial_definition;
using mi::neuraylib::IValue_string;
using mi::neuraylib::IValue;
using mi::neuraylib::IExpression_constant;
using mi::neuraylib::IDatabase;
using mi::neuraylib::IMdl_discovery_api;
using mi::neuraylib::IMdl_discovery_result;
using mi::neuraylib::IMdl_package_info;
using mi::neuraylib::IMdl_info;
using mi::neuraylib::IMdl_module_info;
using mi::neuraylib::IMdl_impexp_api;
using mi::IString;
using std::string;
using std::vector;
using std::stack;
using std::cout;
using tinyxml2::XMLElement;
using tinyxml2::XMLDocument;
using tinyxml2::XMLDeclaration;
using tinyxml2::XMLText;
using tinyxml2::XMLNode;

// A context which generates an XLIFF document from traversal of MDL elements
class Create_xliff_context : public Traversal_context
{
    bool m_output_context = true;
public:
    Create_xliff_context(const string & locale)
        : m_locale(locale)
        , m_doc(NULL)
        , m_body(NULL)
        , m_unique_id(0)
    {
        create_document();
        XMLElement * xliff = add_new_element("xliff");
        xliff->SetAttribute("version", "1.2");
        xliff->SetAttribute("xmlns", "urn:oasis:names:tc:xliff:document:1.2");
        xliff->SetAttribute("xmlns:trolltech", "urn:trolltech:names:ts:document:1.0");
        XMLElement * file = add_new_element("file", xliff);
        file->SetAttribute("original", "");
        file->SetAttribute("datatype", "plaintext");
        file->SetAttribute("source-language", "en");
        file->SetAttribute("target-language", locale.c_str());
        m_body = add_new_element("body", file);
    }

    ~Create_xliff_context()
    {
        delete m_doc;
        m_doc = NULL;
    }

public: // Settings and controls

    bool finalize_document(const string & filename)
    {
        if (m_doc)
        {
            tinyxml2::XMLError err(m_doc->SaveFile(filename.c_str()));
            return err == tinyxml2::XML_SUCCESS;
        }
        return false;
    }

    void set_output_context(bool output_context)
    {
        m_output_context = output_context;
    }

    void set_qualified_name(const std::string & qualified_name)
    {
        m_qualified_name = qualified_name;
    }

public: // From Traversal_context

    void push_annotation(const char* name, const char* value, const char* note = NULL) override
    {
        if (name && value && translated_annotation(name))
        {
            if (!m_output_context)
            {
                XMLElement * trans_unit = already_output(value);
                if (trans_unit)
                {
                    // Do not duplicate source strings if no context
                    // Useless to have 10 times "Color 1"

                    XMLElement * elt = trans_unit->FirstChildElement("note");
                    if (elt)
                    {
                        XMLNode * noteval = elt->FirstChild();
                        const char * val = noteval->Value();
                        if (val)
                        {
                            string new_val(val);
                            new_val += string("\n") + (note ? note : name);
                            noteval->SetValue(new_val.c_str());
                        }
                    }

                    return;
                }
            }

            XMLElement * trans_unit = add_new_trans_unit();

            //<source xml : space = "preserve">AEC - Carpet - Pattern - Circle - Brown< / source>
            XMLElement * src = add_new_source(value, trans_unit);
            check_success(src != NULL);
            XMLElement * target = add_new_target(trans_unit);
            check_success(target != NULL);
            const char* note_text = (note ? note : name);
            XMLElement * note_elt = add_new_note(note_text, trans_unit);
            check_success(note_elt != NULL);
            if (m_group.m_element)
            {
                m_group.m_count++;
            }
        }
    }

    void push_qualified_name(const char* name) override
    {
        string resname;
        if (name)
        {
            resname = name;
            // Remove the qualified name from the name
            if (resname.find(m_qualified_name) == 0)
            {
                resname = resname.substr(m_qualified_name.size() + 2); // Remove the :: as well
            }
            if (m_output_context)
            {
                m_group.m_element = add_new_element("group", m_body);
                m_group.m_element->SetAttribute("restype", "x-trolltech-linguist-context");
                m_group.m_element->SetAttribute("resname", resname.c_str());
            }
        }  
        m_qualified_name_stack.push(resname.c_str());
    }

    void pop_qualified_name() override
    {
        m_qualified_name_stack.pop();
        // Remove empty groups
        if (m_group.m_element)
        {
            if (m_group.m_count == 0)
            {
                m_doc->DeleteNode(m_group.m_element);
            }
            m_group.m_element = NULL;
            m_group.m_count = 0;
        }
    }

    const char* top_qualified_name() const override
    {
        return m_qualified_name_stack.top().c_str();
    }

private:
    string m_locale;
    XMLDocument * m_doc;
    XMLElement * m_body;
    struct Group
    {
        Group()
            : m_element(NULL)
            , m_count(0)
        {}
        XMLElement * m_element;
        unsigned int m_count;// Number of annotations in this group

    } m_group;
    
    unsigned int m_unique_id;
    std::map<string /*source*/ , XMLElement * /*trans_unit*/ > m_already_output;
    string m_qualified_name;
    std::stack<string> m_qualified_name_stack;
private:
    XMLElement * add_new_element(const char * name, XMLElement * parent = 0)
    {
        XMLElement * element = m_doc->NewElement(name);
        XMLElement * parent_node = (parent ? parent : (XMLElement *)m_doc);
        parent_node->LinkEndChild(element);
        return element;
    }
    XMLElement * add_new_trans_unit()
    {
        XMLElement * parent_node = (m_group.m_element ? m_group.m_element : m_body);
        XMLElement * trans_unit = add_new_element("trans-unit", parent_node);
        std::ostringstream id;
        id << m_unique_id++;
        trans_unit->SetAttribute("id", id.str().c_str());
        return trans_unit;
    }
    XMLElement * add_new_source(const char * text, XMLElement * trans_unit)
    {
        XMLElement * src(NULL);
        if (text)
        {
            src = add_new_element("source", trans_unit);
            src->SetAttribute("xml:space", "preserve");
            XMLText * text_element = m_doc->NewText(text);
            src->LinkEndChild(text_element);
            m_already_output[text] = trans_unit;
        }
        return src;
    }

    XMLElement * add_new_target(XMLElement * trans_unit)
    {
        XMLElement * target = add_new_element("target", trans_unit);
        target->SetAttribute("xml:space", "preserve");
        XMLText * text_element = m_doc->NewText("");
        target->LinkEndChild(text_element);
        return target;
    }

    XMLElement * add_new_note(const char * text, XMLElement * trans_unit)
    {
        XMLElement * note = add_new_element("note", trans_unit);
        note->SetAttribute("from", "translator");
        XMLText * text_element = m_doc->NewText(text);
        note->LinkEndChild(text_element);
        return note;
    }

    void create_document()
    {
        m_doc = new XMLDocument;
        m_doc->LinkEndChild(m_doc->NewDeclaration());
    }

    bool translated_annotation(const char* name)
    {
        // WARNING: To keep in sync with: 
        //      mdl\integration\i18n\i18n_translator.cpp
        //      mdl\integration\i18n\i_i18n.h
        //      mdl\integration\i18n\i18n_translator.h
        static std::set<std::string> translation =
        {
            "::anno::display_name(string)"
            , "::anno::in_group(string)"
            , "::anno::in_group(string,string)"
            , "::anno::in_group(string,string,string)"
            , "::anno::key_words(string[N])"
            , "::anno::copyright_notice(string)"
            , "::anno::description(string)"
            , "::anno::author(string)"
            , "::anno::contributor(string)"
            , "::anno::unused(string)"
            , "::anno::deprecated(string)"
        };
        return (translation.find(name) != translation.end());
    }

    XMLElement * already_output(const char* value) const
    {
        if (value)
        {
            std::map<string /*source*/, XMLElement * /*trans_unit*/ >::const_iterator it =
                m_already_output.find(value);
            if (it != m_already_output.end())
            {
                return it->second;
            }
        }
        return NULL;
    }
};

// Get annotations from module
class Module_annotations
{
    std::string m_qualified_name;
private:
    mi::Sint32 load_module(mi::neuraylib::ITransaction * transaction) const
    {
        Handle<IMdl_impexp_api> mdl_impexp_api(
            Application::theApp().neuray()->get_api_component<IMdl_impexp_api>());
        return mdl_impexp_api->load_module(transaction, m_qualified_name.c_str());
    }
public:
    Module_annotations(const std::string & qualified_name)
        : m_qualified_name(qualified_name)
    {
    }

    mi::Sint32 traverse_module(Traversal_context & context) const
    {
        INeuray * nr(Application::theApp().neuray());
        Handle<IDatabase> database(nr->get_api_component<IDatabase>());
        Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
        Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());
        std::string db_name = i18n::Util::add_mdl_db_prefix(m_qualified_name);
        Handle<const IModule> module(transaction->access<IModule>(db_name.c_str()));
        check_success(module != 0);

        Annotation_traversal traversal(transaction.get());
        traversal.set_context(& context);
        traversal.handle_module(module.get());
        module = NULL;
        transaction->commit();
        return 0;
    }
};

// Attempt to load the given qualified name as a module
// Return true if success, return false if failure
bool load_module(const std::string & qualified_name)
{
    INeuray * nr(Application::theApp().neuray());
    Handle<IDatabase> database(nr->get_api_component<IDatabase>());
    Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());
    Handle<IMdl_impexp_api> mdl_impexp_api(nr->get_api_component<IMdl_impexp_api>());
    const mi::Sint32 rtn = mdl_impexp_api->load_module(transaction.get(), qualified_name.c_str());
    transaction->commit();
    return (0 == rtn);
}

class Search_element
{
    Handle<const IMdl_info> m_element;
    Handle<const IMdl_package_info> m_root;
    typedef enum
    {
        NONE, PACKAGE, MODULE

    } FILTER;
    FILTER m_filter;
protected:
    bool accept_package() const
    {
        return m_filter == NONE || m_filter == PACKAGE;
    }
    bool accept_module() const
    {
        return m_filter == NONE || m_filter == MODULE;
    }
    bool find_internal(const IMdl_package_info * package, const string & qname)
    {
        const char * name(package->get_qualified_name());
        if (name == qname && accept_package())
        {
            m_element = mi::base::make_handle_dup(package);
            return true;
        }
        for (mi::Size i = 0; i < package->get_child_count(); i++)
        {
            Handle<const IMdl_info> child(package->get_child(i));

            const IMdl_info::Kind kind(child->get_kind());

            if (kind == IMdl_info::DK_PACKAGE)
            {
                Handle<const IMdl_package_info> child_package(
                    child->get_interface<IMdl_package_info>());
                if (find_internal(child_package.get(), qname) == true)
                {
                    return true;
                }
            }
            else if (kind == IMdl_info::DK_MODULE)
            {
                Handle<const IMdl_module_info> child_module(
                    child->get_interface<IMdl_module_info>());
                const char * name(child_module->get_qualified_name());
                if (name == qname && accept_module())
                {
                    m_element = mi::base::make_handle_dup(child_module.get());
                    return true;
                }
            }
        }
        return false;
    }

    bool init_search()
    {
        m_element = NULL;
        if (!m_root)
        {
            INeuray * nr(Application::theApp().neuray());
            // Use discovery API to get some information regarding MDL elements
            Handle<IMdl_discovery_api> discovery(nr->get_api_component<IMdl_discovery_api>());
            Handle<const IMdl_discovery_result> discovery_result(discovery->discover());
            m_root = Handle<const IMdl_package_info>(discovery_result->get_graph());
        }
        return true;
    }
public:
    bool find(const string & element)
    {
        m_filter = NONE;
        init_search();
        return find_internal(m_root.get(), element);
    }
    bool find_module(const string & element)
    {
        m_filter = MODULE;
        init_search();
        return find_internal(m_root.get(), element);
    }
    bool find_package(const string & element)
    {
        m_filter = PACKAGE;
        init_search();
        return find_internal(m_root.get(), element);
    }
    const IMdl_info * get_element()
    {
        return (m_element ? m_element.get() : NULL);
    }
    bool list_modules_and_packages_from_package(
          const IMdl_info * elem
        , vector<string> & modules
        , vector<string> & packages
        , const char * mdl_path_filter
        , bool recursive
    )
    {
        if (elem)
        {
            Handle<const IMdl_package_info> p(elem->get_interface<IMdl_package_info>());
            if (p)
            {
                for (mi::Size i = 0; i < p->get_child_count(); i++)
                {
                    Handle<const IMdl_info> child(p->get_child(i));
                    const IMdl_info::Kind kind(child->get_kind());
                    if (kind == IMdl_info::DK_MODULE)
                    {
                        Handle<const IMdl_module_info> m(child->get_interface<IMdl_module_info>());
                        string search_path(m->get_search_path());
                        if (!mdl_path_filter || (mdl_path_filter && search_path == mdl_path_filter))
                        {
                            const char * name(child->get_qualified_name());
                            modules.push_back(name);
                        }
                    }
                    else if (kind == IMdl_info::DK_PACKAGE)
                    {
                        const char * name(child->get_qualified_name());
                        packages.push_back(name);
                        if (recursive)
                        {
                            list_modules_and_packages_from_package(
                                child.get()
                                , modules
                                , packages
                                , mdl_path_filter
                                , recursive
                            );
                        }
                    }
                }
                return true;
            }
        }
        return false;
    }

    bool list_modules_from_package(
          const IMdl_info * elem
        , vector<string> & modules
        , const char * mdl_path_filter
        , bool recursive)
    {
        vector<string> packages;
        return list_modules_and_packages_from_package(
            elem, modules, packages, mdl_path_filter, recursive);
    }
    bool in_archive(const IMdl_info * element) const
    {
        bool in_archive(false);
        const IMdl_info::Kind kind(element->get_kind());
        if (kind == IMdl_info::DK_MODULE)
        {
            Handle<const IMdl_module_info> module(element->get_interface<IMdl_module_info>());
            in_archive = module->in_archive();
        }
        else if (kind == IMdl_info::DK_PACKAGE)
        {
            Handle<const IMdl_package_info> package(element->get_interface<IMdl_package_info>());
            for (mi::Size i = 0; i < package->get_search_path_index_count(); i++)
            {
                in_archive |= package->in_archive(i);
            }
        }
        return in_archive;
    }
    bool get_search_paths(const IMdl_package_info * package, vector<string> & search_path) const
    {
        for (mi::Size i = 0; i < package->get_search_path_index_count(); i++)
        {
            search_path.push_back(package->get_search_path(i));
        }
        return true;
    }
};

Create_xliff_command::Create_xliff_command()
{}

void Create_xliff_command::log_settings() const
{
    Util::log_info("Create XLIFF Settings:");
    Util::log_info("\tModules:");
    for (auto & module : m_modules)
    {
        Util::log_info("\t\t" + module);
    }
    Util::log_info("\tPackages:");
    for (auto & package : m_packages)
    {
        Util::log_info("\t\t" + package);
    }
    Util::log_info("\tLocale: " + m_locale);
    Util::log_info(string("\tRecursive traversal: ") + (m_recursive ? "Yes" : "No"));
    Util::log_info(string("\tOutput context: ") + (m_output_context ? "Yes" : "No"));
    Util::log_report(string("\tDry run: ") + (m_dry_run ? "Yes" : "No"));
    Util::log_report(string("\tForce: ") + (m_force ? "Yes" : "No"));
    i18n::Search_path sp(Application::theApp().neuray());
    sp.snapshot();
    Util::log_info("\tMDL Search paths: ");
    for (auto & p : sp.paths())
    {
        Util::log_info(string("\t\t") + p);
    }
}

int Create_xliff_command::execute()
{
    if (m_locale.empty())
    {
        Util::log_error("Locale is missing, use --locale | -l <locale string>");
        return MISSING_LOCALE;
    }
    if (m_packages.empty() && m_modules.empty())
    {
        Util::log_error("Package and/or module is missing, \
use --module | -m <module> or --package | -p <package>");
        return MISSING_PACKAGE_OR_MODULE;
    }
    if ((!m_packages.empty() && !m_modules.empty())
        || m_packages.size() > 1
        || m_modules.size() > 1
        )
    {
        Util::log_error("Can only process one package or one module at a time.");
        return ONLY_ONE_PACKAGE_OR_MODULE;
    }
    
    log_settings();

    int rtn_code = SUCCESS;
    for (auto & package : m_packages)
    {
        Util::log_info("Processing package : " + package);
        rtn_code = handle_package(package);
    }
    for (auto & module : m_modules)
    {
        Util::log_info("Processing module : " + module);
        rtn_code = handle_module(module);
    }

    return rtn_code;
}

bool Create_xliff_command::check_file(const string & filename) const
{
    if (Util::File(filename).exist())
    {
        if (m_force)
        {
            Util::log_info("Overwriting existing file: " + filename);
            return true;
        }
        else
        {
            Util::log_error("The file already exists: " + filename);
            return false;
        }
    }
    return true;
}

bool Create_xliff_command::build_filename(const IMdl_module_info * module, string & filename) const
{
    Handle<const IString> ispath(module->get_resolved_path());
    string path(ispath->get_c_str());
    Util::File f(path);
    string dir(f.get_directory());
    filename = string(module->get_simple_name()) + "_" + m_locale + ".xlf";
    filename = Util::path_appends(dir, filename);
    return true;
}

bool Create_xliff_command::build_filename(
    const IMdl_package_info * package, const string & search_path, string & filename) const
{
    for (mi::Size i = 0; i < package->get_search_path_index_count(); i++)
    {
        if (package->get_search_path(i) == search_path)
        {
            Handle<const IString> ispath(package->get_resolved_path(i));
            string dir(ispath->get_c_str());
            filename = m_locale + ".xlf";
            filename = Util::path_appends(dir, filename);
            return true;
        }
    }
    return false;
}

int Create_xliff_command::handle_module(const std::string & module)
{
    // Double check the module exists
    Search_element helper;
    if (!helper.find_module(module))
    {
        Util::log_error("The module can not be found: " + module);
        return MODULE_NOT_FOUND;
    }

    Handle<const IMdl_info> element(mi::base::make_handle_dup(helper.get_element()));
    if (helper.in_archive(element.get()))
    {
        Util::log_error("Archive are not supported, extract archive and try again.");
        return ARCHIVE_NOT_SUPPORTED;
    }
    const IMdl_info::Kind kind(element->get_kind());
    check_success(kind == IMdl_info::DK_MODULE);
    Handle<const IMdl_module_info> module_info(element->get_interface<IMdl_module_info>());
    string filename;
    if (build_filename(module_info.get(), filename))
    {
        if (!check_file(filename))
        {
            return FILE_ALREADY_EXISTS;
        }
        const std::vector<std::string> modules = { module };
        return handle_modules(modules, filename, module);
    }
    return UNSPECIFIED_FAILURE;
}

int Create_xliff_command::handle_package(const std::string & package)
{
    // Double check the package exists
    Search_element helper;
    if (!helper.find_package(package))
    {
        Util::log_error("The package can not be found: " + package);
        return PACKAGE_NOT_FOUND;
    }

    Handle<const IMdl_info> element(mi::base::make_handle_dup(helper.get_element()));
    if (helper.in_archive(element.get()))
    {
        Util::log_error("Archives are not supported, extract archive and try again.");
        return ARCHIVE_NOT_SUPPORTED;
    }

    const IMdl_info::Kind kind(element->get_kind());
    check_success(kind == IMdl_info::DK_PACKAGE);
    Handle<const IMdl_package_info> package_info(element->get_interface<IMdl_package_info>());

    vector<string> search_path;
    if (helper.get_search_paths(package_info.get(), search_path))
    {
        if (search_path.size() > 1)
        {
            // The code below can create multiple XLIFF files (one per search path),
            // but to avoid confusion, we do not allow to create multiple XLIFF files.
            Util::log_error(
                "Package " + package +
                " is present in multiple MDL search paths, this is not allowed.");
            for (auto & p : search_path)
            {
                Util::log_error("\tSearch path: " + p);
            }
            return PACKAGE_IN_MANY_SEARCH_PATH;
        }

        for (auto & p : search_path)
        {
            Util::log_info("---------------------------------------------");
            Util::log_info("Handle modules from MDL path: " + p);
            std::vector<std::string> modules;
            helper.list_modules_from_package(element.get(), modules, p.c_str(), m_recursive);
            for (auto & m : modules)
            {
                Util::log_info(m);
            }
            string filename;
            if (build_filename(package_info.get(), p, filename))
            {
                if (check_file(filename) != 0)
                {
                    std::vector<std::string> modules;
                    helper.list_modules_from_package(
                        element.get(), modules, p.c_str(), m_recursive);
                    handle_modules(modules, filename, package);
                }
            }
        }
    }
    return SUCCESS;
}

int Create_xliff_command::handle_modules(
    const std::vector<std::string> & modules, const std::string & filename
    , const std::string & qualified_name)
{
    int rtn(SUCCESS);
    Create_xliff_context context(m_locale);
    context.set_output_context(m_output_context);
    context.set_qualified_name(qualified_name);
    for (auto& m : modules)
    {
        if (!m_dry_run && load_module(m))
        {
            // The module was loaded properly
            Util::log_info("Succesfully loaded module: " + m);

            Module_annotations ma(m);
            rtn |= ma.traverse_module(context);
        }
    }
    if (!m_dry_run)
    {
        if (!context.finalize_document(filename))
        {
            Util::log_info("Failed to create XLIFF file: " + filename);
            rtn = FAILED_TO_CREATE_XLIFF_FILE;
        }
        else
        { 
            Util::log_info("Succesfully created XLIFF file: " + filename);
        }
    }
    else
    {
        Util::log_report("Dry run, do not create XLIFF file: " + filename);
    }

    return rtn;
}
