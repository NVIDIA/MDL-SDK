/******************************************************************************
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

// examples/mdl_sdk/shared/example_materialx_shared.cpp

#include "example_materialx_shared.h"

#include "example_shared.h"

#include <MaterialXCore/Material.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXGenMdl/MdlShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Library.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Util.h>


namespace mi {namespace examples { namespace materialx {

namespace mx = MaterialX;

bool is_materialx_resource(const std::string& s)
{
    if (mi::examples::strings::ends_with(s, ".mtlx"))
        return true;
    if( s.find( ".mtlx?") != std::string::npos)
        return true;
    return false;
}

// ------------------------------------------------------------------------------------------------

class Mdl_string_resolver;
using Mdl_string_resolver_ptr = std::shared_ptr<Mdl_string_resolver>;

class Mdl_string_resolver : public mx::StringResolver
{
    Mdl_string_resolver(mi::neuraylib::IMdl_configuration* mdl_configuration)
        : m_mdl_configuration(make_handle_dup(mdl_configuration))
    {}

public:

    /// Create a new string resolver.
    static Mdl_string_resolver_ptr create(mi::neuraylib::IMdl_configuration* mdl_configuration)
    {
        return Mdl_string_resolver_ptr(new Mdl_string_resolver(mdl_configuration));
    }
    ~Mdl_string_resolver() = default;

    void initialize(mx::DocumentPtr document)
    {
        // remove duplicates and keep order by using a set
        auto less = [](const mx::FilePath& lhs, const mx::FilePath& rhs)
            { return lhs.asString() < rhs.asString(); };
        std::set<mx::FilePath, decltype(less)> mtlx_paths(less);
        m_mtlx_document_paths.clear();
        m_mdl_search_paths.clear();

        // use the source search paths as base
        mx::FilePath p = mx::FilePath(document->getSourceUri()).getParentPath().getNormalized();
        mtlx_paths.insert(p);
        m_mtlx_document_paths.append(p);

        for (auto sp : mx::getSourceSearchPath(document))
        {
            sp = sp.getNormalized();
            if(sp.exists() && mtlx_paths.insert(sp).second)
                m_mtlx_document_paths.append(sp);
        }

        // add all search paths known to MDL
        for (size_t i = 0, n = m_mdl_configuration->get_mdl_paths_length(); i < n; i++)
        {
            mi::base::Handle<const mi::IString> sp_istring(m_mdl_configuration->get_mdl_path(i));
            p = mx::FilePath(sp_istring->get_c_str()).getNormalized();
            if (p.exists() && mtlx_paths.insert(p).second)
                m_mtlx_document_paths.append(p);

            // keep a list of MDL search paths for resource resolution
            m_mdl_search_paths.append(p);
        }
    }

    std::string resolve(const std::string& str, const std::string& type) const override
    {
        mx::FilePath normalized_path = mx::FilePath(str).getNormalized();
        std::string resource_path;

        // in case the path is absolute we need to find a proper search path to put the file in
        if (normalized_path.isAbsolute())
        {
            // find the highest priority search path that is a prefix of the resource path
            for (const auto& sp : m_mdl_search_paths)
            {
                if (sp.size() > normalized_path.size())
                    continue;

                bool is_parent = true;
                for (size_t i = 0; i < sp.size(); ++i)
                {
                    if (sp[i] != normalized_path[i])
                    {
                        is_parent = false;
                        break;
                    }
                }

                if (!is_parent)
                    continue;

                // found a search path that is a prefix of the resource
                resource_path = normalized_path.asString(mx::FilePath::FormatPosix).substr(
                    sp.asString(mx::FilePath::FormatPosix).size());
                if (resource_path[0] != '/')
                    resource_path = "/" + resource_path;
                return resource_path;
            }
        }
        else
        {
            // for relative paths we can try to find them in the MDL search paths, assuming
            // they are specified "relative" to a search path root.
            mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> resolver(
                m_mdl_configuration->get_entity_resolver());

            resource_path = str;
            if (resource_path[0] != '/')
                resource_path = "/" + resource_path;

            mi::base::Handle<const mi::neuraylib::IMdl_resolved_resource> result(
                resolver->resolve_resource(resource_path.c_str(), nullptr, nullptr, 0, 0));

            if (result && result->get_count() > 0)
                return resource_path;
        }

        print_message(mi::base::MESSAGE_SEVERITY_ERROR,
            "Failed to resolve a MaterialX resource via an MDL search path. "
            "Dropping the resource from the Material. Resource Path: " + normalized_path.asString());

        // drop the resource by returning the empty string.
        // alternatively, the resource could be copied into an MDL search path,
        // maybe even only temporary.
        return "";
    }

    // Get the MaterialX paths used to load the current document as well the current MDL search
    // paths in order to resolve resources by the MaterialX SDK.
    const mx::FileSearchPath& get_search_paths() const { return m_mtlx_document_paths; }

private:

    // API component to get access to the entity resolver and the search path configuration.
    mi::base::Handle<mi::neuraylib::IMdl_configuration> m_mdl_configuration;

    // List of paths from which MaterialX can locate resources.
    // This includes the document folder and the search paths used to load the document.
    mx::FileSearchPath m_mtlx_document_paths;

    // List of MDL search paths from which we can locate resources.
    // This is only a subset of the MaterialX document paths and needs to be extended by using the
    // `--mdl_path` option when starting the application if needed.
    mx::FileSearchPath m_mdl_search_paths;
};

// ------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

void Mdl_generator::set_add_std_path(bool add_paths)
{
    m_add_mtlx_std_path = add_paths;
}

// ------------------------------------------------------------------------------------------------

void Mdl_generator::add_path(const std::string& mtlx_path)
{
    m_mtlx_search_paths.push_back(mtlx_path);
#ifdef MI_PLATFORM_WINDOWS
    std::replace(m_mtlx_search_paths.back().begin(), m_mtlx_search_paths.back().end(), '/', '\\');
#endif
}

// ------------------------------------------------------------------------------------------------

void Mdl_generator::add_library(const std::string& mtlx_library)
{
    m_mtlx_relative_library_paths.push_back(mtlx_library);
#ifdef MI_PLATFORM_WINDOWS
    std::replace(m_mtlx_relative_library_paths.back().begin(),
        m_mtlx_relative_library_paths.back().end(), '/', '\\');
#endif
}

// ------------------------------------------------------------------------------------------------

void Mdl_generator::set_mdl_version(mi::neuraylib::Mdl_version mdl_version)
{
    m_mdl_version = mdl_version;
}

// ------------------------------------------------------------------------------------------------

void Mdl_generator::set_materialxtest_mode(bool enabled)
{
    m_materialxtest_mode = enabled;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_generator::set_source(const std::string& mtlx_file, const std::string& material_name)
{
    m_mtlx_file = mtlx_file;
    m_mtlx_material_name = material_name;
    std::replace(m_mtlx_file.begin(), m_mtlx_file.end(), '/', '\\');
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_generator::generate(
    mi::neuraylib::IMdl_configuration* mdl_configuration, Mdl_generator_result& inout_result) const
{
    // Initialize the standard library
    mx::DocumentPtr mtlx_std_lib;
    mx::StringSet mtlx_include_files;
    mx::FilePathVec mtlx_library_folders = { "libraries" };
    mx::FileSearchPath mtlx_search_path;

    // add additional search paths
    for (auto& p : m_mtlx_search_paths)
        mtlx_search_path.append(mx::FilePath{ p });

    // add the libraries in the applications binary folder
    if (m_add_mtlx_std_path)
    {
        std::string mdl_search_path_for_materialx_support
            = mi::examples::mdl::find_resource_directory(
                m_mdl_example_relative_directory.c_str(), "materialx");
        mtlx_search_path.append(mx::FilePath{ mdl_search_path_for_materialx_support });
    }

    // add additional relative library paths
    for (auto& l : m_mtlx_relative_library_paths)
        mtlx_library_folders.push_back(mx::FilePath{ l });

    try
    {
        mtlx_std_lib = mx::createDocument();
        mtlx_include_files = mx::loadLibraries(
            mtlx_library_folders, mtlx_search_path, mtlx_std_lib);
        if (mtlx_include_files.empty())
        {
            print_message(mi::base::MESSAGE_SEVERITY_ERROR,
                "[MTLX] Could not find standard data libraries on the given search path: "
                + mtlx_search_path.asString(), __FILE__, __LINE__);
        }

    }
    catch (std::exception& e)
    {
        print_message(mi::base::MESSAGE_SEVERITY_ERROR,
            std::string( "[MTLX] Failed to initialize standard libraries: ") + e.what(),
            __FILE__, __LINE__);
    }

    // Initialize unit management.
    mx::UnitConverterRegistryPtr mtlx_unit_registry = mx::UnitConverterRegistry::create();
    mx::UnitTypeDefPtr distance_type_def = mtlx_std_lib->getUnitTypeDef("distance");
    mx::LinearUnitConverterPtr distance_unit_converter
        = mx::LinearUnitConverter::create(distance_type_def);
    mtlx_unit_registry->addUnitConverter(distance_type_def, distance_unit_converter);
    mx::UnitTypeDefPtr angle_type_def = mtlx_std_lib->getUnitTypeDef("angle");
    mx::LinearUnitConverterPtr angle_converter = mx::LinearUnitConverter::create(angle_type_def);
    mtlx_unit_registry->addUnitConverter(angle_type_def, angle_converter);

    // Create the list of supported distance units.
    mx::StringVec distance_unit_options;
    const std::unordered_map<std::string, float>& unitScales
        = distance_unit_converter->getUnitScale();
    distance_unit_options.resize(unitScales.size());
    for (const auto& unitScale : unitScales)
    {
        int location = distance_unit_converter->getUnitAsInteger(unitScale.first);
        distance_unit_options[location] = unitScale.first;
    }

    // Initialize the generator contexts.
    mx::GenContext generator_context = mx::MdlShaderGenerator::create();

    // Initialize search paths.
    for (const mx::FilePath& path : mtlx_search_path)
    {
        for (const auto& folder : mtlx_library_folders)
        {
            if (folder.size() > 0)
                generator_context.registerSourceCodeSearchPath(path / folder);
        }
    }
    // Initialize color management.
    mx::DefaultColorManagementSystemPtr cms = mx::DefaultColorManagementSystem::create(
        generator_context.getShaderGenerator().getTarget());
    cms->loadLibrary(mtlx_std_lib);
    generator_context.getShaderGenerator().setColorManagementSystem(cms);
    generator_context.getOptions().targetColorSpaceOverride = "lin_rec709";


#if MATERIALX_VERSION_INDEX <= MATERIALX_GENERATE_INDEX(1, 39, 5)
    // Flipping the texture lookups for the test renderer only.
    // This is because OSL testrender does not allow to change the UV layout of their sphere (yet)
    // and the MaterialX test suite adopts the OSL behavior in order to produce comparable results.
    // This means that raw texture coordinates, or procedurals that use the texture coordinates, do
    // not match what might be expected when reading the MaterialX spec:
    //    "[...] the image is mapped onto the geometry based on geometry UV coordinates, with the
    //    lower-left corner of an image  mapping to the (0,0) UV coordinate [...]"
    // This means for MDL: when used with the MaterialXTest suite, we flip the UV coordinates of
    // mesh using the `--uv_flip` option of the renderer, and to correct the image orientation, we
    // apply `fileTextureVerticalFlip`. In regular MDL integrations this is not needed because MDL
    // and MaterialX define the texture space equally with the origin at the bottom left.
    generator_context.getOptions().fileTextureVerticalFlip = m_materialxtest_mode;
#endif // MATERIALX_VERSION_INDEX <= MATERIALX_GENERATE_INDEX(1, 39, 5)

    // Initialize unit management.
    mx::UnitSystemPtr unit_system = mx::UnitSystem::create(
        generator_context.getShaderGenerator().getTarget());
    unit_system->loadLibrary(mtlx_std_lib);
    unit_system->setUnitConverterRegistry(mtlx_unit_registry);
    generator_context.getShaderGenerator().setUnitSystem(unit_system);
    generator_context.getOptions().targetDistanceUnit = "meter";

    // load the actual material
    if (m_mtlx_file.empty())
    {
        print_message(mi::base::MESSAGE_SEVERITY_ERROR,
            "[MTLX] Source file not specified.", __FILE__, __LINE__);
        return false;
    }

    // Set up read options.
    mx::XmlReadOptions read_options;
    read_options.readXIncludeFunction = [](mx::DocumentPtr doc, const mx::FilePath& filename,
        const mx::FileSearchPath& search_path, const mx::XmlReadOptions* options)
    {
        mx::FilePath resolved_filename = search_path.find(filename);
        if (resolved_filename.exists())
        {
            readFromXmlFile(doc, resolved_filename, search_path, options);
        }
        else
        {
            print_message(mi::base::MESSAGE_SEVERITY_ERROR,
                "[MTLX] Include file not found: " + filename.asString(), __FILE__, __LINE__);
        }
    };

    // Clear user data on the generator.
    generator_context.clearUserData();

    // Specify the MDL target version, for MaterialX 1.38.9 and later.
#if MATERIALX_VERSION_INDEX >= MATERIALX_GENERATE_INDEX(1, 38, 9)
    mx::GenMdlOptionsPtr gen_mdl_options = std::make_shared<mx::GenMdlOptions>();

    if (m_mdl_version == mi::neuraylib::MDL_VERSION_1_6)
        gen_mdl_options->targetVersion = mx::GenMdlOptions::MdlVersion::MDL_1_6;
    else if (m_mdl_version == mi::neuraylib::MDL_VERSION_1_7)
        gen_mdl_options->targetVersion = mx::GenMdlOptions::MdlVersion::MDL_1_7;
    else if (m_mdl_version == mi::neuraylib::MDL_VERSION_1_8)
        gen_mdl_options->targetVersion = mx::GenMdlOptions::MdlVersion::MDL_1_8;
#if MATERIALX_VERSION_INDEX >= MATERIALX_GENERATE_INDEX(1, 39, 2)
    else if (m_mdl_version == mi::neuraylib::MDL_VERSION_1_9)
        gen_mdl_options->targetVersion = mx::GenMdlOptions::MdlVersion::MDL_1_9;
#if MATERIALX_VERSION_INDEX >= MATERIALX_GENERATE_INDEX(1, 39, 3)
    else if (m_mdl_version == mi::neuraylib::MDL_VERSION_1_10)
        gen_mdl_options->targetVersion = mx::GenMdlOptions::MdlVersion::MDL_1_10;
#endif // MATERIALX_VERSION_INDEX >= MATERIALX_GENERATE_INDEX(1, 39, 3)
#endif // MATERIALX_VERSION_INDEX >= MATERIALX_GENERATE_INDEX(1, 39, 2)
    else if (m_mdl_version == mi::neuraylib::MDL_VERSION_LATEST)
        gen_mdl_options->targetVersion = mx::GenMdlOptions::MdlVersion::MDL_LATEST;
    else
        print_message(mi::base::MESSAGE_SEVERITY_WARNING,
            "Ignoring unexpected MDL version.", __FILE__, __LINE__);

    generator_context.pushUserData(mx::GenMdlOptions::GEN_CONTEXT_USER_DATA_KEY, gen_mdl_options);
#endif // MATERIALX_VERSION_INDEX >= MATERIALX_GENERATE_INDEX(1, 38, 9)

    // Load source document.
    mx::DocumentPtr material_document = mx::createDocument();
    mx::FilePath material_filename = m_mtlx_file;
    mx::readFromXmlFile(material_document, m_mtlx_file, mtlx_search_path, &read_options);

    // Import libraries.
    material_document->importLibrary(mtlx_std_lib);

    // flatten the resource paths of the document using a custom resolver allows
    // the change the resource URIs into valid MDL paths.
    auto custom_resolver = Mdl_string_resolver::create(mdl_configuration);
    custom_resolver->initialize(material_document);
    mx::flattenFilenames(material_document, custom_resolver->get_search_paths(), custom_resolver);

    // list the active set of search paths
    std::string refs_string = "MaterialX documents referenced:";
    mx::StringSet refs = material_document->getReferencedSourceUris();
    mi::Size n = refs.size();
    if (n == 0)
        refs_string += "\n                      - <none>";
    for (const std::string& r : refs)
    {
        refs_string += "\n                      - ";
        refs_string += r;
    }
    print_message(mi::base::MESSAGE_SEVERITY_VERBOSE, refs_string);

    // Validate the document.
    std::string message;
    if (!material_document->validate(&message))
    {
        // materialX validation failures do not mean that content can not be rendered.
        // it points to mtlx authoring errors but rendering could still be fine.
        // since MDL is robust against erroneous code we just continue. If there are problems
        // in the generated code, we detect it on module load and use a fall-back material.
        print_message(mi::base::MESSAGE_SEVERITY_WARNING,
            "[MTLX] Validation warnings for '" + m_mtlx_file + "'\n" + message,
            __FILE__, __LINE__);
    }

    // find (selected) renderable nodes
    mx::TypedElementPtr element_to_generate_code_for;
    if (!m_mtlx_material_name.empty())
    {
        mx::ElementPtr elem = material_document->getRoot();
        std::vector<std::string> path = mi::examples::strings::split(m_mtlx_material_name, '/');
        for (size_t i = 0; i < path.size(); ++i)
        {
            elem = elem->getChild(path[i]);
            if (!elem)
                break;
        }
        // if a node is specified properly, there is only one
        if (elem)
        {
            mx::TypedElementPtr typedElem = elem ? elem->asA<mx::TypedElement>() : nullptr;
            if (typedElem)
                element_to_generate_code_for = typedElem;
        }
    }
    else
    {
        // find the first render-able element
        std::vector<mx::TypedElementPtr> elems;
        elems = mx::findRenderableElements(material_document);
        if (elems.size() > 0)
        {
            element_to_generate_code_for = elems[0];
        }
    }

    if (!element_to_generate_code_for)
    {
        if (!m_mtlx_material_name.empty())
            print_message(mi::base::MESSAGE_SEVERITY_ERROR,
                "[MTLX] Code generation failure: no material named '"
                    + m_mtlx_material_name + "' found in '" + m_mtlx_file + "'");
        else
            print_message(mi::base::MESSAGE_SEVERITY_ERROR,
                "[MTLX] Code generation failure: no material found in '" + m_mtlx_file + "'");

        return false;
    }

    // Clear cached implementations, in case libraries on the file system have changed.
    generator_context.clearNodeImplementations();

    std::string material_name = element_to_generate_code_for->getNamePath();
    material_name = mi::examples::strings::replace(material_name, '/', '_');

    mx::ShaderPtr shader = nullptr;
    try
    {
#if MATERIALX_VERSION_INDEX >= MATERIALX_GENERATE_INDEX(1, 39, 3)
        generator_context.getShaderGenerator().registerTypeDefs(material_document);
#endif // MATERIALX_VERSION_INDEX >= MATERIALX_GENERATE_INDEX(1, 39, 3)
        shader = generator_context.getShaderGenerator().generate(
            material_name, element_to_generate_code_for, generator_context);
    }
    catch (mx::Exception& e)
    {
        print_message(mi::base::MESSAGE_SEVERITY_ERROR,
            std::string( "[MTLX] Code generation failure: ") + e.what(), __FILE__, __LINE__);
        return false;
    }

    if (!shader)
    {
        print_message(mi::base::MESSAGE_SEVERITY_ERROR,
            "[MTLX] Failed to generate shader for element: " + material_name, __FILE__, __LINE__);
        return false;
    }

    const std::string& generated = shader->getSourceCode("pixel");
    if (generated.empty())
    {
        print_message(mi::base::MESSAGE_SEVERITY_ERROR,
            "[MTLX] Failed to generate source code for stage.", __FILE__, __LINE__);
        return false;
    }

    inout_result.materialx_file_name = m_mtlx_file;
    inout_result.materialx_material_name = material_name;
    inout_result.generated_mdl_code
        = std::string("// generated from MaterialX using the SDK version ")
        + MaterialX::getVersionString() + "\n\n"
        + generated;
    inout_result.generated_mdl_name = shader->getStage("pixel").getFunctionName();
    return true;
}

}}}
