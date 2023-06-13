/******************************************************************************
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_generator.h"

#include "../common.h"
#include "../mdl_sdk.h"
#include <mi/mdl_sdk.h>

#include <MaterialXCore/Material.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXGenMdl/MdlShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Library.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Util.h>


namespace mi {namespace examples { namespace mdl_d3d12 { namespace materialx
{
namespace mx = MaterialX;


class MdlStringResolver;
using MdlStringResolverPtr = std::shared_ptr<MdlStringResolver>;

class MdlStringResolver : public mx::StringResolver
{
public:

    /// Create a new string resolver.
    static MdlStringResolverPtr create()
    {
        return MdlStringResolverPtr(new MdlStringResolver());
    }
    ~MdlStringResolver() = default;

    void initialize(mx::DocumentPtr document, mi::neuraylib::IMdl_configuration* config)
    {
        // remove duplicates and keep order by using a set
        auto less = [](const mx::FilePath& lhs, const mx::FilePath& rhs) { return lhs.asString() < rhs.asString(); };
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
        for (size_t i = 0, n = config->get_mdl_paths_length(); i < n; i++)
        {
            mi::base::Handle<const mi::IString> sp_istring(config->get_mdl_path(i));
            p = mx::FilePath(sp_istring->get_c_str()).getNormalized();
            if (p.exists() && mtlx_paths.insert(p).second)
                m_mtlx_document_paths.append(p);

            // keep a list of MDL search paths for resource resolution
            m_mdl_search_paths.append(p);
        }
    }

    std::string resolve(const std::string& str, const std::string& type) const override
    {
        mx::FilePath normalizedPath = mx::FilePath(str).getNormalized();

        // in case the path is absolute we need to find a proper search path to put the file in
        if (normalizedPath.isAbsolute())
        {
            // find the highest priority search path that is a prefix of the resource path
            for (const auto& sp : m_mdl_search_paths)
            {
                if (sp.size() > normalizedPath.size())
                    continue;

                bool isParent = true;
                for (size_t i = 0; i < sp.size(); ++i)
                {
                    if (sp[i] != normalizedPath[i])
                    {
                        isParent = false;
                        break;
                    }
                }

                if (!isParent)
                    continue;

                // found a search path that is a prefix of the resource
                std::string resource_path =
                    normalizedPath.asString(mx::FilePath::FormatPosix).substr(
                        sp.asString(mx::FilePath::FormatPosix).size());
                if (resource_path[0] != '/')
                    resource_path = "/" + resource_path;
                return resource_path;
            }
        }

        log_error("MaterialX resource can not be accessed through an MDL search path. "
            "Dropping the resource from the Material. Resource Path: " + normalizedPath.asString());

        // drop the resource by returning the empty string.
        // alternatively, the resource could be copied into an MDL search path,
        // maybe even only temporary.
        return "";
    }

    // Get the MaterialX paths used to load the current document as well the current MDL search
    // paths in order to resolve resources by the MaterialX SDK.
    const mx::FileSearchPath& get_search_paths() const { return m_mtlx_document_paths; }

private:

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

Mdl_generator::Mdl_generator()
    : m_mtlx_search_paths()
    , m_mtlx_relative_library_paths()
{
}

// ------------------------------------------------------------------------------------------------

void Mdl_generator::add_path(const std::string& mtlx_path)
{
    m_mtlx_search_paths.push_back(mtlx_path);
    std::replace(m_mtlx_search_paths.back().begin(), m_mtlx_search_paths.back().end(), '/', '\\');
}

// ------------------------------------------------------------------------------------------------

void Mdl_generator::add_library(const std::string& mtlx_library)
{
    m_mtlx_relative_library_paths.push_back(mtlx_library);
    std::replace(m_mtlx_relative_library_paths.back().begin(),
        m_mtlx_relative_library_paths.back().end(), '/', '\\');
}

// ------------------------------------------------------------------------------------------------

bool Mdl_generator::set_source(const std::string& mtlx_material, const std::string& material_name)
{
    if (!mi::examples::io::file_exists(mtlx_material))
    {
        log_error("[MTLX] Material path does not exist: " + mtlx_material, SRC);
        return false;
    }

    m_mtlx_source = mtlx_material;
    m_mtlx_material_name = material_name;
    std::replace(m_mtlx_source.begin(), m_mtlx_source.end(), '/', '\\');
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_generator::generate(Mdl_sdk& mdl_sdk, Mdl_generator_result& inout_result) const
{
    // Initialize the standard library
    mx::DocumentPtr mtlx_std_lib;
    mx::StringSet mtlx_include_files;
    mx::FilePathVec mtlx_library_folders = { "libraries" };
    mx::FileSearchPath mtlx_search_path;
    mtlx_search_path.append(
        mx::FilePath{ mi::examples::io::get_executable_folder() + "/autodesk_materialx" });

    // add additional search paths
    for (auto& p : m_mtlx_search_paths)
        mtlx_search_path.append(mx::FilePath{ p });

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
            log_error("[MTLX] Could not find standard data libraries on the given search path: " +
                mtlx_search_path.asString(), SRC);
        }

    }
    catch (std::exception& e)
    {
        log_error("[MTLX] Failed to initialize standard libraries:", e, SRC);
    }

    // Initialize unit management.
    mx::UnitConverterRegistryPtr mtlx_unit_registry = mx::UnitConverterRegistry::create();
    mx::UnitTypeDefPtr distanceTypeDef = mtlx_std_lib->getUnitTypeDef("distance");
    mx::LinearUnitConverterPtr _distanceUnitConverter = mx::LinearUnitConverter::create(distanceTypeDef);
    mtlx_unit_registry->addUnitConverter(distanceTypeDef, _distanceUnitConverter);
    mx::UnitTypeDefPtr angleTypeDef = mtlx_std_lib->getUnitTypeDef("angle");
    mx::LinearUnitConverterPtr angleConverter = mx::LinearUnitConverter::create(angleTypeDef);
    mtlx_unit_registry->addUnitConverter(angleTypeDef, angleConverter);

    // Create the list of supported distance units.
    mx::StringVec _distanceUnitOptions;
    auto unitScales = _distanceUnitConverter->getUnitScale();
    _distanceUnitOptions.resize(unitScales.size());
    for (auto unitScale : unitScales)
    {
        int location = _distanceUnitConverter->getUnitAsInteger(unitScale.first);
        _distanceUnitOptions[location] = unitScale.first;
    }

    // Initialize the generator contexts.
    mx::GenContext generator_context = mx::MdlShaderGenerator::create();

    // Initialize search paths.
    for (const mx::FilePath& path : mtlx_search_path)
    {
        for (const auto folder : mtlx_library_folders)
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
    generator_context.getOptions().fileTextureVerticalFlip = false;

    // Initialize unit management.
    mx::UnitSystemPtr unitSystem = mx::UnitSystem::create(
        generator_context.getShaderGenerator().getTarget());
    unitSystem->loadLibrary(mtlx_std_lib);
    unitSystem->setUnitConverterRegistry(mtlx_unit_registry);
    generator_context.getShaderGenerator().setUnitSystem(unitSystem);
    generator_context.getOptions().targetDistanceUnit = "meter";

    // load the actual material

    if (m_mtlx_source.empty())
    {
        log_error("[MTLX] Source file not specified.", SRC);
        return false;
    }

    // Set up read options.
    mx::XmlReadOptions readOptions;
    readOptions.readXIncludeFunction = [](mx::DocumentPtr doc, const mx::FilePath& filename,
        const mx::FileSearchPath& searchPath, const mx::XmlReadOptions* options)
    {
        mx::FilePath resolvedFilename = searchPath.find(filename);
        if (resolvedFilename.exists())
        {
            readFromXmlFile(doc, resolvedFilename, searchPath, options);
        }
        else
        {
            log_error("[MTLX] Include file not found: " + filename.asString(), SRC);
        }
    };

    // Clear user data on the generator.
    generator_context.clearUserData();

    // Load source document.
    mx::DocumentPtr material_document = mx::createDocument();
    mx::FilePath material_filename = m_mtlx_source;
    mx::readFromXmlFile(material_document, m_mtlx_source, mtlx_search_path, &readOptions);

    // Import libraries.
    material_document->importLibrary(mtlx_std_lib);

    // flatten the resource paths of the document using a custom resolver allows
    // the change the resource URIs into valid MDL paths.
    auto custom_resolver = MdlStringResolver::create();
    custom_resolver->initialize(material_document, &mdl_sdk.get_config());
    mx::flattenFilenames(material_document, custom_resolver->get_search_paths(), custom_resolver);

    // Validate the document.
    std::string message;
    if (!material_document->validate(&message))
    {
        // materialX validation failures do not mean that content can not be rendered.
        // it points to mtlx authoring errors but rendering could still be fine.
        // since MDL is robust against erroneous code we just continue. If there are problems
        // in the generated code, we detect it on module load and use a fall-back material.
        log_warning("[MTLX] Validation warnings for '" + m_mtlx_source + "'\n" + message, SRC);
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
        mx::findRenderableElements(material_document, elems);
        if (elems.size() > 0)
        {
            element_to_generate_code_for = elems[0];
        }
    }

    if (!element_to_generate_code_for)
    {
        if (!m_mtlx_material_name.empty())
            log_error("[MTLX] Code generation failure: no material named '" +
                m_mtlx_material_name + "' found in '" + m_mtlx_source + "'");
        else
            log_error("[MTLX] Code generation failure: no material found in '"
                + m_mtlx_source + "'");

        return false;
    }

    // Clear cached implementations, in case libraries on the file system have changed.
    generator_context.clearNodeImplementations();

    std::string material_name = element_to_generate_code_for->getNamePath();
    material_name = mi::examples::strings::replace(material_name, '/', '_');

    mx::ShaderPtr shader = nullptr;
    try
    {
        shader =
            generator_context.getShaderGenerator().generate(material_name, element_to_generate_code_for, generator_context);
    }
    catch (mx::Exception& e)
    {
        log_error("[MTLX] Code generation failure:", e, SRC);
        return false;
    }

    if (!shader)
    {
        log_error("[MTLX] Failed to generate shader for element: " + material_name, SRC);
        return false;
    }

    auto generated = shader->getSourceCode("pixel");
    if (generated.empty())
    {
        log_error("[MTLX] Failed to generate source code for stage.", SRC);
        return false;
    }

    inout_result.materialx_file_name = m_mtlx_source;
    inout_result.materialx_material_name = material_name;
    inout_result.generated_mdl_code = generated;
    inout_result.generated_mdl_name = shader->getStage("pixel").getFunctionName();
    return true;
}

}}}}
