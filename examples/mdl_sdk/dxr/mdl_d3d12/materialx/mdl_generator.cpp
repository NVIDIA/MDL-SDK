/******************************************************************************
 * Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
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

#include <example_shared.h>
#include <utils/io.h>

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

bool Mdl_generator::set_source(const std::string& mtlx_material)
{
    if (!mi::examples::io::file_exists(mtlx_material))
    {
        log_error("[MTLX] Material path does not exist: " + mtlx_material, SRC);
        return false;
    }

    m_mtlx_source = mtlx_material;
    std::replace(m_mtlx_source.begin(), m_mtlx_source.end(), '/', '\\');
    return true;
}

// ------------------------------------------------------------------------------------------------

bool Mdl_generator::generate(Mdl_generator_result& inout_result) const
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
    {
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

        // Validate the document.
        std::string message;
        if (!material_document->validate(&message))
        {
            log_error("[MTLX] Validation warnings for '" + m_mtlx_source + "'", SRC);
            std::cerr << message;
            return false;
        }

        // Find renderable elements.
        mx::StringVec renderablePaths;
        std::vector<mx::TypedElementPtr> elems;
        std::vector<mx::NodePtr> materialNodes;
        mx::findRenderableElements(material_document, elems);
        if (elems.empty())
        {
            log_error("[MTLX] No renderable elements found in '" + m_mtlx_source + "'", SRC);
            return false;
        }
        for (mx::TypedElementPtr elem : elems)
        {
            mx::TypedElementPtr renderableElem = elem;
            mx::NodePtr node = elem->asA<mx::Node>();
            if (node && node->getType() == mx::MATERIAL_TYPE_STRING)
            {
                std::vector<mx::NodePtr> shaderNodes = getShaderNodes(node, mx::SURFACE_SHADER_TYPE_STRING);
                if (!shaderNodes.empty())
                {
                    renderableElem = *shaderNodes.begin();
                }
                materialNodes.push_back(node);
            }
            else
            {
                materialNodes.push_back(nullptr);
            }
            renderablePaths.push_back(renderableElem->getNamePath());
        }

        struct Mtlx_material
        {
            mx::DocumentPtr document;
            mx::TypedElementPtr element;
            mx::NodePtr material_node;
        };
        std::vector<Mtlx_material> materials_to_genereate;

        // Create new materials.
        for (size_t i = 0; i < renderablePaths.size(); i++)
        {
            const auto& renderablePath = renderablePaths[i];
            mx::ElementPtr elem = material_document->getDescendant(renderablePath);
            mx::TypedElementPtr typedElem = elem ? elem->asA<mx::TypedElement>() : nullptr;
            if (!typedElem)
            {
                continue;
            }

            materials_to_genereate.push_back(Mtlx_material{});
            Mtlx_material& mat = materials_to_genereate.back();
            mat.document = material_document;
            mat.element = typedElem;
            mat.material_node = materialNodes[i];
        }

        for (Mtlx_material& mat : materials_to_genereate)
        {
            // Clear cached implementations, in case libraries on the file system have changed.
            generator_context.clearNodeImplementations();

            mx::ShaderPtr shader = nullptr;
            try
            {
                shader =
                    generator_context.getShaderGenerator().generate("Shader", mat.element, generator_context);
            }
            catch (mx::Exception& e)
            {
                log_error("[MTLX] Code generation failure:", e, SRC);
                shader = nullptr;
            }
            if (!shader)
            {
                log_error("[MTLX] Failed to generate shader for element: " +
                    mat.element->getNamePath(), SRC);
                continue;
            }

            auto generated = shader->getSourceCode("pixel");
            if (generated.empty())
            {
                log_error("[MTLX] Failed to generate source code for stage.", SRC);
                continue;
            }

            inout_result.generated_mdl_code.push_back(generated);
            inout_result.materialx_file_name.push_back(m_mtlx_source);
        }
    }
    return inout_result.generated_mdl_code.size() > 0;
}

}}}}
