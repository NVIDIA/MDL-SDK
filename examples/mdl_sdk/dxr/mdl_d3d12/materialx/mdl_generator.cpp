/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <MaterialXCore/MaterialNode.h>
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
    : m_dependencies()
{
}

// ------------------------------------------------------------------------------------------------

bool Mdl_generator::add_dependency(const std::string& mtlx_library)
{
    if (!mi::examples::io::file_exists(mtlx_library))
    {
        log_error("[MTLX] Library path does not exist: " + mtlx_library, SRC);
        return false;
    }

    m_dependencies.push_back(mtlx_library);
    std::replace(m_dependencies.back().begin(), m_dependencies.back().end(), '/', '\\');
    return true;
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
    auto generator = mx::MdlShaderGenerator::create();
    const std::string language = generator->getLanguage();
    mx::GenOptions generateOptions;

    // standard library
    std::string libSearchPath_s =
        mi::examples::io::get_executable_folder() + "/autodesk_materialx/libraries";
    const mx::FilePath libSearchPath_fp = mx::FilePath(libSearchPath_s);
    const mx::FileSearchPath libSearchPath(libSearchPath_fp.asString());
    if (!libSearchPath_fp.exists())
    {
        log_error("[MTLX] Library path does not exist: " + libSearchPath_s, SRC);
        return false;
    }

    // target language standard library
    mx::FileSearchPath srcSearchPath = libSearchPath;
    srcSearchPath.append(libSearchPath_fp / mx::FilePath("stdlib/genmdl"));

    // setup dependencies
    auto _dependLib = mx::createDocument();
    {
        // load the standard libraries.
        const mx::FilePathVec libraries = { "stdlib", "pbrlib", "lights" };
        mx::loadLibraries(libraries, libSearchPath, _dependLib);

        // add color management
        auto _colorManagementSystem = mx::DefaultColorManagementSystem::create(language);
        if (_colorManagementSystem)
        {
            generator->setColorManagementSystem(_colorManagementSystem);
            _colorManagementSystem->loadLibrary(_dependLib);
            generateOptions.targetColorSpaceOverride = "lin_rec709";
        }
        else
        {
            log_error("[MTLX] Failed to create color management system.", SRC);
            return false;
        }

        // add unit system
        auto _unitSystem = mx::UnitSystem::create(language);
        if (_unitSystem)
        {
            generator->setUnitSystem(_unitSystem);
            _unitSystem->loadLibrary(_dependLib);
            _unitSystem->setUnitConverterRegistry(mx::UnitConverterRegistry::create());
            mx::UnitTypeDefPtr distanceTypeDef = _dependLib->getUnitTypeDef("distance");
            _unitSystem->getUnitConverterRegistry()->addUnitConverter(distanceTypeDef, mx::LinearUnitConverter::create(distanceTypeDef));
            generateOptions.targetDistanceUnit = "meter";
            mx::UnitTypeDefPtr angleTypeDef = _dependLib->getUnitTypeDef("angle");
            _unitSystem->getUnitConverterRegistry()->addUnitConverter(angleTypeDef, mx::LinearUnitConverter::create(angleTypeDef));
        }
        else
        {
            log_error("[MTLX] Failed to create unit system.", SRC);
            return false;
        }

        // Load dependencies of the current material
        for(auto& dep : m_dependencies)
            mx::loadLibrary(mx::FilePath(dep), _dependLib);
    }

    // load the actual material
    mx::DocumentPtr material_document = mx::createDocument();
    {
        if (m_mtlx_source.empty())
        {
            log_error("[MTLX] Mtlx source file not specified.", SRC);
            return false;
        }

        mx::XmlReadOptions readOptions;
        mx::FileSearchPath searchPath(libSearchPath);
        mx::FilePath material_path(m_mtlx_source);
        mx::readFromXmlFile(material_document, material_path, searchPath, &readOptions);

        // Add in dependent libraries
        bool importedLibrary = false;
        try
        {
            material_document->importLibrary(_dependLib);
            importedLibrary = true;
        }
        catch (std::exception& e)
        {
            log_error("[MTLX] Failed to import libraries into material:", e, SRC);
            return false;
        }
    }

    // Find elements to render in the document
    std::vector<mx::TypedElementPtr> elements;
    try
    {
        mx::findRenderableElements(material_document, elements);
    }
    catch (mx::Exception & e)
    {
        log_error("[MTLX] Renderables search errors:", e, SRC);
        return false;
    }

    // generate code
    {
        // Map to replace "/" in Element path names with "_".
        mx::StringMap pathMap;
        pathMap["/"] = "_";

        mx::GenContext context(generator);
        context.getOptions() = generateOptions;
        context.registerSourceCodeSearchPath(srcSearchPath);

        for (const auto& element : elements)
        {
            mx::TypedElementPtr targetElement = element;
            mx::OutputPtr output = targetElement->asA<mx::Output>();
            mx::ShaderRefPtr shaderRef = targetElement->asA<mx::ShaderRef>();
            mx::NodePtr outputNode = targetElement->asA<mx::Node>();
            mx::NodeDefPtr nodeDef = nullptr;
            if (output)
            {
                outputNode = output->getConnectedNode();
                // Handle connected upstream material nodes later on.
                if (outputNode->getType() != mx::MATERIAL_TYPE_STRING)
                {
                    nodeDef = outputNode->getNodeDef();
                }
            }
            else if (shaderRef)
            {
                nodeDef = shaderRef->getNodeDef();
            }

            // Handle material node checking. For now only check first surface shader if any
            if (outputNode)
            {
                const std::string& type = outputNode->getType();
                if (type == mx::MATERIAL_TYPE_STRING)
                {
                    std::unordered_set<mx::NodePtr> shaderNodes = getShaderNodes(outputNode, mx::SURFACE_SHADER_TYPE_STRING);
                    if (!shaderNodes.empty())
                    {
                        auto first = shaderNodes.begin();
                        nodeDef = (*first)->getNodeDef();
                        targetElement = *first;
                    }
                }
            }

            const std::string namePath(targetElement->getNamePath());
            if (nodeDef)
            {
                mx::string elementName = mx::replaceSubstrings(namePath, pathMap);
                elementName = mx::createValidName(elementName);

                mx::InterfaceElementPtr impl = nodeDef->getImplementation(generator->getTarget(), language);
                if (impl)
                {
                    // generate mdl code
                    {
                        mx::ShaderPtr shader = nullptr;
                        try
                        {
                            shader = generator->generate(elementName, targetElement, context);
                        }
                        catch (mx::Exception & e)
                        {
                            log_error("[MTLX] Code generation failure:", e, SRC);
                            shader = nullptr;
                        }
                        if (!shader)
                        {
                            log_error("[MTLX] Failed to generate shader for element: " +
                                targetElement->getNamePath(), SRC);
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
            }
        }
    }
    return inout_result.generated_mdl_code.size() > 0;
}

}}}}
