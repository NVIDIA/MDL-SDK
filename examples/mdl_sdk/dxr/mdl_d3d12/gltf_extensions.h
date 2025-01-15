/******************************************************************************
 * Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

// examples/mdl_sdk/dxr/mdl_d3d12/gltf_nv_materials_mdl.h

#ifndef MDL_D3D12_GLTF_EXTENSIONS_H
#define MDL_D3D12_GLTF_EXTENSIONS_H

#include "common.h"

#include <fx/gltf.h>

namespace fx { namespace gltf
{
    namespace detail
    {
        constexpr char const* const MimetypeApplicationIES_LM_63 = "data:application/x-ies-lm-63;base64";
        constexpr char const* const MimetypeApplicationMDL = "data:application/vnd.mdl;base64";
        constexpr char const* const MimetypeApplicationMBSDF = "data:application/vnd.mdl-mbsdf;base64";
        constexpr char const* const MimetypeImageEXR = "data:image/x-exr;base64";
    }

    template <typename ResourceT>
    void materialize_data(const ResourceT& gltf_resource, std::vector<uint8_t>& data);

    struct NV_MaterialsMDL
    {
        struct Type
        {
            enum class Modifier
            {
                Auto,
                Varying,
                Uniform
            };

            int32_t module{ -1 };
            std::string typeName;
            int32_t arraySize{ -1 };
            Modifier modifier{ Modifier::Auto };

            nlohmann::json extensionsAndExtras{};
        };

        struct Module
        {
            int32_t bufferView{ -1 };

            std::string uri;
            std::string modulePath;
            std::string mimeType;
            std::string name{};

            nlohmann::json extensionsAndExtras{};

            bool IsEmbeddedResource() const noexcept
            {
                return uri.find(detail::MimetypeApplicationMDL) == 0;
            }

            void MaterializeData(std::vector<uint8_t>& data) const
            {
                materialize_data(*this, data);
            }
        };

        struct Argument
        {
            // added for convenience (not part of the schema)
            // allows to quickly differentiate between `functionCall` and `value`
            enum class Kind
            {
                None,
                FunctionCall,
                Value
            };

            struct Value
            {
                enum class Kind
                {
                    None,
                    Boolean,
                    Decimal,
                    Integer,
                    String
                };

                Kind kind{ Kind::None };

                union Element
                {
                    bool boolean;
                    int integer;
                    float decimal;
                };

                std::vector<Element> data = std::vector<Element>(0, Element());
                std::vector<std::string> dataStrings = std::vector<std::string>(0, "");

                nlohmann::json extensionsAndExtras{};
            };

            std::string name{};
            Type type{};

            Kind kind{ Kind::None };
            int32_t functionCall{ -1 };
            Value value{};

            nlohmann::json extensionsAndExtras{};
        };

        struct FunctionCall
        {
            int32_t module{ -1 };
            std::string functionName{};
            Type type{};
            std::vector<Argument> arguments{};
            std::string name{};

            nlohmann::json extensionsAndExtras{};
        };

        struct BsdfMeasurement
        {
            int32_t bufferView{ -1 };

            std::string name;
            std::string uri;
            std::string mimeType;

            nlohmann::json extensionsAndExtras{};

            bool IsEmbeddedResource() const noexcept
            {
                return uri.find(detail::MimetypeApplicationMBSDF) == 0;
            }

            void MaterializeData(std::vector<uint8_t>& data) const
            {
                materialize_data(*this, data);
            }
        };


        /// Extensions that appears in the top level "extensions" list of an glTF file.
        struct Gltf_extension
        {
            static constexpr const char* EXTENSION_NAME = "NV_materials_mdl";

            std::vector<Module> modules{};
            std::vector<FunctionCall> functionCalls{};
            std::vector<BsdfMeasurement> bsdfMeasurements{};

            nlohmann::json extensionsAndExtras{};
        };

        /// Extensions that appears in the "extensions" list of an glTF material.
        struct Material_extension
        {
            static constexpr const char* EXTENSION_NAME = "NV_materials_mdl";

            int32_t functionCall{ -1 };

            nlohmann::json extensionsAndExtras{};
        };

        static std::string convert_module_uri_to_package(const std::string& uri)
        {
            // remove .mdl
            std::string result = (uri.rfind(".mdl") != std::string::npos)
                ? uri.substr(0, uri.size() - 4) : uri;

            // remove mdl schema
            if (result.find("mdl:///") != std::string::npos)
                result = result.substr(6); // remove "mdl://", leave one '/'
            else if (result.find("mdl:/") != std::string::npos)
                result = result.substr(4); // remove "mdl:", leave the '/'

            // remove preceding . and ..
            if (result[0] == '.')
            {
                if (result[1] == '.')
                    result = result.substr(2);
                else
                    result = result.substr(1);
            }

            // prepend slash (/) if missing
            if (result[0] != '/')
                result = '/' + result;

            // replace all slashes (/) with a double colon (::)
            for (size_t pos = 0; pos < result.size(); pos++)
            {
                if (result[pos] != '/')
                    continue;

                // find end of multi slash
                size_t end = pos + 1;
                while (end < result.size() && result[end] == '/')
                    end++;

                result.replace(pos, end - pos, "::");
            }

            return result;
        }
    };

    struct EXT_LightsIES
    {
        struct Light
        {
            int32_t bufferView{ -1 };

            std::string name;
            std::string uri;
            std::string mimeType;

            nlohmann::json extensionsAndExtras{};

            bool IsEmbeddedResource() const noexcept
            {
                return uri.find(detail::MimetypeApplicationIES_LM_63) == 0;
            }

            void MaterializeData(std::vector<uint8_t>& data) const
            {
                materialize_data(*this, data);
            }
        };

        /// Extensions that appears in the top level "extensions" list of an glTF file.
        struct Gltf_extension
        {
            static constexpr const char* EXTENSION_NAME = "EXT_lights_ies";

            std::vector<Light> lights{};

            nlohmann::json extensionsAndExtras{};
        };

        /// Extensions that appears in the "extensions" list of an glTF node.
        struct Node_extension
        {
            static constexpr const char* EXTENSION_NAME = "EXT_lights_ies";

            int32_t light{ -1 };
            float multiplier{ 1.0f };
            std::array<float, 3> color{ 1.0f, 1.0f, 1.0f };

            nlohmann::json extensionsAndExtras{};
        };
    };

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Type::Modifier& modifier)
    {
        std::string name = json.get<std::string>();

        if (name == "varying")
        {
            modifier = NV_MaterialsMDL::Type::Modifier::Varying;
        }
        else if (name == "uniform")
        {
            modifier = NV_MaterialsMDL::Type::Modifier::Uniform;
        }
        else
        {
            modifier = NV_MaterialsMDL::Type::Modifier::Auto;
        }
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Type& type)
    {
        detail::ReadOptionalField("module", json, type.module);
        detail::ReadRequiredField("typeName", json, type.typeName);
        detail::ReadOptionalField("modifier", json, type.modifier);
        detail::ReadOptionalField("arraySize", json, type.arraySize);

        detail::ReadExtensionsAndExtras(json, type.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Module& module)
    {
        detail::ReadOptionalField("uri", json, module.uri);
        detail::ReadOptionalField("bufferView", json, module.bufferView);
        detail::ReadOptionalField("mimeType", json, module.mimeType);
        detail::ReadOptionalField("modulePath", json, module.modulePath);
        detail::ReadOptionalField("name", json, module.name);

        detail::ReadExtensionsAndExtras(json, module.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Argument::Value& argumentValue)
    {
        if (json.is_array())
        {
            size_t size = json.size();
            if (size == 0)
                return;

            if (json[0].is_boolean())
            {
                argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Boolean;
                argumentValue.data.resize(size);
            }
            else if (json[0].is_number_float())
            {
                argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Decimal;
                argumentValue.data.resize(size);
            }
            else if (json[0].is_number_integer())
            {
                argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Integer;
                argumentValue.data.resize(size);
            }
            else if (json[0].is_string())
            {
                argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::String;
                argumentValue.dataStrings.resize(size);
            }
            else if (json[0].is_array())
            {
                size_t element_size = json[0].size();
                if (element_size == 0)
                    return;

                if (json[0][0].is_boolean())
                {
                    argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Boolean;
                    argumentValue.data.resize(size * element_size);
                }
                else if (json[0][0].is_number_float())
                {
                    argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Decimal;
                    argumentValue.data.resize(size * element_size);
                }
                else if (json[0][0].is_number_integer())
                {
                    argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Integer;
                    argumentValue.data.resize(size * element_size);
                }
            }

            for (size_t i = 0; i < size; i++)
            {
                if (json[i].is_array())
                {
                    for (size_t k = 0; k < json[i].size(); k++)
                    {
                        size_t data_index = i * json[i].size() + k;

                        switch (argumentValue.kind)
                        {
                        case NV_MaterialsMDL::Argument::Value::Kind::Boolean:
                            argumentValue.data[data_index].boolean = json[i][k].get<bool>();
                            break;
                        case NV_MaterialsMDL::Argument::Value::Kind::Decimal:
                            argumentValue.data[data_index].decimal = json[i][k].get<float>();
                            break;
                        case NV_MaterialsMDL::Argument::Value::Kind::Integer:
                            argumentValue.data[data_index].integer = json[i][k].get<int>();
                            break;
                        default:
                            break;
                        }
                    }
                }
                else
                {
                    switch (argumentValue.kind)
                    {
                    case NV_MaterialsMDL::Argument::Value::Kind::Boolean:
                        argumentValue.data[i].boolean = json[i].get<bool>();
                        break;
                    case NV_MaterialsMDL::Argument::Value::Kind::Decimal:
                        argumentValue.data[i].decimal = json[i].get<float>();
                        break;
                    case NV_MaterialsMDL::Argument::Value::Kind::Integer:
                        argumentValue.data[i].integer = json[i].get<int>();
                        break;
                    case NV_MaterialsMDL::Argument::Value::Kind::String:
                        argumentValue.dataStrings[i] = json[i].get<std::string>();
                        break;
                    default:
                        break;
                    }
                }
            }
        }
        else
        {
            if (json.is_boolean())
            {
                argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Boolean;
                argumentValue.data.resize(1);
                argumentValue.data[0].boolean = json.get<bool>();
            }
            else if (json.is_number_float())
            {
                argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Decimal;
                argumentValue.data.resize(1);
                argumentValue.data[0].decimal = json.get<float>();
            }
            else if (json.is_number_integer())
            {
                argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::Integer;
                argumentValue.data.resize(1);
                argumentValue.data[0].integer = json.get<int>();
            }
            else if (json.is_string())
            {
                argumentValue.kind = NV_MaterialsMDL::Argument::Value::Kind::String;
                argumentValue.dataStrings.resize(1);
                argumentValue.dataStrings[0] = json.get<std::string>();
            }
        }
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Argument& argument)
    {
        detail::ReadRequiredField("name", json, argument.name);

        bool isFunctionCall = json.contains("functionCall");
        bool isValue = json.contains("value");

        if (isFunctionCall == isValue)
        {
            throw invalid_gltf_document("Argument has to be either a 'functionCall' or 'value'");
        }

        if (isFunctionCall)
        {
            if (json.contains("type"))
                throw invalid_gltf_document(
                    "Arguments with a 'functionCall' must not contain a 'type'");

            argument.kind = NV_MaterialsMDL::Argument::Kind::FunctionCall;
            detail::ReadRequiredField("functionCall", json, argument.functionCall);
            argument.value = {};
        }
        else if (isValue)
        {
            argument.kind = NV_MaterialsMDL::Argument::Kind::Value;
            detail::ReadRequiredField("value", json, argument.value);
            detail::ReadRequiredField("type", json, argument.type);
            argument.functionCall = { -1 };
        }

        detail::ReadExtensionsAndExtras(json, argument.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::FunctionCall& function_call)
    {
        detail::ReadOptionalField("module", json, function_call.module);
        detail::ReadRequiredField("functionName", json, function_call.functionName);
        detail::ReadRequiredField("type", json, function_call.type);
        detail::ReadOptionalField("arguments", json, function_call.arguments);
        detail::ReadOptionalField("name", json, function_call.name);

        detail::ReadExtensionsAndExtras(json, function_call.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::BsdfMeasurement& mbsdf)
    {
        detail::ReadOptionalField("uri", json, mbsdf.uri);
        detail::ReadOptionalField("bufferView", json, mbsdf.bufferView);
        detail::ReadOptionalField("mimeType", json, mbsdf.mimeType);
        detail::ReadOptionalField("name", json, mbsdf.name);

        detail::ReadExtensionsAndExtras(json, mbsdf.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Gltf_extension& gltf_extension)
    {
        detail::ReadOptionalField("modules", json, gltf_extension.modules);
        detail::ReadOptionalField("functionCalls", json, gltf_extension.functionCalls);
        detail::ReadOptionalField("bsdfMeasurements", json, gltf_extension.bsdfMeasurements);

        detail::ReadExtensionsAndExtras(json, gltf_extension.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Material_extension& material_extension)
    {
        detail::ReadRequiredField("functionCall", json, material_extension.functionCall);

        detail::ReadExtensionsAndExtras(json, material_extension.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, EXT_LightsIES::Light& light)
    {
        detail::ReadOptionalField("uri", json, light.uri);
        detail::ReadOptionalField("bufferView", json, light.bufferView);
        detail::ReadOptionalField("mimeType", json, light.mimeType);
        detail::ReadOptionalField("name", json, light.name);

        detail::ReadExtensionsAndExtras(json, light.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, EXT_LightsIES::Gltf_extension& gltf_extension)
    {
        detail::ReadOptionalField("lights", json, gltf_extension.lights);

        detail::ReadExtensionsAndExtras(json, gltf_extension.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, EXT_LightsIES::Node_extension& node_extension)
    {
        detail::ReadOptionalField("light", json, node_extension.light);
        detail::ReadOptionalField("multiplier", json, node_extension.multiplier);
        detail::ReadOptionalField("color", json, node_extension.color);

        detail::ReadExtensionsAndExtras(json, node_extension.extensionsAndExtras);
    }

    template<typename TParent, typename TExtension>
    bool read_extension(const TParent& parent, TExtension& extension)
    {
        const auto it = parent.extensionsAndExtras.find("extensions");
        if (it != parent.extensionsAndExtras.end())
        {
            if (it->contains(TExtension::EXTENSION_NAME))
            {
                from_json(*it->find(TExtension::EXTENSION_NAME), extension);
                return true;
            }
        }
        return false;
    }

    // Version of Image::IsEmbeddedResource that adds support for NV_image_exr.
    inline bool is_embedded_resource(const Image& image) noexcept
    {
        return image.IsEmbeddedResource() || image.uri.find(detail::MimetypeImageEXR) == 0;
    }

    inline bool is_embedded_resource(const EXT_LightsIES::Light& light) noexcept
    {
        return light.IsEmbeddedResource();
    }

    inline bool is_embedded_resource(const NV_MaterialsMDL::Module& module) noexcept
    {
        return module.IsEmbeddedResource();
    }

    inline bool is_embedded_resource(const NV_MaterialsMDL::BsdfMeasurement& mbsdf) noexcept
    {
        return mbsdf.IsEmbeddedResource();
    }

    template <typename ResourceT>
    void materialize_data(const ResourceT& gltf_resource, std::vector<uint8_t>& data)
    {
        std::size_t startPos = gltf_resource.uri.find(";base64,");
        if (startPos == std::string::npos)
        {
            throw std::runtime_error("Only base64 embedded data is supported");
        }
        startPos += 7;

    #if defined(FX_GLTF_HAS_CPP_17)
        const std::size_t base64Length = gltf_resource.uri.length() - startPos - 1;
        const bool success = base64::TryDecode({ &gltf_resource.uri[startPos + 1], base64Length }, data);
    #else
        const bool success = base64::TryDecode(gltf_resource.uri.substr(startPos + 1), data);
    #endif
        if (!success)
        {
            throw invalid_gltf_document("Invalid buffer.uri value", "malformed base64");
        }
    }
}}

namespace mi { namespace examples { namespace mdl_d3d12
{

}}} // mi::examples::mdl_d3d12
#endif
