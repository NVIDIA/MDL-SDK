/******************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_D3D12_GLTF_NV_MATERIAL_MDL_H
#define MDL_D3D12_GLTF_NV_MATERIAL_MDL_H

#include "common.h"

#include <fx/gltf.h>

namespace fx { namespace gltf
{
    struct NV_MaterialsMDL
    {
        struct Type
        {
            // added for convenience (not part of the schema)
            // allows to quickly differentiate between `BuiltinType` and `UserType`
            enum class Kind
            {
                None,
                BuiltinType,
                UserType
            };

            enum class Modifier
            {
                Auto,
                Varying,
                Uniform
            };

            Kind kind{ Kind::None };
            std::string builtinType;
            int32_t userType;
            int32_t arraySize{ -1 };
            Modifier modifier{ Modifier::Auto };

            nlohmann::json extensionsAndExtras{};
        };

        struct UserType
        {
            int32_t module;
            std::string typeName;
            std::string name{};

            nlohmann::json extensionsAndExtras{};
        };

        struct Module
        {
            std::string uri;
            std::string name{};

            nlohmann::json extensionsAndExtras{};
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


        /// Extensions that appears in the top level "extensions" list of an glTF file.
        struct Gltf_extension
        {
            static constexpr const char* EXTENSION_NAME = "NV_materials_mdl";

            std::vector<Module> modules{};
            std::vector<FunctionCall> functionCalls{};
            std::vector<UserType> userTypes{};

            nlohmann::json extensionsAndExtras{};
        };

        /// Extensions that appears in the "extensions" list of an glTF material.
        struct Material_extension
        {
            static constexpr const char* EXTENSION_NAME = "NV_materials_mdl";

            int32_t functionCall{ -1 };

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
        detail::ReadOptionalField("modifier", json, type.modifier);

        bool isBuiltinType = json.contains("builtinType");
        bool isUserType = json.contains("userType");
        detail::ReadOptionalField("arraySize", json, type.arraySize);

        if (isBuiltinType == isUserType)
        {
            throw invalid_gltf_document("Type has to be either 'builtinType' or 'userType'");
        }

        if (isBuiltinType)
        {
            type.kind = NV_MaterialsMDL::Type::Kind::BuiltinType;
            detail::ReadRequiredField("builtinType", json, type.builtinType);
            type.userType = { -1 };
        }
        else if (isUserType)
        {
            type.kind = NV_MaterialsMDL::Type::Kind::UserType;
            detail::ReadRequiredField("userType", json, type.userType);
            type.builtinType = "";
        }

        detail::ReadExtensionsAndExtras(json, type.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::UserType& userType)
    {
        detail::ReadRequiredField("module", json, userType.module);
        detail::ReadRequiredField("typeName", json, userType.typeName);
        detail::ReadOptionalField("name", json, userType.name);

        detail::ReadExtensionsAndExtras(json, userType.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Module& module)
    {
        detail::ReadRequiredField("uri", json, module.uri);
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

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Gltf_extension& gltf_extension)
    {
        detail::ReadOptionalField("modules", json, gltf_extension.modules);
        detail::ReadOptionalField("functionCalls", json, gltf_extension.functionCalls);
        detail::ReadOptionalField("userTypes", json, gltf_extension.userTypes);

        detail::ReadExtensionsAndExtras(json, gltf_extension.extensionsAndExtras);
    }

    inline void from_json(nlohmann::json const& json, NV_MaterialsMDL::Material_extension& material_extension)
    {
        detail::ReadRequiredField("functionCall", json, material_extension.functionCall);

        detail::ReadExtensionsAndExtras(json, material_extension.extensionsAndExtras);
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

}}

namespace mi { namespace examples { namespace mdl_d3d12
{

}}} // mi::examples::mdl_d3d12
#endif
