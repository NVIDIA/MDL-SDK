/***************************************************************************************************
* Copyright 2020 NVIDIA Corporation. All rights reserved.
**************************************************************************************************/

#include "mdl_arnold.h"
#include "mdl_arnold_interface.h"
#include "mdl_arnold_bsdf.h"
#include "mdl_arnold_utils.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <set>

#include <ai.h>
#include <ai_shaderglobals.h>
#include <ai_array.h>
#include <ai_nodes.h>
#include <ai_params.h>
#include <ai_shader_userdef.h>

AI_SHADER_NODE_EXPORT_METHODS(Mdl_arnold_bsdf_shader_methods);

// load the MDL SDK when the plugin is loaded
node_plugin_initialize
{
    Mdl_sdk_interface* mdl_sdk = new Mdl_sdk_interface();
    *plugin_data = mdl_sdk;
    return mdl_sdk->get_sdk_state() == EMdl_sdk_state::loaded;
}

// unload the MDL SDK when the plugin is unloaded
node_plugin_cleanup
{
    Mdl_sdk_interface* mdl_sdk = static_cast<Mdl_sdk_interface*>(plugin_data);
    delete mdl_sdk;
}

namespace
{
    namespace CONST_STRINGS
    {
        const AtString empty("");
        const AtString qualified_name("qualified_name");
    }
}

// Per node data, used for CPU and GPU renderer.
// Holds the entire MDL related state of an MDL Node.
struct MdlLocalNodeData
{
    MdlLocalNodeData()
        : current_qualified_name(CONST_STRINGS::empty)
        , material_db_name("")
        , current_user_param_values()
        , user_parameters_changed(true)
        , material_hash{0, 0, 0, 0}
        , require_code_update(true)
        , require_texture_update(true)
        , data_cpu()
    {
    }

    // keep the current parameter name to detect changes
    AtString current_qualified_name;

    // store the db name
    std::string material_db_name;

    // handle changes of materials in interactive sessions
    std::unordered_map<size_t, AtParamValue> current_user_param_values;
    bool user_parameters_changed;

    // handle changes in the compiled material (second check after parameters changed)
    mi::base::Uuid material_hash;
    bool require_code_update;
    bool require_texture_update;

    // part of the node data that is passed to the BSDF
    MdlShaderNodeDataCPU data_cpu;
};


// Parameter handling converting Arnold node parameters to MDL values.
template<typename TParam>
mi::base::Handle<mi::neuraylib::IValue> create_value(
    Mdl_sdk_interface& mdl_sdk,
    const TParam& param);

template<>
mi::base::Handle<mi::neuraylib::IValue> create_value(Mdl_sdk_interface& mdl_sdk, const AtRGB& param)
{
    return mi::base::Handle<mi::neuraylib::IValue>(
        mdl_sdk.get_value_factory().create_color(param.r, param.g, param.b));
}

template<>
mi::base::Handle<mi::neuraylib::IValue> create_value(Mdl_sdk_interface& mdl_sdk, const float& param)
{
    return mi::base::Handle<mi::neuraylib::IValue>(mdl_sdk.get_value_factory().create_float(param));
}

template<>
mi::base::Handle<mi::neuraylib::IValue> create_value(Mdl_sdk_interface& mdl_sdk, const int& param)
{
    return mi::base::Handle<mi::neuraylib::IValue>(mdl_sdk.get_value_factory().create_int(param));
}

template<>
mi::base::Handle<mi::neuraylib::IValue> create_value(Mdl_sdk_interface& mdl_sdk, const bool& param)
{
    return mi::base::Handle<mi::neuraylib::IValue>(mdl_sdk.get_value_factory().create_bool(param));
}

template<>
mi::base::Handle<mi::neuraylib::IValue> create_value(
    Mdl_sdk_interface& mdl_sdk, const AtString& param)
{
    return mi::base::Handle<mi::neuraylib::IValue>(
        mdl_sdk.get_value_factory().create_string(param));
}

template<>
mi::base::Handle<mi::neuraylib::IValue> create_value(
    Mdl_sdk_interface& mdl_sdk, const AtVector& param)
{
    mi::base::Handle<const mi::neuraylib::IType_float> ft(
        mdl_sdk.get_type_factory().create_float());
    mi::base::Handle<const mi::neuraylib::IType_vector> vt(
        mdl_sdk.get_type_factory().create_vector(ft.get(), 3));

    mi::base::Handle<mi::neuraylib::IValue_vector> res(
        mdl_sdk.get_value_factory().create_vector(vt.get()));
    res->set_value(0, create_value(mdl_sdk, param.x).get());
    res->set_value(1, create_value(mdl_sdk, param.y).get());
    res->set_value(2, create_value(mdl_sdk, param.z).get());

    return res;
}

template<>
mi::base::Handle<mi::neuraylib::IValue> create_value(
    Mdl_sdk_interface& mdl_sdk, const AtVector2& param)
{
    mi::base::Handle<const mi::neuraylib::IType_float> ft(
        mdl_sdk.get_type_factory().create_float());
    mi::base::Handle<const mi::neuraylib::IType_vector> vt(
        mdl_sdk.get_type_factory().create_vector(ft.get(), 2));

    mi::base::Handle<mi::neuraylib::IValue_vector> res(
        mdl_sdk.get_value_factory().create_vector(vt.get()));
    res->set_value(0, create_value(mdl_sdk, param.x).get());
    res->set_value(1, create_value(mdl_sdk, param.y).get());

    return res;
}

// Convert and add a named Arnold node parameter to an MDL expression list.
template<typename TParam>
void add_argument(
    Mdl_sdk_interface& mdl_sdk,
    mi::neuraylib::IExpression_list* parameter_list,
    const char* parameter_name,
    const TParam& param)
{

    auto value = create_value(mdl_sdk, param);

    mi::base::Handle<mi::neuraylib::IExpression> expr(
        mdl_sdk.get_expr_factory().create_constant(value.get()));

    parameter_list->add_expression(parameter_name, expr.get());
}

// Update the user parameter set with the given entry.
// Returns true, if the user parameters actually changed with this update.
bool update_current_user_param_value(
    MdlLocalNodeData& data, AtNode* node, const AtUserParamEntry* entry)
{
    if (entry == nullptr)
        return false;

    uint8_t param_type = AiUserParamGetType(entry);
    const char* param_name = AiUserParamGetName(entry);
    size_t hash = std::hash<std::string>{}(param_name);

    // parameter not known yet? -> just set it
    bool found = data.current_user_param_values.find(hash) != data.current_user_param_values.end();

    switch (param_type)
    {
    case AI_TYPE_BOOLEAN:
    {
        bool value = AiNodeGetBool(node, param_name);
        if (found && data.current_user_param_values[hash].BOOL() == value) return false;
        data.current_user_param_values[hash].BOOL() = value; return true;
    }
    case AI_TYPE_BYTE:
    {
        uint8_t value = AiNodeGetByte(node, param_name);
        if (found && data.current_user_param_values[hash].BYTE() == value) return false;
        data.current_user_param_values[hash].BYTE() = value; return true;
    }
    case AI_TYPE_INT:
    {
        int value = AiNodeGetInt(node, param_name);
        if (found && data.current_user_param_values[hash].INT() == value) return false;
        data.current_user_param_values[hash].INT() = value; return true;
    }
    case AI_TYPE_UINT:
    {
        unsigned int value = AiNodeGetUInt(node, param_name);
        if (found && data.current_user_param_values[hash].UINT() == value) return false;
        data.current_user_param_values[hash].UINT() = value; return true;
    }
    case AI_TYPE_FLOAT:
    {
        float value = AiNodeGetFlt(node, param_name);
        if (found && data.current_user_param_values[hash].FLT() == value) return false;
        data.current_user_param_values[hash].FLT() = value; return true;
    }
    case AI_TYPE_RGB:
    {
        AtRGB value = AiNodeGetRGB(node, param_name);
        if (found && data.current_user_param_values[hash].RGB() == value) return false;
        data.current_user_param_values[hash].RGB() = value; return true;
    }
    case AI_TYPE_RGBA:
    {
        AtRGBA value = AiNodeGetRGBA(node, param_name);
        if (found && data.current_user_param_values[hash].RGBA() == value) return false;
        data.current_user_param_values[hash].RGBA() = value; return true;
    }
    case AI_TYPE_VECTOR:
    {
        AtVector value = AiNodeGetVec(node, param_name);
        if (found && data.current_user_param_values[hash].VEC() == value) return false;
        data.current_user_param_values[hash].VEC() = value; return true;
    }
    case AI_TYPE_VECTOR2:
    {
        AtVector2 value = AiNodeGetVec2(node, param_name);
        if (found && data.current_user_param_values[hash].VEC2() == value) return false;
        data.current_user_param_values[hash].VEC2() = value; return true;
    }
    case AI_TYPE_POINTER:
    {

        void* value = AiNodeGetPtr(node, param_name);
        if (found && data.current_user_param_values[hash].PTR() == value) return false;
        data.current_user_param_values[hash].PTR() = value; return true;
    }
    case AI_TYPE_STRING:
    {
        AtString value = AiNodeGetStr(node, param_name);
        if (found && data.current_user_param_values[hash].STR() == value) return false;
        data.current_user_param_values[hash].STR() = value; return true;
    }
    case AI_TYPE_MATRIX:  // TODO: implement
    case AI_TYPE_ARRAY:   // TODO: implement
    case AI_TYPE_CLOSURE: // TODO: implement
    default:
        return false;
    }
}

// Loads the material specified by the "qualified_name" parameter of the given node
// and sets the database name of the material in the local node data.
// Returns true, if loading the material was successful.
bool load_material(AtNode* node, Mdl_sdk_interface& mdl_sdk)
{
    MdlLocalNodeData *data = static_cast<MdlLocalNodeData*>(AiNodeGetLocalData(node));

    // update search paths in case they changed
    std::lock_guard<std::mutex> lock(mdl_sdk.get_loading_mutex());
    mdl_sdk.set_search_paths();

    // set default material name
    if (data->material_db_name == "")
        data->material_db_name = mdl_sdk.get_default_material_db_name();

    // determine MDL module and material name from "qualified_name" node parameter
    std::string mdl_name(AiNodeGetStr(node, CONST_STRINGS::qualified_name).c_str());

    std::string module_name, material_name;
    if (!mi::examples::mdl::parse_cmd_argument_material_name(mdl_name, module_name, material_name, false))
    {
        AiMsgWarning("[mdl] module name to load '%s' is invalid", module_name.c_str());
        return false;
    }

    // try to get the database name of the required module
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> mdl_context(mdl_sdk.create_context());

    mi::base::Handle<const mi::IString> module_db_name(
        mdl_sdk.get_factory().get_db_module_name(module_name.c_str()));

    // parameter is no a valid qualified module name
    if (!module_db_name)
    {
        AiMsgWarning("[mdl] module name to load '%s' is invalid", module_name.c_str());
        return false;
    }

    // check if the module is already loaded
    mi::base::Handle<const mi::neuraylib::IModule> module(
        mdl_sdk.get_transaction().access<mi::neuraylib::IModule>(module_db_name->get_c_str()));

    // not found? -> load it
    // otherwise, it's loaded already because of an other material in the scene
    if (!module)
    {
        mi::Sint32 load_res_code = mdl_sdk.get_impexp_api().load_module(
            &mdl_sdk.get_transaction(), module_name.c_str(), mdl_context.get());

        if (!mdl_sdk.log_messages(mdl_context.get()))
        {
            AiMsgWarning("[mdl] failed to load module '%s' (Code: %d)",
                       module_name.c_str(), load_res_code);

            std::string current_search_paths = "";
            size_t count = mdl_sdk.get_config().get_mdl_paths_length();
            for (size_t i = 0; i < count; ++i)
                current_search_paths += std::string("\n - ") +
                std::string(mdl_sdk.get_config().get_mdl_path(i)->get_c_str());

            AiMsgWarning("[mdl] using error material as '%s' can not be resolved or loaded.\n"
                       "current search paths:%s", mdl_name.c_str(), current_search_paths.c_str());
            return false;
        }
    }

    // get the material by the material db name and ensure it exists
    std::string material_db_name = module_db_name->get_c_str();
    material_db_name += "::" + material_name;
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        mdl_sdk.get_transaction().access<mi::neuraylib::IMaterial_definition>(
        material_db_name.c_str()));

    if (!material_definition)
    {
        AiMsgWarning("[mdl] material '%s' not found in module '%s'",
                   material_name.c_str(), module_name.c_str());
        return false;
    }

    data->material_db_name = material_db_name;
    return true;
}


// Create an MDL expression list for the material arguments of the given material
// according to the user parameters specified in the given node.
mi::base::Handle<mi::neuraylib::IExpression_list> create_material_argument_list(
    AtNode* node,
    Mdl_sdk_interface& mdl_sdk,
    const mi::neuraylib::IMaterial_definition* material_definition)
{
    MdlLocalNodeData *data = static_cast<MdlLocalNodeData*>(AiNodeGetLocalData(node));

    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        mdl_sdk.get_expr_factory().create_expression_list());

    mi::base::Handle<const mi::neuraylib::IType_list> parameter_types(
        material_definition->get_parameter_types());

    // set of additional helper parameters which will not be treated as material parameters
    std::set<std::string> helper_params;

    AtUserParamIterator* it = AiNodeGetUserParamIterator(node);
    while (!AiUserParamIteratorFinished(it))
    {
        const AtUserParamEntry* entry = AiUserParamIteratorGetNext(it);
        const char* name = AiUserParamGetName(entry);
        size_t hash = std::hash<std::string>{}(name);
        uint8_t type = AiUserParamGetType(entry);

        // checking `node->isParamModified(i)` or `node->isNodeModified()` is not enough.
        // the report changes even when only the camera position changed
        if (update_current_user_param_value(*data, node, entry))
            data->user_parameters_changed = true;

        mi::Size param_index = material_definition->get_parameter_index(name);
        if (param_index == mi::Size(-1))
        {
            if (helper_params.find(name) == helper_params.end() &&
                strcmp(name, "nodeName") != 0) // special name in 3Ds max
                    AiMsgWarning("[mdl] parameter '%s' not found in material '%s'. Ignoring ...",
                                 name, material_definition->get_mdl_name());
            continue;
        }

        mi::base::Handle<const mi::neuraylib::IType> param_type(
            parameter_types->get_type(param_index));
        param_type = param_type->skip_all_type_aliases();
        switch (type)
        {
        case AI_TYPE_BOOLEAN:
        {
            if (param_type->get_kind() != mi::neuraylib::IType::TK_BOOL)
            {
                AiMsgWarning("[mdl] parameter '%s' is not of type bool. Ignoring ...", name);
                continue;
            }
            add_argument(
                mdl_sdk, arguments.get(), name, data->current_user_param_values[hash].BOOL());
            break;
        }

        case AI_TYPE_FLOAT:
        {
            if (param_type->get_kind() != mi::neuraylib::IType::TK_FLOAT)
            {
                AiMsgWarning("[mdl] parameter '%s' is not of type float. Ignoring ...", name);
                continue;
            }
            add_argument(
                mdl_sdk, arguments.get(), name, data->current_user_param_values[hash].FLT());
            break;
        }

        case AI_TYPE_INT:
        {
            if (param_type->get_kind() != mi::neuraylib::IType::TK_INT)
            {
                AiMsgWarning("[mdl] parameter '%s' is not of type int. Ignoring ...", name);
                continue;
            }
            add_argument(
                mdl_sdk, arguments.get(), name, data->current_user_param_values[hash].INT());
            break;
        }

        case AI_TYPE_RGB:
        {
            if (param_type->get_kind() != mi::neuraylib::IType::TK_COLOR)
            {
                AiMsgWarning("[mdl] parameter '%s' is not of type color. Ignoring ...", name);
                continue;
            }
            add_argument(
                mdl_sdk, arguments.get(), name, data->current_user_param_values[hash].RGB());
            break;
        }

        case AI_TYPE_VECTOR:
        {
            if (param_type->get_kind() != mi::neuraylib::IType::TK_VECTOR)
            {
                AiMsgWarning("[mdl] parameter '%s' is not of type vector. Ignoring ...", name);
                continue;
            }
            add_argument(
                mdl_sdk, arguments.get(), name, data->current_user_param_values[hash].VEC());
            break;
        }

        case AI_TYPE_VECTOR2:
        {
            if (param_type->get_kind() != mi::neuraylib::IType::TK_VECTOR)
            {
                AiMsgWarning("[mdl] parameter '%s' is not of type vector. Ignoring ...", name);
                continue;
            }
            add_argument(
                mdl_sdk, arguments.get(), name, data->current_user_param_values[hash].VEC2());
            break;
        }

        case AI_TYPE_STRING:
        {
            if (param_type->get_kind() == mi::neuraylib::IType::TK_TEXTURE)
            {
                // register gamma_mode helper parameter to be ignored by main parameter loop
                std::string tex_gamma_param_name(name);
                tex_gamma_param_name += "_gamma_mode";
                helper_params.insert(tex_gamma_param_name);

                AtString filename = data->current_user_param_values[hash].STR();
                if (filename.empty())
                    continue;

                // handle optional gamma_mode parameter
                std::string tex_gamma_str = "_default";
                float gamma = 0.0f; // default
                AtString tex_gamma_param_name_at(tex_gamma_param_name.c_str());
                const AtUserParamEntry* param =
                    AiNodeLookUpUserParameter(node, tex_gamma_param_name_at);
                if (param)
                {
                    AtString gamma_mode = AiNodeGetStr(node, tex_gamma_param_name_at);
                    if (strcmp(gamma_mode.c_str(), "gamma_srgb") == 0)
                    {
                        gamma = 2.2f;
                        tex_gamma_str = "_srgb";
                    }
                    else if (strcmp(gamma_mode.c_str(), "gamma_linear") == 0)
                    {
                        gamma = 1.0f;
                        tex_gamma_str = "_linear";
                    }
                }

                std::string texture_db_name = filename.c_str() + tex_gamma_str;
                mi::base::Handle<const mi::neuraylib::ITexture> tex_access;
                {
                    std::lock_guard<std::mutex> lock(mdl_sdk.get_loading_mutex());

                    tex_access = mi::base::Handle<const mi::neuraylib::ITexture>(
                        mdl_sdk.get_transaction().access<mi::neuraylib::ITexture>(
                        texture_db_name.c_str()));

                    // texture not known by database, yet? -> load it
                    if (!tex_access.is_valid_interface())
                    {
                        mi::base::Handle<mi::neuraylib::ITexture> tex(
                            mdl_sdk.get_transaction().create<mi::neuraylib::ITexture>("Texture"));
                        mi::base::Handle<mi::neuraylib::IImage> image(
                            mdl_sdk.get_transaction().create<mi::neuraylib::IImage>("Image"));

                        if (image->reset_file(filename.c_str()) != 0)
                        {
                            AiMsgWarning("[mdl] Texture file '%s' could not be found ...",
                                         filename.c_str());
                            continue;
                        }

                        std::string image_name = texture_db_name.c_str() + std::string("_image");
                        mdl_sdk.get_transaction().store(image.get(), image_name.c_str());

                        tex->set_image(image_name.c_str());
                        tex->set_gamma(gamma);
                        mdl_sdk.get_transaction().store(tex.get(), texture_db_name.c_str());
                    }
                }

                // for now, lets just assume this is 2D (TODO)
                mi::base::Handle<const mi::neuraylib::IType_texture> texture_type(
                    mdl_sdk.get_type_factory().create_texture(mi::neuraylib::IType_texture::TS_2D));

                mi::base::Handle<mi::neuraylib::IValue_texture> texture_value(
                    mdl_sdk.get_value_factory().create_texture(
                    texture_type.get(), texture_db_name.c_str()));

                mi::base::Handle<mi::neuraylib::IExpression_constant> expr(
                    mdl_sdk.get_expr_factory().create_constant(texture_value.get()));

                arguments->add_expression(name, expr.get());
            }
            else
            {
                if (param_type->get_kind() == mi::neuraylib::IType::TK_STRING)
                {
                    add_argument(mdl_sdk, arguments.get(), name, 
                                 data->current_user_param_values[hash].STR());
                }
                else
                {
                    AiMsgWarning("[mdl] parameter '%s' is not of type string. Ignoring ...", name);
                    continue;
                }
            }
            break;
        }
        default:
            AiMsgWarning("[mdl] parameter '%s' type %d is not supported, yet. Ignoring ...",
                         name, type);
            break;
        }
    }

    return arguments;
}

// Compile the MDL material associated with the given node and generate target code for the
// current render device.
bool compile_material(AtNode* node, Mdl_sdk_interface& mdl_sdk)
{
    MdlLocalNodeData* data = static_cast<MdlLocalNodeData*>(AiNodeGetLocalData(node));
    if (data->material_db_name.empty())
        return false;

    mi::base::Handle<mi::neuraylib::IMdl_execution_context> mdl_context(mdl_sdk.create_context());

    // get the material definition which should be valid after loading the module
    mi::base::Handle<const mi::neuraylib::IMaterial_definition> material_definition(
        mdl_sdk.get_transaction().access<mi::neuraylib::IMaterial_definition>(
            data->material_db_name.c_str()));
    if (!material_definition)
        return false;  // should not happen

    // create material arguments from node user data
    mi::base::Handle<mi::neuraylib::IExpression_list> arguments(
        create_material_argument_list(node, mdl_sdk, material_definition.get()));

    // only compile if it is required
    if (!data->user_parameters_changed)
        return true;

    // create a material instance (with default parameters, if non are specified)
    mi::Sint32 ret = 0;
    mi::base::Handle<mi::neuraylib::IMaterial_instance> material_instance(
        material_definition->create_material_instance(
            (arguments->get_size() == 0) ? nullptr : arguments.get(), &ret));
    if (ret != 0)
    {
        AiMsgWarning("[mdl] instantiating material '%s' failed", data->material_db_name.c_str());
        return false;
    }

    // compile the material instance (using instance compilation)
    mi::Uint32 flags = mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_material(
        material_instance->create_compiled_material(flags, mdl_context.get()));
    if (!mdl_sdk.log_messages(mdl_context.get()))
        return false;

    // nothing changed -> we can continue using the already existing programs
    mi::base::Uuid currentHash = compiled_material->get_hash();
    if (data->material_hash == currentHash)
    {
        data->user_parameters_changed = false;   // parameter changes didn't have any impact
        return true;
    }

    // generate target code
    // --------------------------------------------------------------------------------------------

    // determine, whether cutout_opacity is constant or if it needs to be evaluated
    // for opacity there is a special function to figure that out
    data->data_cpu.cutout_opacity_constant = 1.0f;
    bool need_cutout_opacity_evaluation = !compiled_material->get_cutout_opacity(
        &data->data_cpu.cutout_opacity_constant);

    // determine, whether thin_walled is constant or if it needs to be evaluated
    mi::base::Handle<mi::neuraylib::IExpression const> thin_walled(
        compiled_material->lookup_sub_expression("thin_walled"));

    bool need_thin_walled_evaluation = true;
    data->data_cpu.thin_walled_constant = false;
    if (thin_walled->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
    {
        mi::base::Handle<mi::neuraylib::IExpression_constant const> thin_walled_const(
            thin_walled->get_interface<mi::neuraylib::IExpression_constant const>());
        mi::base::Handle<mi::neuraylib::IValue_bool const> thin_walled_bool(
            thin_walled_const->get_value<mi::neuraylib::IValue_bool>());

        need_thin_walled_evaluation = false;
        data->data_cpu.thin_walled_constant = thin_walled_bool->get_value();
    }

    // back faces could be different for thin walled materials 
    bool need_backface_bsdf = false;
    bool need_backface_edf = false;
    bool need_backface_emission_intensity = false;
    if (need_thin_walled_evaluation || data->data_cpu.thin_walled_constant)
    {
        // first, backfaces dfs are only considered for thin_walled materials

        // second, we only need to generate new code if surface and backface are different
        need_backface_bsdf =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_SCATTERING) !=
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_SCATTERING);
        need_backface_edf =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_EDF_EMISSION) !=
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_EDF_EMISSION);
        need_backface_emission_intensity =
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_SURFACE_EMISSION_INTENSITY) !=
            compiled_material->get_slot_hash(mi::neuraylib::SLOT_BACKFACE_EMISSION_INTENSITY);

        // third, either the bsdf or the edf need to be non-default (black)
        mi::base::Handle<mi::neuraylib::IExpression const> scattering_expr(
            compiled_material->lookup_sub_expression("backface.scattering"));
        mi::base::Handle<mi::neuraylib::IExpression const> emission_expr(
            compiled_material->lookup_sub_expression("backface.emission.emission"));

        if (scattering_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT &&
            emission_expr->get_kind() == mi::neuraylib::IExpression::EK_CONSTANT)
        {
            mi::base::Handle<mi::neuraylib::IExpression_constant const> scattering_expr_constant(
                scattering_expr->get_interface<mi::neuraylib::IExpression_constant>());
            mi::base::Handle<mi::neuraylib::IValue const> scattering_value(
                scattering_expr_constant->get_value());

            mi::base::Handle<mi::neuraylib::IExpression_constant const> emission_expr_constant(
                emission_expr->get_interface<mi::neuraylib::IExpression_constant>());
            mi::base::Handle<mi::neuraylib::IValue const> emission_value(
                emission_expr_constant->get_value());

            if (scattering_value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF &&
                emission_value->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF)
            {
                need_backface_bsdf = false;
                need_backface_edf = false;
                need_backface_emission_intensity = false;
            }
        }
    }

    // create a link unit (to keep it simple we use one per material)
    mi::neuraylib::IMdl_backend *backend = &mdl_sdk.get_native_backend();
    mi::base::Handle<mi::neuraylib::ILink_unit> link_unit(
        backend->create_link_unit(&mdl_sdk.get_transaction(), mdl_context.get()));
    if (!mdl_sdk.log_messages(mdl_context.get()))
        return false;

    // select expressions to generate code for
    std::vector<mi::neuraylib::Target_function_description> descs;
    descs.push_back(mi::neuraylib::Target_function_description("surface.scattering"));
    descs.push_back(mi::neuraylib::Target_function_description("surface.emission.emission"));
    descs.push_back(mi::neuraylib::Target_function_description("surface.emission.intensity"));

    size_t backface_scattering_index = ~0;
    if (need_backface_bsdf)
    {
        backface_scattering_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("backface.scattering"));
    }

    size_t backface_edf_index = ~0;
    if (need_backface_edf)
    {
        backface_edf_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("backface.emission.emission"));
    }

    size_t backface_emission_intensity_index = ~0;
    if (need_backface_emission_intensity)
    {
        backface_emission_intensity_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("backface.emission.intensity"));
    }

    size_t cutout_opacity_desc_index = ~0;
    if (need_cutout_opacity_evaluation)
    {
        cutout_opacity_desc_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("geometry.cutout_opacity"));
    }

    size_t thin_walled_desc_index = ~0;
    if (need_thin_walled_evaluation)
    {
        thin_walled_desc_index = descs.size();
        descs.push_back(mi::neuraylib::Target_function_description("thin_walled"));
    }

    // add the material to the link unit
    link_unit->add_material(
        compiled_material.get(),
        descs.data(), descs.size(),
        mdl_context.get());
    if (!mdl_sdk.log_messages(mdl_context.get()))
        return false;

    // translate link unit
    mi::base::Handle<const mi::neuraylib::ITarget_code> code(
        backend->translate_link_unit(link_unit.get(), mdl_context.get()));
    if (!mdl_sdk.log_messages(mdl_context.get()))
        return false;

    // destroy the old target code if existing
    if (data->data_cpu.target_code)
        data->data_cpu.target_code->release();
    data->data_cpu.target_code = nullptr;

    // store the new one
    code->retain();
    data->data_cpu.target_code = code.get();

    // TODO: implement read-only data handling if required
    if (code->get_ro_data_segment_count() != 0)
        return false;

    // set CPU function references
    data->data_cpu.surface_bsdf_function_index = descs[0].function_index;
    data->data_cpu.surface_edf_function_index = descs[1].function_index;
    data->data_cpu.surface_emission_intensity_function_index = descs[2].function_index;

    data->data_cpu.backface_bsdf_function_index = need_backface_bsdf
        ? descs[backface_scattering_index].function_index
        : descs[0].function_index;

    data->data_cpu.backface_edf_function_index = need_backface_edf
        ? descs[backface_edf_index].function_index
        : descs[1].function_index;

    data->data_cpu.backface_emission_intensity_function_index = need_backface_emission_intensity
        ? descs[backface_emission_intensity_index].function_index
        : descs[2].function_index;

    data->data_cpu.cutout_opacity_function_index = need_cutout_opacity_evaluation 
        ? descs[cutout_opacity_desc_index].function_index 
        : ~0;

    data->data_cpu.thin_walled_function_index = need_thin_walled_evaluation 
        ? descs[thin_walled_desc_index].function_index 
        : ~0;

    AiMsgInfo("[mdl] compiled material '%s'", data->material_db_name.c_str());

    data->material_hash = currentHash;      // store the hash to identify changes
    data->user_parameters_changed = false;  // parameter changes have been applied
    data->require_code_update = true;       // recreate OptiX programs from the new PTX code
    data->require_texture_update = true;    // reload textures (TODO, more granularly)
    return true;
}

node_parameters
{
    // define default values
    AiParameterStr(CONST_STRINGS::qualified_name, "::ai_mdl::not_available");

    AiMetaDataSetStr(nentry, CONST_STRINGS::qualified_name,     "description", "Fully qualified MDL material name of the form: ::package::sub_package::module::material_name");
    AiMetaDataSetBool(nentry, CONST_STRINGS::qualified_name,    "linkable", false);
    AiMetaDataSetStr(nentry, CONST_STRINGS::qualified_name,     "maya.name", "qualified_name");
    AiMetaDataSetStr(nentry, CONST_STRINGS::qualified_name,     "maya.shortname", "qname");
    AiMetaDataSetStr(nentry, CONST_STRINGS::qualified_name,     "max.label", "Qualified Name");

    AiMetaDataSetStr(nentry, NULL, "description",           "Arnold MDL Material Shader");
    AiMetaDataSetInt(nentry, NULL, "maya.id",               0x0010E63F); // temporary test id!
    AiMetaDataSetStr(nentry, NULL, "maya.classification",   "shader/surface");
    AiMetaDataSetStr(nentry, NULL, "maya.output_name",      "outColor");
    AiMetaDataSetStr(nentry, NULL, "maya.output_shortname", "out");
    AiMetaDataSetStr(nentry, NULL, "maya.name",             "aiMDL");
    AiMetaDataSetStr(nentry, NULL, "max.category",          "Surface");
}

node_initialize
{
    // allocate node data
    MdlLocalNodeData* data = new MdlLocalNodeData();
    data->current_qualified_name = CONST_STRINGS::empty;
    AiNodeSetLocalData(node, data);

    // get the sdk
    Mdl_sdk_interface& mdl_sdk = *static_cast<Mdl_sdk_interface*>(AiNodeGetPluginData(node));

    // get the selected material
    auto selected_material = AiNodeGetStr(node, CONST_STRINGS::qualified_name);

    // load the specified material and compile it
    if (!load_material(node, mdl_sdk) || !compile_material(node, mdl_sdk))
    {
        AiMsgWarning("[mdl] compiling the shader for material '%s' failed.\n",
                   selected_material.c_str());

        data->material_db_name = mdl_sdk.get_default_material_db_name();
        compile_material(node, mdl_sdk);
        return;
    }

    // store the selected material to detect changes later
    data->current_qualified_name = selected_material;
}

node_update
{
    MdlLocalNodeData *data = static_cast<MdlLocalNodeData*>(AiNodeGetLocalData(node));
    Mdl_sdk_interface& mdl_sdk = *static_cast<Mdl_sdk_interface*>(AiNodeGetPluginData(node));

    // check if the selected material changed
    auto selected_material = AiNodeGetStr(node, CONST_STRINGS::qualified_name);

    // load the specified material of not loaded already
    if (data->current_qualified_name != selected_material && !load_material(node, mdl_sdk))
    {
        AiMsgWarning("[mdl] loading or compiling the shader for material '%s' failed.\n",
                   selected_material.c_str());

        data->material_db_name = mdl_sdk.get_default_material_db_name();
        compile_material(node, mdl_sdk);
        return;
    }

    // handle parameter updates
    if (!compile_material(node, mdl_sdk))
    {
        AiMsgWarning("[mdl] compiling the shader for material '%s' failed.\n",
                   selected_material.c_str());

        data->material_db_name = mdl_sdk.get_default_material_db_name();
        compile_material(node, mdl_sdk);
        return;
    }

    // store the selected material to detect changes later
    data->current_qualified_name = selected_material;
}

node_finish
{
    MdlLocalNodeData *data = static_cast<MdlLocalNodeData*>(AiNodeGetLocalData(node));
    if (data == nullptr)
        return;

    if (data->data_cpu.target_code)
        data->data_cpu.target_code->release();
    data->data_cpu.target_code = nullptr;

    delete(data);
}

AtBSDF* MdlBSDFCreate(
    const AtShaderGlobals* sg, 
    const MdlShaderNodeDataCPU* shader_data)
{
    AtBSDF* bsdf = AiBSDF(sg, AI_RGB_WHITE, MdlBSDFData::methods, sizeof(MdlBSDFData));

    MdlBSDFData* data = (MdlBSDFData*) AiBSDFGetData(bsdf);
    data->shader = *shader_data;

    // currently we sample and evaluate all lobes at once
    // and we use AI_RAY_SPECULAR_REFLECT, hoping that we don't cut any light paths.
    // when using diffuse instead, MDL surfaces have not been visible in reflections on
    // standard surfaces with metallic = 1
    // update: try to split into diffuse and glossy
    //         adding further lobes for specular reflect and transmit fails because of the sampling
    //         singular lobes are sampled only once and when our sample not the right one, we need
    //         to reject. In that case it's not sampled again ever...

    data->lobe_info[0].ray_type = AI_RAY_DIFFUSE_REFLECT;
    data->lobe_info[0].flags = 0;
    data->lobe_info[0].label = AtString();

    // TODO: data->lobe_info[1].ray_type = AI_RAY_SPECULAR_REFLECT;
    // TODO: data->lobe_info[1].flags = 0;
    // TODO: data->lobe_info[1].label = AtString();
    return bsdf;
}

shader_evaluate
{
    // when the host compiler sees this code, the local data contains the entire node state
    MdlLocalNodeData *data = static_cast<MdlLocalNodeData*>(AiNodeGetLocalData(node));
    if (data->data_cpu.target_code == nullptr)
        return;

    // handle opacity
    // as documented in https://docs.arnoldrenderer.com/display/A5ARP/Closures
    AtRGB opacity = AtRGB(MdlOpacityCreate(sg, &data->data_cpu));
    opacity = AiShaderGlobalsStochasticOpacity(sg, opacity);
    if (opacity != AI_RGB_WHITE)
    {
        sg->out.CLOSURE() = AiClosureTransparent(sg, AI_RGB_WHITE - opacity);
        // early out for nearly fully transparent objects
        if (AiAll(opacity < AI_OPACITY_EPSILON))
            return;
    }

    // early out for shadow rays
    if (sg->Rt & AI_RAY_SHADOW)
        return;

    // create shader closures
    AtClosureList closures;
    closures.add(MdlBSDFCreate(sg, &data->data_cpu));
    closures.add(MdlEDFCreate(sg, &data->data_cpu));

    // write closures
    if (opacity != AI_RGB_WHITE)
    {
        closures *= opacity;
        sg->out.CLOSURE().add(closures);
    }
    else
    {
        sg->out.CLOSURE() = closures;
    }
}
