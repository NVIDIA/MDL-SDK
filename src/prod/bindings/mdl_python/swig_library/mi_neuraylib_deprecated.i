/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

// These functions should map to the latest version and issue a warning.
// Deprecated functions will be dropped with one of the next major releases.

// Strategy for (more complex) changes:
// - create a deprecated function that maps to the new signature just like in the native API. Call it `_deprecated_FOO(...)`
// - add type checks to all parameters in this deprecated function. Keep in mind that some parameters are allowed to be None.
// - rename the current function to start with an underscore. Call it `_FOO(...)`
// - create a new function that tries to call the current function and in case of an TypeError, calls the deprecated one. Call it `FOO(...)`
//   here try to handle as many parameters directly at the beginning of the signature

%pythoncode %{
    def _isIInterfaceOfOrNone(value: "IInterface", type) -> bool:
        """Improves readability by combining the two checks."""
        if value == None:  # None could be a default and is okay
            return True
        if isinstance(value, type):  # Type matches exactly
            return True
        if getattr(value, "get_interface", None) == None:  # type has no 'get_interface' member, can not be a in IInterface
            return False
        return value.get_interface(type).is_valid_interface()  # Type is compatible (usually derived from)
%}

// MI_NEURAYLIB_DEPRECATED_15_1
// ----------------------------------------------------------------------------
%ignore mi::neuraylib::IBaker::bake_texture;
%rename(_bake_texture) mi::neuraylib::IBaker::bake_texture(ICanvas*,Float32,Float32,Float32,Float32,Float32,Uint32) const;
%extend SmartPtr<mi::neuraylib::IBaker> {
    %pythoncode %{

        def _deprecated_bake_texture(self, canvas: "ICanvas", min_u: float = 0.0, max_u: float = 1.0, min_v: float = 0.0, max_v: float = 1.0, animation_time: float = 0.0, samples: int = 1) -> int:
            if _isIInterfaceOfOrNone(canvas, ICanvas) and \
                isinstance(min_u, float) and \
                isinstance(max_u, float) and \
                isinstance(min_v, float) and \
                isinstance(max_v, float) and \
                isinstance(animation_time, float) and \
                isinstance(samples, int):
                warnings.warn("Breaking change in `IBaker.bake_texture`. Add all parameters explicitly.", DeprecationWarning)
                return self._bake_texture(canvas, min_u, max_u, min_v, max_v, animation_time, samples)
            raise TypeError("Wrong argument types passed to `IBaker.bake_texture`.")

        @_post_swig_move_to_end_of_class
        @_copy_docs_from(_bake_texture)
        def bake_texture(self, canvas: "ICanvas", *args, **kwargs) -> int:
            try:
                return self._bake_texture(canvas, *args, **kwargs)
            except TypeError:
                pass
            try:  # old default value for samples
                if len(args) == 1 and isinstance(args[0], int) and len(kwargs) == 0:
                    return self._deprecated_bake_texture(canvas, samples=args[0])
                if len(kwargs) == 1 and "samples" in kwargs.keys() and len(args) == 0:
                    return self._deprecated_bake_texture(canvas, samples=kwargs["samples"])
            except TypeError:
                pass
            return self._deprecated_bake_texture(canvas, *args, **kwargs)
    %}
}

// MI_NEURAYLIB_DEPRECATED_15_0
// ----------------------------------------------------------------------------

// ITransaction
%extend SmartPtr<mi::neuraylib::ITransaction> {

    %pythoncode %{

        @_post_swig_move_to_end_of_class
        @_copy_docs_from(_create)
        def create(self, type_name: str, *args, **kwargs):
            if len(args) > 0 or len(kwargs) > 0:
                warnings.warn("Breaking change in `ITransaction.create`. Optional arguments can not be passed anymore.", DeprecationWarning)
            return self._create(type_name)

        @_post_swig_move_to_end_of_class
        @_copy_docs_from(_create_as)
        def create_as(self, type, type_name: str, *args, **kwargs):
            if len(args) > 0 or len(kwargs) > 0:
                warnings.warn("Breaking change in `ITransaction.create_as`, Optional arguments can not be passed anymore.", DeprecationWarning)
            return self._create_as(type, type_name)
    %}
}

// IMdl_module_builder
%rename(_add_function) mi::neuraylib::IMdl_module_builder::add_function;
%rename(_add_struct_type) mi::neuraylib::IMdl_module_builder::add_struct_type;
%rename(_add_variant) mi::neuraylib::IMdl_module_builder::add_variant;
%extend SmartPtr<mi::neuraylib::IMdl_module_builder> {
    %pythoncode %{

        def _deprecated_add_function(self, name: str, body: "IExpression", parameters: "IType_list", defaults: "IExpression_list", parameter_annotations: "IAnnotation_list", annotations: "IAnnotation_block", return_annotations: "IAnnotation_block", is_exported: bool, frequency_qualifier: "IType.Modifier", context: "IMdl_execution_context") -> int:
            if isinstance(name, str) and \
                _isIInterfaceOfOrNone(body, IExpression) and \
                _isIInterfaceOfOrNone(parameters, IType_list) and \
                _isIInterfaceOfOrNone(defaults, IExpression_list) and \
                _isIInterfaceOfOrNone(parameter_annotations, IAnnotation_list) and \
                _isIInterfaceOfOrNone(annotations, IAnnotation_block) and \
                _isIInterfaceOfOrNone(return_annotations, IAnnotation_block) and \
                isinstance(is_exported, bool) and \
                isinstance(frequency_qualifier, IType.Modifier) and \
                _isIInterfaceOfOrNone(context, IMdl_execution_context):
                warnings.warn("Breaking change in `IMdl_module_builder.add_function`. Add the two new parameters: `temporaries` and `is_declarative`.", DeprecationWarning)
                return self._add_function(name, body, None, parameters, defaults, parameter_annotations, annotations, return_annotations, is_exported, False, frequency_qualifier, context)
            raise TypeError("Wrong argument types passed to `IMdl_module_builder.add_function`.")

        @_post_swig_move_to_end_of_class
        @_copy_docs_from(_add_function)
        def add_function(self, name: str, body: "IExpression", *args, **kwargs) -> int:
            try:
                return self._add_function(name, body, *args, **kwargs)
            except TypeError as e:
                return self._deprecated_add_function(name, body, *args, **kwargs)


        def _deprecated_add_struct_type(self, name: str, fields: "IType_list", field_defaults: "IExpression_list", field_annotations: "IAnnotation_list", annotations: "IAnnotation_block", is_exported: bool, context: "IMdl_execution_context") -> int:
            if isinstance(name, str) and \
                _isIInterfaceOfOrNone(fields, IType_list) and \
                _isIInterfaceOfOrNone(field_defaults, IExpression_list) and \
                _isIInterfaceOfOrNone(field_annotations, IAnnotation_list) and \
                _isIInterfaceOfOrNone(annotations, IAnnotation_block) and \
                isinstance(is_exported, bool) and \
                _isIInterfaceOfOrNone(context, IMdl_execution_context):
                warnings.warn("Breaking change in `IMdl_module_builder.add_struct_type`. Add the two new parameters: `is_declarative` and `struct_category`.", DeprecationWarning)
                return self._add_struct_type(name, fields, field_defaults, field_annotations, annotations, is_exported, False, None, context)
            raise TypeError("Wrong argument types passed to `IMdl_module_builder.add_struct_type`.")

        @_post_swig_move_to_end_of_class
        @_copy_docs_from(_add_struct_type)
        def add_struct_type(self, name: str, fields: "IType_list", field_defaults: "IExpression_list", field_annotations: "IAnnotation_list", annotations: "IAnnotation_block", is_exported: bool, *args, **kwargs) -> int:
            try:
                return self._add_struct_type(name, fields, field_defaults, field_annotations, annotations, is_exported, *args, **kwargs)
            except TypeError as e:
                return self._deprecated_add_struct_type(name, fields, field_defaults, field_annotations, annotations, is_exported, *args, **kwargs)



        def _deprecated_add_variant(self, name: str, prototype_name: str, defaults: "IExpression_list", annotations: "IAnnotation_block", return_annotations: "IAnnotation_block", is_exported: bool, context: "IMdl_execution_context") -> int:
            if isinstance(name, str) and \
                isinstance(prototype_name, str) and \
                _isIInterfaceOfOrNone(defaults, IExpression_list) and \
                _isIInterfaceOfOrNone(annotations, IAnnotation_block) and \
                _isIInterfaceOfOrNone(return_annotations, IAnnotation_block) and \
                isinstance(is_exported, bool) and \
                _isIInterfaceOfOrNone(context, IMdl_execution_context):
                warnings.warn("Breaking change in `IMdl_module_builder.add_variant`. Add the new parameter: `is_declarative`.", DeprecationWarning)
                return self._add_variant(name, prototype_name, defaults, annotations, return_annotations, is_exported, False, context)
            raise TypeError("Wrong argument types passed to `IMdl_module_builder.add_variant`.")

        @_post_swig_move_to_end_of_class
        @_copy_docs_from(_add_variant)
        def add_variant(self, name: str, prototype_name: str, defaults: "IExpression_list", annotations: "IAnnotation_block", return_annotations: "IAnnotation_block", is_exported: bool, *args, **kwargs) -> int:
            try:
                return self._add_variant(name, prototype_name, defaults, annotations, return_annotations, is_exported, *args, **kwargs)
            except TypeError as e:
                return self._deprecated_add_variant(name, prototype_name, defaults, annotations, return_annotations, is_exported, *args, **kwargs)
    %}
}

// IMdl_impexp_api
%rename(_export_canvas) mi::neuraylib::IMdl_impexp_api::export_canvas;
%extend SmartPtr<mi::neuraylib::IMdl_impexp_api> {

    mi::Sint32 _deprecated_export_canvas(
        const char* filename,
        const mi::neuraylib::ICanvas* canvas,
        mi::Uint32 quality = 100,
        bool force_default_gamma = false)
    {
        mi::base::Handle<mi::neuraylib::IFactory> factory(g_neuray->get_api_component<mi::neuraylib::IFactory>());

        mi::base::Handle<mi::IUint32> option_jpg_quality(factory->create<mi::IUint32>());
        option_jpg_quality->set_value(quality);
        mi::base::Handle<mi::IBoolean> option_force_default_gamma(factory->create<mi::IBoolean>());
        option_force_default_gamma->set_value(force_default_gamma);

        mi::base::Handle export_options(factory->create<mi::IMap>("Map<Interface>"));
        export_options->insert("jpg:quality", option_jpg_quality.get());
        export_options->insert("force_default_gamma", option_force_default_gamma.get());

        return $self->get()->export_canvas(filename, canvas, export_options.get());
    }

    %pythoncode{
        @_post_swig_move_to_end_of_class
        @_copy_docs_from(_export_canvas)
        def export_canvas(self, filename: str, canvas: "ICanvas", *args, **kwargs) -> bool:
            try:
                return self._export_canvas(filename, canvas, *args, **kwargs)
            except TypeError as e:
                warnings.warn("Breaking change in `IMdl_impexp_api.export_canvas`. Use the `export_options` parameter to pass export options.", DeprecationWarning)
                return self._deprecated_export_canvas(filename, canvas, *args, **kwargs)
    }
}


// *_with_ret functions that should not be used anymore since 2023.1
// ----------------------------------------------------------------------------

%extend SmartPtr<mi::neuraylib::IExpression_factory> {
    %pythoncode {
        def create_cast_with_ret(self, src_expr, target_type, cast_db_name, force_cast, direct_call):
            warnings.warn("Use `create_cast` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._create_cast(src_expr, target_type, cast_db_name, force_cast, direct_call)

        def create_direct_call_with_ret(self, name, arguments):
            warnings.warn("Use `create_direct_call` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._create_direct_call(name, arguments)
    }
}

%extend SmartPtr<mi::neuraylib::IFunction_definition> {
    %pythoncode {
        def create_function_call_with_ret(self, arguments):
            warnings.warn("Use `create_function_call` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._create_function_call(arguments)
    }
}

%extend SmartPtr<mi::neuraylib::IMdl_distiller_api> {
    %pythoncode {
        def distill_material_with_ret(self, material, target, distiller_options = None) :
            warnings.warn("Use `distill_material` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._distill_material(material, target, distiller_options)
    }
}

%extend SmartPtr<mi::neuraylib::ICompiled_material> {
    %pythoncode {
        def get_connected_function_db_name_with_ret(self, material_instance_name, parameter_index):
            warnings.warn("Use `get_connected_function_db_name` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._get_connected_function_db_name(material_instance_name, parameter_index)
    }
}

%extend SmartPtr<mi::neuraylib::IType_enumeration> {
    %pythoncode {
        def get_value_code_with_ret(self, index):
            warnings.warn("Use `get_value_code` instead using the `errors: ReturnCode` parameter.", DeprecationWarning)
            return self._get_value_code(index)
    }
}


// inherited deprecated functions
// ----------------------------------------------------------------------------
// just remove them, they are not in the public interfaces

%ignore mi::neuraylib::IImage_api::deprecated_create_buffer_from_canvas;
%ignore mi::neuraylib::IMdl_configuration::deprecated_set_logger;
%ignore mi::neuraylib::IMdl_configuration::deprecated_get_logger;
%ignore mi::neuraylib::IMdl_impexp_api::deprecated_export_canvas;
%ignore mi::neuraylib::IMdl_impexp_api::deprecated_uvtile_string_to_marker;
%ignore mi::neuraylib::IMdl_impexp_api::deprecated_frame_string_to_marker;
%ignore mi::neuraylib::IModule::deprecated_get_resource_type;
%ignore mi::neuraylib::IModule::deprecated_get_resource_mdl_file_path;
%ignore mi::neuraylib::IModule::deprecated_get_resource_name;


// other changes
// ----------------------------------------------------------------------------
%extend SmartPtr<mi::neuraylib::IFunction_definition> {
    %pythoncode {
        def get_mdl_mangled_name(self) -> str:
            warnings.warn("Use `get_mangled_name` instead.", DeprecationWarning)
            return self.get_mangled_name()
    }
}
