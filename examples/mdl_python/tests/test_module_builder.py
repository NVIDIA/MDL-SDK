import unittest
import os
import tempfile
import uuid

try:  # pragma: no cover
    # testing from within a package or CI
    from .setup import SDK, BindingModule
    from .unittest_base import UnittestBase
    pymdlsdk = BindingModule
except ImportError:  # pragma: no cover
    # local testing
    from setup import SDK
    from unittest_base import UnittestBase
    import pymdlsdk


class MainModuleBuilder(UnittestBase):
    sdk: SDK = None
    tf: pymdlsdk.IType_factory = None
    vf: pymdlsdk.IValue_factory = None
    ef: pymdlsdk.IExpression_factory = None
    imp_exp: pymdlsdk.IMdl_impexp_api = None
    loading_base_return_code: int = -1

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=True, loadImagePlugins=True)
        self.tf = self.sdk.mdlFactory.create_type_factory(self.sdk.transaction)
        self.vf = self.sdk.mdlFactory.create_value_factory(self.sdk.transaction)
        self.ef = self.sdk.mdlFactory.create_expression_factory(self.sdk.transaction)
        self.imp_exp: pymdlsdk.IMdl_impexp_api = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_impexp_api)
        self.loading_base_return_code = self.imp_exp.load_module(self.sdk.transaction, "::base")  # load something to get builtins

    @classmethod
    def tearDownClass(self):
        self.imp_exp = None
        self.vf = None
        self.tf = None
        self.ef = None
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def test_setupIsDone(self):
        self.assertIsNotNone(self.sdk)
        self.assertIsValidInterface(self.sdk.neuray)
        self.assertIsValidInterface(self.sdk.transaction)
        self.assertIsValidInterface(self.sdk.mdlFactory)
        self.assertIsValidInterface(self.tf)
        self.assertIsValidInterface(self.vf)
        self.assertIsValidInterface(self.ef)
        self.assertEqual(self.sdk.neuray.get_status(), pymdlsdk.INeuray.Status.STARTED)

    def create_module_builder(self, context: pymdlsdk.IMdl_execution_context):
        module_name: str = f"mdl::new_module_{str(uuid.uuid4()).replace('-', '_')}"
        module_builder: pymdlsdk.IMdl_module_builder = self.sdk.mdlFactory.create_module_builder(
            self.sdk.transaction,
            module_name,
            pymdlsdk.Mdl_version.MDL_VERSION_1_8,
            pymdlsdk.Mdl_version.MDL_VERSION_LATEST,
            context)
        self.assertIsValidInterface(module_builder)
        return module_builder

    def test_enums(self):
        context: pymdlsdk.IExpression_constant = self.sdk.mdlFactory.create_execution_context()
        module_builder: pymdlsdk.IMdl_module_builder = self.create_module_builder(context)
        enumerators: pymdlsdk.IExpression_list = self.ef.create_expression_list()
        enumerators.add_expression("No",
            self.ef.create_constant(self.ef.get_value_factory().create_int(0)))
        enumerators.add_expression("Yes",
            self.ef.create_constant(self.ef.get_value_factory().create_int(1)))
        res: int = module_builder.add_enum_type("NewBool", enumerators, None, None, True, context)
        self.assertContextNoErrors(context)
        self.assertZero(res)

    def test_const(self):
        context: pymdlsdk.IExpression_constant = self.sdk.mdlFactory.create_execution_context()
        module_builder: pymdlsdk.IMdl_module_builder = self.create_module_builder(context)
       
        res: int = module_builder.add_constant("ONE", 
            self.ef.create_constant(self.ef.get_value_factory().create_int(1)), None, True,
            context)
        self.assertContextNoErrors(context)
        self.assertZero(res)

        res = module_builder.remove_entity("ONE", 0, context)
        self.assertContextNoErrors(context)
        self.assertZero(res)

    def test_module_basics(self):
        context: pymdlsdk.IExpression_constant = self.sdk.mdlFactory.create_execution_context()
        module_builder: pymdlsdk.IMdl_module_builder = self.create_module_builder(context)
       
        anno_block: pymdlsdk.IAnnotation_block = self.ef.create_annotation_block()
        anno: pymdlsdk.IAnnotation = self.ef.create_annotation("::unused()", 
            self.ef.create_expression_list())
        anno_block.add_annotation(anno)
        res: int = module_builder.set_module_annotations(anno_block, context)
        self.assertContextNoErrors(context)
        self.assertZero(res)

        # Rudimentary check that analyze_uniform() is callable.
        uniforms: pymdlsdk.IArray = module_builder.analyze_uniform(
            self.ef.create_constant(self.ef.get_value_factory().create_int(1)), True, context)
        self.assertIsValidInterface(uniforms)
        self.assertZero(uniforms.get_length())

    def test_module_builder_struct_categories(self):
        module_name = "mdl::new_module_42"
        context: pymdlsdk.IExpression_constant = self.sdk.mdlFactory.create_execution_context()
        module_builder: pymdlsdk.IMdl_module_builder = self.sdk.mdlFactory.create_module_builder(
            self.sdk.transaction,
            module_name,
            pymdlsdk.Mdl_version.MDL_VERSION_1_8,
            pymdlsdk.Mdl_version.MDL_VERSION_LATEST,
            context)
        self.assertIsValidInterface(module_builder)
        module_builder.add_struct_category("custom42_category", None, True, context)
        self.assertContextNoErrors(context)
        mdl_module: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, module_name)
        struct_cat_list: pymdlsdk.IStruct_category_list = mdl_module.get_struct_categories()
        self.assertEqual(struct_cat_list.get_size(), 1)
        custom42_category: pymdlsdk.IStruct_category = struct_cat_list.get_struct_category(0)
        self.assertEqual(custom42_category.get_predefined_id(), pymdlsdk.IStruct_category.Predefined_id.CID_USER)
        self.assertEqual(custom42_category.get_symbol(), '::new_module_42::custom42_category')
        custom42_fields: pymdlsdk.IType_list = self.tf.create_type_list()
        custom42_fields.add_type("example_aov", self.tf.create_color())
        custom42_fields.add_type("example_aov2", self.tf.create_float())
        # current, correct signature
        module_builder.add_struct_type("custom42", custom42_fields, None, None, None, True, True, custom42_category, context)
        module_builder.add_struct_type("custom43", custom42_fields, None, None, None, True, context=context, is_declarative=True, struct_category=custom42_category)
        # deprecated signature(s)
        self.assertDeprecationWarning(lambda: module_builder.add_struct_type("custom44", custom42_fields, None, None, None, True, context))
        self.assertDeprecationWarning(lambda: module_builder.add_struct_type("custom45", custom42_fields, None, None, None, True, None))
        # invalid signature
        self.assertException(TypeError, lambda: module_builder.add_struct_type("custom46", custom42_fields, None, None, None, True))
        self.assertException(TypeError, lambda: module_builder.add_struct_type("custom47", custom42_fields, None, None, None, True, context=context, is_declarative=False)) # missing struct_category
        self.assertException(TypeError, lambda: module_builder.add_struct_type("custom48", custom42_fields, None, None, None, True, custom42_category))  # weird mix
        custom42_category = None
        mdl_module = self.sdk.transaction.access_as(pymdlsdk.IModule, module_name)
        module_types: pymdlsdk.IType_list = mdl_module.get_types()
        self.assertEqual(module_types.get_size(), 4)
        type0: pymdlsdk.IType = module_types.get_type(0)
        custom42_type: pymdlsdk.IType_structure = type0.get_interface(pymdlsdk.IType_structure)
        self.assertEqual(custom42_type.get_symbol(), '::new_module_42::custom42')
        custom42_type_cat: pymdlsdk.IStruct_category = custom42_type.get_struct_category()
        self.assertIsValidInterface(custom42_type_cat)
        custom42_category: pymdlsdk.IStruct_category = struct_cat_list.get_struct_category(0) # because the module changed
        self.assertIsValidInterface(custom42_category)
        self.assertEqual(self.tf.compare(custom42_category, custom42_type_cat), 0)

        module_builder.clear_module(context)
        module_builder = None


    def test_udim_export(self):
        example_module_name: str = "::nvidia::sdk_examples::tutorials"
        self.assertNotEqual(self.load_module(example_module_name), "")
        func_def: pymdlsdk.IFunction_definition = \
            self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, f"mdl{example_module_name}::example_tiled(texture_2d)")
        self.assertIsValidInterface(func_def)
        ret: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        func_call: pymdlsdk.IFunction_call = func_def.create_function_call(None, errors=ret)
        self.assertEqual(ret.value, 0)
        self.assertIsValidInterface(func_call)
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        new_module_name: str = f"mdl::new_module_{str(uuid.uuid4()).replace('-', '_')}"
        moduleBuilder: pymdlsdk.IMdl_module_builder = \
            self.sdk.mdlFactory.create_module_builder(
                self.sdk.transaction,
                new_module_name,
                pymdlsdk.Mdl_version.MDL_VERSION_1_6, pymdlsdk.Mdl_version.MDL_VERSION_LATEST,
                context)
        ef: pymdlsdk.IExpression_factory = self.sdk.mdlFactory.create_expression_factory(self.sdk.transaction)
        e: pymdlsdk.IAnnotation_block = ef.create_annotation_block()

        moduleBuilder.clear_module(context)
        self.assertZero(moduleBuilder.add_annotation("author", None, None, None, None, False, None))

        self.assertZero(moduleBuilder.add_variant( "NewVariant0", func_call.get_function_definition(), func_call.get_arguments(), e, e, True, False, context))
        self.assertZero(moduleBuilder.add_variant( "NewVariant1", func_call.get_function_definition(), func_call.get_arguments(), e, e, True, False, None))
        self.assertZero(self.assertDeprecationWarning(lambda: moduleBuilder.add_variant( "NewVariant2", func_call.get_function_definition(), func_call.get_arguments(), e, e, True, context)))  # old signature
        self.assertZero(self.assertDeprecationWarning(lambda: moduleBuilder.add_variant( "NewVariant3", func_call.get_function_definition(), func_call.get_arguments(), e, e, True, None)))   # old signature
        self.assertZero(self.assertDeprecationWarning(lambda: moduleBuilder.add_variant( "NewVariant4", func_call.get_function_definition(), func_call.get_arguments(), e, e, True, context=context)))  # old signature
        self.assertException(TypeError, lambda: moduleBuilder.add_variant( "NewVariant5", func_call.get_function_definition(), func_call.get_arguments(), e, e, True))  # missing parameters
        self.assertException(TypeError, lambda: moduleBuilder.add_variant( "NewVariant6", func_call.get_function_definition(), func_call.get_arguments(), e, e, True, False))  # missing parameters

        body_constant: pymdlsdk.IValue_color = self.vf.create_color()
        body_expression: pymdlsdk.IExpression_constant = self.ef.create_constant(body_constant)
        # new and correct signature
        self.assertZero(moduleBuilder.add_function("NewFunction0", body_expression, None, None, None, None, e, e, True, True, pymdlsdk.IType.Modifier.MK_UNIFORM, context))
        self.assertZero(moduleBuilder.add_function("NewFunction1", body_expression, None, None, None, None, e, e, True, False, pymdlsdk.IType.Modifier.MK_UNIFORM, None))
        # old signature
        self.assertZero(self.assertDeprecationWarning(lambda: moduleBuilder.add_function( "NewFunction2", body_expression, None, None, None, e, e, True, pymdlsdk.IType.Modifier.MK_UNIFORM, context)))  # old signature
        self.assertZero(self.assertDeprecationWarning(lambda: moduleBuilder.add_function( "NewFunction3", body_expression, None, None, None, e, e, True, pymdlsdk.IType.Modifier.MK_UNIFORM, None)))  # old signature
        self.assertZero(self.assertDeprecationWarning(lambda: moduleBuilder.add_function( "NewFunction4", body_expression, None, None, None, e, e, True, pymdlsdk.IType.Modifier.MK_UNIFORM, context=context)))  # old signature
        self.assertZero(self.assertDeprecationWarning(lambda: moduleBuilder.add_function( "NewFunction5", body_expression, None, None, None, e, e, True, frequency_qualifier=pymdlsdk.IType.Modifier.MK_UNIFORM, context=context)))  # old signature
        self.assertZero(self.assertDeprecationWarning(lambda: moduleBuilder.add_function( "NewFunction6", body_expression, None, None, None, e, e, is_exported=True, frequency_qualifier=pymdlsdk.IType.Modifier.MK_UNIFORM, context=context)))  # old signature
        # invalid signature
        self.assertException(TypeError, lambda: moduleBuilder.add_function("NewFunction7", body_expression, None, None, None, None, e, e, True, False))  # missing parameters
        self.assertException(TypeError, lambda: moduleBuilder.add_function("NewFunction8", body_expression, None, None, None, None, e, e, True, False, pymdlsdk.IType.Modifier.MK_UNIFORM))  # missing parameters
        self.assertException(TypeError, lambda: moduleBuilder.add_function("NewFunction9", body_expression, None, None, None, None, e, e, True, False, context=context))  # missing parameters

        with tempfile.TemporaryDirectory() as temp_dir:
            impExpApi: pymdlsdk.IMdl_impexp_api = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_impexp_api)
            context.set_option("bundle_resources", True)
            result = impExpApi.export_module(self.sdk.transaction, new_module_name, f'{temp_dir}/exported_module.mdl', context)
            self.assertEqual(result, 0, "failed to export module")
            self.assertMd5Hash(f'{temp_dir}/exported_module_tiled_resource.0.1001.png', 'cfa1005eb726fde90b4bd9abcc5b6ea2')
            self.assertMd5Hash(f'{temp_dir}/exported_module_tiled_resource.0.1002.png', '838956dba88fa6b7df3f89fe6dc2ed8d')
            self.assertMd5Hash(f'{temp_dir}/exported_module_tiled_resource.0.1011.png', 'f3642713deb0af9f41b6fe76f2911805')
            self.assertMd5Hash(f'{temp_dir}/exported_module_tiled_resource.2.1012.png', '589931a48a8bd2157ee36879d6315ac8')

    def test_mdl_factory_analyse_uniform(self):
        ef: pymdlsdk.IExpression_factory = self.sdk.mdlFactory.create_expression_factory(self.sdk.transaction)

        # build a small call graph starting with state normal
        self.assertNotEqual(self.load_module("::state"), "")
        state_normal_def: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, "mdl::state::normal()")
        state_normal_call: pymdlsdk.IFunction_call = state_normal_def.create_function_call(None)
        state_normal_call_db_name: str = "state_normal_call"
        self.assertZero(self.sdk.transaction.store(state_normal_call, state_normal_call_db_name))
        state_normal_call = None  # not accessible after store, needs access(..)

        # access the x field
        access_field_def: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, "mdl::float3.x(float3)")
        state_normal_call_expr: pymdlsdk.IExpression_call = ef.create_call(state_normal_call_db_name)
        args: pymdlsdk.IExpression_list = ef.create_expression_list()
        param_name: str = access_field_def.get_parameter_name(0)
        args.add_expression(param_name, state_normal_call_expr)
        access_field_call: pymdlsdk.IFunction_call = access_field_def.create_function_call(args)
        access_field_call_db_name: str = "access_field_call"
        self.assertZero(self.sdk.transaction.store(access_field_call, access_field_call_db_name))
        access_field_call = None  # not accessible after store, needs access(..)

        # convert to bool
        access_field_call_expr: pymdlsdk.IExpression_call = ef.create_call(access_field_call_db_name)
        bool_from_float_def: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, "mdl::bool(float)")
        param_name:str = bool_from_float_def.get_parameter_name(0)
        args: pymdlsdk.IExpression_list = ef.create_expression_list()
        args.add_expression(param_name, access_field_call_expr)
        bool_from_float_call: pymdlsdk.IFunction_call = bool_from_float_def.create_function_call(args)
        bool_from_float_call_db_name: str = "bool_from_float"
        self.assertZero(self.sdk.transaction.store(bool_from_float_call, bool_from_float_call_db_name))
        bool_from_float_call = None

        # create a new material from constructor
        material_def: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, "mdl::material(bool,material_surface,material_surface,color,material_volume,material_geometry,hair_bsdf)")
        bool_from_float_call_expr: pymdlsdk.IExpression_call = ef.create_call(bool_from_float_call_db_name)
        args: pymdlsdk.IExpression_list = ef.create_expression_list()
        args.add_expression("thin_walled", bool_from_float_call_expr)
        material_instance: pymdlsdk.IFunction_call = material_def.create_function_call(args)
        material_db_name: str = "varying_bool_at_uniform_parameter"
        self.assertZero(self.sdk.transaction.store(material_instance, material_db_name))

        query_result: bool
        error_path: str 
        # The subgraph starting at "analyze_uniform_color" is ok
        (query_result, error_path) = self.sdk.mdlFactory.analyze_uniform(self.sdk.transaction, bool_from_float_call_db_name, False, None, None)
        self.assertFalse(query_result)
        self.assertEqual(error_path, "")

        # But attached to the uniform "thin_walled" slot of a material, it is broken.
        (query_result, error_path) = self.sdk.mdlFactory.analyze_uniform(self.sdk.transaction, material_db_name, False, None, None)
        self.assertFalse(query_result)
        self.assertEqual(error_path, "thin_walled.value.s")

        # access the state_normal call expression directly
        access_field_call: pymdlsdk.IFunction_call = self.sdk.transaction.access_as(pymdlsdk.IFunction_call, access_field_call_db_name)
        args: pymdlsdk.IExpression_list = access_field_call.get_arguments()
        state_normal_call_expr: pymdlsdk.IExpression = args.get_expression("s")
        (query_result, error_path) = self.sdk.mdlFactory.analyze_uniform(self.sdk.transaction, material_db_name, False, state_normal_call_expr, None)
        self.assertTrue(query_result)
        self.assertEqual(error_path, "thin_walled.value.s")

        # Access the "thin_walled" node (arguments of the root expression need a special handling internally)
        material_instance: pymdlsdk.IFunction_call = self.sdk.transaction.access_as(pymdlsdk.IFunction_call, material_db_name)
        args: pymdlsdk.IExpression_list = material_instance.get_arguments()
        bool_from_float_call_expr: pymdlsdk.IExpression = args.get_expression("thin_walled")
        (query_result, error_path) = self.sdk.mdlFactory.analyze_uniform(self.sdk.transaction, material_db_name, False, bool_from_float_call_expr, None)
        self.assertTrue(query_result)
        self.assertEqual(error_path, "thin_walled.value.s")


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
