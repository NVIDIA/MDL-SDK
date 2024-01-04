import unittest
import os

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


class MainBaseMdl(UnittestBase):
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

    def test_ITransaction(self):
        self.assertTrue(self.loading_base_return_code == 0 or self.loading_base_return_code == 1)
        baseModule: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, "mdl::base")
        self.assertIsValidInterface(baseModule)
        baseModule2_iinterface: pymdlsdk.IInterface = self.sdk.transaction.access("mdl::base")
        baseModule2: pymdlsdk.IModule = baseModule2_iinterface.get_interface(pymdlsdk.IModule)
        self.assertIsValidInterface(baseModule2)
        self.assertEqual(baseModule.get_iid(), baseModule2.get_iid())
        self.assertEqual(baseModule.get_mdl_name(), baseModule2.get_mdl_name())

    def run_ISceneElement_test(self, iinterface: pymdlsdk.IScene_element):
        self.assertIsValidInterface(iinterface)
        elementType: pymdlsdk.Element_type = iinterface.get_element_type()
        self.assertNotEqual(elementType, pymdlsdk.Element_type.ELEMENT_TYPE_FORCE_32_BIT)
        # TODO test attribute set
        # test type hierarchy because we don't see the inheritance in python, all functions are replicated
        if type(iinterface).IID() != pymdlsdk.IScene_element.IID():
            sceneElement: pymdlsdk.IScene_element = iinterface.get_interface(pymdlsdk.IScene_element)
            self.run_ISceneElement_test(sceneElement)

    def run_expression_test(self, expr: pymdlsdk.IExpression):
        self.assertIsValidInterface(expr)
        self.assertIsInstance(expr.get_kind(), pymdlsdk.IExpression.Kind)
        exprType: pymdlsdk.IType = expr.get_type()
        exprType2: pymdlsdk.IType = expr.get_type_as(pymdlsdk.IType_float)
        self.assertIsValidInterface(exprType)
        if exprType2.is_valid_interface():
            self.assertEqual(exprType.get_kind(), exprType2.get_kind())
        if expr.get_kind() == pymdlsdk.IExpression.Kind.EK_CONSTANT:
            exprConst: pymdlsdk.IExpression_constant = expr.get_interface(pymdlsdk.IExpression_constant)
            self.assertIsValidInterface(exprConst)
            value: pymdlsdk.IValue = exprConst.get_value()
            self.assertIsValidInterface(value)
        if expr.get_kind() == pymdlsdk.IExpression.Kind.EK_CALL:
            exprCall: pymdlsdk.IExpression_call = expr.get_interface(pymdlsdk.IExpression_call)
            self.assertIsValidInterface(exprCall)
            value: str = exprCall.get_call()

    def run_expression_list_test(self, exprList: pymdlsdk.IExpression_list):
        self.assertIsValidInterface(exprList)
        for i in range(exprList.get_size()):
            name: str = exprList.get_name(i)
            self.assertEqual(exprList.get_index(name), i)
            expr1: pymdlsdk.IExpression = exprList.get_expression(name)
            expr2: pymdlsdk.IExpression = exprList.get_expression(i)
            self.assertIsValidInterface(expr1)
            self.assertIsValidInterface(expr2)
            self.assertIsInstance(expr1.get_kind(), pymdlsdk.IExpression.Kind)
            self.assertIsInstance(expr2.get_kind(), pymdlsdk.IExpression.Kind)
            self.assertEqual(expr1.get_kind(), expr2.get_kind())
            if expr1.get_kind() == pymdlsdk.IExpression.Kind.EK_CONSTANT:
                exprCast = exprList.get_expression_as(pymdlsdk.IExpression_constant, name)
                self.assertIsValidInterface(exprCast)
            if expr1.get_kind() == pymdlsdk.IExpression.Kind.EK_CALL:
                exprCast = exprList.get_expression_as(pymdlsdk.IExpression_call, name)
                self.assertIsValidInterface(exprCast)
            if expr1.get_kind() == pymdlsdk.IExpression.Kind.EK_DIRECT_CALL:
                exprCast = exprList.get_expression_as(pymdlsdk.IExpression_direct_call, name)
                self.assertIsValidInterface(exprCast)
            if expr1.get_kind() == pymdlsdk.IExpression.Kind.EK_PARAMETER:
                exprCast = exprList.get_expression_as(pymdlsdk.IExpression_parameter, name)
                self.assertIsValidInterface(exprCast)
            if expr1.get_kind() == pymdlsdk.IExpression.Kind.EK_TEMPORARY:
                exprCast = exprList.get_expression_as(pymdlsdk.IExpression_temporary, name)
                self.assertIsValidInterface(exprCast)
            invalid: pymdlsdk.IInterface = exprList.get_expression_as(pymdlsdk.IExpression_list, i)
            self.assertIsNotNone(invalid)
            self.assertFalse(invalid.is_valid_interface())
            self.run_expression_test(expr1)
            self.run_expression_test(exprCast)

    def run_call_test(self, definition: pymdlsdk.IFunction_definition):
        call: pymdlsdk.IFunction_call = definition.create_function_call(None)
        returnCode: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        call2: pymdlsdk.IFunction_call = definition.create_function_call(None, returnCode)

        defaults: pymdlsdk.IExpression_list = definition.get_defaults()
        if definition.get_parameter_count() > defaults.get_size():
            self.assertFalse(call.is_valid_interface())
            self.assertFalse(call2.is_valid_interface())
            self.assertNotEqual(returnCode.value, 0)
        else:
            self.assertIsValidInterface(call)
            self.assertIsValidInterface(call2)
            self.assertEqual(returnCode.value, 0)

        args: pymdlsdk.IExpression_list = self.ef.create_expression_list()
        args.add_expression("foo", self.ef.create_constant(self.vf.create_bool(False)))
        args.set_expression("foo", self.ef.create_constant(self.vf.create_bool(True)))

        call3: pymdlsdk.IFunction_call = definition.create_function_call(args)
        self.assertIsNotNone(call3)
        self.assertFalse(call3.is_valid_interface())

        call4: pymdlsdk.IFunction_call = definition.create_function_call(args, returnCode)
        self.assertIsNotNone(call4)
        self.assertNotEqual(returnCode.value, 0)
        self.assertFalse(call4.is_valid_interface())

    def run_function_test(self, dbName: str):
        fDefinition: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, dbName)
        self.assertIsValidInterface(fDefinition)
        pTypes: pymdlsdk.IType_list = fDefinition.get_parameter_types()
        self.assertIsValidInterface(pTypes)
        fAnnos: pymdlsdk.IAnnotation_block = fDefinition.get_annotations()
        self.assertIsNotNone(fAnnos)
        pAnnos: pymdlsdk.IAnnotation_list = fDefinition.get_parameter_annotations()
        self.assertIsValidInterface(pAnnos)
        since: pymdlsdk.IModule
        removed: pymdlsdk.IModule
        (since, removed) = fDefinition.get_mdl_version()
        self.assertNotEqual(since, pymdlsdk.Mdl_version.MDL_VERSION_INVALID)
        self.assertEqual(removed, pymdlsdk.Mdl_version.MDL_VERSION_INVALID)
        elementType: pymdlsdk.Element_type = fDefinition.get_element_type()
        self.assertEqual(elementType, pymdlsdk.Element_type.ELEMENT_TYPE_FUNCTION_DEFINITION)
        for pIndex in range(fDefinition.get_parameter_count()):
            pName: str = fDefinition.get_parameter_name(pIndex)
            self.assertEqual(fDefinition.get_parameter_index(pName), pIndex)
            pType1: pymdlsdk.IType = pTypes.get_type(pIndex)
            pType2: pymdlsdk.IType = pTypes.get_type(pName)
            self.assertEqual(pType1.get_iid(), pType2.get_iid())
            self.assertIsNotNone(fDefinition.get_mdl_parameter_type_name(pIndex))
            pAnno1: pymdlsdk.IAnnotation_block = pAnnos.get_annotation_block(pName)
            pAnno2: pymdlsdk.IAnnotation_block = pAnnos.get_annotation_block(pAnnos.get_index(pName))
            self.assertEqual(pAnno1.is_valid_interface(), pAnno2.is_valid_interface())
            if pAnno1.is_valid_interface():
                self.assertEqual(pAnno1.get_iid(), pAnno2.get_iid())
            for uIndex in range(fDefinition.get_enable_if_users(pIndex)):
                self.assertIsInstance(fDefinition.get_enable_if_user(pIndex, uIndex), int)
            invalid: int = fDefinition.get_enable_if_user(pIndex, 1337)
            self.assertEqual(invalid, -1)

        self.run_ISceneElement_test(fDefinition)
        self.assertIsNotNone(fDefinition.get_mdl_name())
        self.assertIsNotNone(fDefinition.get_mdl_simple_name())
        self.assertStartswith(fDefinition.get_module(), "mdl::")
        self.assertIsNotNone(fDefinition.get_mdl_module_name())
        self.assertIsNotNone(fDefinition.get_mdl_mangled_name())
        self.assertIsInstance(fDefinition.is_array_constructor(), bool)
        self.assertIsInstance(fDefinition.is_exported(), bool)
        self.assertIsInstance(fDefinition.is_uniform(), bool)
        self.assertIsInstance(fDefinition.is_material(), bool)
        semantic: pymdlsdk.IFunction_definition.Semantics = fDefinition.get_semantic()
        self.assertIsInstance(semantic, pymdlsdk.IFunction_definition.Semantics)
        self.assertNotEqual(semantic, pymdlsdk.IFunction_definition.Semantics.DS_FORCE_32_BIT)
        self.assertTrue(fDefinition.is_valid(None))  # no reload of base
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        self.assertTrue(fDefinition.is_valid(context))  # no reload of base
        self.assertContextNoErrors(context)
        self.assertIsNone(fDefinition.get_prototype())  # not in base
        self.assertIsNone(fDefinition.get_thumbnail())  # not in base
        returnType: pymdlsdk.IType = fDefinition.get_return_type()
        self.assertIsValidInterface(returnType)
        returnAnnos: pymdlsdk.IAnnotation_block = fDefinition.get_return_annotations()
        self.assertIsNotNone(returnAnnos)
        defaults: pymdlsdk.IExpression_list = fDefinition.get_defaults()
        self.run_expression_list_test(defaults)
        self.assertIsValidInterface(defaults)
        enableIfConditions: pymdlsdk.IExpression_list = fDefinition.get_enable_if_conditions()
        self.assertIsValidInterface(enableIfConditions)
        fBody: pymdlsdk.IExpression = fDefinition.get_body()
        if fBody.is_valid_interface():
            self.assertIsValidInterface(fBody)
        for i in range(fDefinition.get_temporary_count()):
            temp: pymdlsdk.IExpression = fDefinition.get_temporary(i)
            self.assertIsValidInterface(temp)
            if fDefinition.get_temporary_name(i) is not None:
                self.assertNotEqual(fDefinition.get_temporary_name(i), "")
        self.run_call_test(fDefinition)

    def run_module_test(self, mdlModule: pymdlsdk.IModule):
        elementType: pymdlsdk.Element_type = mdlModule.get_element_type()
        self.assertEqual(elementType, pymdlsdk.Element_type.ELEMENT_TYPE_MODULE)
        imports: int = mdlModule.get_import_count()
        self.assertIsNone(mdlModule.get_import(42))
        for i in range(imports):
            importDbName: str = mdlModule.get_import(i)
            self.assertTrue(len(importDbName) > 3)

        constants: pymdlsdk.IValue_list = mdlModule.get_constants()
        self.assertIsValidInterface(constants)

        self.assertGreater(mdlModule.get_function_count(), 0)
        self.assertIsNone(mdlModule.get_function(1337))
        for i in range(mdlModule.get_function_count()):
            fDbName: str = mdlModule.get_function(i)
            self.run_function_test(fDbName)

        self.assertLess(mdlModule.get_material_count(), 1337)
        self.assertIsNone(mdlModule.get_material(1337))
        for i in range(mdlModule.get_material_count()):
            fDbName: str = mdlModule.get_material(i)
            self.run_function_test(fDbName)

        self.assertLess(mdlModule.get_resources_count(), 1337)
        invalidResource: pymdlsdk.IValue_resource = mdlModule.get_resource(1337)
        self.assertIsNotNone(invalidResource)
        self.assertFalse(invalidResource.is_valid_interface())
        for i in range(mdlModule.get_resources_count()):
            resource: pymdlsdk.IValue_resource = mdlModule.get_resource(i)
            self.assertIsValidInterface(resource)

        annos: pymdlsdk.IAnnotation_block = mdlModule.get_annotations()
        self.assertIsNotNone(annos)
        self.run_ISceneElement_test(mdlModule)

    def test_IAnnotations(self):
        self.assertTrue(self.loading_base_return_code == 0 or self.loading_base_return_code == 1)
        annoModule: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, "mdl::anno")
        self.assertIsNone(annoModule.get_filename())  # only because its base.mdl
        self.assertLess(annoModule.get_annotation_definition_count(), 1337)
        invalidAnnoDef: pymdlsdk.IAnnotation_definition = annoModule.get_annotation_definition(1337)
        self.assertIsNotNone(invalidAnnoDef)
        self.assertFalse(invalidAnnoDef.is_valid_interface())
        for i in range(annoModule.get_annotation_definition_count()):
            annoDef: pymdlsdk.IAnnotation_definition = annoModule.get_annotation_definition(i)
            self.assertIsValidInterface(annoDef)
            since: pymdlsdk.IModule
            removed: pymdlsdk.IModule
            (since, removed) = annoDef.get_mdl_version()
            self.assertNotEqual(since, pymdlsdk.Mdl_version.MDL_VERSION_INVALID)

    def run_compiled_material_test(self, compiledMaterial: pymdlsdk.ICompiled_material, instanceDbName: str):
        self.assertIsInstance(compiledMaterial.depends_on_global_distribution(), bool)
        self.assertIsInstance(compiledMaterial.depends_on_uniform_scene_data(), bool)
        self.assertIsInstance(compiledMaterial.depends_on_state_object_id(), bool)
        self.assertIsInstance(compiledMaterial.depends_on_state_transform(), bool)
        sceneDataCount: int = compiledMaterial.get_referenced_scene_data_count()
        self.assertIsNone(compiledMaterial.get_referenced_scene_data_name(sceneDataCount + 42))
        for i in range(sceneDataCount):
            self.assertIsNotNone(compiledMaterial.get_referenced_scene_data_name(i))
        paramCount: int = compiledMaterial.get_parameter_count()
        self.assertIsNone(compiledMaterial.get_parameter_name(paramCount + 42))
        for p in range(paramCount):
            self.assertIsNotNone(compiledMaterial.get_parameter_name(p))
            value: pymdlsdk.IValue = compiledMaterial.get_argument(i)
            self.assertIsValidInterface(value)
        hash: pymdlsdk.Uuid = compiledMaterial.get_hash()
        self.assertNotEqual(hash, pymdlsdk.Uuid())
        surfaceScatteringSlotHash: pymdlsdk.Uuid = compiledMaterial.get_slot_hash(pymdlsdk.Material_slot.SLOT_SURFACE_SCATTERING)
        self.assertNotEqual(hash, surfaceScatteringSlotHash)
        subExpr: pymdlsdk.IExpression = compiledMaterial.lookup_sub_expression('surface.scattering')
        if subExpr.is_valid_interface():
            self.run_expression_test(subExpr)
        subExpr2: pymdlsdk.IExpression = compiledMaterial.lookup_sub_expression('surface.NOT_EXISTING')
        self.assertIsNotNone(subExpr2)
        self.assertFalse(subExpr2.is_valid_interface())
        subExpr3: pymdlsdk.IExpression = compiledMaterial.lookup_sub_expression('NOT_EXISTING_EITHER')
        self.assertIsNotNone(subExpr3)
        self.assertFalse(subExpr3.is_valid_interface())
        self.assertNotEqual(compiledMaterial.get_mdl_meters_per_scene_unit(), 0)
        self.assertNotEqual(compiledMaterial.get_mdl_wavelength_min(), 0)
        self.assertNotEqual(compiledMaterial.get_mdl_wavelength_max(), 0)
        self.assertIsInstance(compiledMaterial.get_opacity(), pymdlsdk.Material_opacity)
        self.assertIsInstance(compiledMaterial.get_surface_opacity(), pymdlsdk.Material_opacity)
        success: bool = False
        opacity_value: float = -1.0
        success, opacity_value = compiledMaterial.get_cutout_opacity()
        if success:
            self.assertGreaterEqual(opacity_value, 0.0)
            self.assertLessEqual(opacity_value, 1.0)
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        self.assertTrue(compiledMaterial.is_valid(context))  # no reloading here
        self.assertTrue(compiledMaterial.is_valid(None))
        with self.assertRaises(TypeError):
            compiledMaterial.is_valid()  # missing context paramter
        self.assertTrue(compiledMaterial.get_element_type(), pymdlsdk.Element_type.ELEMENT_TYPE_COMPILED_MATERIAL)
        body: pymdlsdk.IExpression_direct_call = compiledMaterial.get_body()
        self.run_expression_test(body)
        tempCount: int = compiledMaterial.get_temporary_count()
        tempOutOfBounds: pymdlsdk.IExpression = compiledMaterial.get_temporary(tempCount + 42)
        self.assertIsNotNone(tempOutOfBounds)
        self.assertFalse(tempOutOfBounds.is_valid_interface())
        for t in range(tempCount):
            tempExpr: pymdlsdk.IExpression = compiledMaterial.get_temporary(t)
            self.run_expression_test(tempExpr)

        returnCode: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        idbname1: pymdlsdk.IString = compiledMaterial.get_connected_function_db_name(instanceDbName, 0)
        idbname2: pymdlsdk.IString = compiledMaterial.get_connected_function_db_name(instanceDbName, 0, returnCode)
        self.assertIsNotNone(idbname1)
        self.assertIsNotNone(idbname2)
        idbname3: pymdlsdk.IString
        errorCode: int = -1337
        idbname3, errorCode = compiledMaterial.get_connected_function_db_name_with_ret(instanceDbName, 0)
        self.assertIsNotNone(idbname2)
        self.assertNotEqual(errorCode, -1337)

    def run_material_test(self, mdlModule: pymdlsdk.IModule, materialSimpleName: str):
        qualifiedName: str = mdlModule.get_mdl_name() + "::" + materialSimpleName
        idbName: pymdlsdk.IString = self.sdk.mdlFactory.get_db_definition_name(qualifiedName)
        self.assertIsValidInterface(idbName)
        dbName: str = idbName.get_c_str()
        ioverloads: pymdlsdk.IArray = mdlModule.get_function_overloads(dbName)
        self.assertGreater(ioverloads.get_length(), 0)
        ifirstOverload: pymdlsdk.IString = ioverloads.get_element_as(pymdlsdk.IString, 0)
        definition: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, ifirstOverload.get_c_str())
        self.assertIsValidInterface(definition)
        res: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        call: pymdlsdk.IFunction_call = definition.create_function_call(None, res)
        self.assertEqual(res.value, 0)
        self.assertIsValidInterface(call)
        self.assertTrue(call.is_material())
        materialInstance: pymdlsdk.IMaterial_instance = call.get_interface(pymdlsdk.IMaterial_instance)
        self.assertIsValidInterface(materialInstance)

        instanceFlags: int = pymdlsdk.IMaterial_instance.Compilation_options.DEFAULT_OPTIONS.value
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        instanceCompiledMaterial: pymdlsdk.ICompiled_material = materialInstance.create_compiled_material(instanceFlags, context)
        self.assertIsValidInterface(instanceCompiledMaterial)
        self.assertContextNoErrors(context)

        classFlags: int = pymdlsdk.IMaterial_instance.Compilation_options.CLASS_COMPILATION.value
        classCompiledMaterial: pymdlsdk.ICompiled_material = materialInstance.create_compiled_material(classFlags, context)
        self.assertIsValidInterface(classCompiledMaterial)
        self.assertContextNoErrors(context)

        dbName: str = qualifiedName + "_INSTANCE"
        self.assertEqual(self.sdk.transaction.store(materialInstance, dbName), 0)
        self.run_compiled_material_test(instanceCompiledMaterial, dbName)
        self.run_compiled_material_test(classCompiledMaterial, dbName)

    def test_tutorials_mdl(self):
        moduleQualifier = "::nvidia::sdk_examples::tutorials"
        moduleDbName = self.load_module(moduleQualifier)
        self.assertEqual(moduleDbName, self.sdk.mdlFactory.get_db_module_name(moduleQualifier).get_c_str())
        mdlModule: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, moduleDbName)
        self.assertIsNotNone(mdlModule.get_filename())
        self.assertEqual(mdlModule.get_mdl_name(), moduleQualifier)
        self.assertEqual(mdlModule.get_mdl_simple_name(), "tutorials")
        self.assertFalse(mdlModule.is_standard_module())
        self.assertFalse(mdlModule.is_mdle_module())
        self.assertGreater(mdlModule.get_mdl_package_component_count(), 0)
        self.assertIsNone(mdlModule.get_mdl_package_component_name(42))
        self.run_module_test(mdlModule)

    def test_gltf_support_mdl(self):
        moduleQualifier = "::nvidia::sdk_examples::gltf_support"
        moduleDbName = self.load_module(moduleQualifier)
        self.assertEqual(moduleDbName, self.sdk.mdlFactory.get_db_module_name(moduleQualifier).get_c_str())
        mdlModule: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, moduleDbName)
        self.assertIsNotNone(mdlModule.get_filename())
        self.assertEqual(mdlModule.get_mdl_name(), moduleQualifier)
        self.assertEqual(mdlModule.get_mdl_simple_name(), "gltf_support")
        self.assertFalse(mdlModule.is_standard_module())
        self.assertFalse(mdlModule.is_mdle_module())
        self.assertGreater(mdlModule.get_mdl_package_component_count(), 0)
        self.assertIsNone(mdlModule.get_mdl_package_component_name(42))
        self.run_module_test(mdlModule)
        self.run_material_test(mdlModule, "gltf_material")


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
