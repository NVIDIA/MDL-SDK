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


class MainBaseModules(UnittestBase):
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
        self.sdk.load(addExampleSearchPath=False, loadImagePlugins=False)
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
        # UPCOMING self.assertNotEqual(elementType, pymdlsdk.Element_type.ELEMENT_TYPE_FORCE_32_BIT)
        # MEANWHILE
        self.assertNotEqual(elementType, pymdlsdk.ELEMENT_TYPE_FORCE_32_BIT)
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
        # UPCOMING returnCode: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        # UPCOMING call2: pymdlsdk.IFunction_call = definition.create_function_call(None, returnCode)
        # MEANWHILE
        call2, returnCode = definition.create_function_call_with_ret(None)

        defaults: pymdlsdk.IExpression_list = definition.get_defaults()
        if definition.get_parameter_count() > defaults.get_size():
            self.assertFalse(call.is_valid_interface())
            self.assertFalse(call2.is_valid_interface())
            # UPCOMING self.assertNotEqual(returnCode.value, 0)
            # MEANWHILE
            self.assertNotEqual(returnCode, 0)
        else:
            self.assertIsValidInterface(call)
            self.assertIsValidInterface(call2)
            # UPCOMING self.assertEqual(returnCode.value, 0)
            # MEANWHILE
            self.assertEqual(returnCode, 0)

        args: pymdlsdk.IExpression_list = self.ef.create_expression_list()
        args.add_expression("foo", self.ef.create_constant(self.vf.create_bool(False)))
        args.set_expression("foo", self.ef.create_constant(self.vf.create_bool(True)))

        call3: pymdlsdk.IFunction_call = definition.create_function_call(args)
        self.assertIsNotNone(call3)
        self.assertFalse(call3.is_valid_interface())

        # UPCOMING call4: pymdlsdk.IFunction_call = definition.create_function_call(args, returnCode)
        # MEANWHILE
        call4, returnCode = definition.create_function_call_with_ret(args)
        self.assertIsNotNone(call4)
        self.assertFalse(call4.is_valid_interface())
        # UPCOMING self.assertNotEqual(returnCode.value, 0)
        # MEANWHILE
        self.assertNotEqual(returnCode, 0)

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
        elementType: pymdlsdk.Element_type = fDefinition.get_element_type()
        # UPCOMING self.assertNotEqual(since, pymdlsdk.Mdl_version.MDL_VERSION_INVALID)
        # UPCOMING self.assertEqual(removed, pymdlsdk.Mdl_version.MDL_VERSION_INVALID)
        # UPCOMING self.assertEqual(elementType, pymdlsdk.Element_type.ELEMENT_TYPE_FUNCTION_DEFINITION)
        # MEANWHILE
        self.assertNotEqual(since, pymdlsdk.MDL_VERSION_INVALID)
        self.assertEqual(removed, pymdlsdk.MDL_VERSION_INVALID)
        self.assertEqual(elementType, pymdlsdk.ELEMENT_TYPE_FUNCTION_DEFINITION)
        # MEANWHILE END
        for pIndex in range(fDefinition.get_parameter_count()):
            pName: str = fDefinition.get_parameter_name(pIndex)
            self.assertEqual(fDefinition.get_parameter_index(pName), pIndex)
            pType1: pymdlsdk.IType = pTypes.get_type(pIndex)
            pType2: pymdlsdk.IType = pTypes.get_type(pName)
            self.assertEqual(pType1.get_iid(), pType2.get_iid())
            self.assertIsNotNone(fDefinition.get_mdl_parameter_type_name(pIndex))
            pAnno1: pymdlsdk.IAnnotation_block = pAnnos.get_annotation_block(pIndex)
            pAnno2: pymdlsdk.IAnnotation_block = pAnnos.get_annotation_block(pName)
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

    def test_IModule(self):
        self.assertTrue(self.loading_base_return_code == 0 or self.loading_base_return_code == 1)
        baseModule: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, "mdl::base")
        self.assertIsNone(baseModule.get_filename())  # only because its base.mdl
        self.assertEqual(baseModule.get_mdl_name(), "::base")
        self.assertEqual(baseModule.get_mdl_simple_name(), "base")
        self.assertFalse(baseModule.is_standard_module())
        self.assertFalse(baseModule.is_mdle_module())
        self.assertEqual(baseModule.get_mdl_package_component_count(), 0)
        self.assertIsNone(baseModule.get_mdl_package_component_name(42))
        version: pymdlsdk.Mdl_version = baseModule.get_mdl_version()
        elementType: pymdlsdk.Element_type = baseModule.get_element_type()
        # UPCOMING self.assertNotEqual(version, pymdlsdk.Mdl_version.MDL_VERSION_1_6)
        # UPCOMING self.assertEqual(version, pymdlsdk.Mdl_version.MDL_VERSION_1_7)  # might change at some point
        # UPCOMING self.assertEqual(elementType, pymdlsdk.Element_type.ELEMENT_TYPE_MODULE)
        # MEANWHILE
        self.assertNotEqual(version, pymdlsdk.MDL_VERSION_1_6)
        self.assertEqual(version, pymdlsdk.MDL_VERSION_1_7)
        self.assertEqual(elementType, pymdlsdk.ELEMENT_TYPE_MODULE)
        # MEANWHILE END
        imports: int = baseModule.get_import_count()
        self.assertIsNone(baseModule.get_import(42))
        for i in range(imports):
            importDbName: str = baseModule.get_import(i)
            self.assertTrue(len(importDbName) > 3)

        constants: pymdlsdk.IValue_list = baseModule.get_constants()
        self.assertIsValidInterface(constants)

        self.assertGreater(baseModule.get_function_count(), 0)
        self.assertIsNone(baseModule.get_function(1337))
        for i in range(baseModule.get_function_count()):
            fDbName: str = baseModule.get_function(i)
            self.run_function_test(fDbName)

        self.assertLess(baseModule.get_material_count(), 1337)
        self.assertIsNone(baseModule.get_material(1337))
        for i in range(baseModule.get_material_count()):
            fDbName: str = baseModule.get_material(i)
            self.run_function_test(fDbName)

        self.assertLess(baseModule.get_resources_count(), 1337)
        invalidResource: pymdlsdk.IValue_resource = baseModule.get_resource(1337)
        self.assertIsNotNone(invalidResource)
        self.assertFalse(invalidResource.is_valid_interface())
        for i in range(baseModule.get_resources_count()):
            resource: pymdlsdk.IValue_resource = baseModule.get_resource(i)
            self.assertIsValidInterface(resource)

        annos: pymdlsdk.IAnnotation_block = baseModule.get_annotations()
        self.assertIsNotNone(annos)
        self.run_ISceneElement_test(baseModule)

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
            # UPCOMING self.assertNotEqual(since, pymdlsdk.Mdl_version.MDL_VERSION_INVALID)
            # MEANWHILE
            self.assertNotEqual(since, pymdlsdk.MDL_VERSION_INVALID)


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
