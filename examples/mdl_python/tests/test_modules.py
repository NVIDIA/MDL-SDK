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

import warnings

class MainModulesMdl(UnittestBase):
    sdk: SDK = None
    tf: pymdlsdk.IType_factory = None
    vf: pymdlsdk.IValue_factory = None
    ef: pymdlsdk.IExpression_factory = None
    imp_exp: pymdlsdk.IMdl_impexp_api = None
    evaluator: pymdlsdk.IMdl_evaluator_api = None
    loading_base_return_code: int = -1

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=True, addExampleResourcePath=True, loadImagePlugins=True)
        self.tf = self.sdk.mdlFactory.create_type_factory(self.sdk.transaction)
        self.vf = self.sdk.mdlFactory.create_value_factory(self.sdk.transaction)
        self.ef = self.sdk.mdlFactory.create_expression_factory(self.sdk.transaction)
        self.imp_exp: pymdlsdk.IMdl_impexp_api = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_impexp_api)
        self.evaluator = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_evaluator_api)

        self.loading_base_return_code = self.imp_exp.load_module(self.sdk.transaction, "::base")  # load something to get builtins

    @classmethod
    def tearDownClass(self):
        self.evaluator = None
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
        self.assertIsValidInterface(self.evaluator)
        self.assertEqual(self.sdk.neuray.get_status(), pymdlsdk.INeuray.Status.STARTED)

    def test_execution_context(self):
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        self.assertIsValidInterface(context)

        self.assertEqual(context.set_option("opt", 42), -1)
        self.assertEqual(context.set_option("optimization_level", 1), 0)
        #self.assertEqual(context.get_option("opt"), -1) # FIXME: how to call?
        #self.assertZero(context.get_option("optimization_level")) # FIXME: how to call?
        
        typ: str = context.get_option_type("warning")
        self.assertEqual(typ, "String")
        optCount: int = context.get_option_count()
        for i in range(optCount):
            name: str = context.get_option_name(i)

    def test_ITransaction(self):
        self.assertTrue(self.loading_base_return_code == 0 or self.loading_base_return_code == 1)
        self.assertTrue(self.sdk.transaction.is_open())

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
            self.assertNotNullOrEmpty(value)
            self.assertTrue(exprCall.set_call(value) < 0)

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

            call3, returnCode2 = self.assertDeprecationWarning(lambda: definition.create_function_call_with_ret(None))
            self.assertZero(returnCode2)

            isDecl: bool = call.is_declarative()
            self.assertTrue(isDecl == False or isDecl == True)
            isDef: bool = call.is_default()
            self.assertTrue(isDef == False or isDef == True)

            elementType: pymdlsdk.Element_type = call.get_element_type()
            self.assertNotEqual(elementType, pymdlsdk.Element_type.ELEMENT_TYPE_FORCE_32_BIT)

            funDefName: str = call.get_mdl_function_definition()
            self.assertNotNullOrEmpty(funDefName)
            self.assertFalse(call.is_array_constructor())
            self.assertEqual(call.get_parameter_count(), definition.get_parameter_count())
            for i in range(call.get_parameter_count()):
                aname: str = call.get_parameter_name(i)
                pname: str = definition.get_parameter_name(i)
                self.assertEqual(aname, pname)
                self.assertEqual(call.get_parameter_index(aname), i)
                enabled: pymdlsdk.IValue_bool = self.evaluator.is_function_parameter_enabled(
                    self.sdk.transaction, self.vf, call, i, returnCode)
                self.assertIsValidInterface(enabled)
            ptypes: pymdlsdk.IType_list = call.get_parameter_types()
            for i in range(call.get_parameter_count()):
                self.assertIsValidInterface(ptypes.get_type(i))

            retType: pymdlsdk.IType = call.get_return_type()
            self.assertIsValidInterface(retType)

            if not isDef:
                # Do not try this for calls in default argument, will fail
                args: pymdlsdk.IExpression_list = call.get_arguments()
                for i in range(call.get_parameter_count()):
                    arg: pymdlsdk.IExpression = args.get_expression(i)
                    call.set_argument(i, arg)
                for i in range(call.get_parameter_count()):
                    self.assertZero(call.reset_argument(i))

                args: pymdlsdk.IExpression_list = call.get_arguments()
                self.assertZero(call.set_arguments(args))

                context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
                call.repair(0, context)
                self.assertContextNoErrors(context)
                self.assertTrue(call.is_valid(context))
                self.assertContextNoErrors(context)

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
        self.assertTrue(isinstance(fDefinition.is_declarative(), bool))
        mangled: str = fDefinition.get_mangled_name()
        self.assertTrue(isinstance(mangled, str))
        self.assertNotEqual(mangled, "")
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
        self.assertIsNotNone(self.assertDeprecationWarning(lambda: fDefinition.get_mdl_mangled_name()))
        self.assertIsNotNone(fDefinition.get_mangled_name())
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
        # Note: Temporary names are tested in a separate class below because this
        # requires special configuration.
        self.run_call_test(fDefinition)

        # test deprecated function
        direct_call: pymdlsdk.IExpression_direct_call
        code: int
        direct_call, code = self.assertDeprecationWarning(lambda: self.ef.create_direct_call_with_ret(dbName, defaults))

        outReturnCode: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        direct_call2: pymdlsdk.IExpression_direct_call = self.ef.create_direct_call(dbName, defaults, outReturnCode)
        self.assertEqual(code, outReturnCode.value)

        if fDefinition.get_parameter_count() > defaults.get_size():
            self.assertIsNotValidInterface(direct_call)
            self.assertIsNotValidInterface(direct_call2)
            self.assertNotEqual(code, 0)
        else:
            self.assertIsValidInterface(direct_call)
            self.assertIsValidInterface(direct_call2)
            self.assertEqual(code, 0)

            # Also test expression factory method.
            returnCode: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
            direct_call_ = self.ef.create_direct_call(dbName, defaults, returnCode)
            self.assertIsValidInterface(direct_call_)
            self.assertZero(returnCode.value)

            # Since this is the only place where we have a direct call, use it
            # to improve coverage on run_expression_list_test().
            expr_list: pymdlsdk.IExpression_list = self.ef.create_expression_list()
            expr_list.add_expression("name", direct_call)
            self.run_expression_list_test(expr_list)

            # Some more tests on direct calls.
            defDbName: str = direct_call.get_definition()
            self.assertEqual(dbName, defDbName)
            args: pymdlsdk.IExpression_list = direct_call.get_arguments()
            self.assertEqual(args.get_size(), defaults.get_size())

    def run_module_test(self, mdlModule: pymdlsdk.IModule):
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        self.assertTrue(mdlModule.is_valid(context))
        self.assertContextNoErrors(context)
        self.assertIsNotNone(mdlModule.get_mdl_version())

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

        categories: pymdlsdk.IStruct_category_list = mdlModule.get_struct_categories()
        self.assertIsNotNone(categories)

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
            semantic: pymdlsdk.IAnnotation_definition.Semantics = annoDef.get_semantic()
            self.assertIsInstance(semantic, pymdlsdk.IAnnotation_definition.Semantics)
            code: int = semantic.value
            since: pymdlsdk.IModule
            removed: pymdlsdk.IModule
            (since, removed) = annoDef.get_mdl_version()
            self.assertNotEqual(since, pymdlsdk.Mdl_version.MDL_VERSION_INVALID)
            self.assertNotNullOrEmpty(annoDef.get_module())
            self.assertNotNullOrEmpty(annoDef.get_name())
            self.assertNotNullOrEmpty(annoDef.get_mdl_module_name())
            self.assertNotNullOrEmpty(annoDef.get_mdl_simple_name())
            self.assertTrue(annoDef.is_exported())
            for j in range(annoDef.get_parameter_count()):
                self.assertNotNullOrEmpty(annoDef.get_mdl_parameter_type_name(j))
                pname: str = annoDef.get_parameter_name(j)
                self.assertNotNullOrEmpty(pname)
                self.assertEqual(j, annoDef.get_parameter_index(pname))
            types: pymdlsdk.IType_list = annoDef.get_parameter_types()
            self.assertEqual(types.get_size(), annoDef.get_parameter_count())
            defaults: pymdlsdk.IExpression_list = annoDef.get_defaults()
            annotations: pymdlsdk.IAnnotation_list = annoDef.get_annotations()

        unusedDef: pymdlsdk.IAnnotation_definition = annoModule.get_annotation_definition("::anno::unused()")
        unusedExprList: pymdlsdk.IExpression_list = self.ef.create_expression_list()
        unusedAnno: pymdlsdk.IAnnotation = annoDef.create_annotation(unusedExprList)
        self.assertIsValidInterface(unusedAnno)
        self.assertTrue(unusedAnno.get_arguments().get_size() == 1)
        self.assertIsValidInterface(unusedAnno.get_definition())

        anno_args: pymdlsdk.IExpression_list = self.ef.create_expression_list()
        anno: pymdlsdk.IAnnotation = self.ef.create_annotation("::anno::unused()", anno_args)
        self.assertIsValidInterface(anno)
        self.assertEqual(anno.get_name(), "::anno::unused()")
        anno.set_name("other_name()")
        self.assertEqual(anno.get_name(), "other_name()")

        anno_list: pymdlsdk.IAnnotation_list = self.ef.create_annotation_list()
        self.assertIsValidInterface(anno_list)
        self.assertZero(anno_list.get_size())

        anno_block: pymdlsdk.IAnnotation_block = self.ef.create_annotation_block()
        self.assertIsValidInterface(anno_block)

        anno_args2: pymdlsdk.IExpression_list = self.ef.create_expression_list()
        anno2: pymdlsdk.IAnnotation = self.ef.create_annotation("::anno::unused()", anno_args2)
        self.assertZero(anno_block.get_size())
        self.assertZero(anno_block.add_annotation(anno2))
        self.assertEqual(anno_block.get_size(), 1)
        self.assertIsValidInterface(anno_block.get_annotation(0))
        anno_args3: pymdlsdk.IExpression_list = self.ef.create_expression_list()
        anno3: pymdlsdk.IAnnotation = self.ef.create_annotation("::anno::unused()", anno_args3)
        self.assertZero(anno_block.set_annotation(0, anno3)) # test overwriting
        self.assertIsValidInterface(anno_block.get_annotation(0))

        anno_list.add_annotation_block("test", anno_block)
        self.assertEqual(anno_list.get_size(), 1)

        anno_block2: pymdlsdk.IAnnotation_list = anno_list.get_annotation_block(0)
        self.assertIsValidInterface(anno_block2)

        anno_block3: pymdlsdk.IAnnotation_list = anno_list.get_annotation_block("test")
        self.assertIsValidInterface(anno_block3)

        self.assertEqual(anno_list.get_name(0), "test")

        anno_block_new: pymdlsdk.IAnnotation_block = self.ef.create_annotation_block()
        self.assertIsValidInterface(anno_block_new)
        self.assertZero(anno_list.set_annotation_block(0, anno_block_new))
        self.assertZero(anno_list.set_annotation_block("test", anno_block_new))
        self.assertEqual(anno_list.set_annotation_block("test", None), -1)
        self.assertEqual(anno_list.set_annotation_block(1337, anno_block_new), -2)
        self.assertEqual(anno_list.set_annotation_block("1337", anno_block_new), -2)

    class MaterialTestSpec(object):
        class AnalysisTestSpec(object):
            cutOutOpacityIsConstant: bool = False
            cutOutOpacityConstantValue: float = -1.0
            opacity: pymdlsdk.Material_opacity = pymdlsdk.Material_opacity.OPACITY_FORCE_32_BIT
            surfaceOpacity: pymdlsdk.Material_opacity = pymdlsdk.Material_opacity.OPACITY_FORCE_32_BIT

        simpleName: str = ""
        instanceCompile: AnalysisTestSpec = AnalysisTestSpec()
        classCompile: AnalysisTestSpec = AnalysisTestSpec()

    def run_compiled_material_test(self, compiledMaterial: pymdlsdk.ICompiled_material, instanceDbName: str, matSpec: MaterialTestSpec, isClassCompiled: bool):
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
            value: pymdlsdk.IValue = compiledMaterial.get_argument(p)
            self.assertIsValidInterface(value)
        hash: pymdlsdk.Uuid = compiledMaterial.get_hash()
        self.assertNotEqual(hash, pymdlsdk.Uuid())
        hash2: pymdlsdk.Uuid = compiledMaterial.get_sub_expression_hash('')
        self.assertEqual(hash, hash2)
        surfaceScatteringSlotHash: pymdlsdk.Uuid = compiledMaterial.get_slot_hash(pymdlsdk.Material_slot.SLOT_SURFACE_SCATTERING)
        self.assertNotEqual(hash, surfaceScatteringSlotHash)
        surfaceScatteringSlotHash2: pymdlsdk.Uuid = compiledMaterial.get_sub_expression_hash('surface.scattering')
        self.assertEqual(surfaceScatteringSlotHash, surfaceScatteringSlotHash2)
        notExistingPathHash: pymdlsdk.Uuid = compiledMaterial.get_sub_expression_hash('surface.scattering.NOT_EXISTING')
        self.assertEqual(notExistingPathHash, pymdlsdk.Uuid())
        iorHash: pymdlsdk.Uuid = compiledMaterial.get_sub_expression_hash('ior')
        self.assertNotEqual(iorHash, pymdlsdk.Uuid())
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
        is_constant: bool = False
        opacity_value: float = -1.0
        if isClassCompiled:
            self.assertEqual(compiledMaterial.get_opacity(), matSpec.classCompile.opacity)
            self.assertEqual(compiledMaterial.get_surface_opacity(), matSpec.classCompile.surfaceOpacity)
            is_constant, opacity_value = compiledMaterial.get_cutout_opacity()
            self.assertEqual(matSpec.classCompile.cutOutOpacityIsConstant, is_constant)
            if is_constant:
                self.assertAlmostEqual(matSpec.classCompile.cutOutOpacityConstantValue, opacity_value)
        else:
            self.assertEqual(compiledMaterial.get_opacity(), matSpec.instanceCompile.opacity)
            self.assertEqual(compiledMaterial.get_surface_opacity(), matSpec.instanceCompile.surfaceOpacity)
            is_constant, opacity_value = compiledMaterial.get_cutout_opacity()
            self.assertEqual(matSpec.instanceCompile.cutOutOpacityIsConstant, is_constant)
            if is_constant:
                self.assertAlmostEqual(matSpec.instanceCompile.cutOutOpacityConstantValue, opacity_value)

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
        idbname3, errorCode = self.assertDeprecationWarning(lambda: compiledMaterial.get_connected_function_db_name_with_ret(instanceDbName, 0))
        self.assertIsNotNone(idbname2)
        self.assertNotEqual(errorCode, -1337)

    def run_material_test(self, mdlModule: pymdlsdk.IModule, matSpec: MaterialTestSpec):
        qualifiedName: str = mdlModule.get_mdl_name() + "::" + matSpec.simpleName
        idbName: pymdlsdk.IString = self.sdk.mdlFactory.get_db_definition_name(qualifiedName)
        self.assertIsValidInterface(idbName)
        dbName: str = idbName.get_c_str()
        ioverloads: pymdlsdk.IArray = mdlModule.get_function_overloads(matSpec.simpleName)
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
        self.assertEqual(materialInstance.get_element_type(), pymdlsdk.Element_type.ELEMENT_TYPE_MATERIAL_INSTANCE)

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
        self.run_compiled_material_test(instanceCompiledMaterial, dbName, matSpec, False)
        self.run_compiled_material_test(classCompiledMaterial, dbName, matSpec, True)

    def run_module_and_material_test(self, moduleQualifier: str, matSpec: MaterialTestSpec):
        moduleDbName = self.load_module(moduleQualifier)
        self.assertEqual(moduleDbName, self.sdk.mdlFactory.get_db_module_name(moduleQualifier).get_c_str())
        mdlModule: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, moduleDbName)
        self.assertIsNotNone(mdlModule.get_filename())
        self.assertEqual(mdlModule.get_mdl_name(), moduleQualifier)
        pos: int = moduleQualifier.rfind(':')
        simpleName: str = moduleQualifier[pos + 1:]
        self.assertEqual(mdlModule.get_mdl_simple_name(), simpleName)
        self.assertFalse(mdlModule.is_standard_module())
        self.assertFalse(mdlModule.is_mdle_module())
        self.assertGreater(mdlModule.get_mdl_package_component_count(), 0)
        self.assertIsNone(mdlModule.get_mdl_package_component_name(4200))
        self.run_module_test(mdlModule)
        self.run_material_test(mdlModule, matSpec)

    def test_gltf_support_mdl(self):
        matSpec: MainModulesMdl.MaterialTestSpec = MainModulesMdl.MaterialTestSpec()
        matSpec.simpleName = "gltf_material"
        matSpec.classCompile.cutOutOpacityIsConstant = False
        matSpec.classCompile.opacity = pymdlsdk.Material_opacity.OPACITY_UNKNOWN
        matSpec.classCompile.surfaceOpacity = pymdlsdk.Material_opacity.OPACITY_UNKNOWN
        matSpec.instanceCompile.cutOutOpacityIsConstant = False # ??
        matSpec.instanceCompile.opacity = pymdlsdk.Material_opacity.OPACITY_UNKNOWN  # ??
        matSpec.instanceCompile.surfaceOpacity = pymdlsdk.Material_opacity.OPACITY_OPAQUE
        self.run_module_and_material_test("::nvidia::sdk_examples::gltf_support", matSpec)

    def test_dxr_sphere_mat_mdl(self):
        matSpec: MainModulesMdl.MaterialTestSpec = MainModulesMdl.MaterialTestSpec()
        matSpec.simpleName = "dxr_sphere_mat"
        matSpec.classCompile.cutOutOpacityIsConstant = True
        matSpec.classCompile.cutOutOpacityConstantValue = 1.0
        matSpec.classCompile.opacity = pymdlsdk.Material_opacity.OPACITY_OPAQUE
        matSpec.classCompile.surfaceOpacity = pymdlsdk.Material_opacity.OPACITY_OPAQUE
        matSpec.instanceCompile = matSpec.classCompile
        self.run_module_and_material_test("::nvidia::sdk_examples::tutorials", matSpec)

    def test_aov_material(self):
        moduleQualifier = "::nvidia::sdk_examples::tutorials_aov"
        moduleDbName = self.load_module(moduleQualifier)
        self.assertEqual(moduleDbName, self.sdk.mdlFactory.get_db_module_name(moduleQualifier).get_c_str())
        mdlModule: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, moduleDbName)
        self.assertIsValidInterface(mdlModule)
        for matIndex in range(mdlModule.get_material_count()):
            matDbName: str = mdlModule.get_material(matIndex)
            if matDbName.endswith('::green()'):
                funcDef: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, matDbName)
                self.assertIsValidInterface(funcDef)
                funcCall: pymdlsdk.IFunction_call = funcDef.create_function_call(None)
                self.assertIsValidInterface(funcCall)
                matInstance: pymdlsdk.IMaterial_instance = funcCall.get_interface(pymdlsdk.IMaterial_instance)
                self.assertIsValidInterface(matInstance)
                classFlags: int = pymdlsdk.IMaterial_instance.Compilation_options.CLASS_COMPILATION.value
                # compiled materials without specifying a target type will only have the return type fields
                compiledMaterial: pymdlsdk.ICompiled_material = matInstance.create_compiled_material(classFlags)
                self.assertIsValidInterface(compiledMaterial)
                aovExpression: pymdlsdk.IExpression = compiledMaterial.lookup_sub_expression("example")
                self.assertIsValidInterface(aovExpression)
                # when specifying the target type, the target fields are available
                tMaterial: pymdlsdk.IType_structure = self.tf.get_predefined_struct(pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL)
                context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
                context.set_option("target_type", tMaterial)
                compiledMaterialTargeted: pymdlsdk.ICompiled_material = matInstance.create_compiled_material(classFlags, context)
                aovExpression: pymdlsdk.IExpression = compiledMaterialTargeted.lookup_sub_expression("example")
                self.assertIsNotValidInterface(aovExpression)  # not a valid field of the target type
                thinWalledExpression: pymdlsdk.IExpression = compiledMaterialTargeted.lookup_sub_expression("thin_walled")
                self.assertIsValidInterface(thinWalledExpression)  # present in the target type

    def test_reload(self):
        moduleQualifier = "::nvidia::sdk_examples::tutorials_aov"
        moduleDbName = self.load_module(moduleQualifier)
        self.assertEqual(moduleDbName, self.sdk.mdlFactory.get_db_module_name(moduleQualifier).get_c_str())
        mdlModule: pymdlsdk.IModule = self.sdk.transaction.edit_as(pymdlsdk.IModule, moduleDbName)
        self.assertZero(mdlModule.reload(False, None)) # we just want to check reload binding.

        module_source: str = "mdl 1.8; export const int PI = 3;"
        self.assertEqual(mdlModule.reload_from_string(module_source, False, None), -1) # we just want to check reload binding.

    def test_ivalue_factory_resources(self):
        vTexture2d1: pymdlsdk.IValue_texture = self.sdk.mdlFactory.create_texture(self.sdk.transaction, "/nvidia/sdk_examples/resources/example.png", pymdlsdk.IType_texture.Shape.TS_2D, 2.2, None, False, None)
        self.assertIsValidInterface(vTexture2d1)
        mdlFilePath: str = vTexture2d1.get_file_path()
        dbName: str  = vTexture2d1.get_value()
        self.assertNotNullOrEmpty(mdlFilePath)
        self.assertNotNullOrEmpty(dbName)
        tTexture2d: pymdlsdk.IType_texture = self.tf.create_texture(pymdlsdk.IType_texture.Shape.TS_2D)
        vTexture2d2: pymdlsdk.IValue_texture = self.vf.create_texture(tTexture2d, None)
        self.assertIsValidInterface(vTexture2d2)
        self.assertIsNone(vTexture2d2.get_file_path())
        self.assertIsNone(vTexture2d2.get_value())
        self.assertZero(vTexture2d2.set_value(dbName))
        self.assertEqual(mdlFilePath, vTexture2d2.get_file_path())
        self.assertEqual(dbName, vTexture2d2.get_value())
        self.assertEqual(vTexture2d2.get_owner_module(), '')  # None unless from a weak relative import in MDL 1.5 or older
        self.assertEqual(vTexture2d2.get_selector(), None)
        self.assertAlmostEqual(vTexture2d2.get_gamma(), 2.2)

        vLP1: pymdlsdk.IValue_light_profile = self.sdk.mdlFactory.create_light_profile(self.sdk.transaction, "/nvidia/sdk_examples/resources/example_modules.ies", False, None)
        self.assertIsValidInterface(vLP1)
        mdlFilePath: str = vLP1.get_file_path()
        dbName: str  = vLP1.get_value()
        self.assertNotNullOrEmpty(mdlFilePath)
        self.assertNotNullOrEmpty(dbName)
        vLP2: pymdlsdk.IValue_light_profile = self.vf.create_light_profile(None)
        self.assertIsValidInterface(vLP2)
        self.assertIsNone(vLP2.get_file_path())
        self.assertIsNone(vLP2.get_value())
        self.assertZero(vLP2.set_value(dbName))
        self.assertEqual(mdlFilePath, vLP2.get_file_path())
        self.assertEqual(dbName, vLP2.get_value())
        self.assertEqual(vLP2.get_owner_module(), '')

        vMbsdf1: pymdlsdk.IValue_bsdf_measurement = self.sdk.mdlFactory.create_bsdf_measurement(self.sdk.transaction, "/nvidia/sdk_examples/resources/example_modules.mbsdf", False, None)
        self.assertIsValidInterface(vMbsdf1)
        mdlFilePath: str = vMbsdf1.get_file_path()
        dbName: str  = vMbsdf1.get_value()
        self.assertNotNullOrEmpty(mdlFilePath)
        self.assertNotNullOrEmpty(dbName)
        vMbsdf2: pymdlsdk.IValue_bsdf_measurement = self.vf.create_bsdf_measurement(None)
        self.assertIsValidInterface(vMbsdf2)
        self.assertIsNone(vMbsdf2.get_file_path())
        self.assertIsNone(vMbsdf2.get_value())
        self.assertZero(vMbsdf2.set_value(dbName))
        self.assertEqual(mdlFilePath, vMbsdf2.get_file_path())
        self.assertEqual(dbName, vMbsdf2.get_value())
        self.assertEqual(vMbsdf2.get_owner_module(), '')
        bsdfMeasurement: pymdlsdk.IBsdf_measurement = self.sdk.transaction.edit_as(pymdlsdk.IBsdf_measurement, vMbsdf1.get_value())
        self.assertIsValidInterface(bsdfMeasurement)
        self.assertEqual(bsdfMeasurement.get_element_type(), pymdlsdk.Element_type.ELEMENT_TYPE_BSDF_MEASUREMENT)
        filename: str = bsdfMeasurement.get_filename()
        self.assertTrue(filename.endswith("example_modules.mbsdf"))
        self.assertIsNone(bsdfMeasurement.get_original_filename())  # because we did not use reset

        self.assertZero(bsdfMeasurement.reset_file(filename))
        cfg: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        er: pymdlsdk.IMdl_entity_resolver = cfg.get_entity_resolver()
        resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(mdlFilePath, "", "", 0, 0, None)
        self.assertIsValidInterface(resource)
        resourceElement: pymdlsdk.IMdl_resolved_resource_element = resource.get_element(0)
        mbsdfReader: pymdlsdk.IReader = resourceElement.create_reader(0)
        self.assertIsValidInterface(mbsdfReader)
        self.assertZero(bsdfMeasurement.reset_reader(mbsdfReader))

        refl: pymdlsdk.IInterface = bsdfMeasurement.get_reflection()
        trans: pymdlsdk.IInterface = bsdfMeasurement.get_transmission()
        self.assertIsNotValidInterface(trans)  # mbsdf file does not contain transmission
        reflIso: pymdlsdk.IBsdf_isotropic_data = refl.get_interface(pymdlsdk.IBsdf_isotropic_data)
        transIso: pymdlsdk.IBsdf_isotropic_data = trans.get_interface(pymdlsdk.IBsdf_isotropic_data)
        self.assertIsNotValidInterface(transIso)  # still invalid after casting
        self.assertIsValidInterface(reflIso)
        self.assertZero(bsdfMeasurement.set_reflection(reflIso))
        self.assertZero(bsdfMeasurement.set_transmission(reflIso))

        self.assertEqual(reflIso.get_resolution_phi(), 90)
        self.assertEqual(reflIso.get_resolution_theta(), 45)
        self.assertEqual(reflIso.get_type(), pymdlsdk.Bsdf_type.BSDF_RGB)
        buffer: pymdlsdk.IBsdf_buffer = reflIso.get_bsdf_buffer()
        self.assertIsValidInterface(buffer)

        vResource: pymdlsdk.IValue_resource = vMbsdf1.get_interface(pymdlsdk.IValue_resource)
        self.assertIsValidInterface(vResource)
        self.assertEqual(vResource.get_kind(), pymdlsdk.IValue.Kind.VK_BSDF_MEASUREMENT)
        self.assertEqual(vResource.get_file_path(), vMbsdf2.get_file_path())
        self.assertEqual(vResource.get_value(), vMbsdf2.get_value())
        self.assertEqual(vResource.get_owner_module(), vMbsdf2.get_owner_module())
        self.assertEqual(vResource.get_type().get_iid(), vMbsdf2.get_type().get_iid())
        self.assertZero(vResource.set_value(None))
        self.assertIsNone(vResource.get_file_path())
        self.assertIsNone(vResource.get_value())

    def test_IMessage(self):
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        context.add_message(pymdlsdk.IMessage.Kind.MSG_COMILER_CORE, pymdlsdk.Message_severity.MESSAGE_SEVERITY_WARNING, 23, "this is a warning")
        context.add_message(pymdlsdk.IMessage.Kind.MSG_COMILER_CORE, pymdlsdk.Message_severity.MESSAGE_SEVERITY_ERROR, 42, "this is an error")
        self.assertEqual(context.get_messages_count(), 2)
        self.assertEqual(context.get_error_messages_count(), 1)

        err_msg: pymdlsdk.IMessage = context.get_error_message(0)
        self.assertIsValidInterface(err_msg)
        self.assertEqual(err_msg.get_kind(), pymdlsdk.IMessage.Kind.MSG_COMILER_CORE)
        self.assertEqual(err_msg.get_severity(), pymdlsdk.Message_severity.MESSAGE_SEVERITY_ERROR)
        self.assertEqual(err_msg.get_code(), 42)

        warn_msg: pymdlsdk.IMessage = context.get_message(0)
        self.assertIsValidInterface(warn_msg)
        self.assertEqual(warn_msg.get_kind(), pymdlsdk.IMessage.Kind.MSG_COMILER_CORE)
        self.assertEqual(warn_msg.get_severity(), pymdlsdk.Message_severity.MESSAGE_SEVERITY_WARNING)

        self.assertZero(warn_msg.get_notes_count())
        note: pymdlsdk.IMessage = warn_msg.get_note(42)
        self.assertIsNotValidInterface(note)

        context.clear_messages()
        self.assertEqual(context.get_error_messages_count(), 0)

    def test_expressions(self):
        # Test some expression kinds that are not covered by the general tests.
        list: pymdlsdk.IExpression_list = self.ef.create_expression_list()
        self.assertIsValidInterface(list)
        int_t: pymdlsdk.IType = self.tf.create_int()
        self.assertIsValidInterface(int_t)
        param: pymdlsdk.IExpression = self.ef.create_parameter(int_t, 0)
        self.assertIsValidInterface(param)
        self.assertEqual(param.get_index(), 0)
        param.set_index(1)
        self.assertEqual(param.get_index(), 1)

        temp: pymdlsdk.IExpression = self.ef.create_temporary(int_t, 1)
        self.assertIsValidInterface(temp)
        self.assertEqual(temp.get_index(), 1)
        temp.set_index(0)
        self.assertEqual(temp.get_index(), 0)

        list.add_expression("exp1", param)
        list.add_expression("exp2", temp)
        self.run_expression_list_test(list)

        self.assertTrue(self.ef.compare(param, temp) < 0)
        self.assertTrue(self.ef.compare(temp, param) > 0)
        self.assertTrue(self.ef.compare(temp, temp) == 0)
        tempDump: str = self.ef.dump(temp, "name").get_c_str()
        self.assertTrue(len(tempDump) > 0)
        paramDump: str = self.ef.dump(param, "name").get_c_str()
        self.assertTrue(len(paramDump) > 0)

        cloned_list: pymdlsdk.IExpression_list = self.ef.clone(list)
        self.assertIsValidInterface(cloned_list)
        cloned_param: pymdlsdk.IExpression = self.ef.clone(param)
        self.assertIsValidInterface(cloned_param)

        to_cast: pymdlsdk.IExpression = self.ef.create_constant(self.ef.get_value_factory().create_int(42))
        self.assertIsValidInterface(to_cast)
        cast_call, ret_code = self.ef.create_cast_with_ret(to_cast, int_t, None, False, False)
        self.assertZero(ret_code)
        self.assertIsValidInterface(cast_call)
        returnCode: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        cast_call = self.ef.create_cast(to_cast, int_t, None, False, False, returnCode)
        self.assertZero(returnCode.value)
        self.assertIsValidInterface(cast_call)

        to_cast.set_value(self.ef.get_value_factory().create_int(11))
        val: pymdlsdk.IValue_int = to_cast.get_value().get_interface(pymdlsdk.IValue_int)
        self.assertEqual(val.get_value(), 11)
        val = to_cast.get_value_as(pymdlsdk.IValue_int)
        self.assertEqual(val.get_value(), 11)

    def test_decl_cast(self):
        moduleQualifier = "::nvidia::sdk_examples::tutorials_aov"
        moduleDbName = self.load_module(moduleQualifier)
        mdlModule: pymdlsdk.IModule = self.sdk.transaction.access_as(pymdlsdk.IModule, moduleDbName)
        self.assertIsValidInterface(mdlModule)

        types: pymdlsdk.IType_list = mdlModule.get_types()
        self.assertIsValidInterface(types)
        sty: pymdlsdk.IType_structure
        for i in range(types.get_size()):
            ty: pymdlsdk.IType = types.get_type(i)
            self.assertIsValidInterface(ty)
            sty = ty.get_interface(pymdlsdk.IType_structure)
            if sty.is_valid_interface():
                name = sty.get_symbol()
                break
        self.assertIsValidInterface(sty)

        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        returnCode: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        srcExpr: pymdlsdk.IExpression = self.ef.create_constant(self.ef.get_value_factory().create_int(23))
        expr: pymdlsdk.IExpression = self.ef.create_decl_cast(srcExpr, sty, None, True, True, returnCode)
        self.assertEqual(returnCode.value, -2) # cannot cast int to struct type

        matTy: pymdlsdk.IType = self.tf.create_struct("::material")
        self.assertIsValidInterface(matTy)
        matVal: pymdlsdk.IValue = self.ef.get_value_factory().create_struct(matTy)
        self.assertIsValidInterface(matVal)
        srcExpr = self.ef.create_constant(matVal)
        self.assertIsValidInterface(srcExpr)
        expr: pymdlsdk.IExpression = self.ef.create_decl_cast(srcExpr, sty, None, True, True, returnCode)
        self.assertEqual(returnCode.value, 0)
        pass

    def test_module_transformer(self):
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()

        moduleQualifier = "::nvidia::sdk_examples::tutorials"
        moduleDbName = self.load_module(moduleQualifier)
        transformer: pymdlsdk.IMdl_module_transfomer = self.sdk.mdlFactory.create_module_transformer(self.sdk.transaction,
            moduleDbName, context)
        self.assertContextNoErrors(context)
        self.assertIsValidInterface(transformer)

        result: int = transformer.upgrade_mdl_version(pymdlsdk.Mdl_version.MDL_VERSION_LATEST, context)
        self.assertContextNoErrors(context)
        self.assertZero(result)

        result = transformer.use_absolute_import_declarations(None, None, context)
        self.assertContextNoErrors(context)
        self.assertZero(result)

        result = transformer.use_relative_import_declarations(None, None, context)
        self.assertContextNoErrors(context)
        self.assertZero(result)

        result = transformer.use_absolute_resource_file_paths(None, None, context)
        self.assertContextNoErrors(context)
        self.assertZero(result)

        result = transformer.use_relative_resource_file_paths(None, ".*tiled_resource.*", context)
        self.assertContextNoErrors(context)
        self.assertZero(result)

        result = transformer.inline_imported_modules(None, None, False, context)
        self.assertContextNoErrors(context)
        self.assertZero(result)

        result = transformer.export_module("outmod.mdl", context)
        self.assertContextNoErrors(context)
        self.assertZero(result)

        out_mod: pymdlsdk.IString = self.sdk.transaction.create_as(pymdlsdk.IString, "String")
        self.assertIsValidInterface(out_mod)
        result = transformer.export_module_to_string(out_mod, context)
        self.assertContextNoErrors(context)
        self.assertZero(result)
        content: str = out_mod.get_c_str()
        self.assertNotNullOrEmpty(content)

    def test_serialized_names(self):
        fc = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, "mdl::bool(int)")
        self.assertIsValidInterface(fc)
        orig_argument_types = fc.get_parameter_types()
        self.assertIsValidInterface(orig_argument_types)
        context = self.sdk.mdlFactory.create_execution_context()

        sfn = self.imp_exp.serialize_function_name(
            "mdl::bool(int)", None, None, context)
        self.assertContextNoErrors(context)
        self.assertIsValidInterface(sfn)
        self.assertEqual(sfn.get_function_name(), "mdl::bool(int)")
        self.assertEqual(sfn.get_module_name(), "mdl::%3Cbuiltins%3E")
        self.assertEqual(sfn.get_function_name_without_module_name(), "bool(int)")

        dfn = self.imp_exp.deserialize_function_name(
            self.sdk.transaction, sfn.get_function_name(), context)
        self.assertContextNoErrors(context)
        self.assertIsValidInterface(dfn)
        self.assertEqual(dfn.get_db_name(), "mdl::bool(int)")
        argument_types = dfn.get_argument_types()
        self.assertIsValidInterface(argument_types)
        self.assertEqual(self.tf.compare(argument_types, orig_argument_types), 0)

        dfn2 = self.imp_exp.deserialize_function_name(
            self.sdk.transaction,
            sfn.get_module_name(),
            sfn.get_function_name_without_module_name(),
            context)
        self.assertContextNoErrors(context)
        self.assertIsValidInterface(dfn2)
        self.assertEqual(dfn2.get_db_name(), "mdl::bool(int)")
        argument_types2 = dfn2.get_argument_types()
        self.assertIsValidInterface(argument_types2)
        self.assertEqual(self.tf.compare(argument_types2, orig_argument_types), 0)

        dmn = self.imp_exp.deserialize_module_name(
            sfn.get_module_name(), context)
        self.assertContextNoErrors(context)
        self.assertIsValidInterface(dmn)
        self.assertEqual(dmn.get_db_name(), "mdl::%3Cbuiltins%3E")
        self.assertEqual(dmn.get_load_module_argument(), "::%3Cbuiltins%3E")

    def test_imdl_factory(self):
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        self.assertIsValidInterface(context)
        context2: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.clone(context)
        self.assertIsValidInterface(context2)

        iname: pymdlsdk.IString = self.sdk.mdlFactory.decode_name("::mdl::%3Cbuiltins%3E")
        self.assertEqual(iname.get_c_str(), "::mdl::<builtins>")

        encModName: pymdlsdk.IString = self.sdk.mdlFactory.encode_module_name("::mdl::<builtins>")
        self.assertEqual(encModName.get_c_str(), "::mdl::%3Cbuiltins%3E")
        
        encTypeName: pymdlsdk.IString = self.sdk.mdlFactory.encode_type_name("bool")
        self.assertEqual(encTypeName.get_c_str(), "bool")

        pTypes: pymdlsdk.IArray = self.sdk.transaction.create_as(pymdlsdk.IArray, "Float32[0]")
        encFuncName: pymdlsdk.IString = self.sdk.mdlFactory.encode_function_definition_name("func", pTypes)
        self.assertEqual(encFuncName.get_c_str(), "func()")

        self.assertTrue(self.sdk.mdlFactory.is_valid_mdl_identifier("test"))

# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
