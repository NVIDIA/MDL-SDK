import unittest
import pymdlsdk
import pymdl
from setup import SDK

# Purpose is to perform a test coverage of the API for distilling and baking materials

class Main(unittest.TestCase):
    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=True, loadImagePlugins=True, loadDistillerPlugin=True)

    @classmethod
    def tearDownClass(self):
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def compileMaterialInstance(self, functionCall: pymdlsdk.IFunction_call, classCompilation: bool) -> pymdlsdk.ICompiled_material:
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        if classCompilation:
            flags = pymdlsdk._pymdlsdk._IMaterial_instance_CLASS_COMPILATION
        else:
            flags = pymdlsdk._pymdlsdk._IMaterial_instance_DEFAULT_OPTIONS
        materialInstance: pymdlsdk.IMaterial_instance = functionCall.get_interface(pymdlsdk.IMaterial_instance)
        self.assertIsNotNone(materialInstance)
        self.assertTrue(materialInstance.is_valid_interface())
        compiledMaterial: pymdlsdk.ICompiled_material = materialInstance.create_compiled_material(flags, context)
        self.assertIsNotNone(compiledMaterial)
        self.assertTrue(compiledMaterial.is_valid_interface())
        return compiledMaterial

    def distillMaterial(self, compiledMaterial: pymdlsdk.ICompiled_material, targetModel: str) -> pymdlsdk.ICompiled_material:
        distillingApi: pymdlsdk.IMdl_distiller_api = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_distiller_api)
        self.assertIsNotNone(distillingApi)
        self.assertTrue(distillingApi.is_valid_interface())
        (distilledMaterial, res) = distillingApi.distill_material_with_ret(compiledMaterial, targetModel, None)
        self.assertEqual(res, 0)
        self.assertIsNotNone(distilledMaterial)
        self.assertTrue(distilledMaterial.is_valid_interface())
        return distilledMaterial

    def test_setupIsDone(self):
        self.assertIsNotNone(self.sdk)
        self.assertIsNotNone(self.sdk.neuray)
        self.assertIsNotNone(self.sdk.transaction)

    def _distillAndBake(self, qualifiedModuleName: str, simpleFunctionName: str, targetModel: str, classCompilation: bool):
        qualifiedFunctionName: str = qualifiedModuleName + simpleFunctionName
        # load and fetch
        moduleDbName: str = self.sdk.load_module(qualifiedModuleName)
        self.assertNotEqual(moduleDbName, "")
        functionDbName: pymdlsdk.IString = self.sdk.mdlFactory.get_db_definition_name(qualifiedFunctionName)
        fct_definition: pymdlsdk.IFunction_definition = self.sdk.transaction.access_as(pymdlsdk.IFunction_definition, functionDbName.get_c_str())
        self.assertTrue(fct_definition.is_valid_interface())
        # create instance (call)
        functionCall: pymdlsdk.IFunction_call
        (functionCall, ret) = fct_definition.create_function_call_with_ret(None)
        self.assertEqual(ret, 0)
        self.assertIsNotNone(functionCall)
        self.assertTrue(functionCall.is_valid_interface())
        # compile instance
        compiledMaterial: pymdlsdk.ICompiled_material = self.compileMaterialInstance(functionCall, classCompilation=classCompilation)
        # distill instance
        distilledMaterial: pymdlsdk.ICompiled_material = self.distillMaterial(compiledMaterial, targetModel=targetModel)
        self.assertTrue(distilledMaterial.is_valid_interface())

    def test_distillAndBake_example_material(self):
        self._distillAndBake("::nvidia::sdk_examples::tutorials", "::example_material(color,float)", "transmissive_pbr", classCompilation=True)
        self._distillAndBake("::nvidia::sdk_examples::tutorials", "::example_material(color,float)", "transmissive_pbr", classCompilation=False)
        self._distillAndBake("::nvidia::sdk_examples::tutorials", "::example_material(color,float)", "ue4", classCompilation=True)
        self._distillAndBake("::nvidia::sdk_examples::tutorials", "::example_material(color,float)", "ue4", classCompilation=False)

# run all tests of this file
if __name__ == '__main__':
    unittest.main()