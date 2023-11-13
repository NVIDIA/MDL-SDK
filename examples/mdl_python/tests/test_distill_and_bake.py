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


# Purpose is to perform a test coverage of the API for distilling and baking materials

class Main(UnittestBase):
    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
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
        moduleDbName: str = self.load_module(qualifiedModuleName)
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

    def test_image_and_canvas(self):
        imageApi: pymdlsdk.IImage_api = self.sdk.neuray.get_api_component(pymdlsdk.IImage_api)
        self.assertIsValidInterface(imageApi)
        
        tile: pymdlsdk.ITile = imageApi.create_tile("Rgba", 16, 16)
        self.assertIsValidInterface(tile)
        self.assertEqual(tile.get_type(), "Rgba")
        self.assertEqual(tile.get_resolution_x(), 16)
        self.assertEqual(tile.get_resolution_y(), 16)
        pixelValue: pymdlsdk.Color_struct = tile.get_pixel(2, 3)
        self.assertEqual(pixelValue.r, 0)
        self.assertEqual(pixelValue.g, 0)
        self.assertEqual(pixelValue.b, 0)
        self.assertEqual(pixelValue.a, 0)
        pixelValue.r = 0.25
        pixelValue.g = 0.5
        pixelValue.b = 0.75
        pixelValue.a = 1.0
        tile.set_pixel(4, 5, pixelValue)
        pixelValue2: pymdlsdk.Color_struct = tile.get_pixel(4, 5)
        self.assertAlmostEqual(pixelValue2.r, 0.25, delta=0.01)  # 8bit
        self.assertAlmostEqual(pixelValue2.g, 0.5, delta=0.01)
        self.assertAlmostEqual(pixelValue2.b, 0.75, delta=0.01)
        self.assertAlmostEqual(pixelValue2.a, 1.0, delta=0.01)

        tile: pymdlsdk.ITile = imageApi.create_tile("Float32", 13, 17)
        self.assertIsValidInterface(tile)
        self.assertEqual(tile.get_type(), "Float32")
        self.assertEqual(tile.get_resolution_x(), 13)
        self.assertEqual(tile.get_resolution_y(), 17)
        pixelValue: pymdlsdk.Color_struct = tile.get_pixel(0, 1)
        pixelValue.r = 0.25
        tile.set_pixel(0, 1, pixelValue)
        pixelValue2: pymdlsdk.Color_struct = tile.get_pixel(0, 1)
        self.assertEqual(pixelValue2.r, 0.25)  # 32bit float
        self.assertEqual(pixelValue2.g, 0.25)  # same as r
        self.assertEqual(pixelValue2.b, 0.25)  # same as r
        self.assertEqual(pixelValue2.a, 1.0)  # alpha defaults to 1.0

        tile.set_pixel(1337, 1338, pixelValue)
        pixelValue3: pymdlsdk.Color_struct = tile.get_pixel(1337, 1338)
        self.assertIsNotNone(pixelValue3)  # returns garbage, but not None


# run all tests of this file
if __name__ == '__main__':  # pragma: no cover
    unittest.main()
