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
class MainDistillAndBake(UnittestBase):
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
        res: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        distilledMaterial = distillingApi.distill_material(compiledMaterial, targetModel)
        distilledMaterial = distillingApi.distill_material(compiledMaterial, targetModel, None)
        distilledMaterial = distillingApi.distill_material(compiledMaterial, targetModel, None, res)
        distilledMaterial = distillingApi.distill_material(compiledMaterial, targetModel, errors=res)
        self.assertEqual(res.value, 0)
        self.assertIsNotNone(distilledMaterial)
        self.assertTrue(distilledMaterial.is_valid_interface())
        
        # Note: the following function is deprecated and this test will be removed when
        # the function is not available anymore.
        distilledMaterial_ignored = self.assertDeprecationWarning(
            lambda: distillingApi.distill_material_with_ret(compiledMaterial, targetModel))
        
        return distilledMaterial

    def bakeMaterial(self, compiledMaterial: pymdlsdk.ICompiled_material):
        # not baking for now but at least create the bakers
        distillingApi: pymdlsdk.IMdl_distiller_api = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_distiller_api)
        baker0: pymdlsdk.IBaker = distillingApi.create_baker(compiledMaterial, "geometry.normal")
        self.assertIsValidInterface(baker0)

        pt: str = baker0.get_pixel_type()
        self.assertEqual(pt, "Float32<3>")
        self.assertTrue(baker0.is_uniform())

        imageApi: pymdlsdk.IImage_api = self.sdk.neuray.get_api_component(pymdlsdk.IImage_api)
        self.assertIsValidInterface(imageApi)
        canvas: pymdlsdk.ICanvas = imageApi.create_canvas(pt, 16, 16)
        self.assertIsValidInterface(canvas)

        self.assertZero(self.assertDeprecationWarning(lambda: baker0.bake_texture(canvas)))
        self.assertZero(self.assertDeprecationWarning(lambda: baker0.bake_texture(canvas, 2)))  # try the other overload too
        self.assertZero(self.assertDeprecationWarning(lambda: baker0.bake_texture(canvas, samples=2)))  # and named
        tile: pymdlsdk.ITile = canvas.get_tile()
        self.assertIsValidInterface(tile)
        pixelValue: pymdlsdk.Color_struct = tile.get_pixel(2, 3)
        self.assertEqual(pixelValue.r, 0)
        self.assertEqual(pixelValue.g, 0)
        self.assertEqual(pixelValue.b, 1)
        self.assertEqual(pixelValue.a, 1)

        canvas2: pymdlsdk.ICanvas = imageApi.create_canvas(pt, 16, 16)
        res: int = baker0.bake_texture(canvas2, 0.0, 1.0, 0.0, 1.0, 1)
        self.assertZero(res)
        tile: pymdlsdk.ITile = canvas2.get_tile()
        self.assertIsValidInterface(tile)
        pixelValue: pymdlsdk.Color_struct = tile.get_pixel(2, 3)
        self.assertEqual(pixelValue.r, 0)
        self.assertEqual(pixelValue.g, 0)
        self.assertEqual(pixelValue.b, 1)
        self.assertEqual(pixelValue.a, 1)


        # FIXME: The following results in
        # TypeError: Wrong number or type of arguments for overloaded function 'IBaker_bake_constant'.
        # Possible C/C++ prototypes are:
        #   mi::neuraylib::IBaker::bake_constant(mi::IData *,mi::Uint32) const
        #   mi::neuraylib::IBaker::bake_constant(mi::IData *) const
        # Why is no Color variant overload available?
        # a: pymdlsdk.IColor = self.sdk.transaction.create_as(pymdlsdk.IColor, "Color")
        # self.assertIsValidInterface(a)
        # value: pymdlsdk.Color_struct = a.get_value()
        # value.r = 0.25
        # value.g = 0.5
        # value.b = 0.75
        # value.a = 1.0
        # a.set_value(value)
        # canvas: pymdlsdk.ICanvas = imageApi.create_canvas("Float32<3>", 16, 16)
        # self.assertIsValidInterface(canvas)
        # res: int = baker0.bake_constant(canvas, a)
        # self.assertZero(res)
        # tile: pymdlsdk.ITile = canvas.get_tile()
        # self.assertIsValidInterface(tile)
        # pixelValue: pymdlsdk.Color_struct = tile.get_pixel(2, 3)
        # self.assertEqual(pixelValue.r, 0.25)
        # self.assertEqual(pixelValue.g, 0.5)
        # self.assertEqual(pixelValue.b, 0.75)
        # self.assertEqual(pixelValue.a, 1)

        baker1: pymdlsdk.IBaker = distillingApi.create_baker(compiledMaterial, "geometry.normal", pymdlsdk.Baker_resource.BAKE_ON_CPU)
        self.assertIsValidInterface(baker1)
        baker2: pymdlsdk.IBaker = distillingApi.create_baker(compiledMaterial, "geometry.normal", pymdlsdk.Baker_resource.BAKE_ON_CPU, 0)
        self.assertIsValidInterface(baker2)
        baker3: pymdlsdk.IBaker = distillingApi.create_baker(compiledMaterial, "thin_walled", pymdlsdk.Baker_resource.BAKE_ON_CPU)
        self.assertIsValidInterface(baker3)
        this_walled_baked: pymdlsdk.IBoolean = self.sdk.transaction.create_as(pymdlsdk.IBoolean, "Boolean")
        self.assertZero(baker3.bake_constant(this_walled_baked))
        self.assertFalse(this_walled_baked.get_value())


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
        res: pymdlsdk.ReturnCode = pymdlsdk.ReturnCode()
        functionCall = fct_definition.create_function_call(None)
        functionCall = fct_definition.create_function_call(None, res)
        self.assertEqual(res.value, 0)
        self.assertIsNotNone(functionCall)
        self.assertTrue(functionCall.is_valid_interface())
        # compile instance
        compiledMaterial: pymdlsdk.ICompiled_material = self.compileMaterialInstance(functionCall, classCompilation=classCompilation)
        # distill instance
        distilledMaterial: pymdlsdk.ICompiled_material = self.distillMaterial(compiledMaterial, targetModel=targetModel)
        self.assertTrue(distilledMaterial.is_valid_interface())
        # create baker
        self.bakeMaterial(compiledMaterial)

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
        
        canvas: pymdlsdk.ICanvas = imageApi.create_canvas("Rgba", 16, 16)
        self.assertIsValidInterface(canvas)
        mdlImpExpApi: pymdlsdk.IMdl_impexp_api = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_impexp_api)
        self.assertIsValidInterface(mdlImpExpApi)
        tile: pymdlsdk.ITile = canvas.get_tile()
        pixelValue.r = 0.25
        pixelValue.g = 0.5
        pixelValue.b = 0.75
        pixelValue.a = 1.0
        for x in range(16):
            for y in range(8):
                tile.set_pixel(x, y, pixelValue)
        pixelValue.b = 0.0
        for x in range(16):
            for y in range(8, 16):
                tile.set_pixel(x, y, pixelValue)

        export_options: pymdlsdk.IMap = self.sdk.transaction.create_as(pymdlsdk.IMap, "Map<Interface>")
        quality: pymdlsdk.IUint32 = self.sdk.transaction.create_as(pymdlsdk.IUint32, "Uint32")
        quality.set_value(100)
        export_options.insert('jpg:quality', quality)
        force_default_gamma: pymdlsdk.IBoolean = self.sdk.transaction.create_as(pymdlsdk.IBoolean, "Boolean")
        force_default_gamma.set_value(True)
        export_options.insert('force_default_gamma', force_default_gamma)

        # current, correct signatures
        self.assertZero(mdlImpExpApi.export_canvas("canvas_test.png", canvas, export_options))
        self.assertZero(mdlImpExpApi.export_canvas("canvas_test1.png", canvas, None))
        self.assertZero(mdlImpExpApi.export_canvas("canvas_test2.png", canvas))  # no optional

        # deprecated signatures
        self.assertEqual(self.assertDeprecationWarning(lambda: mdlImpExpApi.export_canvas("canvas_test3.png", canvas, 99)), 0)
        self.assertEqual(self.assertDeprecationWarning(lambda: mdlImpExpApi.export_canvas("canvas_test4.png", canvas, quality=96)), 0)
        self.assertEqual(self.assertDeprecationWarning(lambda: mdlImpExpApi.export_canvas("canvas_test5.png", canvas, 98, True)), 0)
        self.assertEqual(self.assertDeprecationWarning(lambda: mdlImpExpApi.export_canvas("canvas_test6.png", canvas, 97, force_default_gamma=True)), 0)
        self.assertEqual(self.assertDeprecationWarning(lambda: mdlImpExpApi.export_canvas("canvas_test7.png", canvas, force_default_gamma=True)), 0)
        self.assertEqual(self.assertDeprecationWarning(lambda: mdlImpExpApi.export_canvas("canvas_test8.png", canvas, force_default_gamma=True, quality=95)), 0)
        self.assertEqual(self.assertDeprecationWarning(lambda: mdlImpExpApi.export_canvas("canvas_test9.png", quality=94, force_default_gamma=True, canvas=canvas)), 0)

        canvas: pymdlsdk.ICanvas = imageApi.create_canvas("Rgba", 16, 16)
        self.assertIsValidInterface(canvas)
        self.assertEqual(canvas.get_layers_size(), 1)
        pt: str = canvas.get_type()
        self.assertEqual(pt, "Rgba")
        gamma: float = canvas.get_gamma()
        self.assertAlmostEqual(gamma, 2.2)
        canvas.set_gamma(gamma - 0.2)
        self.assertAlmostEqual(canvas.get_gamma(), gamma - 0.2)

    def test_buffer(self):
        imageApi: pymdlsdk.IImage_api = self.sdk.neuray.get_api_component(pymdlsdk.IImage_api)
        self.assertIsValidInterface(imageApi)
        canvas: pymdlsdk.ICanvas = imageApi.create_canvas("Rgba", 16, 16)

        buffer: pymdlsdk.IBuffer = imageApi.create_buffer_from_canvas(canvas, "jpg", "Rgba")
        self.assertIsValidInterface(buffer)
        data_size: int = buffer.get_data_size()
        # data_size may be different for different plugin versions
        self.assertTrue(data_size > 0)

        canvas2: pymdlsdk.ICanvas = imageApi.create_canvas_from_buffer(buffer, "jpg")
        self.assertIsValidInterface(canvas2)
        pass

    def test_image_api(self):
        imageApi: pymdlsdk.IImage_api = self.sdk.neuray.get_api_component(pymdlsdk.IImage_api)
        self.assertIsValidInterface(imageApi)

        canvas: pymdlsdk.ICanvas = imageApi.create_canvas("Rgba", 16, 16)
        self.assertIsValidInterface(canvas)

        self.assertEqual(imageApi.get_components_per_pixel("Rgba"), 4)
        self.assertEqual(imageApi.get_bytes_per_component("Rgba"), 1)
        self.assertEqual(imageApi.get_pixel_type_for_channel("Rgba", "A"), "Sint8")

        mipmap: pymdlsdk.IArray = imageApi.create_mipmap(canvas)
        self.assertIsValidInterface(mipmap)

        channel: pymdlsdk.ITile = imageApi.extract_channel(canvas, "A")
        self.assertIsValidInterface(channel)
        
        canvas2: pymdlsdk.ICanvas = imageApi.clone_canvas(canvas)
        self.assertIsValidInterface(canvas2)
        canvas_base: pymdlsdk.ICanvas_base = canvas2.get_interface(pymdlsdk.ICanvas_base)
        self.assertIsValidInterface(canvas_base)
        self.assertEqual(canvas_base.get_resolution_x(), 16)
        self.assertEqual(canvas_base.get_resolution_y(), 16)
        self.assertEqual(canvas_base.get_layers_size(), 1)
        self.assertEqual(canvas_base.get_type(), "Rgba")
        orgGamma: float = canvas_base.get_gamma()
        canvas_base.set_gamma(4.0)
        self.assertAlmostEqual(canvas_base.get_gamma(), 4.0)
        canvas_base.set_gamma(orgGamma)
        self.assertAlmostEqual(canvas_base.get_gamma(), orgGamma)

        imageApi.adjust_gamma(canvas2, 1.0)
        self.assertAlmostEqual(canvas2.get_gamma(), 1.0)

        canvas3: pymdlsdk.ICanvas = imageApi.convert(canvas, "Rgb")
        self.assertIsValidInterface(canvas3)

        tile: pymdlsdk.ITile = imageApi.create_tile("Float32", 16, 16)
        self.assertIsValidInterface(tile)
        tile2: pymdlsdk.ITile = imageApi.clone_tile(tile)
        self.assertIsValidInterface(tile)

        self.assertTrue(imageApi.supports_format_for_decoding("jpg"))
        self.assertFalse(imageApi.supports_format_for_decoding("fantasy"))

        self.assertTrue(imageApi.supports_format_for_encoding("jpg"))
        self.assertFalse(imageApi.supports_format_for_encoding("fantasy"))

    def test_texture_image(self):
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        resourcePath: str = "/nvidia/sdk_examples/resources/metal_cast_iron_roughness.png"
        texVal: pymdlsdk.ITexture = self.sdk.mdlFactory.create_texture(self.sdk.transaction,
            resourcePath, pymdlsdk.IType_texture.Shape.TS_2D, 2.2, None, True, context)
        self.assertContextNoErrors(context)
        self.assertIsValidInterface(texVal)
        textureDbName: str = texVal.get_value()
        tex: pymdlsdk.ITexture = self.sdk.transaction.edit_as(pymdlsdk.ITexture, textureDbName)
        imgDbName: str = tex.get_image()
        self.assertNotNullOrEmpty(imgDbName)
        self.assertZero(tex.set_image(imgDbName))

        vol: str = tex.get_volume()
        self.assertEqual(vol, None)
        self.assertEqual(tex.set_volume("nonexistent"), -2)

        gamma: float = tex.get_gamma()
        tex.set_gamma(gamma + 0.2)
        self.assertAlmostEqual(tex.get_gamma(), gamma + 0.2)
        tex.set_gamma(gamma)
        self.assertAlmostEqual(tex.get_gamma(), gamma)

        effGamma: float = tex.get_effective_gamma(0, 0)
        self.assertAlmostEqual(effGamma, 2.2)

        self.assertEqual(tex.get_selector(), None)

        cmp: pymdlsdk.Texture_compression = tex.get_compression()
        tex.set_compression(pymdlsdk.Texture_compression.TEXTURE_MEDIUM_COMPRESSION)
        self.assertEqual(tex.get_compression(), pymdlsdk.Texture_compression.TEXTURE_MEDIUM_COMPRESSION)
        tex.set_compression(cmp)
        self.assertEqual(tex.get_compression(), cmp)

        et: pymdlsdk.Element_type = tex.get_element_type()
        self.assertEqual(et, pymdlsdk.Element_type.ELEMENT_TYPE_TEXTURE)

        image: pymdlsdk.IImage = self.sdk.transaction.access_as(pymdlsdk.IImage, imgDbName)
        self.assertIsValidInterface(image)
        self.assertFalse(image.is_animated())
        self.assertFalse(image.is_uvtile())
        self.assertIsNone(image.get_selector())
        self.assertEqual(image.get_type(0, 0), "Rgb")
        self.assertEqual(image.get_levels(0, 0), 12)
        self.assertEqual(image.resolution_x(0, 0, 0), 2048)
        self.assertEqual(image.resolution_y(0, 0, 0), 2048)
        self.assertEqual(image.resolution_z(0, 0, 0), 1)
        self.assertEqual(image.get_element_type(), pymdlsdk.Element_type.ELEMENT_TYPE_IMAGE)
        fn: str = image.get_filename(0, 0)
        self.assertNotNullOrEmpty(fn)
        ofn: str = image.get_original_filename()
        self.assertIsNone(ofn)

        image: pymdlsdk.IImage = self.sdk.transaction.edit_as(pymdlsdk.IImage, imgDbName)
        res: int = image.reset_file(fn)
        self.assertZero(res)
        fn: str = image.get_filename(0, 0)
        self.assertNotNullOrEmpty(fn)
        ofn = image.get_original_filename()
        self.assertNotNullOrEmpty(ofn)

        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        self.assertIsValidInterface(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsValidInterface(er)

        resource_path = "/nvidia/sdk_examples/resources/metal_cast_iron_mask.png"
        resolved_resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(
            file_path=resourcePath, owner_file_path=None, owner_name=None, pos_line=0, pos_column=0, context=None)
        self.assertIsValidInterface(resolved_resource)
        resolved_resource_elem: pymdlsdk.IMdl_resolved_resource_element = resolved_resource.get_element(0)
        self.assertIsValidInterface(resolved_resource_elem)
        reader: pymdlsdk.IReader = resolved_resource_elem.create_reader(0)
        res = image.reset_reader(reader, "png")
        self.assertZero(res)
        fn: str = image.get_filename(0, 0)
        self.assertIsNone(fn)
        ofn = image.get_original_filename()
        self.assertIsNone(ofn)

    def test_light_profile(self):
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        resourcePath: str = "/nvidia/sdk_examples/resources/example_modules.ies"
        lightProfileVal: pymdlsdk.IValue_Lightprofile = self.sdk.mdlFactory.create_light_profile(self.sdk.transaction,
            resourcePath, False, None)
        self.assertContextNoErrors(context)
        self.assertIsValidInterface(lightProfileVal)
        lightDbName: str = lightProfileVal.get_value()
        lightProfile: pymdlsdk.ILightprofile = self.sdk.transaction.edit_as(pymdlsdk.ILightprofile, lightDbName)
        self.assertIsValidInterface(lightProfile)

        et: pymdlsdk.Element_type = lightProfile.get_element_type()
        self.assertEqual(et, pymdlsdk.Element_type.ELEMENT_TYPE_LIGHTPROFILE)

        fn: str = lightProfile.get_filename()
        self.assertNotNullOrEmpty(fn)
        ofn: str = lightProfile.get_original_filename()
        self.assertIsNone(ofn)

        self.assertEqual(lightProfile.get_resolution_phi(), 2)
        self.assertEqual(lightProfile.get_resolution_theta(), 19)

        degree: pymdlsdk.Lightprofile_degree = lightProfile.get_degree()
        self.assertEqual(degree, pymdlsdk.Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1)

        flags: pymdlsdk.Lightprofile_degree = lightProfile.get_flags()
        self.assertEqual(flags, pymdlsdk.Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE)

        self.assertAlmostEqual(lightProfile.get_phi(0), 0.0)
        self.assertAlmostEqual(lightProfile.get_phi(2), 0.0)
        self.assertAlmostEqual(lightProfile.get_theta(0), 0.0)
        self.assertAlmostEqual(lightProfile.get_theta(2), 0.34906587)
        self.assertAlmostEqual(lightProfile.get_candela_multiplier(), 10.0)

        d = lightProfile.get_data()
        self.assertIsNotNone(d)

        s: float = lightProfile.sample(0.1, 0.2, False)
        self.assertAlmostEqual(s, 0.0)

        lightProfile: pymdlsdk.IImage = self.sdk.transaction.edit_as(pymdlsdk.ILightprofile, lightDbName)
        res: int = lightProfile.reset_file(fn)
        self.assertZero(res)
        fn: str = lightProfile.get_filename()
        self.assertNotNullOrEmpty(fn)
        ofn = lightProfile.get_original_filename()
        self.assertNotNullOrEmpty(ofn)
        res = lightProfile.reset_file(fn, 0, 0, pymdlsdk.Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1,
            pymdlsdk.Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE)
        self.assertZero(res)
        res = lightProfile.reset_file(fn, 0, 0, pymdlsdk.Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1)
        self.assertZero(res)
        res = lightProfile.reset_file(fn, 0, 0)
        self.assertZero(res)
        res = lightProfile.reset_file(fn, 0)
        self.assertZero(res)

        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        self.assertIsValidInterface(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsValidInterface(er)

        resolved_resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(
            file_path=resourcePath, owner_file_path=None, owner_name=None, pos_line=0, pos_column=0, context=None)
        self.assertIsValidInterface(resolved_resource)
        resolved_resource_elem: pymdlsdk.IMdl_resolved_resource_element = resolved_resource.get_element(0)
        self.assertIsValidInterface(resolved_resource_elem)
        reader: pymdlsdk.IReader = resolved_resource_elem.create_reader(0)
        res = lightProfile.reset_reader(reader)
        self.assertZero(res)
        fn: str = lightProfile.get_filename()
        self.assertIsNone(fn)
        ofn = lightProfile.get_original_filename()
        self.assertIsNone(ofn)
        reader = resolved_resource_elem.create_reader(0)
        res = lightProfile.reset_reader(reader, 0, 0, pymdlsdk.Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1,
            pymdlsdk.Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE)
        self.assertZero(res)
        reader = resolved_resource_elem.create_reader(0)
        res = lightProfile.reset_reader(reader, 0, 0, pymdlsdk.Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1)
        self.assertZero(res)
        reader = resolved_resource_elem.create_reader(0)
        res = lightProfile.reset_reader(reader, 0, 0)
        self.assertZero(res)
        reader = resolved_resource_elem.create_reader(0)
        res = lightProfile.reset_reader(reader, 0)
        self.assertZero(res)

    def test_distiller_api(self):
        distillingApi: pymdlsdk.IMdl_distiller_api = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_distiller_api)
        self.assertIsValidInterface(distillingApi)
        target_count: int = distillingApi.get_target_count()
        self.assertTrue(target_count > 0)
        for i in range(target_count):
            target_name: str = distillingApi.get_target_name(i)
            self.assertNotNullOrEmpty(target_name)
            # Note: we do not have a distilling target that makes use of required modules.
            req_mod_count: int = distillingApi.get_required_module_count(target_name)
            # until then:
            self.assertZero(req_mod_count)
            req_mod_name: str = distillingApi.get_required_module_name(target_name, 42)
            self.assertIsNone(req_mod_name)
            req_mod_code: str = distillingApi.get_required_module_code(target_name, 42)
            self.assertIsNone(req_mod_code)
            # self.assertTrue(req_mod_count >= 0)
            # for j in range(req_mod_count):
            #     req_mod_name: str = distillingApi.get_required_module_count(target_name, j)
            #     self.assertNotNullOrEmpty(req_mod_name)
            #     req_mod_code: str = distillingApi.get_required_module_code(target_name, j)
            #     self.assertNotNullOrEmpty(req_mod_code)

    def test_impexp_api(self):
        mdlImpExpApi: pymdlsdk.IMdl_impexp_api = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_impexp_api)
        self.assertIsValidInterface(mdlImpExpApi)
        context: pymdlsdk.IMdl_execution_context = self.sdk.mdlFactory.create_execution_context()
        module_source: str = "mdl 1.8; export const int PI = 3;"
        res: int = mdlImpExpApi.load_module_from_string(self.sdk.transaction, "::mod1", module_source, context)
        self.assertTrue(res == 0 or res == -1)
        self.assertContextNoErrors(context)

        dbModName: pymdlsdk.IString = self.sdk.mdlFactory.get_db_module_name("::mod1")
        expSrc: pymdlsdk.IString = self.sdk.transaction.create_as(pymdlsdk.IString, "String")
        self.assertIsValidInterface(expSrc)
        res = mdlImpExpApi.export_module_to_string(self.sdk.transaction, dbModName.get_c_str(), expSrc, context)
        self.assertZero(res)
        self.assertContextNoErrors(context)
        self.assertNotNullOrEmpty(expSrc.get_c_str())

        exSp: str = self.sdk.get_examples_search_path()
        mdlModuleName: pymdlsdk.IString = mdlImpExpApi.get_mdl_module_name(exSp + "/nvidia/sdk_examples/tutorials.mdl")
        self.assertEqual(mdlModuleName.get_c_str(), "::nvidia::sdk_examples::tutorials")

        lightProfileVal: pymdlsdk.IValue_Lightprofile = self.sdk.mdlFactory.create_light_profile(self.sdk.transaction,
            "/nvidia/sdk_examples/resources/example_modules.ies", False, None)
        self.assertContextNoErrors(context)
        self.assertIsValidInterface(lightProfileVal)
        lightDbName: str = lightProfileVal.get_value()
        lightProfile: pymdlsdk.ILightprofile = self.sdk.transaction.edit_as(pymdlsdk.ILightprofile, lightDbName)
        self.assertIsValidInterface(lightProfile)
        mdlImpExpApi.export_lightprofile("light_test0.ies", lightProfile)

        fname: pymdlsdk.IString = mdlImpExpApi.frame_uvtile_marker_to_string("tiled_resource.<##>.<UDIM>.png", 23, 0, 1)
        self.assertEqual(fname.get_c_str(), "tiled_resource.23.1011.png")

        self.assertException(TypeError, lambda: mdlImpExpApi.deserialize_function_name())

        vMbsdf1: pymdlsdk.IValue_bsdf_measurement = self.sdk.mdlFactory.create_bsdf_measurement(
            self.sdk.transaction, "/nvidia/sdk_examples/resources/example_modules.mbsdf", False, None)
        self.assertIsValidInterface(vMbsdf1)
        bsdfMeasurement: pymdlsdk.IBsdf_measurement = self.sdk.transaction.access_as(pymdlsdk.IBsdf_measurement, vMbsdf1.get_value())
        self.assertIsValidInterface(bsdfMeasurement)

        refl: pymdlsdk.IInterface = bsdfMeasurement.get_reflection()
        trans: pymdlsdk.IInterface = bsdfMeasurement.get_transmission()
        self.assertIsNotValidInterface(trans)  # mbsdf file does not contain transmission
        reflIso: pymdlsdk.IBsdf_isotropic_data = refl.get_interface(pymdlsdk.IBsdf_isotropic_data)
        transIso: pymdlsdk.IBsdf_isotropic_data = trans.get_interface(pymdlsdk.IBsdf_isotropic_data)
        self.assertIsNotValidInterface(transIso)  # still invalid after casting
        self.assertIsValidInterface(reflIso)

        res = mdlImpExpApi.export_bsdf_data("bsdf_data0.mbsdf", reflIso, transIso)
        self.assertZero(res)
        pass
    
# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
