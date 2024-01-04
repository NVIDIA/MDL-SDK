import unittest
import os
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


class MainResolve(UnittestBase):
    sdk: SDK = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=True, loadImagePlugins=True)

    @classmethod
    def tearDownClass(self):
        self.sdk.unload()
        self.sdk = None
        print(f"\nFinished tests in {__file__}\n")

    def test_load_module_success(self):
        self.assertNotEqual(self.load_module("::df"), "")

    def test_load_module_failure(self):
        self.assertEqual(self.load_module("::NOT_EXISTING"), "")

    def test_resolve_resource_absolute(self):
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        self.assertIsValidInterface(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsValidInterface(er)
        resourcePath: str = "/nvidia/sdk_examples/resources/example_roughness.png"
        resolved_resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(
            file_path=resourcePath, owner_file_path=None, owner_name=None, pos_line=0, pos_column=0, context=None)
        self.assertTrue(resolved_resource.is_valid_interface())
        self.assertEqual(resolved_resource.get_count(), 1)  # not animated
        filename_mask: str = resolved_resource.get_filename_mask().replace("\\", "/")
        self.assertTrue(filename_mask.endswith(resourcePath))
        self.assertTrue(len(filename_mask) > len(resourcePath))
        frame0: pymdlsdk.IMdl_resolved_resource_element = resolved_resource.get_element(0)
        self.assertEqual(frame0.get_count(), 1)  # not tiled
        self.assertEqual(frame0.get_mdl_file_path(0), resourcePath)  # same for not animated and not tiled
        filename_frame0_tile0: str = frame0.get_filename(0).replace("\\", "/")
        self.assertEqual(filename_frame0_tile0, filename_mask)  # same for not animated and not tiled

    def resolve_resource_to_image(self, resourcePath) -> str:
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        imageApi: pymdlsdk.IImage_api = self.sdk.neuray.get_api_component(pymdlsdk.IImage_api)
        self.assertIsValidInterface(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsValidInterface(er)

        # resolve first
        resolved_resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(
            file_path=resourcePath, owner_file_path=None, owner_name=None, pos_line=0, pos_column=0, context=None)
        self.assertIsValidInterface(resolved_resource)

        # create canvases
        class MyCanvas:
            frame: int
            isTiled: bool
            u: int
            v: int
            canvas: pymdlsdk.ICanvas

        canvasList: list[MyCanvas] = []
        for frameIndex in range(resolved_resource.get_count()):
            frame: pymdlsdk.IMdl_resolved_resource_element = resolved_resource.get_element(frameIndex)
            for tileIndex in range(frame.get_count()):
                filename: str = frame.get_filename(tileIndex)
                c: MyCanvas = MyCanvas()
                c.frame = frame.get_frame_number()
                c.isTiled = frame.has_uvtile_uv(tileIndex)
                if c.isTiled:
                    c.u = frame.get_uvtile_u(tileIndex)
                    c.v = frame.get_uvtile_v(tileIndex)
                reader: pymdlsdk.IReader = frame.create_reader(tileIndex)
                self.assertIsValidInterface(reader)
                ext: str = os.path.splitext(filename)[1][1:]
                c.canvas = imageApi.create_canvas_from_reader(reader, ext)
                self.assertIsValidInterface(c.canvas)
                canvasList.append(c)

        # create the image
        image: pymdlsdk.IImage = self.sdk.transaction.create_as(pymdlsdk.IImage, "Image")
        if len(canvasList) == 1:
            res: bool = image.set_from_canvas(canvasList[0].canvas)
            self.assertTrue(res)
        else:
            uvtileArray: pymdlsdk.IDynamic_array = self.sdk.transaction.create_as(pymdlsdk.IDynamic_array, "Uvtile[]")
            uvtileArray.set_length(len(canvasList))
            for c in range(len(canvasList)):
                tile: pymdlsdk.IStructure = uvtileArray.get_element_as(pymdlsdk.IStructure, c)
                u: pymdlsdk.ISint32 = self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")
                v: pymdlsdk.ISint32 = self.sdk.transaction.create_as(pymdlsdk.ISint32, "Sint32")
                f: pymdlsdk.ISize = self.sdk.transaction.create_as(pymdlsdk.ISize, "Size")
                u.set_value(canvasList[c].u)
                v.set_value(canvasList[c].v)
                f.set_value(canvasList[c].frame)
                tile.set_value("u", u)
                tile.set_value("v", v)
                tile.set_value("frame", f)
                tile.set_value("canvas", canvasList[c].canvas)
            res: bool = image.set_from_canvas(uvtileArray, None)
            self.assertTrue(res)

        # store the image
        imageDbName: str = "mdl::image_" + str(uuid.uuid4())
        self.assertZero(self.sdk.transaction.store(image, imageDbName))
        return imageDbName

    def test_resolve_resource_to_image(self):
        resourcePath: str = "/nvidia/sdk_examples/resources/example_roughness.png"
        imageDbName: str = self.resolve_resource_to_image(resourcePath)
        self.assertNotNullOrEmpty(imageDbName)
        image: pymdlsdk.IImage = self.sdk.transaction.access_as(pymdlsdk.IImage, imageDbName)
        self.assertEqual(image.get_length(), 1)
        self.assertEqual(image.get_frame_length(0), 1)
        canvas: pymdlsdk.ICanvas = image.get_canvas(0, 0, 0)
        self.assertEqual(canvas.get_resolution_x(), 2048)
        self.assertEqual(canvas.get_resolution_y(), 2048)

    def check_tiled_resource(self, image: pymdlsdk.IImage):
        self.assertEqual(image.get_length(), 2)
        self.assertEqual(image.get_frame_length(0), 3)
        self.assertEqual(image.get_frame_length(1), 1)
        self.assertFalse(image.has_uvtile_uv(frame_id=42, uvtile_id=0))
        self.assertFalse(image.has_uvtile_uv(frame_id=0, uvtile_id=42))
        self.assertEqual(image.get_frame_number(0), 0)
        self.assertEqual(image.get_frame_length(0), 3)
        self.assertEqual(image.get_frame_length(1), 1)
        # Tile: 1001, Frame: 0
        frameId: int = image.get_frame_id(0)
        self.assertEqual(image.get_uvtile_uv_ranges_min_u(frameId), 0)
        self.assertEqual(image.get_uvtile_uv_ranges_min_v(frameId), 0)
        self.assertEqual(image.get_uvtile_uv_ranges_max_u(frameId), 1)
        self.assertEqual(image.get_uvtile_uv_ranges_max_v(frameId), 1)
        tileId: int = image.get_uvtile_id(frameId, u=0, v=0)
        self.assertTrue(image.has_uvtile_uv(frameId, tileId))
        self.assertEqual(image.get_uvtile_u(frameId, tileId), 0)
        self.assertEqual(image.get_uvtile_v(frameId, tileId), 0)
        canvas: pymdlsdk.ICanvas = image.get_canvas(frameId, tileId, 0)
        self.assertEqual(canvas.get_resolution_x(), 512)
        self.assertEqual(canvas.get_resolution_y(), 512)
        # Tile: 1011, Frame: 0
        tileId: int = image.get_uvtile_id(frameId, u=0, v=1)
        self.assertTrue(image.has_uvtile_uv(frameId, tileId))
        self.assertEqual(image.get_uvtile_u(frameId, tileId), 0)
        self.assertEqual(image.get_uvtile_v(frameId, tileId), 1)
        canvas: pymdlsdk.ICanvas = image.get_canvas(frameId, tileId, 0)
        self.assertEqual(canvas.get_resolution_x(), 512)
        self.assertEqual(canvas.get_resolution_y(), 512)
        # Tile: 1002, Frame: 0
        tileId: int = image.get_uvtile_id(frameId, u=1, v=0)
        self.assertTrue(image.has_uvtile_uv(frameId, tileId))
        self.assertEqual(image.get_uvtile_u(frameId, tileId), 1)
        self.assertEqual(image.get_uvtile_v(frameId, tileId), 0)
        canvas: pymdlsdk.ICanvas = image.get_canvas(frameId, tileId, 0)
        self.assertEqual(canvas.get_resolution_x(), 512)
        self.assertEqual(canvas.get_resolution_y(), 512)
        # Tile: 1012, Frame: 2
        frameId: int = image.get_frame_id(2)
        self.assertEqual(image.get_uvtile_uv_ranges_min_u(frameId), 1)
        self.assertEqual(image.get_uvtile_uv_ranges_min_v(frameId), 1)
        self.assertEqual(image.get_uvtile_uv_ranges_max_u(frameId), 1)
        self.assertEqual(image.get_uvtile_uv_ranges_max_v(frameId), 1)
        tileId: int = image.get_uvtile_id(frameId, u=1, v=1)
        self.assertTrue(image.has_uvtile_uv(frameId, tileId))
        self.assertEqual(image.get_uvtile_u(frameId, tileId), 1)
        self.assertEqual(image.get_uvtile_v(frameId, tileId), 1)
        canvas: pymdlsdk.ICanvas = image.get_canvas(frameId, tileId, 0)
        self.assertEqual(canvas.get_resolution_x(), 512)
        self.assertEqual(canvas.get_resolution_y(), 512)
        # error cases
        self.assertEqual(image.get_frame_id(1), -1)
        self.assertEqual(image.get_frame_id(-1), -1)
        self.assertEqual(image.get_frame_id(42), -1)
        self.assertEqual(image.get_uvtile_id(frameId, u=0, v=0), -1)

    def test_resolve_resource_to_animated_tiled_image(self):
        resourcePath: str = "/nvidia/sdk_examples/resources/tiled_resource.<##>.<UDIM>.png"
        imageDbName: str = self.resolve_resource_to_image(resourcePath)
        self.assertNotNullOrEmpty(imageDbName)
        image: pymdlsdk.IImage = self.sdk.transaction.access_as(pymdlsdk.IImage, imageDbName)
        self.check_tiled_resource(image)

    def test_create_texture_animated_tiled_image(self):
        resourcePath: str = "/nvidia/sdk_examples/resources/tiled_resource.<##>.<UDIM>.png"
        texture: pymdlsdk.IValue_texture = self.sdk.mdlFactory.create_texture(
            self.sdk.transaction, resourcePath, pymdlsdk.IType_texture.Shape.TS_2D, 2.2, None, False, None)
        self.assertIsValidInterface(texture)
        textureDbName: str = texture.get_value()
        itexture: pymdlsdk.ITexture = self.sdk.transaction.access_as(pymdlsdk.ITexture, textureDbName)
        self.assertIsValidInterface(itexture)
        imageDbName: str = itexture.get_image()
        self.assertNotNullOrEmpty(imageDbName)
        image: pymdlsdk.IImage = self.sdk.transaction.access_as(pymdlsdk.IImage, imageDbName)
        self.check_tiled_resource(image)

    def resolve_module(self, mdlName) -> str:
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        self.assertIsValidInterface(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsValidInterface(er)
        resolved_module: pymdlsdk.IMdl_resolved_module = er.resolve_module(mdlName, None, None, 0, 0, None)
        self.assertTrue(resolved_module.is_valid_interface())
        moduleFileName = resolved_module.get_filename()
        self.assertTrue(moduleFileName.replace("\\", "/").endswith(mdlName.replace("::", "/") + ".mdl"))
        return moduleFileName

    def test_resolve_module(self):
        self.assertTrue(len(self.resolve_module("::nvidia::sdk_examples::tutorials")) > 0)

    def test_resolve_resource_relative(self):
        config: pymdlsdk.IMdl_configuration = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_configuration)
        self.assertIsValidInterface(config)
        er: pymdlsdk.IMdl_entity_resolver = config.get_entity_resolver()
        self.assertIsValidInterface(er)
        resourcePath: str = "resources/metal_cast_iron_roughness.png"
        ownerName: str = "::nvidia::sdk_examples::tutorials_distilling"
        ownerFilePath: str = self.resolve_module(ownerName)
        resolved_resource: pymdlsdk.IMdl_resolved_resource = er.resolve_resource(
            file_path=resourcePath, owner_file_path=ownerFilePath, owner_name=ownerName, pos_line=0, pos_column=0, context=None)
        self.assertTrue(resolved_resource.is_valid_interface())
        self.assertEqual(resolved_resource.get_count(), 1)  # not animated
        filename_mask: str = resolved_resource.get_filename_mask().replace("\\", "/")
        self.assertTrue(filename_mask.endswith(resourcePath))
        self.assertTrue(len(filename_mask) > len(resourcePath))
        frame0: pymdlsdk.IMdl_resolved_resource_element = resolved_resource.get_element(0)
        self.assertEqual(frame0.get_count(), 1)  # not tiled
        self.assertEqual(frame0.get_mdl_file_path(0), "/nvidia/sdk_examples/" + resourcePath)  # same for not animated and not tiled
        filename_frame0_tile0: str = frame0.get_filename(0).replace("\\", "/")
        self.assertEqual(filename_frame0_tile0, filename_mask)  # same for not animated and not tiled


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
