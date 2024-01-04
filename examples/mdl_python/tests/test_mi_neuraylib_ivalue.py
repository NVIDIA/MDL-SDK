import unittest
import os
import inspect

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


class MainIValue(UnittestBase):
    sdk: SDK = None
    tf: pymdlsdk.IType_factory = None
    vf: pymdlsdk.IValue_factory = None
    imp_exp: pymdlsdk.IMdl_impexp_api = None

    @classmethod
    def setUpClass(self):
        print(f"Running tests in {__file__} in process with id: {os.getpid()}")
        self.sdk = SDK()
        self.sdk.load(addExampleSearchPath=False, loadImagePlugins=False)
        self.tf = self.sdk.mdlFactory.create_type_factory(self.sdk.transaction)
        self.vf = self.sdk.mdlFactory.create_value_factory(self.sdk.transaction)
        self.imp_exp = self.sdk.neuray.get_api_component(pymdlsdk.IMdl_impexp_api)
        self.imp_exp.load_module(self.sdk.transaction, "::base")  # load something to get builtins

    @classmethod
    def tearDownClass(self):
        self.imp_exp = None
        self.vf = None
        self.tf = None
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

    def test_IValue_bool(self):
        tBool: pymdlsdk.IType_bool = self.tf.create_bool()
        self.assertIsNotNone(tBool)
        self.assertEqual(tBool.get_kind(), pymdlsdk.IType.Kind.TK_BOOL)
        v: pymdlsdk.IValue = self.vf.create(tBool)
        self.assertIsNotNone(v)
        vBool: pymdlsdk.IValue_bool = v.get_interface(pymdlsdk.IValue_bool)
        self.assertIsNotNone(vBool)
        self.assertEqual(vBool.get_kind(), pymdlsdk.IValue.Kind.VK_BOOL)
        self.assertEqual(vBool.get_value(), False)
        vBool.set_value(True)
        self.assertEqual(vBool.get_value(), True)
        vBool = self.vf.create_bool(False)
        self.assertIsNotNone(vBool)
        self.assertEqual(vBool.get_value(), False)
        vBool = self.vf.create_bool(True)
        self.assertIsNotNone(vBool)
        self.assertEqual(vBool.get_value(), True)
        self.assertTrue(True)
        vAtomic: pymdlsdk.IValue_atomic = vBool.get_interface(pymdlsdk.IValue_atomic)
        self.assertEqual(tBool.get_iid(), vBool.get_type().get_iid())
        self.assertTrue(vBool.compare_iid(vBool.get_iid()))
        self.assertTrue(vBool.compare_iid(v.get_iid()))
        self.assertTrue(vBool.compare_iid(vAtomic.get_iid()))
        self.assertFalse(vBool.compare_iid(tBool.get_iid()))

    def run_modifier_tests(self, itype):
        mods: int = itype.get_all_type_modifiers()
        if mods == 0:
            skipped = itype.skip_all_type_aliases()
            self.assertEqual(itype.get_iid(), skipped.get_iid())  # equal iid

    def run_compound_test(self, compoundTypeClass, compoundValueClass, value: pymdlsdk.IValue_compound):
        valueCompound = value.get_interface(compoundValueClass)
        if not valueCompound.is_valid_interface():
            return  # allows to just try every type
        self.assertIsValidInterface(valueCompound)
        for i in range(valueCompound.get_size()):
            element: pymdlsdk.IValue = valueCompound.get_value(i)
            self.assertTrue(element.is_valid_interface())
        self.assertIsInstance(value.get_kind(), pymdlsdk.IValue.Kind)
        self.assertIsInstance(valueCompound.get_kind(), pymdlsdk.IValue.Kind)
        self.assertEqual(value.get_kind(), valueCompound.get_kind())
        self.assertTrue(valueCompound.compare_iid(pymdlsdk.IValue.IID()))
        self.assertTrue(valueCompound.compare_iid(compoundValueClass.IID()))
        self.assertIsNotNone(valueCompound.get_iid())
        with value.get_interface(compoundValueClass) as valueCompound2:
            self.assertIsNotNone(valueCompound2)
            self.assertTrue(valueCompound2.is_valid_interface())
            self.assertEqual(valueCompound.get_iid(), valueCompound2.get_iid())
            component: pymdlsdk.IValue = valueCompound2.get_value(0)
            self.assertIsValidInterface(component)
            self.assertEqual(valueCompound2.set_value(0, component), 0)
        typeCompound = valueCompound.get_type()
        self.assertTrue(typeCompound.is_valid_interface())
        self.assertTrue(typeCompound.compare_iid(pymdlsdk.IType.IID()))
        self.assertTrue(typeCompound.compare_iid(compoundTypeClass.IID()))
        self.run_modifier_tests(typeCompound)
        with typeCompound.get_interface(compoundTypeClass) as typeCompound2:
            self.assertIsNotNone(typeCompound2)
            self.assertTrue(typeCompound2.is_valid_interface())
            self.assertEqual(typeCompound.get_iid(), typeCompound2.get_iid())

    def run_ivalue_test(self, ivalueClass, valueTyped: pymdlsdk.IValue_vector):
        self.assertIsValidInterface(valueTyped)
        value: pymdlsdk.IValue = None
        for funcName, funcObj in inspect.getmembers(ivalueClass):
            if funcName == "get_interface":
                value = funcObj(valueTyped, pymdlsdk.IValue)
        self.assertIsValidInterface(value)
        with value.get_interface(ivalueClass) as value2:
            self.assertIsNotNone(value2)
            self.assertTrue(value2.is_valid_interface())
            self.assertEqual(valueTyped.get_iid(), value2.get_iid())
        pythonType = type(valueTyped)
        castedBack = value.get_interface(pythonType)
        self.assertEqual(castedBack.get_iid(), valueTyped.get_iid())
        self.assertEqual(value.get_kind(), valueTyped.get_kind())
        self.assertEqual(value.get_iid(), valueTyped.get_iid())
        self.assertEqual(value.get_type().get_iid(), valueTyped.get_type().get_iid())
        self.assertIsInstance(value.get_kind(), pymdlsdk.IValue.Kind)
        self.assertIsInstance(valueTyped.get_kind(), pymdlsdk.IValue.Kind)
        self.run_compound_test(pymdlsdk.IType_compound, pymdlsdk.IValue_compound, value)
        self.run_compound_test(pymdlsdk.IType_vector, pymdlsdk.IValue_vector, value)
        self.run_compound_test(pymdlsdk.IType_matrix, pymdlsdk.IValue_matrix, value)
        self.run_compound_test(pymdlsdk.IType_structure, pymdlsdk.IValue_structure, value)
        self.run_compound_test(pymdlsdk.IType_color, pymdlsdk.IValue_color, value)
        self.run_compound_test(pymdlsdk.IType_array, pymdlsdk.IValue_array, value)

    def run_itype_test(self, itypeClass, ivalueClass, typeTyped: pymdlsdk.IType):
        self.assertIsValidInterface(typeTyped)
        itype: pymdlsdk.IType = None
        for funcName, funcObj in inspect.getmembers(itypeClass):
            if funcName == "get_interface":
                itype = funcObj(typeTyped, pymdlsdk.IType)
        self.assertIsNotNone(itype)
        with itype.get_interface(itypeClass) as itype2:
            self.assertIsNotNone(itype2)
            self.assertTrue(itype2.is_valid_interface())
            self.assertEqual(typeTyped.get_iid(), itype2.get_iid())
        self.assertTrue(itype.is_valid_interface())
        pythonType = type(typeTyped)
        castedBack = itype.get_interface(pythonType)
        self.assertEqual(castedBack.get_iid(), typeTyped.get_iid())
        self.assertEqual(itype.get_kind(), typeTyped.get_kind())
        self.assertEqual(itype.get_iid(), typeTyped.get_iid())
        self.assertTrue(typeTyped.compare_iid(pymdlsdk.IType.IID()))
        self.assertTrue(typeTyped.compare_iid(itype.get_iid()))
        self.assertIsInstance(itype.get_kind(), pymdlsdk.IType.Kind)
        self.assertIsInstance(typeTyped.get_kind(), pymdlsdk.IType.Kind)
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_BSDF:
            df: pymdlsdk.IType_df = itype.get_interface(pymdlsdk.IType_df)
            self.assertIsValidInterface(df)
            self.assertIsInstance(df.get_kind(), pymdlsdk.IType.Kind)
            self.assertEqual(df.get_kind(), pymdlsdk.IType.Kind.TK_BSDF)
            return
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_EDF:
            return
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_VDF:
            return
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_HAIR_BSDF:
            return
        if itype.get_kind() == pymdlsdk.IType.Kind.TK_VECTOR:
            elementType: pymdlsdk.IType = typeTyped.get_element_type()
            atomicType: pymdlsdk.IType_atomic = elementType.get_interface(pymdlsdk.IType_atomic)
            self.assertTrue(atomicType.is_valid_interface())
            for i in range(typeTyped.get_size()):
                compType: pymdlsdk.IType = typeTyped.get_component_type(i)
                self.assertEqual(compType.get_kind(), elementType.get_kind())
        if itypeClass == pymdlsdk.IType_texture:
            self.assertEqual(typeTyped.get_shape(), pymdlsdk.IType_texture.Shape.TS_3D)
        if itypeClass == pymdlsdk.IType_structure:
            self.assertEqual(typeTyped.get_symbol(), "::material_surface")
            self.assertEqual(typeTyped.get_predefined_id(), pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL_SURFACE)
            for i in range(typeTyped.get_size()):
                fieldName: str = typeTyped.get_field_name(i)
                self.assertEqual(typeTyped.find_field(fieldName), i)
                annoBlock: pymdlsdk.IAnnotation_block = typeTyped.get_field_annotations(i)
                self.assertIsNotNone(annoBlock)
            fieldType: pymdlsdk.IType = typeTyped.get_field_type(1)
            self.assertEqual(fieldType.get_kind(), pymdlsdk.IType.Kind.TK_STRUCT)
            annoBlock: pymdlsdk.IAnnotation_block = typeTyped.get_annotations()
            self.assertIsNotNone(annoBlock)
        if itypeClass == pymdlsdk.IType_enumeration:
            self.assertEqual(typeTyped.get_symbol(), "::intensity_mode")
            self.assertEqual(typeTyped.get_predefined_id(), pymdlsdk.IType_enumeration.Predefined_id.EID_INTENSITY_MODE)
            for i in range(typeTyped.get_size()):
                name: str = typeTyped.get_value_name(i)
                code: int = typeTyped.get_value_code(i)
                self.assertEqual(typeTyped.find_value(name), i)
                self.assertEqual(typeTyped.find_value(code), i)
                annoBlock: pymdlsdk.IAnnotation_block = typeTyped.get_value_annotations(i)
                self.assertIsNotNone(annoBlock)
            annoBlock: pymdlsdk.IAnnotation_block = typeTyped.get_annotations()
            self.assertIsNotNone(annoBlock)
        isDeferredSizedArray: bool = False
        if itypeClass == pymdlsdk.IType_array:
            elementType: pymdlsdk.IType = typeTyped.get_element_type()
            self.assertIsValidInterface(elementType)
            if typeTyped.is_immediate_sized():
                self.assertEqual(typeTyped.get_size(), 10)
            else:
                self.assertEqual(typeTyped.get_deferred_size(), "X")
                isDeferredSizedArray = True
        asCompound: pymdlsdk.IType_compound = typeTyped.get_interface(pymdlsdk.IType_compound)
        if asCompound.is_valid_interface():
            self.assertIsValidInterface(asCompound)
            self.assertEqual(asCompound.get_kind(), typeTyped.get_kind())
            size: int = asCompound.get_size()
            if isDeferredSizedArray:
                self.assertEqual(size, -1)
            else:
                self.assertGreater(size, 0)
            fieldType: pymdlsdk.IType = asCompound.get_component_type(0)
            self.assertIsValidInterface(fieldType)
        asAtomic: pymdlsdk.IType_atomic = typeTyped.get_interface(pymdlsdk.IType_atomic)
        asResource: pymdlsdk.IType_resource = typeTyped.get_interface(pymdlsdk.IType_resource)
        self.run_modifier_tests(typeTyped)
        self.run_modifier_tests(itype)
        # create a value and run tests on it
        if not isDeferredSizedArray:
            ivalue: pymdlsdk.IValue = self.vf.create(typeTyped)
            ivalueTyped = ivalue.get_interface(ivalueClass)
            self.run_ivalue_test(ivalueClass, ivalueTyped)

    def test_ivalues(self):
        tBool: pymdlsdk.IType_bool = self.tf.create_bool()
        tInt: pymdlsdk.IType_int = self.tf.create_int()
        tFloat: pymdlsdk.IType_float = self.tf.create_float()
        tFloat4: pymdlsdk.IType_vector = self.tf.create_vector(tFloat, 4)
        self.run_itype_test(pymdlsdk.IType_vector, pymdlsdk.IValue_vector, tFloat4)
        self.run_itype_test(pymdlsdk.IType_matrix, pymdlsdk.IValue_matrix, self.tf.create_matrix(tFloat4, 4))
        tDouble: pymdlsdk.IType_double = self.tf.create_double()
        self.run_itype_test(pymdlsdk.IType_bool, pymdlsdk.IValue_bool, tBool)
        self.run_itype_test(pymdlsdk.IType_int, pymdlsdk.IValue_int, tInt)
        self.run_itype_test(pymdlsdk.IType_float, pymdlsdk.IValue_float, tFloat)
        self.run_itype_test(pymdlsdk.IType_double, pymdlsdk.IValue_double, tDouble)
        self.run_itype_test(pymdlsdk.IType_string, pymdlsdk.IValue_string, self.tf.create_string())
        self.run_itype_test(pymdlsdk.IType_color, pymdlsdk.IValue_color, self.tf.create_color())
        self.run_itype_test(pymdlsdk.IType_texture, pymdlsdk.IValue_texture, self.tf.create_texture(pymdlsdk.IType_texture.Shape.TS_3D))
        self.run_itype_test(pymdlsdk.IType_light_profile, pymdlsdk.IValue_light_profile, self.tf.create_light_profile())
        self.run_itype_test(pymdlsdk.IType_bsdf_measurement, pymdlsdk.IValue_bsdf_measurement, self.tf.create_bsdf_measurement())
        self.run_itype_test(pymdlsdk.IType_enumeration, pymdlsdk.IValue_enumeration, self.tf.create_enum("::intensity_mode"))
        self.run_itype_test(pymdlsdk.IType_structure, pymdlsdk.IValue_structure, self.tf.create_struct("::material_surface"))
        self.run_itype_test(pymdlsdk.IType_enumeration, pymdlsdk.IValue_enumeration, self.tf.get_predefined_enum(pymdlsdk.IType_enumeration.Predefined_id.EID_INTENSITY_MODE))
        self.run_itype_test(pymdlsdk.IType_structure, pymdlsdk.IValue_structure, self.tf.get_predefined_struct(pymdlsdk.IType_structure.Predefined_id.SID_MATERIAL_SURFACE))
        self.run_itype_test(pymdlsdk.IType_array, pymdlsdk.IValue_array, self.tf.create_immediate_sized_array(tInt, 10))
        self.run_itype_test(pymdlsdk.IType_array, pymdlsdk.IValue_array, self.tf.create_deferred_sized_array(tInt, "X"))
        tbsdf: pymdlsdk.IType_bsdf = self.tf.create_bsdf()
        self.run_itype_test(pymdlsdk.IType_bsdf, None, tbsdf)
        tedf: pymdlsdk.IType_edf = self.tf.create_edf()
        self.run_itype_test(pymdlsdk.IType_edf, None, tedf)
        tvdf: pymdlsdk.IType_vdf = self.tf.create_vdf()
        self.run_itype_test(pymdlsdk.IType_vdf, None, tvdf)
        thair: pymdlsdk.IType_hair_bsdf = self.tf.create_hair_bsdf()
        self.run_itype_test(pymdlsdk.IType_hair_bsdf, None, thair)


# run all tests of this file
if __name__ == '__main__':
    unittest.main()  # pragma: no cover
