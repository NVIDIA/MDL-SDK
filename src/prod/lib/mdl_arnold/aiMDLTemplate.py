import pymel.core as pm
import mtoa.utils as utils
import mtoa.ui.ae.utils as aeUtils
from mtoa.ui.ae.shaderTemplate import ShaderAETemplate
 
class AEaiMDLTemplate(ShaderAETemplate):
 
    def setup(self):
        # Add the shader swatch to the AE
        self.addSwatch()
        self.beginScrollLayout()
 
        # Add a list that allows to replace the shader for other one
        self.addCustom('message', 'AEshaderTypeNew', 'AEshaderTypeReplace')
 
        self.beginLayout("Material Selection", collapse=False)
        self.addControl("mdl_module_name", label="MDL Module Name")
        self.addControl("mdl_function_name", label="MDL Function Name")
        self.addControl("qualified_name", label="Qualified Name (deprecated)")
        self.endLayout()
 
        # include/call base class/node attributes
        pm.mel.AEdependNodeTemplate(self.nodeName)
 
        # Add Section for the extra controls not displayed before
        self.addExtraControls()
        self.endScrollLayout()
