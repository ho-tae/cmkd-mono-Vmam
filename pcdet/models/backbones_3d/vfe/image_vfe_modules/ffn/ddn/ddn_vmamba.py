from .ddn_template_vmam import DDNTemplateVMam

try:
    import torchvision
except:
    pass


class DDNVMamba(DDNTemplateVMam):    

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes DDNVmamba model
        Args:
            backbone_name: string, VMamba Backbone Name [VMamba-S/VMamba-B]
        """
        # if backbone_name == "VMamba-S":
        #     constructor = torchvision.models.segmentation.deeplabv3_resnet50
        # elif backbone_name == "VMamba-B":
        #     constructor = torchvision.models.segmentation.deeplabv3_resnet101
        # else:
        #     raise NotImplementedError

        super().__init__(backbone_name=backbone_name, **kwargs)
