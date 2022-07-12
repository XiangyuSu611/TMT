import config
from pathlib import Path
import thirdparty.brender.brender as brender
import thirdparty.brender.brender.material 

def cc0texture_to_brender(material_path, **kwargs):
    return brender.material.CC0textureMaterial.from_path(material_path, **kwargs)

def textureHarven_to_brender(material_path, **kwargs):
    return brender.material.TextureHarvenMaterial.from_path(material_path, **kwargs)

def texture_to_brender(material_path, **kwargs):
    return brender.material.TextureMaterial.from_path(material_path, **kwargs)

def texture3D_to_brender(material_path, **kwargs):
    return brender.material.Texture3DMaterial.from_path(material_path, **kwargs)

def material_to_brender(material, **kwargs):
    if material["type"] == 'AITTALA_BECKMANN':
        base_dir = Path(config.MATERIAL_DIR_AITTALA, material["substance"],
                        material["name"])
        return brender.material.AittalaMaterial.from_path(base_dir, **kwargs)
    
    elif material["type"] == 'POLIIGON':
        base_dir = Path(config.MATERIAL_DIR_POLIIGON, material["substance"],
                        material["name"])
        return brender.material.PoliigonMaterial.from_path(base_dir, **kwargs)
    
    elif material["type"] == 'VRAY':
        base_dir = Path(config.MATERIAL_DIR_VRAY, material["substance"],
                        material["params"]['raw_name'], )
        return brender.material.VRayMaterial.from_path(base_dir, **kwargs)
    
    elif (material["type"] == 'MDL' and material["source"] == 'adobe_stock'):
        mdl_path = Path(config.MATERIAL_DIR_ADOBE_STOCK,
                        material["substance"],
                        f'AdobeStock_{material["source_id"]}',
                        f'{material["name"]}.mdl')
        return brender.material.MDLMaterial.from_path(mdl_path, **kwargs)
    
    elif material["type"] == 'PRINCIPLED':
        return brender.material.PrincipledMaterial(
            diffuse_color=material["params"]['diffuse_color'],
            specular=material["params"]['specular'],
            metallic=material["params"]['metallic'],
            roughness=material["params"]['roughness'],
            anisotropy=material["params"]['anisotropy'],
            anisotropic_rotation=material["params"]['anisotropic_rotation'],
            clearcoat=material["params"]['clearcoat'],
            clearcoat_roughness=material["params"]['clearcoat_roughness'],
            ior=material["params"]['ior'],
            **kwargs
        )
    elif material["type"] == 'BLINN_PHONG':
        return brender.material.BlinnPhongMaterial(
            diffuse_albedo=[float(c) for c in material["params"]['diffuse']],
            specular_albedo=[float(c) for c in material["params"]['specular']],
            shininess=float(material["params"]['shininess']),
            **kwargs)
    
    elif material["type"] == 'CC0TEXTURE':
        base_dir = Path(config.MATERIAL_DIR_CCOTEXTURE,
                        material["substance"],
                        f'{material["name"]}')
        return brender.material.CC0textureMaterial.from_path(base_dir, **kwargs)
    
    elif material["type"] == 'TEXTURE3D':
        base_dir = Path(config.MATERIAL_DIR_TEXTURE3D,
                        material["substance"],
                        f'{material["name"]}')
        return brender.material.Texture3DMaterial.from_path(base_dir, **kwargs)

    elif material["type"] == 'TEXTUREHARVEN':
        base_dir = Path(config.MATERIAL_DIR_TEXTUREHARVEN,
                        material["substance"],
                        f'{material["name"]}')
        return brender.material.TextureHarvenMaterial.from_path(base_dir, **kwargs)
    
    elif material["type"] == 'SHARETEXTURE':
        base_dir = Path(config.MATERIAL_DIR_SHARETEXTURE,
                        material["substance"],
                        f'{material["name"]}')
        return brender.material.ShareTextureMaterial.from_path(base_dir, **kwargs)

    else:
        raise ValueError(f'Unsupported material type {material["type"]}')
