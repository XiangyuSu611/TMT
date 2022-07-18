import enum
import math
import ujson
from functools import total_ordering
from pathlib import Path
import numpy as np

import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

import src.terial.database as db
from thirdparty.rendkit.meshkit import wavefront
from src.terial import config
from src.terial.dbutils import BlobDataMixin, _load_image, SerializeMixin


class Exemplar(db.Base, BlobDataMixin):
    # __tablename__ = 'exemplars_table1' # chair? bed? or table1?
    # __tablename__ = 'exemplars_bed' # chair? bed? or table1?
    __tablename__ = 'exemplars' # chair? bed? or table1?

    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    source_path = sa.Column(sa.String)
    source_name = sa.Column(sa.String)
    date_added = sa.Column(sa.DateTime)
    category = sa.Column(sa.String)
    exclude = sa.Column(sa.Boolean, default=False)
    meta = sa.Column(JSONB, default={})

    @property
    def data_path(self):
        # return Path(config.BLOB_ROOT, 'exemplars_table1', str(self.id)) # chair? bed? or table1?
        # return Path(config.BLOB_ROOT, 'exemplars_bed', str(self.id)) # chair? bed? or table1?
        return Path(config.BLOB_ROOT, 'exemplars', str(self.id)) # chair? bed? or table1?

    @property
    def cropped_path(self):
        return self.data_path / 'cropped.jpg'

    @property
    def original_path(self):
        return self.data_path / 'original.jpg'

    def load_cropped_image(self):
        return _load_image(self.cropped_path)

    @classmethod
    def get_image_name(cls, shape):
        return f'cropped_image.{shape[0]}x{shape[1]}.png'

    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'source_name': self.source_name,
            'date_added': self.date_added.isoformat(),
            'category': self.category,
            'exclude': self.exclude,
        }


class Shape(db.Base, BlobDataMixin):
    # __tablename__ = 'shapes_bed'    # chair? bed? or table?
    __tablename__ = 'shapes'    # chair? bed? or table?

    id = sa.Column(sa.Integer, primary_key=True)
    source = sa.Column(sa.String)
    source_id = sa.Column(sa.String)
    category = sa.Column(sa.String)
    exclude = sa.Column(sa.Boolean, default=False)
    split_set = sa.Column(sa.String)
    meta = sa.Column(JSONB, default={})
    azimuth_correction = sa.Column(sa.Float, default=0)

    pairs = relationship('ExemplarShapePair', back_populates='shape')
    result_annotations = relationship('ResultAnnotation')

    def load(self, size=100) -> object:
        mesh = wavefront.read_obj_file(self.obj_path)
        mesh.resize(size)
        materials = wavefront.read_mtl_file(self.mtl_path, mesh)
        return mesh, materials
    
    ## load part shape.
    def load_part(self, size=100) -> object:
        mesh = wavefront.read_obj_file(self.part_obj_path)
        mesh.resize(size)
        materials = wavefront.read_mtl_file(self.part_mtl_path, mesh)
        return mesh, materials

    ## get part shape path.
    @property
    def part_obj_path(self):
        return Path(self.models_dir, 'part_seg.obj')
    @property
    def part_mtl_path(self):
        return Path(self.models_dir, 'part_seg.mtl')


    def get_topk_pairs(self, topk, max_dist):
        pairs = [p for p in self.pairs
                 if not p.exclude and p.distance < max_dist]
        pairs.sort(key=lambda p: p.distance)
        return pairs[:topk]

    @property
    def models_dir(self):
        return Path(self.data_path, 'models')

    @property
    def textures_dir(self):
        return Path(self.data_path, 'textures')

    @property
    def obj_path(self):
        return Path(self.models_dir, 'uvmapped.obj')

    @property
    def mtl_path(self):
        return Path(self.models_dir, 'uvmapped.mtl')

    @property
    def resized_obj_path(self):
        return Path(self.models_dir, 'uvmapped_v2.resized.obj')

    @property
    def uvmapped_obj_path(self):
        return Path(self.data_path, 'uvmapped.obj')

    @property
    def uvmapped_mtl_path(self):
        return Path(self.data_path, 'uvmapped.mtl')

    @property
    def data_path(self):
        # return Path(config.BLOB_ROOT, 'shapes_bed', str(self.id))   # chair? bed? or table?
        return Path(config.BLOB_ROOT, 'shapes', str(self.id))   # chair? bed? or table?

    def get_demo_angles(self):
        azimuth = math.pi/2 + math.pi - math.pi/4
        elevation = math.pi/2 - math.pi/12
        return azimuth + self.azimuth_correction, elevation

    def get_frontal_angles(self):
        azimuth = math.pi/2 + math.pi
        elevation = math.pi / 3
        return azimuth + self.azimuth_correction, elevation

    def get_frontal_camera(self):
        azimuth, elevation = self.get_frontal_angles()
        from thirdparty.toolbox.toolbox.cameras import spherical_coord_to_cam # EXEMPLAR_SUBST_MAP_NAME
        cam = spherical_coord_to_cam(
            fov=50.0,
            azimuth=azimuth,
            elevation=elevation, cam_dist=1.5)
        return cam

    def serialize(self):
        return {
            'id': int(self.id),
            'source': self.source,
            'source_id': self.source_id,
            'category': self.category,
            'exclude': self.exclude,
            'split_set': self.split_set,
            'meta': self.meta,
            'azimuth_correction': self.azimuth_correction,
        }

class partnet_Shape(db.Base, BlobDataMixin):
    # __tablename__ = 'partnet_shapes_table'    # chair? bed? or table?
    # __tablename__ = 'partnet_shapes_bed'    # chair? bed? or table?
    __tablename__ = 'partnet_shapes'    # chair? bed? or table?

    id = sa.Column(sa.Integer, primary_key=True)
    source = sa.Column(sa.String)
    source_id = sa.Column(sa.String)
    category = sa.Column(sa.String)
    exclude = sa.Column(sa.Boolean, default=False)
    split_set = sa.Column(sa.String)
    meta = sa.Column(JSONB, default={})
    azimuth_correction = sa.Column(sa.Float, default=0)

    pairs = relationship('ExemplarpartShapePair', back_populates='shape')
    result_annotations = relationship('part_ResultAnnotation')

    def load(self, size=100) -> object:
        mesh = wavefront.read_obj_file(self.obj_path)
        mesh.resize(size)
        materials = wavefront.read_mtl_file(self.mtl_path, mesh)
        return mesh, materials

    def load_shapenet(self, size=100) -> object:
        mesh = wavefront.read_obj_file(self.obj_shapenet_path)
        mesh.resize(size)
        materials = wavefront.read_mtl_file(self.mtl_shapenet_path, mesh)
        return mesh, materials

    def get_topk_pairs(self, topk, max_dist):
        pairs = [p for p in self.pairs
                 if not p.exclude and p.distance < max_dist]
        pairs.sort(key=lambda p: p.distance)
        return pairs[:topk]

    @property
    def models_dir(self):
        return Path(self.data_path, 'models')
    
    @property
    def shapenet_models_dir(self):
        return Path(self.data_shapenet_path, 'models')

    @property
    def textures_dir(self):
        return Path(self.data_path, 'textures')

    @property
    def obj_path(self):
        return Path(self.models_dir, 'uvmapped_v2.obj')

    @property
    def mtl_path(self):
        return Path(self.models_dir, 'uvmapped_v2.mtl')

    @property
    def obj_shapenet_path(self):
        return Path(self.shapenet_models_dir, 'uvmapped_v2.obj')

    @property
    def mtl_shapenet_path(self):
        return Path(self.shapenet_models_dir, 'uvmapped_v2.mtl')

    @property
    def resized_obj_path(self):
        return Path(self.models_dir, 'uvmapped_v2.obj')

    @property
    def uvmapped_obj_path(self):
        return Path(self.data_path, 'uvmapped.obj')

    @property
    def uvmapped_mtl_path(self):
        return Path(self.data_path, 'uvmapped.mtl')

    @property
    def data_path(self):
        # return Path(config.BLOB_ROOT, 'partnet_target_table', str(self.id))   # chair? bed? or table?
        # return Path(config.BLOB_ROOT, 'partnet_target_bed', str(self.id))   # chair? bed? or table?
        return Path(config.BLOB_ROOT, 'partnet_target', str(self.id))   # chair? bed? or table?

    @property
    def data_shapenet_path(self):
        # return Path(config.BLOB_ROOT, 'shapenet_target_table', str(self.id))   # chair? bed? or table?
        # return Path(config.BLOB_ROOT, 'shapenet_target_bed', str(self.id))   # chair? bed? or table?
        return Path(config.BLOB_ROOT, 'shapenet_target', str(self.id))

    def get_demo_angles(self):
        azimuth = math.pi/2 + math.pi - math.pi/4
        elevation = math.pi/2 - math.pi/12
        return azimuth + self.azimuth_correction, elevation

    def get_frontal_angles(self):
        azimuth = math.pi/2 + math.pi
        elevation = math.pi / 3
        return azimuth + self.azimuth_correction, elevation

    def get_frontal_camera(self):
        azimuth, elevation = self.get_frontal_angles()
        from thirdparty.toolbox.toolbox.cameras import spherical_coord_to_cam   # EXEMPLAR_SUBST_MAP_NAME
        cam = spherical_coord_to_cam(
            fov=50.0,
            azimuth=azimuth,
            elevation=elevation, cam_dist=1.5)
        return cam

    def serialize(self):
        return {
            'id': int(self.id),
            'source': self.source,
            'source_id': self.source_id,
            'category': self.category,
            'exclude': self.exclude,
            'split_set': self.split_set,
            'meta': self.meta,
            'azimuth_correction': self.azimuth_correction,
        }


class ExemplarpartShapePair(db.Base, BlobDataMixin):
    # __tablename__ = 'exemplar_partshape_pair_table1'   # chair? bed? or table1?
    # __tablename__ = 'exemplar_partshape_pair_bed'   # chair? bed? or table1?
    __tablename__ = 'exemplar_partshape_pair'   # chair? bed? or table1?

    id = sa.Column(sa.Integer, primary_key=True)
    exemplar_id = sa.Column(sa.ForeignKey(Exemplar.id), nullable=False)
    shape_id = sa.Column(sa.ForeignKey(partnet_Shape.id), nullable=False)

    azimuth = sa.Column(sa.Float)  # Theta.
    elevation = sa.Column(sa.Float)  # Phi.
    fov = sa.Column(sa.Float)
    distance = sa.Column(sa.Float)
    feature_type = sa.Column(sa.String)

    exemplar = relationship(Exemplar)
    shape = relationship(partnet_Shape, back_populates="pairs")

    num_substances = sa.Column(sa.Integer)
    num_segments = sa.Column(sa.Integer)

    params = sa.Column(sa.JSON)
    meta = sa.Column(JSONB, default={})

    exclude = sa.Column(sa.Boolean, default=False)
    used_for_eval = sa.Column(sa.Boolean, default=False)

    rank = sa.Column(sa.Integer)

    result_annotations = relationship('part_ResultAnnotation')

    @property
    def data_path(self):
        # return Path(config.BLOB_ROOT, 'pairs_partnet_table1', str(self.id))    # chair? bed? or table1?
        # return Path(config.BLOB_ROOT, 'pairs_partnet_bed', str(self.id))    # chair? bed? or table1?
        return Path(config.BLOB_ROOT, 'pairs_partnet', str(self.id))    # chair? bed? or table1?

    def get_camera(self):
        from thirdparty.toolbox.toolbox.cameras import spherical_coord_to_cam
        camera = spherical_coord_to_cam(self.fov, self.azimuth, self.elevation,
                                        cam_dist=2.0)
        return camera

    def serialize(self):
        keys = {
            'id',
            'exemplar_id',
            'shape_id',
            'azimuth',
            'elevation',
            'fov',
            'distance',
            'exclude',
            'rank',
        }
        return {k:v for k, v in self.__dict__.items() if k in keys}

class ExemplarShapePair(db.Base, BlobDataMixin):
    __tablename__ = 'exemplar_shape_pair'

    id = sa.Column(sa.Integer, primary_key=True)
    exemplar_id = sa.Column(sa.ForeignKey(Exemplar.id), nullable=False)
    shape_id = sa.Column(sa.ForeignKey(Shape.id), nullable=False)

    azimuth = sa.Column(sa.Float)  # Theta.
    elevation = sa.Column(sa.Float)  # Phi.
    fov = sa.Column(sa.Float)
    distance = sa.Column(sa.Float)
    feature_type = sa.Column(sa.String)

    exemplar = relationship(Exemplar)
    shape = relationship(Shape, back_populates="pairs")

    num_substances = sa.Column(sa.Integer)
    num_segments = sa.Column(sa.Integer)

    params = sa.Column(sa.JSON)
    meta = sa.Column(JSONB, default={})

    exclude = sa.Column(sa.Boolean, default=False)
    used_for_eval = sa.Column(sa.Boolean, default=False)

    rank = sa.Column(sa.Integer)

    result_annotations = relationship('ResultAnnotation')

    @property
    def data_path(self):
        return Path(config.BLOB_ROOT, 'pairs', str(self.id))

    def get_camera(self):
        from thirdparty.toolbox.toolbox.cameras import spherical_coord_to_cam
        camera = spherical_coord_to_cam(self.fov, self.azimuth, self.elevation,
                                        cam_dist=2.0)
        return camera

    def serialize(self):
        keys = {
            'id',
            'exemplar_id',
            'shape_id',
            'azimuth',
            'elevation',
            'fov',
            'distance',
            'exclude',
            'rank',
        }
        return {k:v for k, v in self.__dict__.items() if k in keys}


class MaterialType(enum.Enum):
    BLINN_PHONG = enum.auto()
    AITTALA_BECKMANN = enum.auto()
    POLIIGON = enum.auto()
    VRAY = enum.auto()
    PRINCIPLED = enum.auto()
    MDL = enum.auto()
    TEXTURE3D = enum.auto()
    CC0TEXTURE = enum.auto()
    TEXTUREHARVEN = enum.auto()
    SHARETEXTURE = enum.auto()


class Material(db.Base, BlobDataMixin, SerializeMixin):
    __tablename__ = 'materials10'

    id = sa.Column(sa.Integer, primary_key=True)
    type = sa.Column(sa.Enum(MaterialType))
    name = sa.Column(sa.String)
    author = sa.Column(sa.String)
    source = sa.Column(sa.String)
    source_id = sa.Column(sa.String)
    source_url = sa.Column(sa.String)

    substance = sa.Column(sa.String)
    default_scale = sa.Column(sa.Float)
    min_scale = sa.Column(sa.Float)
    max_scale = sa.Column(sa.Float)
    spatially_varying = sa.Column(sa.Boolean)
    enabled = sa.Column(sa.Boolean)
    params = sa.Column(sa.JSON)

    @property
    def data_path(self):
        return Path(config.BLOB_ROOT, 'materials', str(self.id))

    def get_base_path(self):
        if self.type == MaterialType.AITTALA_BECKMANN:
            return Path(config.MATERIAL_DIR_AITTALA, self.substance, self.name)
        elif self.type == MaterialType.POLIIGON:
            return Path(config.MATERIAL_DIR_POLIIGON, self.substance, self.name)
        elif self.type == MaterialType.VRAY:
            return Path(config.MATERIAL_DIR_VRAY, self.substance,
                        self.params['raw_name'])
        elif self.type == MaterialType.MDL and self.source == 'adobe_stock':
            return Path(config.MATERIAL_DIR_ADOBE_STOCK, self.substance,
                        f'AdobeStock_{self.source_id}', f'{self.name}.mdl')
        else:
            return None

    def serialize(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.name,
            'author': self.author,
            'source': self.source,
            'source_id': self.source_id,
            'source_url': self.source_url,
            'substance': self.substance,
            'default_scale': self.default_scale,
            'spatially_varying': self.spatially_varying,
            'enabled': self.enabled,
            'params': {k: v for k, v in self.params.items() if 'hist' not in k},
        }


class SplitSet(enum.Enum):
    TRAIN = enum.auto()
    VALIDATION = enum.auto()
    EXCLUDE = enum.auto()


rendering_material_table = sa.Table(
    'rendering_materials', db.Base.metadata,
    sa.Column('rend_id', sa.Integer,
              sa.ForeignKey("renderings.id"),
              primary_key=True),
    sa.Column('material_id', sa.Integer,
              sa.ForeignKey(Material.id),
              primary_key=True)
)

class Envmap(db.Base, BlobDataMixin, SerializeMixin):
    __tablename__ = 'envmaps'

    id = sa.Column(sa.Integer, primary_key=True)
    source = sa.Column(sa.String)
    name = sa.Column(sa.String)
    azimuth = sa.Column(sa.Float, default=0.0)
    enabled = sa.Column(sa.Boolean, default=False)
    split_set = sa.Column(sa.String)

    @property
    def hdr_path(self):
        return self.data_path / 'hdr.exr'

    @property
    def preview_path(self):
        return self.data_path / 'preview.png'

    @property
    def data_path(self):
        return Path(config.BLOB_ROOT, 'envmaps', str(self.id))


class Rendering(db.Base):
    __tablename__ = 'renderings'

    id = sa.Column(sa.Integer, primary_key=True)
    dataset_name = sa.Column(sa.String(64), nullable=False)

    client = sa.Column(sa.String(64), nullable=False)
    epoch = sa.Column(sa.Integer, nullable=False)
    split_set = sa.Column(sa.Enum(SplitSet), nullable=False)
    pair_id = sa.Column(sa.ForeignKey(ExemplarShapePair.id), nullable=False)
    index = sa.Column(sa.Integer, nullable=False)
    prefix = sa.Column(sa.String(64), nullable=False)
    exclude = sa.Column(sa.Boolean, default=False)

    saturated_frac = sa.Column(sa.Float)
    rend_time = sa.Column(sa.Float)

    pair = relationship(ExemplarShapePair)
    materials = relationship("Material",
                             secondary=lambda: rendering_material_table,
                             backref='renderings')

    @property
    def path_prefix(self):
        return str(Path(
            config.BRDF_CLASSIFIER_DATASETS_DIR,
            self.dataset_name,
            f'client={self.client}',
            f'epoch={self.epoch:03d}',
            self.split_set.name.lower(),
            self.prefix
        ))

    def get_json_path(self):
        return Path(f'{self.path_prefix}.params.json')

    def load_params(self):
        with self.get_json_path().open('r') as f:
            return ujson.load(f)

    def get_ldr_path(self, shape, fmt='.jpg'):
        return Path(f'{self.path_prefix}.ldr.{shape[0]}x{shape[1]}{fmt}')

    def get_segment_vis_path(self, shape):
        return Path(f'{self.path_prefix}.segment_vis.{shape[0]}x{shape[1]}.png')

    def get_segment_map_path(self, shape):
        return Path(f'{self.path_prefix}.segment_map.{shape[0]}x{shape[1]}.png')


class ResultSet(db.Base):
    __tablename__ = 'result_sets'

    id = sa.Column(sa.Integer, primary_key=True)

    snapshot_name = sa.Column(sa.String(64), nullable=True)
    model_name = sa.Column(sa.String(512), nullable=True)
    inference_name = sa.Column(sa.String(64), nullable=False)

    @property
    def inference_dir(self):
        if self.model_name is not None:
            return (config.BRDF_CLASSIFIER_DIR_REMOTE / 'inference'
                    / self.snapshot_name / self.model_name / self.inference_name)
        else:
            return (config.BRDF_CLASSIFIER_DIR_REMOTE / 'inference'
                    / self.inference_name)

    @property
    def rendering_dir(self):
        return self.inference_dir / 'renderings-calibrated'

    @property
    def cropped_rendering_dir(self):
        return self.inference_dir / 'renderings-calibrated-cropped'

    @property
    def diagonal_rendering_dir(self):
        return self.inference_dir / 'renderings-diagonal'

    def get_rendering_path(self, pair_id):
        path = self.cropped_rendering_dir / f'{pair_id}.inferred.0000.jpg'
        if path.exists():
            return path
        path = self.rendering_dir / f'{pair_id}.inferred.0000.png'
        if path.exists():
            return path
        path = self.diagonal_rendering_dir / f'{pair_id}.inferred.0000.jpg'
        if path.exists():
            return path


@total_ordering
class ResultQuality(enum.Enum):
    good = enum.auto()
    acceptable = enum.auto()
    wrong_ambiguous = enum.auto()
    wrong_alignment = enum.auto()
    wrong_material = enum.auto()
    wrong_color = enum.auto()
    under_segmented = enum.auto()
    over_segmented = enum.auto()
    bad_shape = enum.auto()
    bad_exemplar = enum.auto()
    limitation = enum.auto()
    not_sure = enum.auto()

    def __gt__(self, other):
        return self.value > other.value


@total_ordering
class ResultAnnotation(db.Base):
    __tablename__ = 'result_annotations1'

    id = sa.Column(sa.Integer, primary_key=True)
    username = sa.Column(sa.String(64), nullable=False)
    category = sa.Column(sa.String(64), nullable=False)
    date_updated = sa.Column(sa.DateTime, onupdate=func.now())

    shape_id = sa.Column(sa.ForeignKey(Shape.id), nullable=False)
    pair_id = sa.Column(sa.ForeignKey(ExemplarShapePair.id), nullable=False)
    result_set_id = sa.Column(sa.ForeignKey(ResultSet.id), nullable=False)

    @property
    def quality(self):
        return ResultQuality[self.category]

    def __gt__(self, other):
        return self.quality.value > other.quality.value

    def serialize(self):
        return dict(
            id=self.id,
            username=self.username,
            category=self.category,
            shape_id=self.shape_id,
            pair_id=self.pair_id,
            result_set_id=self.result_set_id,
            date_updated=self.date_updated.isoformat(),
        )

@total_ordering
class part_ResultAnnotation(db.Base):
    __tablename__ = 'result_annotations'

    id = sa.Column(sa.Integer, primary_key=True)
    username = sa.Column(sa.String(64), nullable=False)
    category = sa.Column(sa.String(64), nullable=False)
    date_updated = sa.Column(sa.DateTime, onupdate=func.now())

    shape_id = sa.Column(sa.ForeignKey(partnet_Shape.id), nullable=False)
    pair_id = sa.Column(sa.ForeignKey(ExemplarpartShapePair.id), nullable=False)
    result_set_id = sa.Column(sa.ForeignKey(ResultSet.id), nullable=False)

    @property
    def quality(self):
        return ResultQuality[self.category]

    def __gt__(self, other):
        return self.quality.value > other.quality.value

    def serialize(self):
        return dict(
            id=self.id,
            username=self.username,
            category=self.category,
            shape_id=self.shape_id,
            pair_id=self.pair_id,
            result_set_id=self.result_set_id,
            date_updated=self.date_updated.isoformat(),
        )