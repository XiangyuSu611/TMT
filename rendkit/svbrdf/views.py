import logging

from flask import Blueprint, abort, render_template
from scipy.misc import imread

from gravel.webutils import image_to_png_response, make_404, make_response
from svbrdf import models

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
blueprint = Blueprint('svbrdf', __name__,
                      template_folder='templates',
                      static_folder='static')


def get_rendering_or_abort(rendering_id):
    rendering = models.SVBRDFRendering.query.get(rendering_id)
    if rendering is None:
        abort(404)
    return rendering


@blueprint.route('/renderings/<rendering_id>/image.png', methods=['GET'])
def rendering_image(rendering_id):
    rendering = get_rendering_or_abort(rendering_id)
    return image_to_png_response(rendering.load_image())


@blueprint.route('/renderings/<rendering_id>/feature_patches.png', methods=['GET'])
def rendering_feature_patches(rendering_id):
    rendering = get_rendering_or_abort(rendering_id)
    feature_dao = models.SVBRDFRenderingFeature.query \
        .filter_by(rendering_id=rendering.id)\
        .first()
    return image_to_png_response(imread(feature_dao.patch_vis_path))
