import json
import logging

import numpy as np
from flask import Blueprint, render_template, request

from gravel.webutils import image_to_png_response, make_response
from .jsd import JSDRenderer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
blueprint = Blueprint('rendkit', __name__,
                      template_folder='templates',
                      static_folder='static')


@blueprint.route('/')
def index():
    return 'rendkit'


@blueprint.route('/render_jsd', methods=['GET'])
def render_jsd_get():
    return render_template('render_jsd.html')


@blueprint.route('/render_jsd', methods=['POST'])
def render_jsd_post():
    try:
        jsd_file = request.files['jsd_file']
    except KeyError:
        message = 'JSD file required.'
        return make_response(422, 'missing_parameters', message)

    uv_scale = float(request.form.get('uv_scale', 1.0))
    logger.info("Rendering with UV Scale {}".format(uv_scale))

    jsd_obj = json.loads(jsd_file.read().decode())

    with JSDRenderer(jsd_obj, gamma=2.2, ssaa=0) as renderer:
        for renderable in renderer.scene.renderables:
            renderable.set_uv_scale(uv_scale)
        image = renderer.render_to_image()

    return image_to_png_response(np.clip(image, 0, 1))
