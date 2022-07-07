import base64
import io
import urllib.parse

from scipy.misc import toimage


def image_to_base64(image):
    f = io.BytesIO()
    toimage(image).save(f, format='png')
    f.seek(0)
    base64_image = base64.b64encode(f.getvalue())
    return 'data:image/png;base64,{}'.format(
        urllib.parse.quote(base64_image))
