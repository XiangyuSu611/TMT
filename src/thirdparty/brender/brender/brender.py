

class Brender:
    _instance = None

    def __init__(self):
        self.initialized = False
        if Brender._instance:
            raise RuntimeError('There can only be one app instance.')

        Brender._instance = self

    def init(self, do_reset=True):
        import bpy
        # Import default scene and settings.
        bpy.ops.wm.read_factory_settings()

        if do_reset:
            # Clear data.
            for bpy_data_iter in (
                    bpy.data.objects,
                    bpy.data.meshes,
                    bpy.data.lamps,
                    bpy.data.cameras,
                    bpy.data.materials,
            ):
                for id_data in bpy_data_iter:
                    bpy_data_iter.remove(id_data, do_unlink=True)

        self.initialized = True
