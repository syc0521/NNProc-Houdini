import io
import os
import h5py
import glooey
import numpy as np

import pyglet
import trimesh
import trimesh.voxel
import trimesh.viewer
import simple_3dviz
from PIL import Image


class ClickableImage(glooey.Image):
    def __init__(self, parent_widget):
        super(ClickableImage, self).__init__()
        self.parent_widget = parent_widget

    def on_mouse_press(self, x, y, button, modifiers):
        i = (655 - y) // 64
        j = (x - 5) // 138
        self.parent_widget.callback(i * 10 + j)
        print(i, j)


def get_widgets(parent_widget):
    with h5py.File(os.path.join('predictions', 'shelf_from_voxel_direct.hdf5'), 'r') as f:
        predictions = f['voxel_preds'][:].squeeze(1)
        targets = f['voxels'][:].squeeze(1)

    summary_img = Image.new('RGBA', (1370, 320))
    for i in range(5):
        for j in range(10):
            index = str(i) + str(j)
            true_image = render_voxels(targets[int(index)])
            pred_image = render_voxels(predictions[int(index)])
            summary_img.paste(true_image, (j * 138, i * 64))
            summary_img.paste(pred_image, (j * 138 + 64, i * 64))
    image_widget = ClickableImage(parent_widget)
    with io.BytesIO() as f:
        summary_img.save(f, format='PNG')
        image_widget.image = pyglet.image.load(filename=None, file=f)
    prediction_widgets = []
    target_widgets = []
    for i in range(50):
        prediction_widgets.append(get_voxel_widget(predictions[i]))
        target_widgets.append(get_voxel_widget(targets[i]))
    return prediction_widgets, target_widgets, image_widget


def get_voxel_widget(voxels):
    scene = trimesh.Scene()
    geom = trimesh.voxel.VoxelGrid(voxels)
    scene.add_geometry(geom.as_boxes())
    widget = trimesh.viewer.SceneWidget(scene)
    return widget


def render_voxels(voxels):
    scene = simple_3dviz.Scene(background=(1.0, 1.0, 1.0, 1.0), size=(64, 64))
    scene.add(simple_3dviz.Mesh.from_voxel_grid(voxels=voxels > 0))
    scene.camera_position = (0, 0, 2)
    scene.up_vector = (0, 1, 0)
    scene.render()
    image = Image.fromarray(scene.frame)
    return image


class Application:
    def __init__(self):
        # create window with padding
        self.prediction_widgets, self.target_widgets, self.image_widget = get_widgets(self)
        self.width, self.height = 1380, 660
        window = self._create_window(width=self.width, height=self.height)

        gui = glooey.Gui(window)

        self.hbox = glooey.HBox()
        vbox = glooey.VBox()
        self.hbox.set_padding(5)

        self.scene_widget1 = self.target_widgets[0]
        self.hbox.add(self.scene_widget1)

        self.scene_widget2 = self.prediction_widgets[0]
        self.hbox.add(self.scene_widget2)

        vbox.add(self.image_widget)
        vbox.add(self.hbox)

        gui.add(vbox)

        #pyglet.clock.schedule_interval(self.callback, 1. / 20)
        pyglet.app.run()

    def callback(self, index):
        #self.hbox.clear()
        #self.hbox.add(self.target_widgets[index])
        #self.hbox.add(self.prediction_widgets[index])
        self.hbox.replace(self.hbox.get_children()[0], self.target_widgets[index])
        self.hbox.replace(self.hbox.get_children()[1], self.prediction_widgets[index])

    def _create_window(self, width, height):
        try:
            config = pyglet.gl.Config(sample_buffers=1,
                                      samples=4,
                                      depth_size=24,
                                      double_buffer=True)
            window = pyglet.window.Window(config=config,
                                          width=width,
                                          height=height)
        except pyglet.window.NoSuchConfigException:
            config = pyglet.gl.Config(double_buffer=True)
            window = pyglet.window.Window(config=config,
                                          width=width,
                                          height=height)

        @window.event
        def on_key_press(symbol, modifiers):
            if modifiers == 0:
                if symbol == pyglet.window.key.Q:
                    window.close()

        return window


if __name__ == '__main__':
    np.random.seed(0)
    Application()
