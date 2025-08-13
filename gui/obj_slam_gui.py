afrom gui.slam_gui import SLAM_GUI
import torch
import open3d as o3d
import imgviz
import open3d.visualization.gui as gui
#import open3d.visualization.rendering as rendering
from utils.logging_utils import Log
from utils.datahandle_utils import move_gaussianpacket_to_gpu

from gui.gui_utils import (
    GaussianPacket,
    Packet_vis2main,
    create_frustum,
    cv_gl,
    get_latest_queue,
)


class OBJ_SLAM_GUI(SLAM_GUI):
    def __init__(self, params_gui=None):
        super().__init__(params_gui)

    def scene_update(self):
        self.receive_data(self.q_main2vis)
        self.render_gui()

    def receive_data(self, q):
        if q is None:
            return

        gaussian_packet = get_latest_queue(q)

        if gaussian_packet is None:
            return

        move_gaussianpacket_to_gpu(gaussian_packet)
        if gaussian_packet.has_gaussians:
            self.gaussian_cur = gaussian_packet
            self.output_info.text = "Number of Gaussians: {}".format(
                self.gaussian_cur.get_xyz.shape[0]
            )
            self.init = True

        if gaussian_packet.current_frame is not None:
            frustum = self.add_camera(
                gaussian_packet.current_frame, name="current", color=[0, 1, 0]
            )
            if self.followcam_chbox.checked:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

        if gaussian_packet.keyframe is not None:
            name = "keyframe_{}".format(gaussian_packet.keyframe.uid)
            frustum = self.add_camera(
                gaussian_packet.keyframe, name=name, color=[0, 0, 1]
            )

        if gaussian_packet.keyframes is not None:
            for keyframe in gaussian_packet.keyframes:
                name = "keyframe_{}".format(keyframe.uid)
                frustum = self.add_camera(keyframe, name=name, color=[0, 0, 1])

        if gaussian_packet.kf_window is not None:
            self.kf_window = gaussian_packet.kf_window
            self._on_kf_window_chbox(is_checked=self.kf_window_chbox.checked)

        if gaussian_packet.gtcolor is not None:
            rgb = torch.clamp(gaussian_packet.gtcolor, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(rgb)
            self.in_rgb_widget.update_image(rgb)

        if gaussian_packet.gtdepth is not None:
            depth = gaussian_packet.gtdepth
            depth = imgviz.depth2rgb(
                depth, min_value=0.1, max_value=5.0, colormap="jet"
            )
            depth = torch.from_numpy(depth)
            depth = torch.permute(depth, (2, 0, 1)).float()
            depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            rgb = o3d.geometry.Image(depth)
            self.in_depth_widget.update_image(rgb)

        if gaussian_packet.finish:
            Log("Received terminate signal", tag="GUI")
            # clean up the pipe
            while not self.q_main2vis.empty():
                self.q_main2vis.get()
            while not self.q_vis2main.empty():
                self.q_vis2main.get()
            self.q_vis2main = None
            self.q_main2vis = None
            self.process_finished = True

def run(params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = OBJ_SLAM_GUI(params_gui)
    app.run()