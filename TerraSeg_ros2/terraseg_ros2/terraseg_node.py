import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from terraseg import TerraSegPredictor

from .pointcloud_conversion import (
    pointcloud2_to_xyz_tensor,
    xyz_and_labels_to_pointcloud2,
)


class TerraSegNode(Node):
    """
    ROS2 node that runs TerraSeg ground segmentation on an incoming LiDAR stream.

    Subscribes to:
        - ``<input_topic>`` (``sensor_msgs/PointCloud2``) : Incoming LiDAR scans in the
          TerraSeg-standardized frame (z = 0 approximately ground-aligned, +x forward).

    Publishes:
        - ``<output_topic>`` (``sensor_msgs/PointCloud2``) : The same XYZ points plus an
          added uint8 ``label`` field (0 = ground, 1 = non-ground).

    Parameters (declared with sensible defaults; override via launch YAML):
        - ``variant`` (str) : ``"B"`` or ``"S"``. Default ``"S"``.
        - ``checkpoint_path`` (str) : Either a local filesystem path to a trained
          ``best.pth``, or a Hugging Face URI of the form
          ``hf://<user>/<repo>/<filename>`` (e.g.
          ``hf://TedLentsch/TerraSeg/terraseg_s.pth``). Required.
        - ``hf_revision`` (str) : Optional revision (branch, tag, or commit hash) when
          loading from Hugging Face. Default ``""`` (latest on ``main``).
        - ``decision_threshold`` (float) : Sigmoid threshold. ``-1.0`` defers to the value
          stored in the checkpoint (recommended).
        - ``grid_size`` (float) : PTv3 voxel grid size. Unit: meters. Default ``0.05``.
        - ``input_topic`` (str) : Input PointCloud2 topic. Default ``/lidar/points``.
        - ``output_topic`` (str) : Output labeled PointCloud2 topic. Default
          ``/terraseg/segmented``.
        - ``device`` (str) : Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
          Default ``"cuda:0"``.
        - ``compile_model`` (bool) : Wrap the model with ``torch.compile`` for an extra
          inference speedup. Adds a one-time compilation cost at startup. Default ``True``.

    TerraSeg runs in FP32. Lower-precision dtypes (BF16, FP16) are not supported because
    PTv3's sparse-convolution path becomes numerically unstable at reduced precision.
    """

    def __init__(self):
        super().__init__("terraseg_node")

        # Declare parameters.
        self.declare_parameter("variant", "S")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("hf_revision", "")
        self.declare_parameter("decision_threshold", -1.0)
        self.declare_parameter("grid_size", 0.05)
        self.declare_parameter("input_topic", "/lidar/points")
        self.declare_parameter("output_topic", "/terraseg/segmented")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("compile_model", True)

        variant = str(self.get_parameter("variant").value)
        checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        hf_revision = str(self.get_parameter("hf_revision").value) or None
        decision_threshold = float(self.get_parameter("decision_threshold").value)
        self.grid_size = float(self.get_parameter("grid_size").value)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        device_str = str(self.get_parameter("device").value)
        compile_model = bool(self.get_parameter("compile_model").value)

        if not checkpoint_path:
            raise RuntimeError(
                "Parameter 'checkpoint_path' is required (local file path or 'hf://...' URI). "
                "Set it in the launch YAML."
            )

        # Build predictor.
        self.get_logger().info(
            f"Loading TerraSeg-{variant} from '{checkpoint_path}' onto '{device_str}' "
            f"(compile={compile_model})..."
        )
        self.predictor = TerraSegPredictor(
            variant=variant,
            checkpoint_path=checkpoint_path,
            device=device_str,
            decision_thres=(decision_threshold if decision_threshold >= 0.0 else None),
            compile_model=compile_model,
            hf_revision=hf_revision,
        )
        self.get_logger().info(
            f"Predictor ready. decision threshold={self.predictor.decision_thres:.3f}."
        )

        # Subscriber + publisher.
        self.sub = self.create_subscription(
            PointCloud2, input_topic, self.on_pointcloud, qos_profile=10
        )
        self.pub = self.create_publisher(PointCloud2, output_topic, qos_profile=10)
        self.get_logger().info(
            f"Subscribed to '{input_topic}', publishing on '{output_topic}'."
        )

    def on_pointcloud(
        self,
        msg: PointCloud2,
    ) -> None:
        """
        Per-scan callback: parse PointCloud2, run TerraSeg, publish labeled PointCloud2.

        Args:
            msg (sensor_msgs.msg.PointCloud2) : Incoming LiDAR scan.
        """
        t_start = time.monotonic()

        device = self.predictor.device
        coord = pointcloud2_to_xyz_tensor(msg=msg, device=device)
        if coord.shape[0] == 0:
            self.get_logger().warn("Received empty PointCloud2; skipping.")
            return

        pred_labels = self.predictor.predict(coord=coord, grid_size=self.grid_size)

        out_msg = xyz_and_labels_to_pointcloud2(
            coord=coord, pred_labels=pred_labels, header=msg.header
        )
        self.pub.publish(out_msg)

        dt_ms = 1000.0 * (time.monotonic() - t_start)
        self.get_logger().debug(
            f"Scan handled: {coord.shape[0]} points, {dt_ms:.1f} ms end-to-end."
        )


def main(args=None) -> None:
    """
    ROS2 entry point.

    Args:
        args (list or None) : Forwarded to :func:`rclpy.init`. Default: ``None``.
    """
    rclpy.init(args=args)
    node = TerraSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
