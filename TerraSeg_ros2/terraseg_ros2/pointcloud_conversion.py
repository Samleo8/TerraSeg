import numpy as np
import torch
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


# Names of the XYZ fields expected on the input PointCloud2.
_XYZ_FIELDS: tuple[str, ...] = ("x", "y", "z")

# Name of the label field added on the output PointCloud2.
LABEL_FIELD_NAME: str = "label"


def pointcloud2_to_xyz_tensor(
    msg: PointCloud2,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a ``sensor_msgs/PointCloud2`` message into an (N, 3) torch tensor of XYZ coordinates.

    Args:
        msg (sensor_msgs.msg.PointCloud2) : Incoming point cloud message.
        device (torch.device) : Target device for the returned tensor.

    Returns:
        coord (torch.Tensor) : (N, 3) per-point XYZ coordinates in float32 on ``device``.
            Unit: meters.
    """
    structured = point_cloud2.read_points(msg, field_names=_XYZ_FIELDS, skip_nans=True)
    structured_np = np.asarray(structured)
    if structured_np.size == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device)
    coord_np = np.stack(
        [structured_np["x"], structured_np["y"], structured_np["z"]], axis=1
    ).astype(np.float32, copy=False)
    return torch.from_numpy(coord_np).to(device, non_blocking=True)


def xyz_and_labels_to_pointcloud2(
    coord: torch.Tensor,
    pred_labels: torch.Tensor,
    header,
) -> PointCloud2:
    """
    Build a ``sensor_msgs/PointCloud2`` carrying XYZ coordinates plus a per-point ``label`` field.

    Args:
        coord (torch.Tensor) : (N, 3) XYZ coordinates. Unit: meters.
        pred_labels (torch.Tensor) : (N,) per-point binary labels in {0, 1} (0 = ground,
            1 = non-ground).
        header (std_msgs.msg.Header) : Header to copy onto the output (frame_id + stamp).

    Returns:
        out (sensor_msgs.msg.PointCloud2) : Output point cloud with four fields (``x``, ``y``,
            ``z``, ``label``) and the given header.
    """
    assert coord.ndim == 2 and coord.shape[1] == 3, (
        f"coord must have shape (N, 3), got {tuple(coord.shape)}!"
    )
    assert pred_labels.shape[0] == coord.shape[0], (
        "coord and pred_labels must agree on N!"
    )

    coord_np = coord.detach().cpu().numpy().astype(np.float32, copy=False)
    labels_np = pred_labels.detach().cpu().numpy().astype(np.uint8, copy=False)

    structured = np.zeros(
        coord_np.shape[0],
        dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("label", np.uint8)],
    )
    structured["x"] = coord_np[:, 0]
    structured["y"] = coord_np[:, 1]
    structured["z"] = coord_np[:, 2]
    structured["label"] = labels_np

    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name=LABEL_FIELD_NAME, offset=12, datatype=PointField.UINT8, count=1),
    ]
    return point_cloud2.create_cloud(header=header, fields=fields, points=structured)
