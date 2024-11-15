import os
import random

import numpy as np
import pandas as pd
from PIL import Image

from img_size import IMG_SIZES

SDD_cols = [
    "trackId",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "frame",
    "lost",
    "occluded",
    "generated",
    "label",
]


def make_df(scene_path):
    scene = (
        scene_path.split("/")[-3] + "_" + scene_path.split("/")[-2].replace("video", "")
    )
    scene_df = pd.read_csv(scene_path, header=0, names=SDD_cols, delimiter=" ")
    # Calculate center point of bounding box
    scene_df["x"] = (scene_df["xmax"] + scene_df["xmin"]) / 2
    scene_df["y"] = (scene_df["ymax"] + scene_df["ymin"]) / 2
    scene_df = scene_df[scene_df["label"] == "Pedestrian"]  # drop non-pedestrians
    scene_df = scene_df[scene_df["lost"] == 0]  # drop lost samples
    scene_df = scene_df.drop(
        columns=[
            "xmin",
            "xmax",
            "ymin",
            "ymax",
            "occluded",
            "generated",
            "label",
            "lost",
        ]
    )
    scene_df["sceneId"] = scene
    # new unique id by combining scene_id and track_id
    scene_df["rec&trackId"] = [
        recId + "_" + str(trackId).zfill(4)
        for recId, trackId in zip(scene_df.sceneId, scene_df.trackId)
    ]
    return scene_df


def create_template(template_size=501, sigma=3):
    template_center = np.array([template_size // 2, template_size // 2])
    template_xycoords = generate_coordsmap(template_size, template_size)
    templates = np.exp(
        -0.5
        * np.linalg.norm(
            template_xycoords - template_center.reshape(1, 1, 2),
            axis=-1,
        )
        ** 2
        / sigma**2
    )
    templates /= templates.sum()

    return templates


def pos2gaussmap_numpy(pos, sigma, xycoords, normalize=False):
    hmap = np.exp(
        -0.5 * np.linalg.norm(xycoords - pos.reshape(1, 1, 2), axis=-1) ** 2 / sigma**2
    )
    if normalize:
        hmap = hmap / hmap.max()

    return hmap


def traj2gaussmap_numpy(
    traj,
    fmap_size,
    sigma,
    xycoords,
    method="sum",
    do_trunc=False,
    normalize=False,
):
    # traj shape = (N, 2)
    traj = traj.copy()
    width, height = fmap_size
    # shape = (height, width, 2=(x, y))
    hmap = [
        pos2gaussmap_numpy(traj[person_i], sigma, xycoords, normalize)
        for person_i in range(traj.shape[0])
    ]

    if len(hmap) == 0:
        hmap += [np.zeros((height, width))]
    if method == "max":
        hmap = np.maximum.reduce(np.array(hmap))
    elif method == "average":
        hmap = np.array(hmap).mean(axis=0)
    else:
        raise ValueError("Bug")

    return hmap.astype(np.float32)


def generate_coordsmap(width, height):
    """Generate coodinates map from (width, height)

    Args:
        width (int): The width of map
        height (int): The height of map

    Returns:
        [ndarray]: Generated map, shape = (height, width, 2=(x, y))
    """
    # create coordinates array
    """
    >>> x = np.arange(3)
    >>> y = np.arange(2)
    >>> xx, yy = np.meshgrid(x, y)
    >>> xx
    array([[0, 1, 2],
        [0, 1, 2]])
    >>> yy
    array([[0, 0, 0],
        [1, 1, 1]])
    >>> np.concatenate(
        [xx.reshape(-1,1), yy.reshape(-1,1)], axis=-1
        ).reshape(2, 3, 2)
        array([[[0, 0],
                [1, 0],
                [2, 0]],

            [[0, 1],
                [1, 1],
                [2, 1]]])
    """
    # height => x, width => y
    x, y = np.arange(height), np.arange(width)
    # note that meshgrid(x, y) will count up y first.
    # xx => height, yy => width
    xx, yy = np.meshgrid(y, x)
    # shape = (height, width, 2=(x, y))
    xycoords = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=-1).reshape(
        height, width, 2
    )

    return xycoords


def generate_heatmap(
    df_frame,
    xycoords,
    fmap_size=(80, 80),
    orig_max_size=(1422, 1422),
    sigma=1,
    normalize=False,
):
    x = df_frame["x"].values
    y = df_frame["y"].values
    traj = np.stack([x, y], axis=1)
    resize = fmap_size[0] / orig_max_size[0]
    traj = resize * traj.copy()
    gaussmap = traj2gaussmap_numpy(
        traj,
        fmap_size,
        xycoords=xycoords,
        sigma=sigma,
        method="sum",
        normalize=normalize,
    )

    return Image.fromarray(gaussmap)


def get_trajectories(root, dataset, split, seq_len=20, obs_frames=8):
    if dataset in ["eth", "univ", "hotel", "zara1", "zara2", "raw"]:
        root = os.path.join(root, "eth_ucy/processed_data", dataset)
    else:
        root = os.path.join(root, dataset)
    split_dir = os.path.join(root, split)
    d_img_sizes = IMG_SIZES[dataset]
    if dataset == "stanford":
        trajectories = get_trajectories_stanford(
            split_dir, d_img_sizes=d_img_sizes, seq_len=seq_len, obs_frames=obs_frames
        )
    if dataset in ["eth", "univ", "hotel", "zara1", "zara2", "raw"]:
        trajectories = get_trajectories_ethucy(
            split_dir, d_img_sizes=d_img_sizes, seq_len=seq_len, obs_frames=obs_frames
        )
    if dataset == "ind-time-split" or dataset == "ind-location-split":
        trajectories = get_trajectories_ind(
            split_dir, d_img_sizes=d_img_sizes, seq_len=seq_len, obs_frames=obs_frames
        )
    if dataset == "fdst":
        trajectories = get_trajectories_fdst(
            split_dir, d_img_sizes=d_img_sizes, seq_len=seq_len, obs_frames=obs_frames
        )
    if dataset == "vscrowd":
        trajectories = get_trajectories_vscrowd(
            split_dir, d_img_sizes=d_img_sizes, seq_len=seq_len, obs_frames=obs_frames
        )
    if dataset == "ht21":
        trajectories = get_trajectories_ht21(
            split_dir, d_img_sizes=d_img_sizes, seq_len=seq_len, obs_frames=obs_frames
        )

    if dataset == "jrdb":
        trajectories = get_trajectories_jrdb(
            split_dir, d_img_sizes=d_img_sizes, seq_len=seq_len, obs_frames=obs_frames
        )

    return trajectories


def get_trajectories_stanford(
    split_dir,
    d_img_sizes,
    step=12,
    seq_len=20,
    stride=1,
    obs_frames=8,
):
    dir_names = [
        dir_name
        for dir_name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, dir_name))
    ]
    trajectories = []
    for dir_name in dir_names:
        anotation_path = os.path.join(
            split_dir,
            dir_name,
            "annotations.txt",
        )
        df = make_df(anotation_path)  # trackId, frame, x, y, sceneId, rec&trackId
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1)]
        frames = frames[::step]  # downsample 30 fps -> 2.5 fps
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len]
            if len(curr_seq_frames) != seq_len:
                continue
            curr_seq_data = np.concatenate(frame_data[idx : idx + seq_len], axis=0)
            frames_in_curr_seq = np.unique(curr_seq_data[:, 1])
            if len(frames_in_curr_seq) != seq_len:
                continue
            peds_in_curr_seq = np.unique(curr_seq_data[:, 0])
            trajectory = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 0] == ped_id, :]
                trajectory.append(curr_ped_seq)

            trajectory = np.concatenate(trajectory, axis=0)
            trajectories.append(
                {
                    "dataset": split_dir.split("/")[-2],
                    "dir_name": dir_name,
                    "trajectory": trajectory[:, 0:4],
                    "img_size": np.array(d_img_sizes[dir_name]),
                    "frames": curr_seq_frames,
                }
            )

    return trajectories


def get_trajectories_ethucy(
    split_dir,
    d_img_sizes,
    seq_len=20,
    stride=1,
    obs_frames=8,
):
    file_names = os.listdir(split_dir)
    step = 10  # downsample 25 fps -> 2.5 fps
    trajectories = []
    for file_name in file_names:
        dir_name = extract_scene_name_from_file_name(file_name)
        df = pd.read_pickle(
            os.path.join(split_dir, file_name)
        )  # trackId, frame, x, y, sceneId, rec&trackId
        df["trackId"] = pd.to_numeric(df["trackId"]).astype("int")
        df["frame"] = pd.to_numeric(df["frame"]).astype("int")
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1, step)]
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len]
            if len(curr_seq_frames) != seq_len:
                continue
            curr_seq_data = np.concatenate(frame_data[idx : idx + seq_len], axis=0)
            frames_in_curr_seq = np.unique(curr_seq_data[:, 1])
            if len(frames_in_curr_seq) != seq_len:
                continue
            peds_in_curr_seq = np.unique(curr_seq_data[:, 0])
            trajectory = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 0] == ped_id, :]
                trajectory.append(curr_ped_seq)
            trajectory = np.concatenate(trajectory, axis=0)
            trajectories.append(
                {
                    "dataset": split_dir.split("/")[-2],
                    "dir_name": dir_name,
                    "trajectory": trajectory[:, 0:4],
                    "img_size": np.array(d_img_sizes[dir_name]),
                    "frames": curr_seq_frames,
                }
            )

    return trajectories


def get_trajectories_ind(
    split_dir,
    d_img_sizes,
    seq_len=20,
    stride=1,
    obs_frames=8,
):
    file_names = os.listdir(split_dir)
    step = 10  # downsample 25 fps -> 2.5 fps
    trajectories = []
    for file_name in file_names:
        dir_name = file_name.split(".")[0]
        df = pd.read_csv(
            os.path.join(split_dir, file_name)
        )  # 'trackId', 'frame', 'xCenterVis', 'yCenterVis', 'recordingId'
        df["trackId"] = pd.to_numeric(df["trackId"]).astype("int")
        df["frame"] = pd.to_numeric(df["frame"]).astype("int")
        df["x"] = pd.to_numeric(df["xCenterVis"])
        df["y"] = pd.to_numeric(df["yCenterVis"])
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1, step)]
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len]
            if len(curr_seq_frames) != seq_len:
                continue
            curr_seq_data = np.concatenate(frame_data[idx : idx + seq_len], axis=0)
            frames_in_curr_seq = np.unique(curr_seq_data[:, 1])
            if len(frames_in_curr_seq) != seq_len:
                continue
            peds_in_curr_seq = np.unique(curr_seq_data[:, 0])
            trajectory = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 0] == ped_id, :]
                trajectory.append(curr_ped_seq)
            trajectory = np.concatenate(trajectory, axis=0)
            trajectories.append(
                {
                    "dataset": split_dir.split("/")[-2],
                    "dir_name": dir_name,
                    "trajectory": trajectory[:, 0:4],
                    "img_size": np.array(d_img_sizes[dir_name]),
                    "frames": curr_seq_frames,
                }
            )

    return trajectories


def get_trajectories_fdst(
    split_dir, d_img_sizes, obs_frames=8, seq_len=20, stride=1, step=5
):
    file_names = os.listdir(split_dir)
    trajectories = []
    for file_name in file_names:
        dir_name = file_name.split(".")[0]
        df = pd.read_csv(
            os.path.join(split_dir, file_name)
        )  # 'trackId'(fake), 'frame', 'x', 'y'
        df["frame"] = pd.to_numeric(df["frame"]).astype("int")
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1)]
        seq_len_ds = seq_len * step  # downsample 30 fps -> 6 fps
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len_ds) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len_ds : step]
            if len(curr_seq_frames) != seq_len:
                continue
            curr_seq_data = np.concatenate(
                frame_data[idx : idx + seq_len_ds : step], axis=0
            )  # 20 frames
            frames_in_curr_seq = np.unique(curr_seq_data[:, 1])
            if len(frames_in_curr_seq) != seq_len:
                continue
            trajectories.append(
                {
                    "dataset": split_dir.split("/")[-2],
                    "dir_name": dir_name,
                    "trajectory": curr_seq_data[:, 0:4],
                    "img_size": np.array(d_img_sizes[dir_name]),
                    "frames": curr_seq_frames,
                }
            )

    return trajectories


def get_trajectories_vscrowd(
    split_dir, d_img_sizes, obs_frames=8, seq_len=20, stride=1, step=5
):
    file_names = os.listdir(split_dir)
    trajectories = []
    for file_name in file_names:
        dir_name = file_name.split(".")[0]
        df = pd.read_csv(
            os.path.join(split_dir, file_name)
        )  # 'trackId'(fake), 'frame', 'x', 'y'
        df["frame"] = pd.to_numeric(df["frame"]).astype("int")
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1)]
        seq_len_ds = seq_len * step  # downsample 25fps -> 5fps
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len_ds) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len_ds : step]
            if len(curr_seq_frames) != seq_len:
                continue
            curr_seq_data = np.concatenate(
                frame_data[idx : idx + seq_len_ds : step], axis=0
            )  # 20 frames
            frames_in_curr_seq = np.unique(curr_seq_data[:, 1])
            if len(frames_in_curr_seq) != seq_len:
                continue
            trajectories.append(
                {
                    "dataset": split_dir.split("/")[-2],
                    "dir_name": dir_name,
                    "trajectory": curr_seq_data[:, 0:4],
                    "img_size": np.array(d_img_sizes[dir_name]),
                    "frames": curr_seq_frames,
                }
            )

    return trajectories


def get_trajectories_ht21(
    split_dir, d_img_sizes, obs_frames=8, seq_len=20, stride=1, step=10
):
    file_names = os.listdir(split_dir)
    trajectories = []
    for file_name in file_names:
        dir_name = file_name.split(".")[0]
        df = pd.read_csv(
            os.path.join(split_dir, file_name)
        )  # 'trackId'(fake), 'frame', 'x', 'y'
        df["frame"] = pd.to_numeric(df["frame"]).astype("int")
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1)]
        seq_len_ds = seq_len * step  # downsample 25fps -> 2.5fps
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len_ds) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len_ds : step]
            if len(curr_seq_frames) != seq_len:
                continue
            curr_seq_data = np.concatenate(
                frame_data[idx : idx + seq_len_ds : step], axis=0
            )  # 20 frames
            frames_in_curr_seq = np.unique(curr_seq_data[:, 1])
            if len(frames_in_curr_seq) != seq_len:
                continue
            trajectories.append(
                {
                    "dataset": split_dir.split("/")[-2],
                    "dir_name": dir_name,
                    "trajectory": curr_seq_data[:, 0:4],
                    "img_size": np.array(d_img_sizes[dir_name]),
                    "frames": curr_seq_frames,
                }
            )

    return trajectories


def get_trajectories_jrdb(
    split_dir, d_img_sizes, obs_frames=8, seq_len=20, stride=1, step=6
):
    file_names = os.listdir(split_dir)
    trajectories = []
    for file_name in file_names:
        dir_name = file_name.split(".")[0]
        df = pd.read_csv(
            os.path.join(split_dir, file_name)
        )  # 'trackId'(fake), 'frame', 'x', 'y'
        df["frame"] = pd.to_numeric(df["frame"]).astype("int")
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])
        frames = sorted(np.unique(df["frame"]).tolist())
        frames = [i for i in range(frames[0], frames[-1] + 1, step)]
        frame_data = []
        for frame in frames:
            df_frame = df.loc[df["frame"] == int(frame), :]
            frame_data.append(df_frame)
        frames_len = len(frames)
        n_chunk = (frames_len - seq_len) // stride + 1
        for idx in range(0, n_chunk * stride + 1, stride):
            curr_seq_frames = frames[idx : idx + seq_len]
            curr_seq_data = np.concatenate(frame_data[idx : idx + seq_len], axis=0)
            if len(curr_seq_data) == 0 or len(curr_seq_frames) != seq_len:
                continue
            peds_in_curr_seq = np.unique(curr_seq_data[:, 0])
            frames_in_curr_seq = np.unique(curr_seq_data[:, 1])
            if len(frames_in_curr_seq) != seq_len:
                continue
            trajectory = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 0] == ped_id, :]
                trajectory.append(curr_ped_seq)
            trajectory = np.concatenate(trajectory, axis=0)
            trajectories.append(
                {
                    "dataset": split_dir.split("/")[-2],
                    "dir_name": dir_name,
                    "trajectory": trajectory[:, 0:4],
                    "img_size": np.array(d_img_sizes[dir_name]),
                    "frames": curr_seq_frames,
                }
            )

    return trajectories





def extract_scene_name_from_file_name(file_name):
    if "eth" in file_name:
        scene = "eth"
    elif "hotel" in file_name:
        scene = "hotel"
    elif "uni_examples" in file_name:
        scene = "uni_examples"
    elif "students001" in file_name:
        scene = "students001"
    elif "students002" in file_name:
        scene = "students002"
    elif "students003" in file_name:
        scene = "students003"
    elif "zara01" in file_name:
        scene = "zara1"
    elif "zara02" in file_name:
        scene = "zara2"
    elif "zara03" in file_name:
        scene = "zara3"

    return scene


def generate_heatmap_from_templates(
    traj, fmap_size=80, img_max_size=1422, templates=None, normalize=False, method="max"
):
    x = traj[:, 2]
    y = traj[:, 3]
    traj = np.stack([x, y], axis=1)
    resize = fmap_size / img_max_size
    traj = resize * traj.copy()
    index = (
        (traj[:, 0] >= 0)
        & (traj[:, 0] <= fmap_size)
        & (traj[:, 1] >= 0)
        & (traj[:, 1] <= fmap_size)
    )
    traj = traj[index]
    gaussmap = traj2gaussmap_from_templates(
        traj,
        (fmap_size, fmap_size),
        templates,
        method=method,
        normalize=normalize,
    )

    return Image.fromarray(gaussmap)


def pos2gaussmap_from_templates(pos, center, templates, fmap_size, normalize=False):
    x_low = center[0] - int(pos[0])
    x_up = x_low + fmap_size[0]
    y_low = center[1] - int(pos[1])
    y_up = y_low + fmap_size[1]
    hmap = templates[y_low:y_up, x_low:x_up]
    if normalize:
        hmap = hmap / hmap.max()

    return hmap


def traj2gaussmap_from_templates(
    traj, fmap_size, templates=None, method="sum", normalize=False
):
    center = np.array([templates.shape[1] // 2, templates.shape[0] // 2])
    height, width = fmap_size
    hmaps = [np.zeros((height, width))]
    for person_i in range(traj.shape[0]):
        if len(traj[person_i]) != 0:
            hmaps.append(
                pos2gaussmap_from_templates(
                    traj[person_i], center, templates, fmap_size, normalize
                )
            )

    if method == "max":
        hmap = np.maximum.reduce(np.array(hmaps))
    elif method == "average":
        hmap = np.array(hmaps).mean(axis=0)
    elif method == "sum":
        hmap = np.array(hmaps).sum(axis=0)
    else:
        raise ValueError("Bug")

    return hmap.astype(np.float32)


def randomnoisedposition(traj, ratio=0.25, sigma=1):
    n = traj.shape[0]
    n_noise = int(n * ratio)
    indexes = random.sample(range(n), n_noise)
    mean = np.array([0, 0])
    sigma = np.array([sigma, sigma])
    covariance = np.diag(sigma**2)
    values = np.random.multivariate_normal(mean, covariance, n_noise)
    noise = np.zeros(traj.shape)
    noise[indexes] = values

    return traj + noise


def randommissingposition(traj, ratio=0.25):
    n = traj.shape[0]
    n_missing = int(n * ratio)
    indexes = random.sample(range(n), n - n_missing)
    traj = traj[indexes]

    return traj


"""stanford
train: 19262 sequences
test: 9633 sequences
-----------------------------
eth
train: 4470 sequences
test: 956 sequences
-----------------------------
univ
train: 4475 sequences
test: 947 sequences
-----------------------------
hotel
train: 4174 sequences
test: 1312 sequences
-----------------------------
zara1
train: 4534 sequences
test: 872 sequences
-----------------------------
zara2
train: 4402 sequences
test: 1033 sequences
-----------------------------
ind-time-split
train: 53243 sequences
test: 15024 sequences
-----------------------------
fdst
train: 3120 sequences
test: 2080 sequences
-----------------------------
vscrowd
train: 1224 sequences
test: 1117 sequences
-----------------------------
ht21
train: 4949 sequences
test: 4733 sequences
-----------------------------
crowdflow
train: 761 sequences
test: 353 sequences
-----------------------------"""

if __name__ == "__main__":
    for dataset in [
        "stanford",
        "eth",
        "univ",
        "hotel",
        "zara1",
        "zara2",
        "ind-time-split",
        "fdst",
        "vscrowd",
        "ht21",
        "crowdflow",
    ]:
        print(f"dataset: {dataset}")
        trajectories = get_trajectories(
            root="data",
            dataset=dataset,
            split="train",
            seq_len=20,
            obs_frames=8,
        )
        print(f"train: {len(trajectories)} sequences")
        trajectories = get_trajectories(
            root="data",
            dataset=dataset,
            split="test",
            seq_len=20,
            obs_frames=8,
        )
        print(f"test: {len(trajectories)} sequences")
        print("-----------------------------")
