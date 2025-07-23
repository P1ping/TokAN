from pathlib import Path


def parse_embed_path(path: str) -> bool:
    """Parse data path which is either a path to
    1. a NPY file
    2. an ARK file with slicing info: "[ark_path]:[offset]"
    (Adapted from fairseq.data.audio.audio_utils.parse_path)

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
          is_ark (bool): whether the path is an ARK file
    """
    suffix = Path(path).suffix
    if suffix == ".npy":
        _path, is_ark = path, False
    else:
        _path, *slice_ptr = path.split(":")
        new_suffix = Path(_path).suffix
        assert len(slice_ptr) == 1, f"Invalid ARK path: {path}"
        assert new_suffix == ".ark", f"Invalid ARK path: {path}"
        is_ark = True
    if not Path(_path).is_file():
        raise FileNotFoundError(f"File not found: {_path}")
    return is_ark
