import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from data.volume_io import load_dicom_series


def _write_series(folder, n=6, rows=8, cols=10, dz=2.5):
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    for k in range(n):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"s{k}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, k * dz]
        ds.PixelSpacing = [0.7, 0.8]
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 0
        ds.RescaleSlope, ds.RescaleIntercept = 1, -1000
        ds.PixelData = np.full((rows, cols), 1000 + k, dtype=np.uint16).tobytes()
        ds.save_as(str(folder / f"s{k}.dcm"), write_like_original=False)


def test_dicom_series_shape_spacing_order(tmp_path):
    _write_series(tmp_path)
    vol, aff, sp = load_dicom_series(tmp_path)
    assert vol.shape == (8, 10, 6)
    assert np.allclose(sp, (0.7, 0.8, 2.5))
    assert np.allclose(vol[0, 0, :], np.arange(6))          # sorted by position, slope/intercept applied
    assert np.allclose(np.linalg.norm(aff[:3, :3], axis=0), (0.7, 0.8, 2.5))   # voxel size along each array axis
